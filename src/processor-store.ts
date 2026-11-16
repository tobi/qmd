import { closeSync, constants, fstatSync, openSync, readSync, realpathSync, statSync } from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { createHash } from "node:crypto";
import type { Database } from "./db.js";
import { syncDocumentMetadata } from "./metadata-store.js";
import { METADATA_EXTRACTION_VERSION } from "./metadata.js";
import {
  insertContent, insertDocument, updateDocument, updateDocumentTitle, deactivateDocument,
  extractTitle, scanWriteBatch, cleanupOrphanedContent,
  type ReindexResult, type ReindexProgress,
} from "./store.js";
import {
  executeProcessor, processorFingerprint, snapshotProcessor, ProcessorFailure, PROCESSOR_INPUT_LIMIT,
  type DocumentProcessor, type ProcessorDocument,
} from "./processors.js";

export function initializeProcessorSchema(db: Database): void {
  db.transaction(() => db.exec(`
    CREATE TABLE IF NOT EXISTS processor_specs (collection TEXT PRIMARY KEY, spec TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS processor_modes (collection TEXT PRIMARY KEY);
    CREATE TABLE IF NOT EXISTS processor_sources (
      collection TEXT NOT NULL, source TEXT NOT NULL, fingerprint TEXT NOT NULL, output_count INTEGER NOT NULL,
      PRIMARY KEY(collection, source)
    );
    CREATE TABLE IF NOT EXISTS processor_outputs (
      collection TEXT NOT NULL, source TEXT NOT NULL, path TEXT NOT NULL, hash TEXT NOT NULL, current INTEGER NOT NULL DEFAULT 1,
      PRIMARY KEY(collection, path)
    );
    CREATE INDEX IF NOT EXISTS idx_processor_outputs_source ON processor_outputs(collection, source);
  `)).immediate();
}

export function hasProcessorMode(db: Database, collection: string): boolean {
  return !!db.prepare("SELECT 1 FROM processor_modes WHERE collection = ?").get(collection);
}

export function requireProcessor(db: Database, collection: string, declared: boolean, processor?: DocumentProcessor): void {
  if (!processor && (declared || hasProcessorMode(db, collection))) {
    throw new Error("Collection processor required: restore the processor, or rebuild by removing and re-adding the collection.");
  }
  if (processor) {
    processorFingerprint(processor.fingerprint);
    if (typeof processor.process !== "function") throw new ProcessorFailure("PROCESSOR_INVALID");
  }
}

export function removeProcessorState(db: Database, collection: string): void {
  for (const table of ["processor_outputs", "processor_sources", "processor_modes", "processor_specs"]) {
    db.prepare(`DELETE FROM ${table} WHERE collection = ?`).run(collection);
  }
}

export function renameProcessorState(db: Database, oldName: string, newName: string): void {
  removeProcessorState(db, newName);
  for (const table of ["processor_outputs", "processor_sources", "processor_modes", "processor_specs"]) {
    db.prepare(`UPDATE ${table} SET collection = ? WHERE collection = ?`).run(newName, oldName);
  }
}

type OutputRow = { path: string; hash: string };
function outputs(db: Database, collection: string, source: string): OutputRow[] {
  return db.prepare("SELECT path, hash FROM processor_outputs WHERE collection = ? AND source = ?").all(collection, source) as OutputRow[];
}

function deleteOwnedOutput(db: Database, collection: string, source: string, path: string): void {
  db.prepare(`DELETE FROM documents WHERE collection = ? AND path = ? AND EXISTS (
    SELECT 1 FROM processor_outputs WHERE collection = ? AND source = ? AND path = ?
  )`).run(collection, path, collection, source, path);
  db.prepare("DELETE FROM processor_outputs WHERE collection = ? AND source = ? AND path = ?").run(collection, source, path);
}

function readSource(filepath: string): { bytes: Uint8Array; modifiedAt: string } {
  // A regular file can be replaced by a FIFO after enumeration: do not block in open.
  const fd = openSync(filepath, constants.O_RDONLY | constants.O_NONBLOCK);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile() || stat.size > PROCESSOR_INPUT_LIMIT) throw new ProcessorFailure("PROCESSOR_INPUT_LIMIT");
    // Read at most limit + 1, even if a file grows after fstat.
    const buffer = Buffer.alloc(PROCESSOR_INPUT_LIMIT + 1);
    let length = 0;
    while (length < buffer.length) {
      const n = readSync(fd, buffer, length, buffer.length - length, null);
      if (!n) break;
      length += n;
    }
    if (length > PROCESSOR_INPUT_LIMIT) throw new ProcessorFailure("PROCESSOR_INPUT_LIMIT");
    return { bytes: new Uint8Array(buffer.subarray(0, length)), modifiedAt: stat.mtime.toISOString() };
  } finally { closeSync(fd); }
}

/** Called only after successful root validation and complete enumeration. */
export async function reindexProcessedFiles(
  db: Database, batch: ReturnType<typeof scanWriteBatch>, root: string, collection: string,
  files: string[], processor: DocumentProcessor,
  options?: { signal?: AbortSignal; onProgress?: (info: ReindexProgress) => void },
): Promise<ReindexResult> {
  const owned = snapshotProcessor(processor);
  const result: ReindexResult = { indexed: 0, updated: 0, unchanged: 0, removed: 0, orphanedCleaned: 0, skipped: 0, skippedFiles: [], metadataErrors: 0 };
  const rootAvailable = (): boolean => {
    try { return statSync(root).isDirectory(); } catch { return false; }
  };
  const unavailable = (): ReindexResult => {
    batch.commit();
    result.skipped++;
    result.skippedFiles.push({ file: root, code: "ROOT_MISSING" });
    return result;
  };
  if (!rootAvailable()) return unavailable();
  batch.next();
  db.prepare("INSERT OR IGNORE INTO processor_modes(collection) VALUES (?)").run(collection);
  batch.commit();
  const seenSources = new Set<string>();
  let processed = 0;
  const skip = (source: string, code: string): void => {
    result.skippedFiles.push({ file: source, code });
    result.skipped++;
    processed++;
    options?.onProgress?.({ file: source, current: processed, total: files.length });
  };
  for (const source of files) {
    seenSources.add(source);
    // Never hold a writer transaction while reading or awaiting external code.
    batch.commit();
    let documents: readonly ProcessorDocument[];
    let input: ReturnType<typeof readSource>;
    let old = outputs(db, collection, source);
    try {
      if (options?.signal?.aborted) throw new ProcessorFailure("PROCESSOR_ABORTED");
      const filepath = realpathSync(resolve(root, source));
      const rel = relative(realpathSync(root), filepath);
      if (isAbsolute(rel) || rel.split(/[\\/]/).includes("..")) throw new ProcessorFailure("OUTSIDE_COLLECTION");
      input = readSource(filepath);
    } catch (error) {
      skip(source, error instanceof ProcessorFailure ? error.code : "PROCESSOR_READ_FAILED");
      continue;
    }
    const modifiedAt = input.modifiedAt;
    const hash = createHash("sha256").update(input.bytes).digest("hex");
    const fingerprint = createHash("sha256").update(JSON.stringify([hash, owned.fingerprint, collection, source, 1, 1, METADATA_EXTRACTION_VERSION])).digest("hex");
    // SQL errors here are fatal storage errors, never processor/read failures.
    const cached = db.prepare("SELECT fingerprint, output_count FROM processor_sources WHERE collection = ? AND source = ?").get(collection, source) as { fingerprint: string; output_count: number } | undefined;
    const invalid = db.prepare(`SELECT 1 FROM processor_outputs o
      LEFT JOIN documents d ON d.collection = o.collection AND d.path = o.path
      LEFT JOIN document_metadata m ON m.document_id = d.id
      WHERE o.collection = ? AND o.source = ? AND
        (o.current != 1 OR d.id IS NULL OR d.active != 1 OR d.hash != o.hash OR m.extraction_version IS NULL OR m.extraction_version != ?) LIMIT 1`).get(collection, source, METADATA_EXTRACTION_VERSION);
    if (cached?.fingerprint === fingerprint && cached.output_count === old.length && !invalid) {
      result.unchanged += old.length;
      processed++;
      options?.onProgress?.({ file: source, current: processed, total: files.length });
      continue;
    }
    try {
      documents = await executeProcessor(owned, { version: 1, collection, relativePath: source, hash, modifiedAt, bytes: input.bytes }, options?.signal);
      if (options?.signal?.aborted) throw new ProcessorFailure("PROCESSOR_ABORTED");
    } catch (error) {
      skip(source, error instanceof ProcessorFailure ? error.code : "PROCESSOR_FAILED");
      continue;
    }
    // Collision checks also cover inactive documents and unowned ordinary rows.
    const collision = documents.some(doc => {
      const path = `${source}/${doc.key}`;
      const owner = db.prepare("SELECT source FROM processor_outputs WHERE collection = ? AND path = ?").get(collection, path) as { source: string } | undefined;
      const existing = db.prepare("SELECT 1 FROM documents WHERE collection = ? AND path = ?").get(collection, path);
      return (owner && owner.source !== source) || (existing && !owner);
    });
    if (collision) { skip(source, "PROCESSOR_PATH_COLLISION"); continue; }

    batch.next();
    db.exec("SAVEPOINT processor_source");
    const counts = { indexed: 0, updated: 0, unchanged: 0, removed: 0, metadataErrors: 0 };
    try {
      // Recheck ownership inside the write transaction, not just before it.
      old = db.prepare("SELECT path, hash FROM processor_outputs WHERE collection = ? AND source = ?").all(collection, source) as OutputRow[];
      const nextPaths = new Set<string>();
      for (const doc of documents) {
        const path = `${source}/${doc.key}`;
        nextPaths.add(path);
        const owner = db.prepare("SELECT source FROM processor_outputs WHERE collection = ? AND path = ?").get(collection, path) as { source: string } | undefined;
        const existing = db.prepare("SELECT id, hash, title, active FROM documents WHERE collection = ? AND path = ?").get(collection, path) as { id: number; hash: string; title: string; active: number } | undefined;
        if ((owner && owner.source !== source) || (existing && !owner)) throw new Error("Processor ownership changed during publication");
        const hash = createHash("sha256").update(doc.markdown).digest("hex");
        const title = extractTitle(doc.markdown, path);
        const now = new Date().toISOString();
        insertContent(db, hash, doc.markdown, now);
        let id: number;
        if (existing) {
          id = existing.id;
          if (existing.hash !== hash || !existing.active) {
            updateDocument(db, id, title, hash, modifiedAt);
            db.prepare("UPDATE documents SET active = 1 WHERE id = ?").run(id);
            counts.updated++;
          } else if (existing.title !== title) {
            updateDocumentTitle(db, id, title, modifiedAt);
            counts.updated++;
          } else counts.unchanged++;
        } else {
          id = insertDocument(db, collection, path, title, hash, now, modifiedAt);
          counts.indexed++;
        }
        const extraction = syncDocumentMetadata(db, id, doc.markdown, path);
        if (extraction?.error) counts.metadataErrors++;
        db.prepare(`INSERT INTO processor_outputs(collection, source, path, hash) VALUES (?, ?, ?, ?)
          ON CONFLICT(collection, path) DO UPDATE SET hash = excluded.hash, current = 1`).run(collection, source, path, hash);
      }
      for (const previous of old) {
        if (nextPaths.has(previous.path)) continue;
        const active = db.prepare("SELECT 1 FROM documents WHERE collection = ? AND path = ? AND active = 1").get(collection, previous.path);
        deleteOwnedOutput(db, collection, source, previous.path);
        if (active) counts.removed++;
      }
      // A collection may have indexed this physical source before processors were enabled.
      const raw = db.prepare(`SELECT 1 FROM documents d WHERE d.collection = ? AND d.path = ? AND d.active = 1
        AND NOT EXISTS (SELECT 1 FROM processor_outputs o WHERE o.collection = d.collection AND o.path = d.path)`).get(collection, source);
      if (raw) { deactivateDocument(db, collection, source); counts.removed++; }
      db.prepare("DELETE FROM file_sync_state WHERE collection = ? AND relative_path = ?").run(collection, source);
      db.prepare(`INSERT INTO processor_sources(collection, source, fingerprint, output_count) VALUES (?, ?, ?, ?)
        ON CONFLICT(collection, source) DO UPDATE SET fingerprint = excluded.fingerprint, output_count = excluded.output_count`).run(collection, source, fingerprint, documents.length);
      db.exec("RELEASE processor_source");
    } catch (error) {
      db.exec("ROLLBACK TO processor_source; RELEASE processor_source");
      throw error;
    }
    for (const key of ["indexed", "updated", "unchanged", "removed", "metadataErrors"] as const) result[key] += counts[key];
    processed++;
    options?.onProgress?.({ file: source, current: processed, total: files.length });
  }
  if (options?.signal?.aborted) { batch.commit(); return result; }
  if (!rootAvailable()) return unavailable();
  const sources = db.prepare("SELECT source FROM processor_sources WHERE collection = ?").all(collection) as { source: string }[];
  for (const { source } of sources) {
    if (seenSources.has(source)) continue;
    batch.next();
    db.exec("SAVEPOINT processor_source");
    let removed = 0;
    try {
      for (const child of outputs(db, collection, source)) {
        if (db.prepare("SELECT 1 FROM documents WHERE collection = ? AND path = ? AND active = 1").get(collection, child.path)) removed++;
        deleteOwnedOutput(db, collection, source, child.path);
      }
      db.prepare("DELETE FROM processor_sources WHERE collection = ? AND source = ?").run(collection, source);
      db.exec("RELEASE processor_source");
    } catch (error) {
      db.exec("ROLLBACK TO processor_source; RELEASE processor_source");
      throw error;
    }
    result.removed += removed;
  }
  // Retire missing physical rows left from an earlier ordinary collection scan.
  const ordinary = db.prepare(`SELECT d.path FROM documents d WHERE d.collection = ? AND d.active = 1
    AND NOT EXISTS (SELECT 1 FROM processor_outputs o WHERE o.collection = d.collection AND o.path = d.path)`).all(collection) as { path: string }[];
  for (const { path } of ordinary) {
    if (seenSources.has(path)) continue;
    batch.next();
    deactivateDocument(db, collection, path);
    db.prepare("DELETE FROM file_sync_state WHERE collection = ? AND relative_path = ?").run(collection, path);
    result.removed++;
  }
  batch.commit();
  result.orphanedCleaned = cleanupOrphanedContent(db);
  return result;
}
