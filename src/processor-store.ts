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

type SourceInput = { bytes: Uint8Array; modifiedAt: string };
function readSource(filepath: string): SourceInput {
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

type DocumentRow = { id: number; hash: string; title: string; active: number };

/**
 * Who may write one output path right now.
 *
 * - `unowned`: no ownership row and no document row, so publication creates both.
 * - `owned`: this source's ownership row; its document row may be missing or
 *   inactive (a previous publication retired it) and is republished.
 * - `foreign`: another source's ownership row, or a document left by a scan
 *   that ran before processors were enabled. Publication may enter neither.
 */
type PathOwnership =
  | { readonly kind: "unowned" }
  | { readonly kind: "owned"; readonly document: DocumentRow | undefined }
  | { readonly kind: "foreign" };

/**
 * Read who may write `path` right now. Called twice per output on purpose: once
 * outside the write transaction to avoid paying for it, and again inside it,
 * because the first reading is outside the lock.
 */
function ownershipOf(db: Database, collection: string, source: string, path: string): PathOwnership {
  const owner = db.prepare("SELECT source FROM processor_outputs WHERE collection = ? AND path = ?").get(collection, path) as { source: string } | undefined;
  const document = db.prepare("SELECT id, hash, title, active FROM documents WHERE collection = ? AND path = ?").get(collection, path) as DocumentRow | undefined;
  if (owner) return owner.source === source ? { kind: "owned", document } : { kind: "foreign" };
  return document ? { kind: "foreign" } : { kind: "unowned" };
}

/**
 * What inspecting one enumerated source concluded, and the only states a scan
 * may hand to publication. `publishable` replaces the source's children;
 * `reused` writes nothing; `skipped` keeps every child and the coverage row the
 * source already has, so a source that failed never loses documents that a
 * later scan would have to recreate.
 */
type SourceVerdict =
  | { readonly kind: "reused"; readonly unchanged: number }
  | { readonly kind: "skipped"; readonly code: string }
  | { readonly kind: "publishable"; readonly documents: readonly ProcessorDocument[]; readonly fingerprint: string; readonly modifiedAt: string };

type PublishableSource = Extract<SourceVerdict, { kind: "publishable" }>;

const skipped = (code: string): SourceVerdict => ({ kind: "skipped", code });

type SourceCounts = { indexed: number; updated: number; unchanged: number; removed: number; metadataErrors: number };

const SOURCE_SAVEPOINT = "processor_source";

/**
 * Run one source's writes in a savepoint inside the batch transaction the
 * caller opened. A source is published completely or not at all: its content,
 * documents, metadata and ownership rows move together.
 */
function withSourceSavepoint<T>(db: Database, write: () => T): T {
  db.exec(`SAVEPOINT ${SOURCE_SAVEPOINT}`);
  try {
    const value = write();
    db.exec(`RELEASE ${SOURCE_SAVEPOINT}`);
    return value;
  } catch (error) {
    rollbackSourceSavepoint(db);
    throw error;
  }
}

function rollbackSourceSavepoint(db: Database): void {
  // A statement that ends the whole transaction (RAISE(ROLLBACK), SQLITE_FULL,
  // SQLITE_IOERR, SQLITE_BUSY) discards the savepoint with it. The batch owns
  // that bookkeeping: its rollback finds no transaction and undoes nothing, so
  // undoing nothing here is already the correct outcome and the statement's own
  // error is the one that must escape.
  if (!db.inTransaction) return;
  try {
    db.exec(`ROLLBACK TO ${SOURCE_SAVEPOINT}; RELEASE ${SOURCE_SAVEPOINT}`);
  } catch { /* the batch rollback that follows discards whatever is left */ }
}

/**
 * Read one enumerated source and decide whether it may be published. Every way
 * of not obtaining trustworthy documents is a skip, never a partial
 * publication. SQL faults are not caught here: they are fatal storage errors,
 * not source failures.
 */
async function inspectSource(
  db: Database, collection: string, root: string, source: string,
  processor: Readonly<DocumentProcessor>, signal: AbortSignal | undefined,
): Promise<SourceVerdict> {
  if (signal?.aborted) return skipped("PROCESSOR_ABORTED");
  let input: SourceInput;
  try {
    const filepath = realpathSync(resolve(root, source));
    const rel = relative(realpathSync(root), filepath);
    if (isAbsolute(rel) || rel.split(/[\\/]/).includes("..")) return skipped("OUTSIDE_COLLECTION");
    input = readSource(filepath);
  } catch (error) {
    return skipped(error instanceof ProcessorFailure ? error.code : "PROCESSOR_READ_FAILED");
  }
  const hash = createHash("sha256").update(input.bytes).digest("hex");
  const fingerprint = createHash("sha256").update(JSON.stringify([hash, processor.fingerprint, collection, source, 1, 1, METADATA_EXTRACTION_VERSION])).digest("hex");
  const cached = db.prepare("SELECT fingerprint, output_count FROM processor_sources WHERE collection = ? AND source = ?").get(collection, source) as { fingerprint: string; output_count: number } | undefined;
  const invalid = db.prepare(`SELECT 1 FROM processor_outputs o
    LEFT JOIN documents d ON d.collection = o.collection AND d.path = o.path
    LEFT JOIN document_metadata m ON m.document_id = d.id
    WHERE o.collection = ? AND o.source = ? AND
      (o.current != 1 OR d.id IS NULL OR d.active != 1 OR d.hash != o.hash OR m.extraction_version IS NULL OR m.extraction_version != ?) LIMIT 1`).get(collection, source, METADATA_EXTRACTION_VERSION);
  if (cached?.fingerprint === fingerprint && cached.output_count === outputs(db, collection, source).length && !invalid) {
    return { kind: "reused", unchanged: cached.output_count };
  }
  try {
    const documents = await executeProcessor(processor, { version: 1, collection, relativePath: source, hash, modifiedAt: input.modifiedAt, bytes: input.bytes }, signal);
    if (signal?.aborted) return skipped("PROCESSOR_ABORTED");
    return { kind: "publishable", documents, fingerprint, modifiedAt: input.modifiedAt };
  } catch (error) {
    return skipped(error instanceof ProcessorFailure ? error.code : "PROCESSOR_FAILED");
  }
}

/**
 * Replace one source's children with the documents a `publishable` verdict
 * carries, atomically. Ownership is re-read inside the write transaction
 * because inspection ran outside it; a path that became `foreign` in between
 * throws, which rolls the savepoint back and leaves the batch to discard its
 * earlier writes.
 */
function publishSource(db: Database, collection: string, source: string, verdict: PublishableSource): SourceCounts {
  return withSourceSavepoint(db, () => {
    const counts: SourceCounts = { indexed: 0, updated: 0, unchanged: 0, removed: 0, metadataErrors: 0 };
    const previous = outputs(db, collection, source);
    const next = new Set<string>();
    for (const doc of verdict.documents) {
      const path = `${source}/${doc.key}`;
      next.add(path);
      const ownership = ownershipOf(db, collection, source, path);
      if (ownership.kind === "foreign") throw new Error("Processor ownership changed during publication");
      const hash = createHash("sha256").update(doc.markdown).digest("hex");
      const title = extractTitle(doc.markdown, path);
      const now = new Date().toISOString();
      insertContent(db, hash, doc.markdown, now);
      const existing = ownership.kind === "owned" ? ownership.document : undefined;
      let id: number;
      if (!existing) {
        id = insertDocument(db, collection, path, title, hash, now, verdict.modifiedAt);
        counts.indexed++;
      } else {
        id = existing.id;
        if (existing.hash !== hash || !existing.active) {
          updateDocument(db, id, title, hash, verdict.modifiedAt);
          db.prepare("UPDATE documents SET active = 1 WHERE id = ?").run(id);
          counts.updated++;
        } else if (existing.title !== title) {
          updateDocumentTitle(db, id, title, verdict.modifiedAt);
          counts.updated++;
        } else counts.unchanged++;
      }
      const extraction = syncDocumentMetadata(db, id, doc.markdown, path);
      if (extraction?.error) counts.metadataErrors++;
      db.prepare(`INSERT INTO processor_outputs(collection, source, path, hash) VALUES (?, ?, ?, ?)
        ON CONFLICT(collection, path) DO UPDATE SET hash = excluded.hash, current = 1`).run(collection, source, path, hash);
    }
    for (const stale of previous) {
      if (next.has(stale.path)) continue;
      const active = db.prepare("SELECT 1 FROM documents WHERE collection = ? AND path = ? AND active = 1").get(collection, stale.path);
      deleteOwnedOutput(db, collection, source, stale.path);
      if (active) counts.removed++;
    }
    // A collection may have indexed this physical source before processors were enabled.
    const raw = db.prepare(`SELECT 1 FROM documents d WHERE d.collection = ? AND d.path = ? AND d.active = 1
      AND NOT EXISTS (SELECT 1 FROM processor_outputs o WHERE o.collection = d.collection AND o.path = d.path)`).get(collection, source);
    if (raw) { deactivateDocument(db, collection, source); counts.removed++; }
    db.prepare("DELETE FROM file_sync_state WHERE collection = ? AND relative_path = ?").run(collection, source);
    db.prepare(`INSERT INTO processor_sources(collection, source, fingerprint, output_count) VALUES (?, ?, ?, ?)
      ON CONFLICT(collection, source) DO UPDATE SET fingerprint = excluded.fingerprint, output_count = excluded.output_count`).run(collection, source, verdict.fingerprint, verdict.documents.length);
    return counts;
  });
}

/**
 * Release a source the finished enumeration no longer sees. Retiring is guarded
 * by ownership, so a path that another source has since claimed keeps its
 * document; only this source's own rows are dropped.
 */
function retireSource(db: Database, collection: string, source: string): number {
  return withSourceSavepoint(db, () => {
    let removed = 0;
    for (const child of outputs(db, collection, source)) {
      if (db.prepare("SELECT 1 FROM documents WHERE collection = ? AND path = ? AND active = 1").get(collection, child.path)) removed++;
      deleteOwnedOutput(db, collection, source, child.path);
    }
    db.prepare("DELETE FROM processor_sources WHERE collection = ? AND source = ?").run(collection, source);
    return removed;
  });
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
  const seen = new Set<string>();
  let processed = 0;
  const advance = (source: string, skipCode?: string): void => {
    if (skipCode !== undefined) {
      result.skippedFiles.push({ file: source, code: skipCode });
      result.skipped++;
    }
    processed++;
    options?.onProgress?.({ file: source, current: processed, total: files.length });
  };
  for (const source of files) {
    seen.add(source);
    // Never hold a writer transaction while reading or awaiting external code.
    batch.commit();
    const verdict = await inspectSource(db, collection, root, source, owned, options?.signal);
    if (verdict.kind === "skipped") { advance(source, verdict.code); continue; }
    if (verdict.kind === "reused") {
      result.unchanged += verdict.unchanged;
      advance(source);
      continue;
    }
    // Cheap ownership look before paying for a write transaction, including
    // inactive documents and unowned ordinary rows; publication repeats it
    // inside the transaction.
    if (verdict.documents.some(doc => ownershipOf(db, collection, source, `${source}/${doc.key}`).kind === "foreign")) {
      advance(source, "PROCESSOR_PATH_COLLISION");
      continue;
    }
    batch.next();
    const counts = publishSource(db, collection, source, verdict);
    for (const key of ["indexed", "updated", "unchanged", "removed", "metadataErrors"] as const) result[key] += counts[key];
    advance(source);
  }
  // Both exits below leave an interrupted or offline scan in place: `seen` is
  // only a complete claim about the root once the enumeration has finished over
  // a root that is still present, so nothing is retired from it.
  if (options?.signal?.aborted) { batch.commit(); return result; }
  if (!rootAvailable()) return unavailable();
  const sources = db.prepare("SELECT source FROM processor_sources WHERE collection = ?").all(collection) as { source: string }[];
  for (const { source } of sources) {
    if (seen.has(source)) continue;
    batch.next();
    result.removed += retireSource(db, collection, source);
  }
  // Retire missing physical rows left from an earlier ordinary collection scan.
  const ordinary = db.prepare(`SELECT d.path FROM documents d WHERE d.collection = ? AND d.active = 1
    AND NOT EXISTS (SELECT 1 FROM processor_outputs o WHERE o.collection = d.collection AND o.path = d.path)`).all(collection) as { path: string }[];
  for (const { path } of ordinary) {
    if (seen.has(path)) continue;
    batch.next();
    deactivateDocument(db, collection, path);
    db.prepare("DELETE FROM file_sync_state WHERE collection = ? AND relative_path = ?").run(collection, path);
    result.removed++;
  }
  batch.commit();
  result.orphanedCleaned = cleanupOrphanedContent(db);
  return result;
}
