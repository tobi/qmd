import { describe, test, expect, afterEach } from "vitest";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, type QMDStore } from "../src/index.js";
import { setDefaultLlamaCpp } from "../src/llm.js";
import { DEFAULT_EMBED_MODEL, hashContent, insertContent, insertDocument, insertEmbedding, searchVec } from "../src/store.js";
import { openDatabase, loadSqliteVec, type Database, type SQLiteValue } from "../src/db.js";
import { replaceDocumentMetadata } from "../src/metadata-store.js";
import { METADATA_EXTRACTION_VERSION } from "../src/metadata.js";

const cleanup: Array<() => Promise<void>> = [];
afterEach(async () => {
  setDefaultLlamaCpp(null);
  for (const close of cleanup.splice(0).reverse()) await close();
});

function fakeLlm() {
  return {
    embedModelName: DEFAULT_EMBED_MODEL,
    complete: false,
    batchStarted: false,
    batches: [] as string[][],
    async tokenize(text: string) { return Array.from(text, c => c.charCodeAt(0)); },
    async detokenize(tokens: readonly number[]) { return String.fromCharCode(...tokens); },
    async embed(_text: string) {
      return this.complete || !this.batchStarted ? { embedding: [1, 0, 0], model: DEFAULT_EMBED_MODEL } : null;
    },
    async embedBatch(texts: string[]) {
      this.batchStarted = true;
      this.batches.push([...texts]);
      return texts.map((_text, i) => this.complete || i === 0 ? { embedding: [1, 0, 0], model: DEFAULT_EMBED_MODEL } : null);
    },
  };
}

async function fixture(): Promise<{ store: QMDStore; llm: ReturnType<typeof fakeLlm> }> {
  const dir = await mkdtemp(join(tmpdir(), "qmd-checkpoint-"));
  cleanup.push(() => rm(dir, { recursive: true, force: true }));
  const a = join(dir, "a"), b = join(dir, "b");
  await mkdir(a); await mkdir(b);
  const body = "# Long document\n\n" + Array.from({ length: 180 }, (_, i) => `Line ${String(i).padStart(4, "0")}: resumable checkpoint data.\n`).join("");
  await writeFile(join(a, "long.md"), body);
  await writeFile(join(b, "long.md"), body);
  const store = await createStore({ dbPath: join(dir, "index.sqlite"), config: { collections: { a: { path: a }, b: { path: b } } } });
  cleanup.push(() => store.close());
  const llm = fakeLlm();
  setDefaultLlamaCpp(llm as any);
  store.internal.llm = llm as any;
  await store.update();
  return { store, llm };
}

function rows(store: QMDStore): any[] { return store.internal.db.prepare("SELECT * FROM content_vectors ORDER BY seq").all(); }
function checkpointColumn(store: QMDStore): void {
  const columns = store.internal.db.prepare("PRAGMA table_info(content_vectors)").all() as Array<{ name: string }>;
  if (!columns.some(c => c.name === "chunk_plan_hash")) store.internal.db.exec("ALTER TABLE content_vectors ADD COLUMN chunk_plan_hash TEXT NOT NULL DEFAULT ''");
}

async function insertVectorDoc(store: QMDStore, collection: string, path: string, body: string, total: number, checkpoint: boolean, vector: number[] = [1, 0, 0]): Promise<string> {
  const db = store.internal.db, now = new Date().toISOString(), hash = await hashContent(body);
  insertContent(db, hash, body, now);
  const id = insertDocument(db, collection, path, path, hash, now, now);
  replaceDocumentMetadata(db, id, { metadata: { eligible: true }, extractionVersion: METADATA_EXTRACTION_VERSION });
  insertEmbedding(db, hash, 0, 0, new Float32Array(vector), DEFAULT_EMBED_MODEL, now, total);
  if (checkpoint) db.prepare("UPDATE content_vectors SET chunk_plan_hash = ? WHERE hash = ?").run("synthetic-checkpoint", hash);
  return hash;
}

describe("opt-in embedding checkpoints", () => {
  test("default embedding retains the existing incomplete-document cleanup", async () => {
    const { store } = await fixture();
    const result = await store.embed();
    expect(result.errors).toBeGreaterThan(0);
    expect(rows(store)).toHaveLength(0);
    expect(result.chunksEmbedded).toBe(0);
  });

  test("retains completed chunks and resumes only the missing actual chunk plan", async () => {
    const { store, llm } = await fixture();
    const first = await store.embed({ resume: true });
    const saved = rows(store);
    expect(saved).toHaveLength(1);
    expect(first.chunksEmbedded).toBe(1);
    expect(saved[0].chunk_plan_hash).toMatch(/^[a-f0-9]{64}$/);
    const expected = saved[0].total_chunks;
    expect(expected).toBeGreaterThan(1);
    const completedText = llm.batches[0]![0]!;
    llm.complete = true;
    const second = await store.embed({ resume: true });
    expect(rows(store)).toHaveLength(expected);
    expect(second.chunksEmbedded).toBe(expected - 1);
    expect(llm.batches.slice(1).flat()).not.toContain(completedText);
    expect(llm.batches.slice(1).flat()).toHaveLength(expected - 1);
    for (const scope of [undefined, "a", "b", ["a", "b"]] as const) {
      expect((await searchVec(store.internal.db, "q", DEFAULT_EMBED_MODEL, 10, scope, undefined, [1, 0, 0])).length).toBeGreaterThan(0);
    }
  });

  for (const column of ["chunk_plan_hash", "model", "embed_fingerprint", "pos", "total_chunks"] as const) {
    test(`rejects a retained checkpoint with changed ${column}`, async () => {
      const { store, llm } = await fixture();
      await store.embed({ resume: true });
      expect(rows(store)).toHaveLength(1);
      const firstText = llm.batches[0]![0]!;
      const invalid = column === "pos" ? 1 : column === "total_chunks" ? 999 : "changed";
      store.internal.db.prepare(`UPDATE content_vectors SET ${column} = ?`).run(invalid);
      llm.complete = true;
      await store.embed({ resume: true });
      expect(llm.batches.slice(1).flat()).toContain(firstText);
      const fresh = rows(store);
      expect(fresh.length).toBe(fresh[0].total_chunks);
    });
  }

  test("a default run discards a previous opt-in partial checkpoint", async () => {
    const { store, llm } = await fixture();
    await store.embed({ resume: true });
    expect(rows(store)).toHaveLength(1);
    llm.batchStarted = false;
    await store.embed();
    expect(rows(store)).toHaveLength(0);
    expect(llm.batches).toHaveLength(2);
  });

  test("legacy metadata remains searchable and gains the checkpoint column lazily", async () => {
    const { store } = await fixture();
    store.internal.ensureVecTable(3);
    const hash = await insertVectorDoc(store, "a", "legacy.md", "legacy content", 1, false);
    const columns = store.internal.db.prepare("PRAGMA table_info(content_vectors)").all() as Array<{ name: string }>;
    if (columns.some(c => c.name === "chunk_plan_hash")) { store.internal.db.exec("DROP INDEX IF EXISTS idx_content_vectors_checkpoints"); store.internal.db.exec("ALTER TABLE content_vectors DROP COLUMN chunk_plan_hash"); };
    const found = await searchVec(store.internal.db, "q", DEFAULT_EMBED_MODEL, 10, "a", undefined, [1, 0, 0]);
    expect(found.map(r => r.hash)).toContain(hash);
    const repaired = store.internal.db.prepare("PRAGMA table_info(content_vectors)").all() as Array<{ name: string }>;
    expect(repaired.map(c => c.name)).toContain("chunk_plan_hash");
  });

  test("a first partial checkpoint committed after the eligibility probe cannot enter the scan", async () => {
    for (const [scope, filter] of [
      [undefined, undefined],
      ["a", undefined],
      [["a", "b"], undefined],
      [undefined, { field: "eligible", operator: "eq", value: true }],
    ] as const) {
      const { store } = await fixture();
      const db = store.internal.db;
      store.internal.ensureVecTable(3);
      checkpointColumn(store);
      const complete = await insertVectorDoc(store, "a", "complete.md", "complete snapshot", 1, false, [0, 1, 0]);
      const now = new Date().toISOString();
      const partial = await hashContent("concurrently published partial");
      insertContent(db, partial, "concurrently published partial", now);
      const id = insertDocument(db, "a", "partial.md", "partial", partial, now, now);
      replaceDocumentMetadata(db, id, { metadata: { eligible: true }, extractionVersion: METADATA_EXTRACTION_VERSION });
      const file = (db.prepare("PRAGMA database_list").all() as Array<{ name: string; file: string }>).find(row => row.name === "main")!.file;
      const writer = openDatabase(file);
      loadSqliteVec(writer);
      cleanup.push(async () => writer.close());
      let committed = false;
      const racingDb: Database = {
        prepare(sql) {
          const statement = db.prepare(sql);
          if (sql !== "SELECT 1 FROM content_vectors WHERE chunk_plan_hash != '' LIMIT 1") return statement;
          return {
            run: statement.run.bind(statement),
            all: statement.all.bind(statement),
            iterate: statement.iterate.bind(statement),
            get<T = unknown>(...params: SQLiteValue[]): T | undefined {
              const result = statement.get<T>(...params);
              if (!committed) {
                expect(result == null).toBe(true);
                writer.transaction(() => {
                  insertEmbedding(writer, partial, 0, 0, new Float32Array([1, 0, 0]), DEFAULT_EMBED_MODEL, now, 2);
                  writer.prepare("UPDATE content_vectors SET chunk_plan_hash = ? WHERE hash = ?").run("synthetic-checkpoint", partial);
                }).immediate();
                committed = true;
              }
              return result;
            },
          };
        },
        exec: sql => db.exec(sql),
        transaction: fn => db.transaction(fn),
        loadExtension: path => db.loadExtension(path),
        close: () => db.close(),
      };
      const found = await searchVec(racingDb, "q", DEFAULT_EMBED_MODEL, 1, scope, undefined, [1, 0, 0], undefined, filter);
      expect(committed).toBe(true);
      expect(writer.prepare("SELECT chunk_plan_hash FROM content_vectors WHERE hash = ?").get(partial)).toEqual({ chunk_plan_hash: "synthetic-checkpoint" });
      expect(found.map(row => row.hash)).toEqual([complete]);
      expect((await searchVec(db, "q", DEFAULT_EMBED_MODEL, 1, scope, undefined, [1, 0, 0], undefined, filter)).map(row => row.hash)).toEqual([complete]);
    }
  });

  test("incomplete checkpoints cannot crowd out complete results in any partition or metadata scope", async () => {
    const { store } = await fixture();
    store.internal.ensureVecTable(3);
    checkpointColumn(store);
    for (let i = 0; i < 4100; i++) await insertVectorDoc(store, "a", `partial-${i}.md`, `partial content ${i}`, 2, true);
    // A shared hash also has an incomplete row in b's real partition.
    await insertVectorDoc(store, "b", "partial-0.md", "partial content 0", 2, true);
    const completeA = await insertVectorDoc(store, "a", "complete.md", "complete a", 1, false, [0, 1, 0]);
    const completeB = await insertVectorDoc(store, "b", "complete.md", "complete b", 1, false, [0, 1, 0]);
    for (const scope of [undefined, "a", "b", ["a", "b"]] as const) {
      for (const filter of [undefined, { field: "eligible", operator: "eq", value: true }] as const) {
        const found = await searchVec(store.internal.db, "q", DEFAULT_EMBED_MODEL, 1, scope, undefined, [1, 0, 0], undefined, filter);
        expect(found).toHaveLength(1);
        expect([completeA, completeB]).toContain(found[0]!.hash);
      }
    }
  });
});
