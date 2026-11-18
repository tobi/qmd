import { test, expect } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, type DocumentProcessor, type ProcessorDocument, type ProcessorInput } from "../src/index.js";

const rows = [{ id: "a", text: "orchard" }, { id: "b", text: "meadow" }];

function render(input: ProcessorInput): ProcessorDocument[] {
  const entries = JSON.parse(Buffer.from(input.bytes).toString()) as Array<{ id: string; text: string }>;
  return entries.map(entry => ({ key: `${entry.id}.md`, markdown: `# ${entry.id}\n${entry.text}\n` }));
}

test("a publication write that ends the batch transaction surfaces its own error, publishes nothing, and leaves no transaction open", async () => {
  const root = mkdtempSync(join(tmpdir(), "qmd-processor-lifecycle-"));
  const docs = join(root, "docs");
  mkdirSync(docs);
  writeFileSync(join(docs, "data.json"), JSON.stringify(rows));
  const store = await createStore({ dbPath: join(root, "index.sqlite"), config: { collections: { docs: { path: docs, pattern: "**/*.json" } } } });
  const db = store.internal.db;
  const processor: DocumentProcessor = { fingerprint: "v1", process: async input => render(input) };
  const update = () => store.update({ processors: { docs: processor } });
  const snapshot = () => ["documents", "content", "document_metadata", "document_metadata_values", "processor_sources", "processor_outputs"]
    .map(table => db.prepare(`SELECT * FROM ${table}`).all());
  try {
    expect((await update()).indexed).toBe(2);
    const before = snapshot();

    // RAISE(ROLLBACK) ends the whole transaction rather than the statement, so it
    // discards the source savepoint as well: the state no rollback path may assume.
    writeFileSync(join(docs, "data.json"), JSON.stringify([{ id: "a", text: "replacement" }, { id: "b", text: "second" }]));
    db.exec(`CREATE TRIGGER abort_batch BEFORE UPDATE ON documents WHEN NEW.path = 'data.json/b.md'
      BEGIN SELECT RAISE(ROLLBACK, 'synthetic transaction abort'); END`);
    await expect(update()).rejects.toThrow("synthetic transaction abort");

    // The batch's rollback finds its transaction already gone and tolerates it,
    // so the storage error is what surfaces and no half-published state or open
    // transaction is left behind.
    expect(db.inTransaction).toBe(false);
    expect(snapshot()).toEqual(before);
    expect(store.internal.searchFTS("replacement")).toHaveLength(0);
    expect(store.internal.searchFTS("orchard")).toHaveLength(1);

    db.exec("DROP TRIGGER abort_batch");
    expect((await update()).updated).toBe(2);
    expect(store.internal.searchFTS("second")).toHaveLength(1);
    expect(store.internal.searchFTS("orchard")).toHaveLength(0);
  } finally { await store.close(); rmSync(root, { recursive: true, force: true }); }
});
