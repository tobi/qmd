/**
 * legacy-adoption-shared-body-worker - runs legacy fingerprint adoption for
 * one legacy chunk whose 1 MiB body sits behind 200 active paths, under a
 * 16 MiB SQLite heap limit.
 *
 * A sample query that carries c.doc through its join copies the body once per
 * active path, far past the budget, while the chunk count stays at one. The
 * index lives in a temp file: the FTS table stores every path's copy of the
 * body, which an in-memory database would count against the heap limit.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, maybeAdoptLegacyEmbeddingFingerprint } from "../../src/store.ts";
import { heapLimitBinds } from "./heap-limit.ts";

const model = "model";
const dir = mkdtempSync(join(tmpdir(), "qmd-legacy-shared-body-"));
const store = createStore(join(dir, "index.sqlite"));
try {
  const body = "word ".repeat(209_716); // About 1 MiB.
  store.insertContent("shared", body, "2026-01-01");
  store.db.transaction(() => {
    for (let path = 0; path < 200; path++) {
      store.insertDocument("test", `shared-${path}.md`, "shared", "shared", "2026-01-01", "2026-01-01");
    }
  })();
  store.ensureVecTable(3);
  store.insertEmbedding("shared", 0, 0, new Float32Array([0.1, 0.2, 0.3]), model, "2026-01-01", 1, "");
  // Adoption only tokenizes, detokenizes and embeds; a stub covers that surface.
  store.llm = {
    async tokenize(text: string) { return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1); },
    async detokenize(tokens: readonly number[]) { return "x".repeat(tokens.length * 16); },
    async embed() { return { embedding: [0.1, 0.2, 0.3], model }; },
  } as any;

  store.db.exec("PRAGMA temp_store = MEMORY");
  store.db.exec("PRAGMA hard_heap_limit = 16777216");
  const result = await maybeAdoptLegacyEmbeddingFingerprint(store, model);
  console.log(JSON.stringify({ checked: result.checked, adopted: result.adopted, limitBinds: heapLimitBinds(store.db, 16777216) }));
} finally {
  store.close();
  rmSync(dir, { recursive: true, force: true });
}
