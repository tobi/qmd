/**
 * legacy-adoption-sample-worker - runs legacy fingerprint adoption under a
 * 128 MiB SQLite heap limit.
 *
 * Spawned by store.test.ts: PRAGMA hard_heap_limit is process-wide and cannot
 * be raised again, so it must not leak into the test runner. The fixture has
 * large bodies with many legacy chunks and duplicate paths; materializing one
 * body per chunk (per path) needs far more than the budget.
 */
import { createStore, maybeAdoptLegacyEmbeddingFingerprint } from "../../src/store.ts";

const model = "model";
const store = createStore(":memory:");
try {
  const body = "word ".repeat(13_107); // About 64 KiB per document.
  const insertChunk = store.db.prepare(`
    INSERT INTO content_vectors (hash, seq, model, embed_fingerprint, embedded_at)
    VALUES (?, ?, ?, '', '2026-01-01')
  `);
  store.db.transaction(() => {
    for (let doc = 0; doc < 16; doc++) {
      const hash = `document-${doc}`;
      store.insertContent(hash, body, "2026-01-01");
      for (let path = 0; path < 8; path++) {
        store.insertDocument("test", `${hash}-${path}.md`, hash, hash, "2026-01-01", "2026-01-01");
      }
      for (let seq = 0; seq < 32; seq++) insertChunk.run(hash, seq, model);
    }
  })();
  // Store the sample chunk's vector: the one the stub embedder returns.
  store.ensureVecTable(3);
  store.insertEmbedding("document-0", 0, 0, new Float32Array([0.1, 0.2, 0.3]), model, "2026-01-01", 32, "");
  // Adoption only tokenizes, detokenizes and embeds; a stub covers that surface.
  store.llm = {
    async tokenize(text: string) { return new Array(Math.max(1, Math.ceil(text.length / 16))).fill(1); },
    async detokenize(tokens: readonly number[]) { return "x".repeat(tokens.length * 16); },
    async embed() { return { embedding: [0.1, 0.2, 0.3], model }; },
  } as any;

  store.db.exec("PRAGMA temp_store = MEMORY");
  store.db.exec("PRAGMA hard_heap_limit = 134217728");
  const result = await maybeAdoptLegacyEmbeddingFingerprint(store, model);
  console.log(JSON.stringify(result));
} finally {
  store.close();
}
