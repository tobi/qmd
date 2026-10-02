import { createStore, getEmbeddingVectorSamples } from "../../src/store.js";

const store = createStore(":memory:");
try {
  const body = "word ".repeat(13_107); // About 64 KiB per document.
  const insertChunk = store.db.prepare(`
    INSERT INTO content_vectors (hash, seq, model, embed_fingerprint, embedded_at)
    VALUES (?, ?, 'model', 'current', '2026-01-01')
  `);
  store.db.transaction(() => {
    for (let doc = 0; doc < 16; doc++) {
      const hash = `document-${doc}`;
      store.insertContent(hash, body, "2026-01-01");
      for (let path = 0; path < 8; path++) {
        store.insertDocument("test", `${hash}-${path}.md`, hash, hash, "2026-01-01", "2026-01-01");
      }
      for (let seq = 0; seq < 32; seq++) insertChunk.run(hash, seq);
    }
  })();
  store.db.exec("PRAGMA temp_store = MEMORY");
  store.db.exec("PRAGMA hard_heap_limit = 33554432");
  const samples = getEmbeddingVectorSamples(store.db, "model", "current");
  console.log(JSON.stringify({
    samples: samples.length,
    distinctChunks: new Set(samples.map(sample => `${sample.hash}:${sample.seq}`)).size,
    bodiesComplete: samples.every(sample => sample.body === body),
  }));
} finally {
  store.close();
}
