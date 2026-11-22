import { createStore, getEmbeddingVectorSamples } from "../../src/store.js";
import { peakRssBytes } from "./heap-limit.js";

/**
 * Samples stored vectors from large documents with duplicate paths, and
 * reports the peak resident memory the sample query added.
 *
 * A query that joins bodies to every chunk and duplicate path sorts a few
 * hundred MiB of text to keep three rows, while limiting chunk identities
 * first reads one body per sample. The check is the kernel's peak-RSS
 * high-water mark, not PRAGMA hard_heap_limit: the limit only binds where
 * SQLite tracks its own memory (Bun's bundled SQLite on Linux stopped doing
 * so), whereas the high-water mark records the spike on every build, and it is
 * read after the query so a transient peak cannot be missed.
 *
 * `temp_store = MEMORY` keeps the sorter such a query would spill into a heap
 * allocation the peak sees.
 */
// What the sample query may add. Three 64 KiB bodies are far below this; the
// per-chunk join sorts about 256 MiB of text.
const budgetBytes = 32 * 1024 * 1024;
// The probe is only trustworthy if it can see a spike this large: one SQLite
// allocation of twice the budget has to move the same high-water mark.
const controlBytes = 2 * budgetBytes;
const body = "word ".repeat(13_107); // About 64 KiB per document.
const store = createStore(":memory:");
try {
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
  const before = peakRssBytes();
  const samples = getEmbeddingVectorSamples(store.db, "model", "current");
  const afterQuery = peakRssBytes();
  const control = store.db.prepare(`SELECT length(randomblob(?)) AS n`).get(controlBytes) as { n: number } | undefined;
  if (!control) throw new Error("randomblob control returned no row");
  const afterControl = peakRssBytes();

  // Peak RSS is only reported where the kernel exposes it; without it the
  // budget cannot be checked, so the worker says so instead of guessing.
  const growth = before === null || afterQuery === null || afterControl === null
    ? null
    : { query: afterQuery - before, control: afterControl - afterQuery };
  console.log(JSON.stringify({
    samples: samples.length,
    distinctChunks: new Set(samples.map(sample => `${sample.hash}:${sample.seq}`)).size,
    bodiesComplete: samples.every(sample => sample.body === body),
    budgetBytes,
    controlBlobBytes: control.n,
    queryGrowthBytes: growth ? growth.query : null,
    controlGrowthBytes: growth ? growth.control : null,
  }));
} finally {
  store.close();
}
