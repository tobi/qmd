/**
 * legacy-adoption-sample-worker - runs legacy fingerprint adoption over a
 * fixture with large bodies, many legacy chunks and duplicate paths, and
 * reports the peak resident memory the adoption query added.
 *
 * The fixture materialises one body per chunk per path when the sample query
 * carries c.doc through its join - here 512 chunks x 8 paths x 64 KiB, about
 * 256 MiB - while reading one sample body by rowid needs one copy. The check is
 * the kernel's peak-RSS high-water mark, not PRAGMA hard_heap_limit: the limit
 * only binds where SQLite tracks its own memory (Bun's bundled SQLite on Linux
 * stopped doing so), whereas the high-water mark records the spike on every
 * build, and it is read after the query so a transient peak cannot be missed.
 *
 * The store is in memory, so the fixture's own copies of the bodies are part of
 * the same process memory this worker measures; the baseline is taken after the
 * fixture so only the query's growth counts against the budget.
 */
import { createStore, maybeAdoptLegacyEmbeddingFingerprint } from "../../src/store.ts";
import { peakRssBytes } from "./heap-limit.ts";
import { LEGACY_ADOPTION_STUB_MODEL, LegacyAdoptionStubLlm } from "./legacy-adoption-stub-llm.ts";

// What the adoption query may add. One sample body is 64 KiB, so a spike
// anywhere near this is the per-chunk (per-path) join.
const budgetBytes = 128 * 1024 * 1024;
// The probe is only trustworthy if it can see a spike this large: one SQLite
// allocation of twice the budget has to move the same high-water mark.
const controlBytes = 2 * budgetBytes;
const model = LEGACY_ADOPTION_STUB_MODEL;
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
  // Adoption only tokenizes, detokenizes and embeds: the stub answers those
  // three calls without loading a model.
  store.llm = new LegacyAdoptionStubLlm();

  store.db.exec("PRAGMA temp_store = MEMORY");
  const before = peakRssBytes();
  const result = await maybeAdoptLegacyEmbeddingFingerprint(store, model);
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
    ...result,
    budgetBytes,
    controlBlobBytes: control.n,
    queryGrowthBytes: growth ? growth.query : null,
    controlGrowthBytes: growth ? growth.control : null,
  }));
} finally {
  store.close();
}
