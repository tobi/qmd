import { createStore } from "../../src/store.js";
import { checkEmbeddingVectorSamples } from "../../src/cli/qmd.js";
import { LlamaCpp, setDefaultLlamaCpp } from "../../src/llm.js";
import { peakRssBytes } from "./heap-limit.js";

/**
 * Runs the doctor's vector-sample check over large documents with duplicate
 * paths, and reports the peak resident memory the check added.
 *
 * A sample query that joins bodies to every chunk and duplicate path sorts a
 * few hundred MiB of text to keep three rows, while limiting chunk identities
 * first reads one body per sample. The check is the kernel's peak-RSS
 * high-water mark, not PRAGMA hard_heap_limit: the limit only binds where
 * SQLite tracks its own memory (Bun's bundled SQLite on Linux stopped doing
 * so), whereas the high-water mark records the spike on every build, and it is
 * read after the check so a transient peak cannot be missed.
 *
 * `temp_store = MEMORY` keeps the sorter such a query would spill into a heap
 * allocation the peak sees.
 */
// What the check may add. Three 64 KiB samples plus their token/embed working
// set are far below this; the per-chunk join sorts about 256 MiB of text.
const budgetBytes = 32 * 1024 * 1024;
// The probe is only trustworthy if it can see a spike this large: one SQLite
// allocation of twice the budget has to move the same high-water mark.
const controlBytes = 2 * budgetBytes;

// Every stored vector is [1, 0] and every re-embedded chunk returns [1, 0], so
// the check passes whenever it can sample, read and chunk the bodies.
class ConstantLlm extends LlamaCpp {
  override async tokenize(text: string) { return new Array(Math.ceil(text.length / 16)).fill(1); }
  override async embed() { return { embedding: [1, 0], model: "model" }; }
}

const store = createStore(":memory:");
try {
  setDefaultLlamaCpp(new ConstantLlm());
  const body = "word ".repeat(13_107); // About 64 KiB per document.
  store.ensureVecTable(2);
  store.db.transaction(() => {
    for (let doc = 0; doc < 16; doc++) {
      const hash = `document-${doc}`;
      store.insertContent(hash, body, "2026-01-01");
      for (let path = 0; path < 8; path++) {
        store.insertDocument("test", `${hash}-${path}.md`, hash, hash, "2026-01-01", "2026-01-01");
      }
      for (let seq = 0; seq < 32; seq++) {
        store.insertEmbedding(hash, seq, 0, new Float32Array([1, 0]), "model", "2026-01-01", 32, "current");
      }
    }
  })();
  store.db.exec("PRAGMA temp_store = MEMORY");
  const before = peakRssBytes();
  const result = await checkEmbeddingVectorSamples(store.db, "model", "current");
  const afterCheck = peakRssBytes();
  const control = store.db.prepare(`SELECT length(randomblob(?)) AS n`).get(controlBytes) as { n: number } | undefined;
  if (!control) throw new Error("randomblob control returned no row");
  const afterControl = peakRssBytes();

  // Peak RSS is only reported where the kernel exposes it; without it the
  // budget cannot be checked, so the worker says so instead of guessing.
  const growth = before === null || afterCheck === null || afterControl === null
    ? null
    : { check: afterCheck - before, control: afterControl - afterCheck };
  console.log(JSON.stringify({
    ...result,
    budgetBytes,
    controlBlobBytes: control.n,
    checkGrowthBytes: growth ? growth.check : null,
    controlGrowthBytes: growth ? growth.control : null,
  }));
} finally {
  setDefaultLlamaCpp(null);
  store.close();
}
