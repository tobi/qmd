/**
 * legacy-adoption-shared-body-worker - runs legacy fingerprint adoption for
 * one legacy chunk whose 1 MiB body sits behind 200 active paths, and reports
 * the peak resident memory the adoption query added.
 *
 * A sample query that carries c.doc through its join keeps one copy of that
 * body live per active path, about 200 MiB, while reading the sample by rowid
 * keeps one. The check is the kernel's peak-RSS high-water mark, not
 * PRAGMA hard_heap_limit: the limit only binds where SQLite tracks its own
 * memory (Bun's bundled SQLite on Linux stopped doing so), whereas the
 * high-water mark records the spike on every build, and it is read after the
 * query so a transient peak cannot be missed.
 *
 * `temp_store = MEMORY` keeps a temp b-tree such a query would spill into a
 * heap allocation the peak sees. The index lives in a temp file: the FTS table
 * stores every path's copy of the body, which an in-memory database would
 * count against the same process memory this worker measures.
 */
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, maybeAdoptLegacyEmbeddingFingerprint } from "../../src/store.ts";
import { peakRssBytes } from "./heap-limit.ts";
import { LEGACY_ADOPTION_STUB_MODEL, LegacyAdoptionStubLlm } from "./legacy-adoption-stub-llm.ts";

// What the adoption query may add. One body copy plus its token/embed working
// set is a few MiB, so a spike anywhere near this is the per-active-path join.
const budgetBytes = 64 * 1024 * 1024;
// The probe is only trustworthy if it can see a spike this large: one SQLite
// allocation of twice the budget has to move the same high-water mark.
const controlBytes = 2 * budgetBytes;
const model = LEGACY_ADOPTION_STUB_MODEL;
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
    checked: result.checked,
    adopted: result.adopted,
    reason: result.reason,
    budgetBytes,
    controlBlobBytes: control.n,
    queryGrowthBytes: growth ? growth.query : null,
    controlGrowthBytes: growth ? growth.control : null,
  }));
} finally {
  store.close();
  rmSync(dir, { recursive: true, force: true });
}
