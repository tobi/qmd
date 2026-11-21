/**
 * wiki-bench v0 — private/local BM25 runner body.
 *
 * Invoked by scripts/run-wiki-bench-local.mjs. Builds a temporary QMD index
 * (collection `wiki-bench`) outside the git tree, runs BM25 via runBenchmark,
 * prints metrics, and exits non-zero if quality floors fail.
 *
 * Never vendors wiki page bodies into the repo.
 */

import {
  existsSync,
  mkdtempSync,
  rmSync,
  statSync,
} from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { createStore } from "../src/index.js";
import { runBenchmark } from "../src/bench/bench.js";
import type { BenchmarkResult, QueryResult } from "../src/bench/types.js";

const COLLECTION = "wiki-bench";
const FLOOR_EXACT_RECALL_AT_3 = 0.85;
const FLOOR_ALIAS_RECALL_AT_3 = 0.80;
const FLOOR_COMBINED_MRR = 0.70;

const root = fileURLToPath(new URL("..", import.meta.url));
const FIXTURE_PATH = join(root, "src", "bench", "fixtures", "wiki-v0.json");

function die(message: string, code = 1): never {
  console.error(`wiki-bench-local: ${message}`);
  process.exit(code);
}

function looksLikeWikiDocsRoot(dir: string): boolean {
  return existsSync(join(dir, "concepts")) || existsSync(join(dir, "sources"));
}

/** Resolve corpus root from env. Never looks under test/wiki-bench-docs. */
function resolveWikiDocsDir(): string {
  const benchDocs = process.env.QMD_WIKI_BENCH_DOCS?.trim();
  if (benchDocs) {
    const abs = resolve(benchDocs);
    if (!existsSync(abs) || !statSync(abs).isDirectory()) {
      die(`QMD_WIKI_BENCH_DOCS is set but not a directory: ${abs}`);
    }
    if (!looksLikeWikiDocsRoot(abs)) {
      die(
        `QMD_WIKI_BENCH_DOCS must contain wiki-relative folders (concepts|sources|...): ${abs}`,
      );
    }
    return abs;
  }

  const wikiPath = process.env.QMD_WIKI_PATH?.trim();
  if (wikiPath) {
    const abs = resolve(wikiPath);
    if (!existsSync(abs) || !statSync(abs).isDirectory()) {
      die(`QMD_WIKI_PATH is set but not a directory: ${abs}`);
    }
    // Spec: use $QMD_WIKI_PATH/wiki (Obsidian wiki/ folder) as the docs root.
    const nested = join(abs, "wiki");
    if (!existsSync(nested) || !statSync(nested).isDirectory()) {
      die(
        `QMD_WIKI_PATH set but missing Obsidian wiki/ folder at: ${nested}\n` +
          `  (expected layout: $QMD_WIKI_PATH/wiki/{concepts,sources,entities,...})`,
      );
    }
    if (!looksLikeWikiDocsRoot(nested)) {
      die(
        `QMD_WIKI_PATH/wiki must contain wiki-relative folders (concepts|sources|...): ${nested}`,
      );
    }
    return nested;
  }

  die(
    "Set QMD_WIKI_PATH (checkout of tobi/wiki) or QMD_WIKI_BENCH_DOCS (markdown root with concepts|sources|...).\n" +
      "  Example: QMD_WIKI_PATH=~/src/wiki node scripts/run-wiki-bench-local.mjs\n" +
      "Private wiki page bodies must never be committed into this repo.",
  );
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function bm25Metrics(results: QueryResult[], type: string): { r3: number; mrr: number; n: number } {
  const subset = results.filter((r) => r.type === type);
  const scores = subset
    .map((r) => r.backends.bm25)
    .filter((b): b is NonNullable<typeof b> => !!b);
  return {
    r3: mean(scores.map((s) => s.recall_at_3)),
    mrr: mean(scores.map((s) => s.mrr)),
    n: scores.length,
  };
}

function printFloorSummary(result: BenchmarkResult): {
  exactR3: number;
  aliasR3: number;
  combinedMrr: number;
} {
  const exact = bm25Metrics(result.results, "exact");
  const alias = bm25Metrics(result.results, "alias");
  const combinedScores = result.results
    .filter((r) => r.type === "exact" || r.type === "alias")
    .map((r) => r.backends.bm25)
    .filter((b): b is NonNullable<typeof b> => !!b);
  const combinedMrr = mean(combinedScores.map((s) => s.mrr));

  console.log("\n[wiki-bench-local] BM25 quality floors (exact + alias):");
  console.log(
    `  exact mean recall@3 = ${exact.r3.toFixed(4)} (n=${exact.n}; floor ${FLOOR_EXACT_RECALL_AT_3})`,
  );
  console.log(
    `  alias mean recall@3 = ${alias.r3.toFixed(4)} (n=${alias.n}; floor ${FLOOR_ALIAS_RECALL_AT_3})`,
  );
  console.log(
    `  exact+alias mean MRR = ${combinedMrr.toFixed(4)} (n=${combinedScores.length}; floor ${FLOOR_COMBINED_MRR})`,
  );

  return { exactR3: exact.r3, aliasR3: alias.r3, combinedMrr };
}

async function main(): Promise<void> {
  if (!existsSync(FIXTURE_PATH)) {
    die(`fixture not found: ${FIXTURE_PATH}`);
  }

  const docsDir = resolveWikiDocsDir();
  const tempDir = mkdtempSync(join(tmpdir(), "qmd-wiki-bench-local-"));
  const dbPath = join(tempDir, "wiki-bench.sqlite");

  console.log(`[wiki-bench-local] docs: ${docsDir}`);
  console.log(`[wiki-bench-local] temp index: ${dbPath}`);
  console.log(`[wiki-bench-local] fixture: ${FIXTURE_PATH}`);

  let exitCode = 0;
  try {
    const store = await createStore({
      dbPath,
      config: {
        collections: {
          [COLLECTION]: { path: docsDir, pattern: "**/*.md" },
        },
      },
    });

    const updateResult = await store.update({ collections: [COLLECTION] });
    await store.close();

    const indexed =
      updateResult.indexed + updateResult.updated + updateResult.unchanged;
    if (indexed <= 0) {
      die(`no markdown documents indexed from ${docsDir}`);
    }
    console.log(
      `[wiki-bench-local] indexed collection '${COLLECTION}': ` +
        `${updateResult.indexed} new, ${updateResult.updated} updated, ` +
        `${updateResult.unchanged} unchanged (${indexed} total)`,
    );

    const benchResult = await runBenchmark(FIXTURE_PATH, {
      backends: ["bm25"],
      collection: COLLECTION,
      dbPath,
    });

    const { exactR3, aliasR3, combinedMrr } = printFloorSummary(benchResult);

    const failures: string[] = [];
    if (exactR3 < FLOOR_EXACT_RECALL_AT_3) {
      failures.push(
        `exact R@3 ${exactR3.toFixed(4)} < ${FLOOR_EXACT_RECALL_AT_3}`,
      );
    }
    if (aliasR3 < FLOOR_ALIAS_RECALL_AT_3) {
      failures.push(
        `alias R@3 ${aliasR3.toFixed(4)} < ${FLOOR_ALIAS_RECALL_AT_3}`,
      );
    }
    if (combinedMrr < FLOOR_COMBINED_MRR) {
      failures.push(
        `exact+alias MRR ${combinedMrr.toFixed(4)} < ${FLOOR_COMBINED_MRR}`,
      );
    }

    if (failures.length > 0) {
      console.error("\nwiki-bench-local: FAILED quality floors:");
      for (const f of failures) console.error(`  - ${f}`);
      exitCode = 1;
    } else {
      console.log("\nwiki-bench-local: PASSED quality floors.");
    }
  } finally {
    rmSync(tempDir, { recursive: true, force: true });
  }

  process.exit(exitCode);
}

main().catch((err) => {
  console.error("wiki-bench-local: unexpected error");
  console.error(err);
  process.exit(1);
});
