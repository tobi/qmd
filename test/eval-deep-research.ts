/**
 * Deep Research Evaluation for QMD
 *
 * Tests end-to-end retrieval quality: query → expansion → reranking → results
 * 
 * These are HARD queries with NO exact keyword matches - they require
 * semantic understanding via query expansion and reranking to succeed.
 *
 * Run: bun test/eval-deep-research.ts
 */

import { execSync } from "child_process";
import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

interface EvalQuery {
  query: string;
  expected_doc: string;
  difficulty: string;
  intent: string;  // Domain context hint for future intent-aware retrieval
  notes: string;
}

interface SearchResult {
  file: string;
  score: number;
  title?: string;
}

function loadQueries(): EvalQuery[] {
  const path = join(__dirname, "eval-deep-research.jsonl");
  const content = readFileSync(path, "utf-8");
  return content
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

function runBM25Search(query: string): SearchResult[] {
  try {
    const output = execSync(
      `bun src/qmd.ts search "${query.replace(/"/g, '\\"')}" -c eval-docs --json -n 5 2>/dev/null`,
      { encoding: "utf-8", timeout: 30000 }
    );
    return JSON.parse(output);
  } catch {
    return [];
  }
}

function runDeepResearch(query: string): SearchResult[] {
  try {
    const output = execSync(
      `bun src/qmd.ts query "${query.replace(/"/g, '\\"')}" -c eval-docs --json -n 5 2>/dev/null`,
      { encoding: "utf-8", timeout: 120000 }
    );
    return JSON.parse(output);
  } catch {
    return [];
  }
}

function matchesExpected(filepath: string, expectedDoc: string): boolean {
  return filepath.toLowerCase().includes(expectedDoc.toLowerCase());
}

function findRank(results: SearchResult[], expectedDoc: string): number {
  for (let i = 0; i < results.length; i++) {
    if (matchesExpected(results[i]!.file, expectedDoc)) {
      return i + 1;
    }
  }
  return -1; // Not found
}

interface MethodResults {
  hit1: number;
  hit3: number;
  hit5: number;
  total: number;
  details: { query: string; rank: number; expected: string; intent?: string }[];
}

function evaluate(
  queries: EvalQuery[],
  searchFn: (q: string) => SearchResult[],
  label: string
): MethodResults {
  const results: MethodResults = {
    hit1: 0,
    hit3: 0,
    hit5: 0,
    total: queries.length,
    details: [],
  };

  console.log(`\n${"=".repeat(60)}`);
  console.log(`  ${label}`);
  console.log(`${"=".repeat(60)}\n`);

  for (const { query, expected_doc, intent, notes } of queries) {
    const searchResults = searchFn(query);
    const rank = findRank(searchResults, expected_doc);

    results.details.push({ query, rank, expected: expected_doc, intent });

    if (rank === 1) results.hit1++;
    if (rank >= 1 && rank <= 3) results.hit3++;
    if (rank >= 1 && rank <= 5) results.hit5++;

    const status =
      rank === 1 ? "✓" : rank > 0 && rank <= 3 ? `@${rank}` : rank > 0 ? `@${rank}` : "✗";
    const statusPad = status.padEnd(4);
    console.log(`  ${statusPad} "${query.slice(0, 45).padEnd(45)}" → ${expected_doc}`);
    if (rank === -1) {
      console.log(`        intent: ${intent} | ${notes}`);
    }
  }

  const hit1Pct = ((results.hit1 / results.total) * 100).toFixed(0);
  const hit3Pct = ((results.hit3 / results.total) * 100).toFixed(0);
  const hit5Pct = ((results.hit5 / results.total) * 100).toFixed(0);

  console.log(`\n  ${"─".repeat(50)}`);
  console.log(`  Hit@1: ${hit1Pct}% (${results.hit1}/${results.total})`);
  console.log(`  Hit@3: ${hit3Pct}% (${results.hit3}/${results.total})`);
  console.log(`  Hit@5: ${hit5Pct}% (${results.hit5}/${results.total})`);

  return results;
}

async function main() {
  console.log("QMD Deep Research Evaluation");
  console.log("=".repeat(60));
  console.log("Testing hard queries that require semantic understanding.");
  console.log("These have NO exact keyword matches in documents.");

  // Check if eval-docs collection exists
  try {
    const status = execSync("bun src/qmd.ts status --json 2>/dev/null", {
      encoding: "utf-8",
    });
    if (!status.includes("eval-docs")) {
      console.log("\n⚠️  eval-docs collection not found. Run:");
      console.log("   qmd collection add test/eval-docs --name eval-docs");
      console.log("   qmd embed");
      process.exit(1);
    }
  } catch {
    console.log("\n⚠️  Could not check status. Make sure qmd is working.");
  }

  const queries = loadQueries();
  console.log(`\nLoaded ${queries.length} hard queries.`);

  // Run BM25 baseline (expected to fail on most)
  const bm25Results = evaluate(queries, runBM25Search, "BM25 BASELINE (keyword search)");

  // Run deep research (expected to succeed via expansion + reranking)
  const deepResults = evaluate(queries, runDeepResearch, "DEEP RESEARCH (expansion + reranking)");

  // Comparison
  console.log(`\n${"=".repeat(60)}`);
  console.log("  COMPARISON");
  console.log(`${"=".repeat(60)}`);
  console.log(`\n  Method              Hit@1    Hit@3    Hit@5`);
  console.log(`  ${"─".repeat(45)}`);
  console.log(
    `  BM25 (baseline)     ${((bm25Results.hit1 / bm25Results.total) * 100).toFixed(0).padStart(3)}%     ${((bm25Results.hit3 / bm25Results.total) * 100).toFixed(0).padStart(3)}%     ${((bm25Results.hit5 / bm25Results.total) * 100).toFixed(0).padStart(3)}%`
  );
  console.log(
    `  Deep Research       ${((deepResults.hit1 / deepResults.total) * 100).toFixed(0).padStart(3)}%     ${((deepResults.hit3 / deepResults.total) * 100).toFixed(0).padStart(3)}%     ${((deepResults.hit5 / deepResults.total) * 100).toFixed(0).padStart(3)}%`
  );

  const improvement = deepResults.hit3 - bm25Results.hit3;
  console.log(`\n  Improvement (Hit@3): +${improvement} queries (${((improvement / bm25Results.total) * 100).toFixed(0)}%)`);

  // Show queries where deep research recovered failures
  const recovered = deepResults.details.filter(
    (d) =>
      d.rank >= 1 &&
      d.rank <= 3 &&
      bm25Results.details.find((b) => b.query === d.query)?.rank === -1
  );

  if (recovered.length > 0) {
    console.log(`\n  Recovered by expansion + reranking (${recovered.length}):`);
    for (const { query, rank, expected } of recovered.slice(0, 5)) {
      console.log(`    @${rank} "${query.slice(0, 40)}..." → ${expected}`);
    }
    if (recovered.length > 5) {
      console.log(`    ... and ${recovered.length - 5} more`);
    }
  }

  // Exit with error if deep research performs poorly
  const deepHit3Pct = (deepResults.hit3 / deepResults.total) * 100;
  if (deepHit3Pct < 60) {
    console.log(`\n❌ Deep research Hit@3 < 60% (${deepHit3Pct.toFixed(0)}%)`);
    process.exit(1);
  } else {
    console.log(`\n✓ Deep research Hit@3 >= 60% (${deepHit3Pct.toFixed(0)}%)`);
  }
}

main();
