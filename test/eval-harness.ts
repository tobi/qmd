/**
 * Evaluation Harness for QMD Search
 *
 * Tests search quality with synthetic queries against known documents.
 * Run: bun test/eval-harness.ts
 */

import { execSync } from "child_process";

// Test queries with expected documents and difficulty
const evalQueries: {
  query: string;
  expectedDoc: string;  // Partial match on filename
  difficulty: "easy" | "medium" | "hard";
  description: string;
}[] = [
  // EASY: Exact keyword matches
  {
    query: "API versioning",
    expectedDoc: "api-design",
    difficulty: "easy",
    description: "Direct keyword match"
  },
  {
    query: "Series A fundraising",
    expectedDoc: "fundraising",
    difficulty: "easy",
    description: "Direct keyword match"
  },
  {
    query: "CAP theorem",
    expectedDoc: "distributed-systems",
    difficulty: "easy",
    description: "Direct keyword match"
  },
  {
    query: "overfitting machine learning",
    expectedDoc: "machine-learning",
    difficulty: "easy",
    description: "Direct keyword match"
  },
  {
    query: "remote work VPN",
    expectedDoc: "remote-work",
    difficulty: "easy",
    description: "Direct keyword match"
  },
  {
    query: "Project Phoenix retrospective",
    expectedDoc: "product-launch",
    difficulty: "easy",
    description: "Direct keyword match"
  },

  // MEDIUM: Semantic/conceptual queries
  {
    query: "how to structure REST endpoints",
    expectedDoc: "api-design",
    difficulty: "medium",
    description: "Conceptual - no exact match"
  },
  {
    query: "raising money for startup",
    expectedDoc: "fundraising",
    difficulty: "medium",
    description: "Conceptual - synonyms"
  },
  {
    query: "consistency vs availability tradeoffs",
    expectedDoc: "distributed-systems",
    difficulty: "medium",
    description: "Conceptual understanding"
  },
  {
    query: "how to prevent models from memorizing data",
    expectedDoc: "machine-learning",
    difficulty: "medium",
    description: "Conceptual - overfitting"
  },
  {
    query: "working from home guidelines",
    expectedDoc: "remote-work",
    difficulty: "medium",
    description: "Synonym match"
  },
  {
    query: "what went wrong with the launch",
    expectedDoc: "product-launch",
    difficulty: "medium",
    description: "Conceptual query"
  },

  // HARD: Vague, partial memory, indirect
  {
    query: "nouns not verbs",
    expectedDoc: "api-design",
    difficulty: "hard",
    description: "Partial phrase recall"
  },
  {
    query: "Sequoia investor pitch",
    expectedDoc: "fundraising",
    difficulty: "hard",
    description: "Indirect reference"
  },
  {
    query: "Raft algorithm leader election",
    expectedDoc: "distributed-systems",
    difficulty: "hard",
    description: "Specific detail in long doc"
  },
  {
    query: "F1 score precision recall",
    expectedDoc: "machine-learning",
    difficulty: "hard",
    description: "Technical detail"
  },
  {
    query: "quarterly team gathering travel",
    expectedDoc: "remote-work",
    difficulty: "hard",
    description: "Specific policy detail"
  },
  {
    query: "beta program 47 bugs",
    expectedDoc: "product-launch",
    difficulty: "hard",
    description: "Specific number recall"
  },
];

interface SearchResult {
  file: string;
  score: number;
  title: string;
}

function runSearch(query: string): SearchResult[] {
  try {
    const output = execSync(
      `bun src/qmd.ts search "${query.replace(/"/g, '\\"')}" --json -n 5 2>/dev/null`,
      { encoding: "utf-8", timeout: 30000 }
    );
    return JSON.parse(output);
  } catch (e) {
    return [];
  }
}

function runQuery(query: string): SearchResult[] {
  try {
    const output = execSync(
      `bun src/qmd.ts query "${query.replace(/"/g, '\\"')}" --json -n 5 2>/dev/null`,
      { encoding: "utf-8", timeout: 60000 }
    );
    return JSON.parse(output);
  } catch (e) {
    return [];
  }
}

function evaluate(mode: "search" | "query") {
  const runFn = mode === "search" ? runSearch : runQuery;
  const results = {
    easy: { total: 0, hit1: 0, hit3: 0, hit5: 0 },
    medium: { total: 0, hit1: 0, hit3: 0, hit5: 0 },
    hard: { total: 0, hit1: 0, hit3: 0, hit5: 0 },
  };

  console.log(`\n=== Evaluating ${mode.toUpperCase()} mode ===\n`);

  for (const { query, expectedDoc, difficulty, description } of evalQueries) {
    const searchResults = runFn(query);
    const ranks = searchResults
      .map((r, i) => ({ rank: i + 1, matches: r.file.toLowerCase().includes(expectedDoc) }))
      .filter(r => r.matches);

    const firstHit = ranks.length > 0 ? ranks[0]!.rank : -1;

    results[difficulty].total++;
    if (firstHit === 1) results[difficulty].hit1++;
    if (firstHit >= 1 && firstHit <= 3) results[difficulty].hit3++;
    if (firstHit >= 1 && firstHit <= 5) results[difficulty].hit5++;

    const status = firstHit === 1 ? "✓" : firstHit > 0 ? `@${firstHit}` : "✗";
    console.log(`[${difficulty.padEnd(6)}] ${status.padEnd(3)} "${query}" → ${description}`);
  }

  console.log("\n--- Summary ---");
  for (const [diff, r] of Object.entries(results)) {
    const hit1Pct = ((r.hit1 / r.total) * 100).toFixed(0);
    const hit3Pct = ((r.hit3 / r.total) * 100).toFixed(0);
    const hit5Pct = ((r.hit5 / r.total) * 100).toFixed(0);
    console.log(`${diff.padEnd(8)}: Hit@1=${hit1Pct}% Hit@3=${hit3Pct}% Hit@5=${hit5Pct}% (n=${r.total})`);
  }

  const total = evalQueries.length;
  const totalHit1 = Object.values(results).reduce((a, r) => a + r.hit1, 0);
  const totalHit3 = Object.values(results).reduce((a, r) => a + r.hit3, 0);
  console.log(`\nOverall: Hit@1=${((totalHit1/total)*100).toFixed(0)}% Hit@3=${((totalHit3/total)*100).toFixed(0)}%`);
}

// Main
console.log("QMD Evaluation Harness");
console.log("=".repeat(50));
console.log(`Testing ${evalQueries.length} queries across 6 documents`);

// Check if eval-docs collection exists
try {
  const status = execSync("bun src/qmd.ts status --json 2>/dev/null", { encoding: "utf-8" });
  if (!status.includes("eval-docs")) {
    console.log("\n⚠️  eval-docs collection not found. Run:");
    console.log("   qmd collection add test/eval-docs --name eval-docs");
    console.log("   qmd embed");
    process.exit(1);
  }
} catch {
  console.log("\n⚠️  Could not check status. Make sure qmd is working.");
}

// Run evaluations
evaluate("search");
evaluate("query");
