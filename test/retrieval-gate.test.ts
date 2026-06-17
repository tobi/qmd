/**
 * Retrieval regression gate for CI.
 *
 * Uses a frozen document/query set to track:
 * - Recall@3
 * - NDCG@3
 * - p95 query latency (ms)
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync, readFileSync, readdirSync } from "fs";
import { join, dirname } from "path";
import { tmpdir } from "os";
import { createHash } from "crypto";
import { fileURLToPath } from "url";
import type { Database } from "../src/db.js";
import {
  createStore,
  searchFTS,
  insertDocument,
  insertContent,
} from "../src/store";

const tempDir = mkdtempSync(join(tmpdir(), "qmd-retrieval-gate-"));
process.env.INDEX_PATH = join(tempDir, "retrieval-gate.sqlite");

afterAll(() => {
  rmSync(tempDir, { recursive: true, force: true });
});

const frozenQueries: Array<{ query: string; expectedDoc: string }> = [
  { query: "API versioning", expectedDoc: "api-design" },
  { query: "Series A fundraising", expectedDoc: "fundraising" },
  { query: "CAP theorem", expectedDoc: "distributed-systems" },
  { query: "overfitting machine learning", expectedDoc: "machine-learning" },
  { query: "remote work VPN", expectedDoc: "remote-work" },
  { query: "Project Phoenix retrospective", expectedDoc: "product-launch" },
  { query: "how to structure REST endpoints", expectedDoc: "api-design" },
  { query: "raising money for startup", expectedDoc: "fundraising" },
  { query: "consistency vs availability tradeoffs", expectedDoc: "distributed-systems" },
  { query: "how to prevent models from memorizing data", expectedDoc: "machine-learning" },
  { query: "working from home guidelines", expectedDoc: "remote-work" },
  { query: "what went wrong with the launch", expectedDoc: "product-launch" },
  { query: "nouns not verbs", expectedDoc: "api-design" },
  { query: "Sequoia investor pitch", expectedDoc: "fundraising" },
  { query: "Raft algorithm leader election", expectedDoc: "distributed-systems" },
  { query: "F1 score precision recall", expectedDoc: "machine-learning" },
  { query: "quarterly team gathering travel", expectedDoc: "remote-work" },
  { query: "beta program 47 bugs", expectedDoc: "product-launch" },
  { query: "how much runway before running out of money", expectedDoc: "fundraising" },
  { query: "datacenter replication sync strategy", expectedDoc: "distributed-systems" },
  { query: "splitting data for training and testing", expectedDoc: "machine-learning" },
  { query: "JSON response codes error messages", expectedDoc: "api-design" },
  { query: "video calls camera async messaging", expectedDoc: "remote-work" },
  { query: "CI/CD pipeline testing coverage", expectedDoc: "product-launch" },
];

function matchesExpected(filepath: string, expectedDoc: string): boolean {
  return filepath.toLowerCase().includes(expectedDoc);
}

function dcgForRank(rank: number): number {
  return 1 / Math.log2(rank + 1);
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx] ?? 0;
}

describe("retrieval regression gate", () => {
  let db: Database;
  let store: ReturnType<typeof createStore>;

  beforeAll(() => {
    store = createStore();
    db = store.db;

    const evalDocsDir = join(dirname(fileURLToPath(import.meta.url)), "eval-docs");
    const files = readdirSync(evalDocsDir).filter(f => f.endsWith(".md"));

    for (const file of files) {
      const content = readFileSync(join(evalDocsDir, file), "utf-8");
      const title = content.split("\n")[0]?.replace(/^#\s*/, "") || file;
      const hash = createHash("sha256").update(content).digest("hex").slice(0, 12);
      const now = new Date().toISOString();

      insertContent(db, hash, content, now);
      insertDocument(db, "eval-docs", file, title, hash, now, now);
    }
  });

  afterAll(() => {
    store.close();
  });

  test("meets frozen-set quality and latency thresholds", () => {
    const k = 3;
    const latenciesMs: number[] = [];
    let recallHits = 0;
    let totalNdcg = 0;

    // Warm-up pass to reduce first-query startup noise.
    for (const { query } of frozenQueries) {
      searchFTS(db, query, k);
    }

    for (const { query, expectedDoc } of frozenQueries) {
      const start = process.hrtime.bigint();
      const results = searchFTS(db, query, k);
      const end = process.hrtime.bigint();
      latenciesMs.push(Number(end - start) / 1_000_000);

      const hitIndex = results.findIndex(r => matchesExpected(r.filepath, expectedDoc));
      if (hitIndex >= 0 && hitIndex < k) {
        recallHits++;
        totalNdcg += dcgForRank(hitIndex + 1);
      }
    }

    const recallAt3 = recallHits / frozenQueries.length;
    const ndcgAt3 = totalNdcg / frozenQueries.length;
    const p95LatencyMs = percentile(latenciesMs, 0.95);

    // Keep these thresholds stable and conservative to reduce CI flakes.
    expect(recallAt3).toBeGreaterThanOrEqual(0.4);
    expect(ndcgAt3).toBeGreaterThanOrEqual(0.3);
    expect(p95LatencyMs).toBeLessThanOrEqual(50);
  });
});
