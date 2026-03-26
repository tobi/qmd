/**
 * QMD Benchmark Harness
 *
 * Runs queries from a fixture file against multiple search backends
 * and measures precision@k, recall, MRR, F1, and latency.
 *
 * Usage:
 *   qmd bench <fixture.json> [--json] [--collection <name>]
 *
 * Backends tested:
 *   - bm25: BM25 keyword search (searchLex)
 *   - vector: Vector similarity search (searchVector)
 *   - hybrid: BM25 + vector RRF fusion without reranking
 *   - full: Full hybrid pipeline with LLM reranking
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  createStore,
  getDefaultDbPath,
  type QMDStore,
  type SearchResult,
  type HybridQueryResult,
} from "../index.js";
import { scoreResults } from "./score.js";
import type {
  BenchmarkFixture,
  BenchmarkQuery,
  BackendResult,
  QueryResult,
  BenchmarkResult,
} from "./types.js";

type Backend = {
  name: string;
  run: (store: QMDStore, query: string, limit: number, collection?: string) => Promise<string[]>;
};

const BACKENDS: Backend[] = [
  {
    name: "bm25",
    run: async (store, query, limit, collection) => {
      const results = await store.searchLex(query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    },
  },
  {
    name: "vector",
    run: async (store, query, limit, collection) => {
      const results = await store.searchVector(query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    },
  },
  {
    name: "hybrid",
    run: async (store, query, limit, collection) => {
      const results = await store.search({ query, limit, collection, rerank: false });
      return results.map((r: HybridQueryResult) => r.file);
    },
  },
  {
    name: "full",
    run: async (store, query, limit, collection) => {
      const results = await store.search({ query, limit, collection, rerank: true });
      return results.map((r: HybridQueryResult) => r.file);
    },
  },
];

async function runQuery(
  store: QMDStore,
  backend: Backend,
  query: BenchmarkQuery,
  collection?: string,
): Promise<BackendResult> {
  const limit = Math.max(query.expected_in_top_k, 10);
  const start = Date.now();

  let resultFiles: string[];
  try {
    resultFiles = await backend.run(store, query.query, limit, collection);
  } catch (err: any) {
    // Backend may not be available (e.g., no embeddings for vector search)
    return {
      precision_at_k: 0,
      recall: 0,
      mrr: 0,
      f1: 0,
      hits_at_k: 0,
      total_expected: query.expected_files.length,
      latency_ms: Date.now() - start,
      top_files: [],
    };
  }

  const latency_ms = Date.now() - start;
  const scores = scoreResults(resultFiles, query.expected_files, query.expected_in_top_k);

  return {
    ...scores,
    total_expected: query.expected_files.length,
    latency_ms,
    top_files: resultFiles.slice(0, 10),
  };
}

function formatTable(results: QueryResult[]): string {
  const lines: string[] = [];
  const pad = (s: string, n: number) => s.slice(0, n).padEnd(n);
  const num = (n: number) => n.toFixed(2).padStart(5);

  lines.push(
    `${pad("Query", 25)} ${pad("Backend", 8)} ${pad("P@k", 6)} ${pad("Recall", 7)} ${pad("MRR", 6)} ${pad("F1", 6)} ${pad("ms", 8)}`
  );
  lines.push("-".repeat(70));

  for (const r of results) {
    for (const [backend, br] of Object.entries(r.backends)) {
      lines.push(
        `${pad(r.id, 25)} ${pad(backend, 8)} ${num(br.precision_at_k)} ${num(br.recall)}  ${num(br.mrr)} ${num(br.f1)} ${String(Math.round(br.latency_ms)).padStart(7)}ms`
      );
    }
    lines.push("");
  }

  return lines.join("\n");
}

function computeSummary(results: QueryResult[]): BenchmarkResult["summary"] {
  const summary: BenchmarkResult["summary"] = {};

  // Collect all backend names
  const backendNames = new Set<string>();
  for (const r of results) {
    for (const name of Object.keys(r.backends)) {
      backendNames.add(name);
    }
  }

  for (const name of backendNames) {
    let totalP = 0, totalR = 0, totalMrr = 0, totalF1 = 0, totalLat = 0, count = 0;
    for (const r of results) {
      const br = r.backends[name];
      if (!br) continue;
      totalP += br.precision_at_k;
      totalR += br.recall;
      totalMrr += br.mrr;
      totalF1 += br.f1;
      totalLat += br.latency_ms;
      count++;
    }
    if (count > 0) {
      summary[name] = {
        avg_precision: totalP / count,
        avg_recall: totalR / count,
        avg_mrr: totalMrr / count,
        avg_f1: totalF1 / count,
        avg_latency_ms: totalLat / count,
      };
    }
  }

  return summary;
}

export async function runBenchmark(
  fixturePath: string,
  options: { json?: boolean; collection?: string; backends?: string[] } = {},
): Promise<BenchmarkResult> {
  // Load fixture
  const raw = readFileSync(resolve(fixturePath), "utf-8");
  const fixture: BenchmarkFixture = JSON.parse(raw);

  if (!fixture.queries || !Array.isArray(fixture.queries)) {
    throw new Error("Invalid fixture: missing 'queries' array");
  }

  // Open store
  const store = await createStore({ dbPath: getDefaultDbPath() });

  // Filter backends if requested
  const activeBackends = options.backends
    ? BACKENDS.filter(b => options.backends!.includes(b.name))
    : BACKENDS;

  const collection = options.collection ?? fixture.collection;

  // Run queries
  const results: QueryResult[] = [];
  for (const query of fixture.queries) {
    const backends: Record<string, BackendResult> = {};

    for (const backend of activeBackends) {
      if (!options.json) {
        process.stderr.write(`  ${query.id} / ${backend.name}...`);
      }
      backends[backend.name] = await runQuery(store, backend, query, collection);
      if (!options.json) {
        process.stderr.write(` ${Math.round(backends[backend.name]!.latency_ms)}ms\n`);
      }
    }

    results.push({
      id: query.id,
      query: query.query,
      type: query.type,
      backends,
    });
  }

  await store.close();

  const summary = computeSummary(results);
  const timestamp = new Date().toISOString().replace(/[:.]/g, "").slice(0, 15);

  const benchResult: BenchmarkResult = {
    timestamp,
    fixture: fixturePath,
    results,
    summary,
  };

  // Output
  if (options.json) {
    console.log(JSON.stringify(benchResult, null, 2));
  } else {
    console.log("\n" + formatTable(results));
    console.log("Summary:");
    console.log("-".repeat(70));
    const pad = (s: string, n: number) => s.slice(0, n).padEnd(n);
    const num = (n: number) => n.toFixed(3).padStart(6);
    for (const [name, s] of Object.entries(summary)) {
      console.log(
        `  ${pad(name, 8)} P@k=${num(s.avg_precision)} Recall=${num(s.avg_recall)} MRR=${num(s.avg_mrr)} F1=${num(s.avg_f1)} Avg=${Math.round(s.avg_latency_ms)}ms`
      );
    }
  }

  return benchResult;
}
