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
  type ExpandedQuery,
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
  run: (store: QMDStore, query: BenchmarkQuery, limit: number, collection?: string) => Promise<string[]>;
};

type ParsedStructuredQuery = {
  searches: ExpandedQuery[];
  intent?: string;
};

function parseStructuredQuery(query: string): ParsedStructuredQuery | undefined {
  const lines = query.split("\n").map((line, idx) => ({
    trimmed: line.trim(),
    number: idx + 1,
  })).filter(line => line.trimmed.length > 0);

  if (lines.length === 0) return undefined;

  const prefixRe = /^(lex|vec|hyde):\s*/i;
  const intentRe = /^intent:\s*/i;
  const searches: ExpandedQuery[] = [];
  let intent: string | undefined;

  for (const line of lines) {
    if (intentRe.test(line.trimmed)) {
      if (intent !== undefined) {
        throw new Error(`Line ${line.number}: only one intent: line is allowed per benchmark query.`);
      }
      intent = line.trimmed.replace(intentRe, "").trim();
      if (!intent) {
        throw new Error(`Line ${line.number}: intent: must include text.`);
      }
      continue;
    }

    const match = line.trimmed.match(prefixRe);
    if (match) {
      const type = match[1]!.toLowerCase() as "lex" | "vec" | "hyde";
      const text = line.trimmed.slice(match[0].length).trim();
      if (!text) {
        throw new Error(`Line ${line.number} (${type}:) must include text.`);
      }
      searches.push({ type, query: text, line: line.number });
      continue;
    }

    if (lines.length === 1) {
      return undefined;
    }

    throw new Error(`Line ${line.number} is missing a lex:/vec:/hyde:/intent: prefix.`);
  }

  if (intent && searches.length === 0) {
    throw new Error("intent: cannot appear alone. Add at least one lex:, vec:, or hyde: line.");
  }

  return searches.length > 0 ? { searches, intent } : undefined;
}

function uniqueFiles(files: string[], limit: number): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const file of files) {
    if (seen.has(file)) continue;
    seen.add(file);
    out.push(file);
    if (out.length >= limit) break;
  }
  return out;
}

const BACKENDS: Backend[] = [
  {
    name: "bm25",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const lexQueries = structured?.searches.filter(q => q.type === "lex");
      if (structured) {
        const files: string[] = [];
        for (const lex of lexQueries ?? []) {
          const results = await store.searchLex(lex.query, { limit, collection });
          files.push(...results.map((r: SearchResult) => r.filepath));
        }
        return uniqueFiles(files, limit);
      }

      const results = await store.searchLex(query.query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    },
  },
  {
    name: "vector",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const vectorQueries = structured?.searches.filter(q => q.type === "vec" || q.type === "hyde");
      if (structured) {
        const files: string[] = [];
        for (const vectorQuery of vectorQueries ?? []) {
          const results = await store.searchVector(vectorQuery.query, { limit, collection });
          files.push(...results.map((r: SearchResult) => r.filepath));
        }
        return uniqueFiles(files, limit);
      }

      const results = await store.searchVector(query.query, { limit, collection });
      return results.map((r: SearchResult) => r.filepath);
    },
  },
  {
    name: "hybrid",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const results = structured
        ? await store.search({ queries: structured.searches, intent: structured.intent, limit, collection, rerank: false })
        : await store.search({ query: query.query, limit, collection, rerank: false });
      return results.map((r: HybridQueryResult) => r.file);
    },
  },
  {
    name: "full",
    run: async (store, query, limit, collection) => {
      const structured = parseStructuredQuery(query.query);
      const results = structured
        ? await store.search({ queries: structured.searches, intent: structured.intent, limit, collection, rerank: true })
        : await store.search({ query: query.query, limit, collection, rerank: true });
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
    resultFiles = await backend.run(store, query, limit, collection);
  } catch {
    // Backend may not be available (e.g., no embeddings for vector search)
    return {
      precision_at_k: 0,
      recall: 0,
      recall_at_1: 0,
      recall_at_3: 0,
      recall_at_5: 0,
      mrr: 0,
      f1: 0,
      hits_at_k: 0,
      total_expected: query.expected_files.length,
      latency_ms: Date.now() - start,
      top_files: [],
      matched_files: [],
      unmatched_expected_files: query.expected_files,
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
    `${pad("Query", 25)} ${pad("Backend", 8)} ${pad("P@k", 6)} ${pad("R@1", 6)} ${pad("R@3", 6)} ${pad("R@5", 6)} ${pad("MRR", 6)} ${pad("F1", 6)} ${pad("ms", 8)}`
  );
  lines.push("-".repeat(88));

  for (const r of results) {
    for (const [backend, br] of Object.entries(r.backends)) {
      lines.push(
        `${pad(r.id, 25)} ${pad(backend, 8)} ${num(br.precision_at_k)} ${num(br.recall_at_1)} ${num(br.recall_at_3)} ${num(br.recall_at_5)} ${num(br.mrr)} ${num(br.f1)} ${String(Math.round(br.latency_ms)).padStart(7)}ms`
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

  for (const name of Array.from(backendNames)) {
    let totalP = 0, totalR = 0, totalR1 = 0, totalR3 = 0, totalR5 = 0, totalMrr = 0, totalF1 = 0, totalLat = 0, count = 0;
    for (const r of results) {
      const br = r.backends[name];
      if (!br) continue;
      totalP += br.precision_at_k;
      totalR += br.recall;
      totalR1 += br.recall_at_1;
      totalR3 += br.recall_at_3;
      totalR5 += br.recall_at_5;
      totalMrr += br.mrr;
      totalF1 += br.f1;
      totalLat += br.latency_ms;
      count++;
    }
    if (count > 0) {
      summary[name] = {
        avg_precision: totalP / count,
        avg_recall: totalR / count,
        avg_recall_at_1: totalR1 / count,
        avg_recall_at_3: totalR3 / count,
        avg_recall_at_5: totalR5 / count,
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
  options: { json?: boolean; collection?: string; backends?: string[]; dbPath?: string; configPath?: string } = {},
): Promise<BenchmarkResult> {
  // Load fixture
  const raw = readFileSync(resolve(fixturePath), "utf-8");
  const fixture: BenchmarkFixture = JSON.parse(raw);

  if (!fixture.queries || !Array.isArray(fixture.queries)) {
    throw new Error("Invalid fixture: missing 'queries' array");
  }

  // Open store
  const store = await createStore({
    dbPath: options.dbPath ?? getDefaultDbPath(),
    ...(options.configPath ? { configPath: options.configPath } : {}),
  });

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
        `  ${pad(name, 8)} P@k=${num(s.avg_precision)} R@1=${num(s.avg_recall_at_1)} R@3=${num(s.avg_recall_at_3)} R@5=${num(s.avg_recall_at_5)} MRR=${num(s.avg_mrr)} F1=${num(s.avg_f1)} Avg=${Math.round(s.avg_latency_ms)}ms`
      );
    }
  }

  return benchResult;
}
