/**
 * Types for the QMD benchmark harness.
 *
 * A benchmark fixture defines queries with expected results.
 * The harness runs each query through multiple search backends
 * and measures precision, recall, MRR, and latency.
 */

export interface BenchmarkQuery {
  /** Unique identifier for the query */
  id: string;
  /** The search query text */
  query: string;
  /** Query difficulty/type for grouping results */
  type: "exact" | "semantic" | "topical" | "cross-domain" | "alias";
  /** Human-readable description of what this tests */
  description: string;
  /** File paths (relative to collection) that should appear in results */
  expected_files: string[];
  /** How many of expected_files should appear in top-k results */
  expected_in_top_k: number;
}

export interface BenchmarkFixture {
  /** Description of the benchmark */
  description: string;
  /** Fixture format version */
  version: number;
  /** Optional collection to search within */
  collection?: string;
  /** The test queries */
  queries: BenchmarkQuery[];
}

export interface BackendResult {
  /** Fraction of top-k results that are relevant */
  precision_at_k: number;
  /** Fraction of expected files found anywhere in results */
  recall: number;
  /** Fraction of expected files found in the first result */
  recall_at_1: number;
  /** Fraction of expected files found in the top 3 results */
  recall_at_3: number;
  /** Fraction of expected files found in the top 5 results */
  recall_at_5: number;
  /** Reciprocal rank of first relevant result (1/rank, 0 if not found) */
  mrr: number;
  /** Harmonic mean of precision_at_k and recall */
  f1: number;
  /** Number of expected files found in top-k */
  hits_at_k: number;
  /** Total expected files */
  total_expected: number;
  /** Wall-clock latency in milliseconds */
  latency_ms: number;
  /** Top result file paths (for inspection) */
  top_files: string[];
  /** Expected files that were found anywhere in the returned result set */
  matched_files: string[];
  /** Expected files missing from the returned result set */
  unmatched_expected_files: string[];
}

export interface QueryResult {
  id: string;
  query: string;
  type: string;
  backends: Record<string, BackendResult>;
}

export interface BenchmarkResult {
  timestamp: string;
  fixture: string;
  results: QueryResult[];
  summary: Record<string, {
    avg_precision: number;
    avg_recall: number;
    avg_recall_at_1: number;
    avg_recall_at_3: number;
    avg_recall_at_5: number;
    avg_mrr: number;
    avg_f1: number;
    avg_latency_ms: number;
  }>;
}
