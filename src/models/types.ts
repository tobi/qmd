/**
 * Type definitions for QMD
 */

// Reranking types
export type LogProb = { token: string; logprob: number };

export type RerankResponse = {
  model: string;
  created_at: string;
  response: string;
  done: boolean;
  done_reason?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
  logprobs?: LogProb[];
};

// Search result types
export type SearchResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
  source: "fts" | "vec";
  chunkPos?: number;
};

export type RankedResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
};

// Output types
export type OutputFormat = "cli" | "csv" | "md" | "xml" | "files" | "json";

export type OutputOptions = {
  format: OutputFormat;
  full: boolean;
  limit: number;
  minScore: number;
  all?: boolean;
};

// Database entity types
export interface Collection {
  id: number;
  pwd: string;
  glob_pattern: string;
  created_at: string;
  context?: string;
}

export interface Document {
  id: number;
  collection_id: number;
  name?: string;
  filepath: string;
  hash: string;
  title: string;
  body: string;
  active: number;
  created_at?: string;
  modified_at: string;
  display_path?: string;
}

export interface ContentVector {
  hash: string;
  seq: number;
  pos: number;
  embedding: Float32Array;
}

export interface PathContext {
  id?: number;
  path_prefix: string;
  context: string;
  created_at?: string;
}

export interface OllamaCache {
  hash: string;
  result: string;
  created_at: string;
}
