// =============================================================================
// llm-types.ts - Shared types for the LLM interface and option bags.
//
// This file was extracted from src/llm.ts as part of the remote-LLM port
// (see docs/qmd-remote-llm-port.md). It contains ONLY types — no runtime
// code, no node-llama-cpp imports — so it is safe to import from any
// backend (local LlamaCpp, remote OpenAI-compatible API, or HybridLLM).
//
// LLM interface (locked decision D1): exactly 9 methods:
//   1. embed
//   2. embedBatch         (new — promoted from LlamaCpp)
//   3. generate
//   4. modelExists
//   5. expandQuery
//   6. rerank
//   7. tokenize           (new — for HybridLLM proxy through store.ts)
//   8. detokenize         (new — for HybridLLM proxy through store.ts)
//   9. dispose
//
// `getDeviceInfo` is intentionally NOT part of the LLM interface. It stays
// a concrete method on LlamaCpp (and a thin proxy on HybridLLM that reads
// from its local backend). qmd status calls it on the concrete singleton.
// =============================================================================

/**
 * Token with log probability
 */
export type TokenLogProb = {
  token: string;
  logprob: number;
};

/**
 * Embedding result
 */
export type EmbeddingResult = {
  embedding: number[];
  model: string;
};

/**
 * Generation result with optional logprobs
 */
export type GenerateResult = {
  text: string;
  model: string;
  logprobs?: TokenLogProb[];
  done: boolean;
};

/**
 * Rerank result for a single document
 */
export type RerankDocumentResult = {
  file: string;
  score: number;
  index: number;
};

/**
 * Batch rerank result
 */
export type RerankResult = {
  results: RerankDocumentResult[];
  model: string;
};

/**
 * Model info
 */
export type ModelInfo = {
  name: string;
  exists: boolean;
  path?: string;
};

/**
 * Options for embedding
 */
export type EmbedOptions = {
  model?: string;
  isQuery?: boolean;
  title?: string;
};

/**
 * Options for query expansion. `intent` is a domain-context hint (e.g.
 * "web page load times") that steers expansion and reranking. It existed on
 * LlamaCpp.expandQuery pre-port; kept on the interface so callers like
 * store.expandQuery can pass it through unchanged.
 */
export type ExpandOptions = {
  context?: string;
  includeLexical?: boolean;
  intent?: string;
};

/**
 * Options for text generation
 */
export type GenerateOptions = {
  model?: string;
  maxTokens?: number;
  temperature?: number;
};

/**
 * Options for reranking
 */
export type RerankOptions = {
  model?: string;
  /** Max characters of each chunk sent to the reranker (remote only). Default: 1200. Override with QMD_RERANK_CHUNK_CHARS. */
  chunkChars?: number;
};

/**
 * Session options for ILLMSession lifecycle
 */
export type LLMSessionOptions = {
  maxDuration?: number;
  name?: string;
};

/**
 * Supported query types for different search backends
 */
export type QueryType = 'lex' | 'vec' | 'hyde';

/**
 * A single query and its target backend type
 */
export type Queryable = {
  type: QueryType;
  text: string;
};

/**
 * Document to rerank
 */
export type RerankDocument = {
  file: string;
  text: string;
  title?: string;
};

// =============================================================================
// LLM Interface (9 methods — locked decision D1)
// =============================================================================

/**
 * Abstract LLM interface — implement this for different backends
 * (LlamaCpp locally, RemoteLLM over HTTP, HybridLLM routing between them).
 *
 * NOTE: `getDeviceInfo()` is intentionally NOT on this interface. It remains a
 * concrete method on LlamaCpp and HybridLLM (proxied to local). The CLI `qmd
 * status` calls it on the concrete singleton. See docs/qmd-remote-llm-port.md
 * §"Interface shape — final decision".
 */
export interface LLM {
  /**
   * Get embeddings for text
   */
  embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null>;

  /**
   * Batch embed multiple texts efficiently. Already implemented on LlamaCpp
   * in mainline; promoted to the interface as part of the port.
   */
  embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]>;

  /**
   * Generate text completion
   */
  generate(prompt: string, options?: GenerateOptions): Promise<GenerateResult | null>;

  /**
   * Check if a model exists/is available
   */
  modelExists(model: string): Promise<ModelInfo>;

  /**
   * Expand a search query into multiple variations for different backends.
   * Returns a list of Queryable objects.
   *
   * `intent` is an optional domain-context hint (e.g. "web page load times")
   * carried through from mainline's pre-port LlamaCpp.expandQuery signature;
   * the local backend uses it to bias expansion, the remote backend ignores
   * it (purely additive).
   */
  expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]>;

  /**
   * Rerank documents by relevance to a query
   * Returns list of documents with relevance scores (higher = more relevant)
   */
  rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult>;

  /**
   * Tokenize text. Used by chunkDocumentByTokens in store.ts. Returns
   * tokenizer tokens (opaque type; the local backend returns node-llama-cpp
   * LlamaToken, the remote backend returns a regex split approximation —
   * both are `readonly any[]` at the interface boundary).
   */
  tokenize(text: string): Promise<readonly any[]>;

  /**
   * Detokenize tokens back to text.
   */
  detokenize(tokens: readonly any[]): Promise<string>;

  /**
   * Dispose of resources
   */
  dispose(): Promise<void>;
}
