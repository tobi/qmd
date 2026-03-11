/**
 * llm-remote.ts - Remote LLM backend for QMD using HTTP endpoints
 *
 * Provides embeddings, text generation, and reranking via remote llama.cpp servers.
 */

import { homedir } from "os";
import { join } from "path";
import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";

import type {
  LLM,
  EmbeddingResult,
  GenerateResult,
  ModelInfo,
  EmbedOptions,
  GenerateOptions,
  RerankOptions,
  RerankDocument,
  RerankDocumentResult,
  RerankResult,
  Queryable,
  QueryType,
  ILLMSession,
  LLMSessionOptions,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

export type RemoteLLMConfig = {
  embedUrl?: string;     // e.g. "http://192.168.1.100:8081"
  rerankUrl?: string;    // e.g. "http://192.168.1.100:8082"
  generateUrl?: string;  // e.g. "http://192.168.1.100:8083" or "http://localhost:4000" (LiteLLM)
  generateModel?: string; // e.g. "gpt-4o-mini" or "ollama/llama3" - required for LiteLLM, optional for llama.cpp
};

// Config file path
const CONFIG_DIR = join(homedir(), ".cache", "qmd");
const CONFIG_FILE = join(CONFIG_DIR, "config.json");

/**
 * Load remote config from file
 */
export function loadRemoteConfig(): RemoteLLMConfig {
  try {
    if (existsSync(CONFIG_FILE)) {
      const data = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
      return data.remote || {};
    }
  } catch (e) {
    // Ignore errors, return empty config
  }
  return {};
}

/**
 * Save remote config to file
 */
export function saveRemoteConfig(config: RemoteLLMConfig): void {
  try {
    if (!existsSync(CONFIG_DIR)) {
      mkdirSync(CONFIG_DIR, { recursive: true });
    }
    
    let data: Record<string, unknown> = {};
    if (existsSync(CONFIG_FILE)) {
      try {
        data = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
      } catch {
        // Start fresh if parse fails
      }
    }
    
    data.remote = config;
    writeFileSync(CONFIG_FILE, JSON.stringify(data, null, 2));
  } catch (e) {
    console.error("Failed to save remote config:", e);
  }
}

/**
 * Clear remote config
 */
export function clearRemoteConfig(): void {
  try {
    if (existsSync(CONFIG_FILE)) {
      let data: Record<string, unknown> = {};
      try {
        data = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
      } catch {
        // Start fresh
      }
      delete data.remote;
      writeFileSync(CONFIG_FILE, JSON.stringify(data, null, 2));
    }
  } catch (e) {
    console.error("Failed to clear remote config:", e);
  }
}

// =============================================================================
// QMD Directory Config (persistent path to .qmd folder)
// =============================================================================

/**
 * Load saved qmdDir from config
 */
export function loadQmdDirConfig(): string | null {
  try {
    if (existsSync(CONFIG_FILE)) {
      const data = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
      return data.qmdDir || null;
    }
  } catch {
    // Ignore errors
  }
  return null;
}

/**
 * Save qmdDir to config
 */
export function saveQmdDirConfig(qmdDir: string): void {
  try {
    if (!existsSync(CONFIG_DIR)) {
      mkdirSync(CONFIG_DIR, { recursive: true });
    }
    
    let data: Record<string, unknown> = {};
    if (existsSync(CONFIG_FILE)) {
      try {
        data = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
      } catch {
        // Start fresh
      }
    }
    
    data.qmdDir = qmdDir;
    writeFileSync(CONFIG_FILE, JSON.stringify(data, null, 2));
  } catch (e) {
    console.error("Failed to save qmdDir config:", e);
  }
}

/**
 * Clear qmdDir from config
 */
export function clearQmdDirConfig(): void {
  try {
    if (existsSync(CONFIG_FILE)) {
      let data: Record<string, unknown> = {};
      try {
        data = JSON.parse(readFileSync(CONFIG_FILE, "utf-8"));
      } catch {
        // Start fresh
      }
      delete data.qmdDir;
      writeFileSync(CONFIG_FILE, JSON.stringify(data, null, 2));
    }
  } catch {
    // Ignore errors
  }
}

/**
 * Check if remote mode is configured
 */
export function isRemoteConfigured(): boolean {
  const config = loadRemoteConfig();
  return !!(config.embedUrl || config.rerankUrl || config.generateUrl);
}

// =============================================================================
// Remote LLM Implementation
// =============================================================================

/**
 * LLM implementation using remote HTTP endpoints (llama.cpp servers)
 */
export class RemoteLLM implements LLM {
  private embedUrl: string | null;
  private rerankUrl: string | null;
  private generateUrl: string | null;
  private generateModel: string | null;

  constructor(config: RemoteLLMConfig = {}) {
    // Load from saved config, then override with explicit config
    const savedConfig = loadRemoteConfig();
    this.embedUrl = config.embedUrl ?? savedConfig.embedUrl ?? null;
    this.rerankUrl = config.rerankUrl ?? savedConfig.rerankUrl ?? null;
    this.generateUrl = config.generateUrl ?? savedConfig.generateUrl ?? null;
    this.generateModel = config.generateModel ?? savedConfig.generateModel ?? null;
  }

  /**
   * Get embeddings via remote server (retries up to 3 times on transient errors)
   */
  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    if (!this.embedUrl) {
      console.error("No embed URL configured");
      return null;
    }

    const MAX_RETRIES = 3;
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const response = await fetch(`${this.embedUrl}/v1/embeddings`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            input: text,
            model: "embeddinggemma",
          }),
        });

        if (response.status === 400) {
          // Client error — retrying won't help
          console.error(`Embed request failed: ${response.status} ${response.statusText}`);
          return null;
        }

        if (!response.ok) {
          console.error(`Embed request failed: ${response.status} ${response.statusText}`);
          if (attempt < MAX_RETRIES - 1) {
            await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
            continue;
          }
          return null;
        }

        const data = await response.json() as {
          data: Array<{ embedding: number[] }>;
          model: string;
        };

        if (!data.data || data.data.length === 0) {
          console.error("No embedding data in response");
          return null;
        }

        return {
          embedding: data.data[0]!.embedding,
          model: data.model || "remote-embed",
        };
      } catch (error) {
        if (attempt < MAX_RETRIES - 1) {
          await new Promise(r => setTimeout(r, 1000 * (attempt + 1)));
          continue;
        }
        console.error("Embedding error:", error);
        return null;
      }
    }
    return null;
  }

  /**
   * Batch embed multiple texts in a single API call.
   * On batch failure, falls back to sequential individual requests (which have their own retries).
   */
  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];
    if (!this.embedUrl) {
      return texts.map(() => null);
    }

    try {
      // Send all texts in a single request (OpenAI API supports array input)
      const response = await fetch(`${this.embedUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: texts,
          model: "embeddinggemma",
        }),
      });

      if (!response.ok) {
        console.error(`Batch embed failed: ${response.status} ${response.statusText}`);
        // Fall back to sequential individual requests (each has retries)
        const results: (EmbeddingResult | null)[] = [];
        for (const text of texts) {
          results.push(await this.embed(text));
        }
        return results;
      }

      const data = await response.json() as {
        data: Array<{ embedding: number[]; index: number }>;
        model: string;
      };

      if (!data.data || data.data.length === 0) {
        console.error("No embedding data in batch response");
        return texts.map(() => null);
      }

      // Map results back to original order (API may return in different order)
      const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);
      for (const item of data.data) {
        if (item.index < texts.length) {
          results[item.index] = {
            embedding: item.embedding,
            model: data.model || "remote-embed",
          };
        }
      }
      return results;
    } catch (error) {
      console.error("Batch embedding error:", error);
      // Fall back to sequential individual requests (each has retries)
      const results: (EmbeddingResult | null)[] = [];
      for (const text of texts) {
        results.push(await this.embed(text));
      }
      return results;
    }
  }

  /**
   * Generate text via remote server
   */
  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    if (!this.generateUrl) {
      console.error("No generate URL configured");
      return null;
    }

    try {
      const body: Record<string, unknown> = {
        prompt,
        max_tokens: options.maxTokens ?? 150,
        temperature: options.temperature ?? 0,
      };
      if (this.generateModel) {
        body.model = this.generateModel;
      }
      const response = await fetch(`${this.generateUrl}/v1/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        console.error(`Generate request failed: ${response.status} ${response.statusText}`);
        return null;
      }

      const data = await response.json() as {
        choices: Array<{ text: string }>;
        model: string;
      };

      if (!data.choices || data.choices.length === 0) {
        console.error("No choices in response");
        return null;
      }

      return {
        text: data.choices[0]!.text,
        model: data.model || "remote-generate",
        done: true,
      };
    } catch (error) {
      console.error("Generate error:", error);
      return null;
    }
  }

  /**
   * Check if model exists (always returns true for remote)
   */
  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  /**
   * Expand a search query into multiple variations
   */
  async expandQuery(
    query: string,
    options: { context?: string; includeLexical?: boolean } = {}
  ): Promise<Queryable[]> {
    if (!this.generateUrl) {
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (options.includeLexical !== false) {
        fallback.unshift({ type: 'lex', text: query });
      }
      return fallback;
    }

    const includeLexical = options.includeLexical ?? true;
    const context = options.context;

    const prompt = `You are a search query optimization expert. Your task is to improve retrieval by rewriting queries and generating hypothetical documents.

Original Query: ${query}

${context ? `Additional Context, ONLY USE IF RELEVANT:\n\n<context>${context}</context>` : ""}

## Step 1: Query Analysis
Identify entities, search intent, and missing context.

## Step 2: Generate Hypothetical Document
Write a focused sentence passage that would answer the query. Include specific terminology and domain vocabulary.

## Step 3: Query Rewrites
Generate 2-3 alternative search queries that resolve ambiguities. Use terminology from the hypothetical document.

## Step 4: Final Retrieval Text
Output MAX ONE 'hyde' line FIRST, then 1-3 'lex' lines, then 1-3 'vec' lines.

<format>
hyde: {complete hypothetical document passage from Step 2 on a SINGLE LINE}
lex: {single search term}
vec: {single vector query}
</format>

<example>
Example (FOR FORMAT ONLY - DO NOT COPY THIS CONTENT):
hyde: This is an example of a hypothetical document passage that would answer the example query. It contains multiple sentences and relevant vocabulary.
lex: example keyword 1
lex: example keyword 2
vec: example semantic query
</example>

<rules>
- DO NOT repeat the same line.
- Each 'lex:' line MUST be a different keyword variation based on the ORIGINAL QUERY.
- Each 'vec:' line MUST be a different semantic variation based on the ORIGINAL QUERY.
- The 'hyde:' line MUST be the full sentence passage from Step 2, but all on one line.
- DO NOT use the example content above.
${!includeLexical ? "- Do NOT output any 'lex:' lines" : ""}
</rules>

Final Output:`;

    try {
      const result = await this.generate(prompt, { maxTokens: 1000, temperature: 1 });
      if (!result) {
        throw new Error("Generation failed");
      }

      const lines = result.text.trim().split("\n");
      const queryables: Queryable[] = lines.map((line: string) => {
        const colonIdx = line.indexOf(":");
        if (colonIdx === -1) return null;
        const type = line.slice(0, colonIdx).trim();
        if (type !== 'lex' && type !== 'vec' && type !== 'hyde') return null;
        const text = line.slice(colonIdx + 1).trim();
        return { type: type as QueryType, text };
      }).filter((q: Queryable | null): q is Queryable => q !== null);

      // Filter out lex entries if not requested
      if (!includeLexical) {
        return queryables.filter(q => q.type !== 'lex');
      }
      return queryables;
    } catch (error) {
      console.error("Query expansion failed:", error);
      // Fallback to original query
      const fallback: Queryable[] = [{ type: 'vec', text: query }];
      if (includeLexical) fallback.unshift({ type: 'lex', text: query });
      return fallback;
    }
  }

  /**
   * Rerank documents by relevance to a query
   */
  async rerank(
    query: string,
    documents: RerankDocument[],
    options: RerankOptions = {}
  ): Promise<RerankResult> {
    if (!this.rerankUrl) {
      // Return documents in original order with default scores
      return {
        results: documents.map((doc, index) => ({
          file: doc.file,
          score: 1 - (index * 0.1), // Decreasing scores
          index,
        })),
        model: "no-rerank",
      };
    }

    // If we have more than 10 documents, batch them to avoid overwhelming the server
    const BATCH_SIZE = 10;
    if (documents.length > BATCH_SIZE) {
      try {
        const allResults: RerankDocumentResult[] = [];
        let modelName = "remote-rerank";

        // Process in batches
        for (let i = 0; i < documents.length; i += BATCH_SIZE) {
          const batch = documents.slice(i, i + BATCH_SIZE);
          const batchResult = await this.rerankBatch(query, batch, i);
          allResults.push(...batchResult.results);
          modelName = batchResult.model;
        }

        // Sort all results by score descending
        allResults.sort((a, b) => b.score - a.score);

        return {
          results: allResults,
          model: modelName,
        };
      } catch (error) {
        console.error("Batch rerank error:", error);
        // Fallback
        return {
          results: documents.map((doc, index) => ({
            file: doc.file,
            score: 1 - (index * 0.1),
            index,
          })),
          model: "rerank-fallback",
        };
      }
    }

    // Single batch - use existing logic
    return this.rerankBatch(query, documents, 0);
  }

  /**
   * Rerank a single batch of documents
   */
  private async rerankBatch(
    query: string,
    documents: RerankDocument[],
    indexOffset: number
  ): Promise<RerankResult> {
    try {
      const texts = documents.map(doc => doc.text);

      const response = await fetch(`${this.rerankUrl}/v1/rerank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          documents: texts,
          model: "qwen3-reranker",
        }),
      });

      if (!response.ok) {
        console.error(`Rerank request failed: ${response.status} ${response.statusText}`);
        // Try to get error details from response body
        try {
          const errorText = await response.text();
          console.error(`Rerank error details: ${errorText}`);
        } catch (e) {
          // Ignore if we can't read the error body
        }
        throw new Error("Rerank request failed");
      }

      const data = await response.json() as {
        results: Array<{ index: number; relevance_score: number }>;
        model: string;
      };

      // Map results back to our format (with adjusted indices)
      const results: RerankDocumentResult[] = data.results.map(item => ({
        file: documents[item.index]!.file,
        score: item.relevance_score,
        index: indexOffset + item.index,
      }));

      // Sort by score descending
      results.sort((a, b) => b.score - a.score);

      return {
        results,
        model: data.model || "remote-rerank",
      };
    } catch (error) {
      console.error("Rerank batch error:", error);
      // Return documents in original order with default scores
      return {
        results: documents.map((doc, index) => ({
          file: doc.file,
          score: 1 - (index * 0.1),
          index: indexOffset + index,
        })),
        model: "rerank-fallback",
      };
    }
  }

  /**
   * Dispose (no-op for remote)
   */
  async dispose(): Promise<void> {
    // Nothing to dispose for remote connections
  }

  /**
   * Check health of remote endpoints
   */
  async checkHealth(): Promise<{ embed: boolean; rerank: boolean; generate: boolean }> {
    const results = { embed: false, rerank: false, generate: false };

    const checkEndpoint = async (url: string | null): Promise<boolean> => {
      if (!url) return false;
      try {
        const response = await fetch(`${url}/health`, { method: "GET" });
        return response.ok;
      } catch {
        return false;
      }
    };

    [results.embed, results.rerank, results.generate] = await Promise.all([
      checkEndpoint(this.embedUrl),
      checkEndpoint(this.rerankUrl),
      checkEndpoint(this.generateUrl),
    ]);

    return results;
  }

  /**
   * Get configured URLs
   */
  getConfig(): RemoteLLMConfig {
    return {
      embedUrl: this.embedUrl || undefined,
      rerankUrl: this.rerankUrl || undefined,
      generateUrl: this.generateUrl || undefined,
      generateModel: this.generateModel || undefined,
    };
  }
}

// =============================================================================
// Remote LLM Session (implements ILLMSession for compatibility)
// =============================================================================

/**
 * Session wrapper for RemoteLLM that implements ILLMSession interface.
 * This allows RemoteLLM to be used with the existing withLLMSession pattern.
 */
class RemoteLLMSession implements ILLMSession {
  private llm: RemoteLLM;
  private released = false;
  private abortController: AbortController;

  constructor(llm: RemoteLLM, _options: LLMSessionOptions = {}) {
    this.llm = llm;
    this.abortController = new AbortController();
  }

  get isValid(): boolean {
    return !this.released && !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  release(): void {
    this.released = true;
    this.abortController.abort();
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    if (!this.isValid) return null;
    return this.llm.embed(text, options);
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (!this.isValid) return texts.map(() => null);
    return this.llm.embedBatch(texts);
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    if (!this.isValid) return [{ type: 'vec', text: query }];
    return this.llm.expandQuery(query, options);
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    if (!this.isValid) {
      return {
        results: documents.map((doc, index) => ({
          file: doc.file,
          score: 1 - (index * 0.1),
          index,
        })),
        model: "session-invalid",
      };
    }
    return this.llm.rerank(query, documents, options);
  }
}

/**
 * Execute a function with a scoped RemoteLLM session.
 * Compatible with the existing withLLMSession pattern.
 */
export async function withRemoteLLMSession<T>(
  fn: (session: ILLMSession) => Promise<T>,
  options?: LLMSessionOptions
): Promise<T> {
  const llm = getDefaultRemoteLLM();
  const session = new RemoteLLMSession(llm, options);

  try {
    return await fn(session);
  } finally {
    session.release();
  }
}

// =============================================================================
// Singleton for default RemoteLLM instance
// =============================================================================

let defaultRemoteLLM: RemoteLLM | null = null;

/**
 * Get the default RemoteLLM instance (creates one if needed)
 */
export function getDefaultRemoteLLM(): RemoteLLM {
  if (!defaultRemoteLLM) {
    defaultRemoteLLM = new RemoteLLM();
  }
  return defaultRemoteLLM;
}

/**
 * Set a custom default RemoteLLM instance
 */
export function setDefaultRemoteLLM(llm: RemoteLLM | null): void {
  defaultRemoteLLM = llm;
}

/**
 * Dispose the default RemoteLLM instance if it exists
 */
export async function disposeDefaultRemoteLLM(): Promise<void> {
  if (defaultRemoteLLM) {
    await defaultRemoteLLM.dispose();
    defaultRemoteLLM = null;
  }
}
