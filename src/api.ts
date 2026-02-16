/**
 * api.ts - API-backed LLM implementation (incremental rollout)
 *
 * Current phase: embeddings (/v1/embeddings), query expansion (/v1/chat/completions),
 * and rerank (/v1/rerank).
 * Query expansion currently prompts model for line-format output ("lex|vec|hyde: ..."),
 * but does not use constrained output. Possibly upgrade to structured output.
 * This path works in current provider-gated tests but is not extensively battle-tested yet.
 * Text generation is intentionally unsupported in this backend for now.
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  QueryType,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

const DEFAULT_EMBED_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_EMBED_MODEL = "text-embedding-3-small";
const DEFAULT_CHAT_BASE_URL = "https://api.openai.com/v1";
const DEFAULT_CHAT_MODEL = "gpt-4o-mini";
const DEFAULT_RERANK_BASE_URL = "https://api.cohere.com/v1";
const DEFAULT_RERANK_MODEL = "rerank-v3.5";

type OpenAIEmbeddingResponse = {
  data?: Array<{ embedding?: number[] }>;
};

type RerankResponse = {
  results?: Array<{ index?: number; relevance_score?: number }>;
  data?: Array<{ index?: number; relevance_score?: number }>;
};

type OpenAIChatResponse = {
  choices?: Array<{
    message?: {
      content?: string | Array<{ type?: string; text?: string }>;
    };
  }>;
};

export type ApiLLMConfig = {
  embedBaseUrl?: string;
  embedApiKey?: string;
  embedModel?: string;
  chatBaseUrl?: string;
  chatApiKey?: string;
  chatModel?: string;
  rerankBaseUrl?: string;
  rerankApiKey?: string;
  rerankModel?: string;
};

/**
 * API-backed LLM implementation.
 * Embeddings/query-expansion/reranking are remote; text generation is unsupported.
 */
export class ApiLLM implements LLM {
  private readonly embedBaseUrl: string;
  private readonly embedApiKey: string;
  private readonly embedModel: string;
  private readonly chatBaseUrl: string;
  private readonly chatApiKey: string;
  private readonly chatModel: string;
  private readonly rerankBaseUrl: string;
  private readonly rerankApiKey: string;
  private readonly rerankModel: string;

  constructor(config: ApiLLMConfig = {}) {
    // Embedding API config
    this.embedBaseUrl = (
      config.embedBaseUrl
      || process.env.QMD_EMBED_BASE_URL
      || DEFAULT_EMBED_BASE_URL
    ).replace(/\/+$/, "");
    this.embedApiKey =
      config.embedApiKey
      || process.env.QMD_EMBED_API_KEY
      || "";
    this.embedModel =
      config.embedModel
      || process.env.QMD_EMBED_MODEL
      || DEFAULT_EMBED_MODEL;
      // Chat API config
    this.chatBaseUrl = (
      config.chatBaseUrl
      || process.env.QMD_CHAT_BASE_URL
      || DEFAULT_CHAT_BASE_URL
    ).replace(/\/+$/, "");
    this.chatApiKey =
      config.chatApiKey
      || process.env.QMD_CHAT_API_KEY
      || this.embedApiKey;
    this.chatModel =
      config.chatModel
      || process.env.QMD_CHAT_MODEL
      || DEFAULT_CHAT_MODEL;
    // Rerank API config
    this.rerankBaseUrl = (
      config.rerankBaseUrl
      || process.env.QMD_RERANK_BASE_URL
      || DEFAULT_RERANK_BASE_URL
    ).replace(/\/+$/, "");
    this.rerankApiKey =
      config.rerankApiKey
      || process.env.QMD_RERANK_API_KEY
      || this.embedApiKey;
    this.rerankModel =
      config.rerankModel
      || process.env.QMD_RERANK_MODEL
      || DEFAULT_RERANK_MODEL;
  }

  private getHeaders(apiKey: string): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    };
  }

  private usesVoyageRerankApi(): boolean {
    // Voyage uses different result shape, if we support more providers maybe add env var selector
    try {
      const hostname = new URL(this.rerankBaseUrl).hostname.toLowerCase();
      return hostname === "api.voyageai.com" || hostname.endsWith(".voyageai.com");
    } catch {
      return this.rerankBaseUrl.toLowerCase().includes("voyageai.com");
    }
  }

  private extractChatContent(response: OpenAIChatResponse): string {
    const content = response.choices?.[0]?.message?.content;
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
      return content
        .filter(part => part.type === "text" && typeof part.text === "string")
        .map(part => part.text as string)
        .join("\n");
    }
    return "";
  }

  private parseExpandedQueries(content: string): Queryable[] {
    const trimmed = content.trim();
    if (!trimmed) return [];

    // Line format: "lex: ...", "vec: ...", "hyde: ..."
    const fromLines = trimmed
      .split("\n")
      .map(line => line.trim())
      .filter(Boolean)
      .map(line => {
        const match = line.match(/^(?:[-*•\d\.\)\s]*)?(lex|vec|hyde)\s*:\s*(.+)$/i);
        if (!match) return null;
        const type = match[1]!.toLowerCase() as QueryType;
        const text = match[2]!.trim();
        if (!text) return null;
        return { type, text };
      })
      .filter((q): q is Queryable => q !== null);

    return fromLines;
  }

  private async requestChatCompletions(
    messages: Array<{ role: "system" | "user"; content: string }>
  ): Promise<string> {
    if (!this.chatApiKey) {
      throw new Error("ApiLLM chat error: missing API key (set QMD_CHAT_API_KEY)");
    }
    let response: OpenAIChatResponse;
    try {
      const payload: Record<string, unknown> = {
        model: this.chatModel,
        messages,
        temperature: 0.2,
      };

      const resp = await fetch(`${this.chatBaseUrl}/chat/completions`, {
        method: "POST",
        headers: this.getHeaders(this.chatApiKey),
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`ApiLLM chat error: ${resp.status} ${resp.statusText} ${body}`.trim());
      }
      response = await resp.json() as OpenAIChatResponse;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`ApiLLM chat request failed: ${detail}`);
    }

    const content = this.extractChatContent(response);
    return content;
  }

  private async requestEmbeddings(texts: string[]): Promise<OpenAIEmbeddingResponse | null> {
    if (!this.embedApiKey) {
      throw new Error("ApiLLM embedding error: missing API key (set QMD_EMBED_API_KEY)");
    }

    try {
      const resp = await fetch(`${this.embedBaseUrl}/embeddings`, {
        method: "POST",
        headers: this.getHeaders(this.embedApiKey),
        body: JSON.stringify({
          model: this.embedModel,
          input: texts,
        }),
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        console.error(`ApiLLM embedding error: ${resp.status} ${resp.statusText} ${body}`.trim());
        return null;
      }
      return await resp.json() as OpenAIEmbeddingResponse;
    } catch (error) {
      console.error("ApiLLM embedding error:", error);
      return null;
    }
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    void options; // Seems used for model override in local backend, ignoring here
    const response = await this.requestEmbeddings([text]);
    const vector = response?.data?.[0]?.embedding;
    if (!vector || !Array.isArray(vector)) return null;

    return {
      embedding: vector,
      model: this.embedModel,
    };
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const response = await this.requestEmbeddings(texts);
    if (!response?.data || !Array.isArray(response.data)) {
      return texts.map(() => null);
    }

    const results: (EmbeddingResult | null)[] = [];
    for (let i = 0; i < texts.length; i++) {
      const vector = response.data[i]?.embedding;
      if (!vector || !Array.isArray(vector)) {
        results.push(null);
      } else {
        results.push({
          embedding: vector,
          model: this.embedModel,
        });
      }
    }
    return results;
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    void prompt;
    void options;
    throw new Error("ApiLLM generate is not implemented for API backend (use QMD_LLM_BACKEND=local)");
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]> {
    const includeLexical = options?.includeLexical ?? true;
    const searchScope = includeLexical ? "lexical and semantic" : "semantic";
    const allowedTypes = includeLexical ? "lex, vec, or hyde" : "vec or hyde";
    const allowedTypesList = includeLexical ? "lex, vec, hyde" : "vec, hyde";
    const lexicalInstruction = includeLexical
      ? "Include at least one lex query."
      : "Do not include any lex queries.";

    const systemPrompt = [
      "You expand search queries for hybrid retrieval.",
      `Produce useful variations for ${searchScope} search.`,
      `Return one query per line in format: type: text, where type is ${allowedTypes}.`,
    ].join(" ");

    const userPrompt = [
      `Original query: ${query}`,
      options?.context ? `Context: ${options.context}` : "",
      lexicalInstruction,
      "Return 2-4 total items. Keep each text concise and relevant.",
      `Allowed types: ${allowedTypesList}.`,
    ].filter(Boolean).join("\n");

    const content = await this.requestChatCompletions([
      { role: "system", content: systemPrompt },
      { role: "user", content: userPrompt },
    ]);

    if (!content.trim()) {
      return [];
    }

    const parsed = this.parseExpandedQueries(content);
    const filteredByLex = includeLexical ? parsed : parsed.filter(q => q.type !== "lex");
    const deduped = Array.from(new Map(
      filteredByLex
        .map(q => ({ ...q, text: q.text.trim() }))
        .filter(q => q.text.length > 0)
        .map(q => [`${q.type}|${q.text.toLowerCase()}`, q] as const)
    ).values());

    if (deduped.length > 0) {
      return deduped;
    }
    console.warn("ApiLLM expandQuery warning: no valid expansions produced; returning empty expansion set");
    return [];
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    void options; // Seems used for model override in local backend, ignoring here
    if (!this.rerankApiKey) {
      throw new Error("ApiLLM rerank error: missing API key (set QMD_RERANK_API_KEY)");
    }
    if (documents.length === 0) {
      return { results: [], model: this.rerankModel };
    }

    const model = this.rerankModel;

    let response: RerankResponse;
    const topCountField = this.usesVoyageRerankApi() ? "top_k" : "top_n";
    try {
      const resp = await fetch(`${this.rerankBaseUrl}/rerank`, {
        method: "POST",
        headers: this.getHeaders(this.rerankApiKey),
        body: JSON.stringify({
          model,
          query,
          documents: documents.map((doc) => doc.text),
          [topCountField]: documents.length,
        }),
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`ApiLLM rerank error: ${resp.status} ${resp.statusText} ${body}`.trim());
      }
      response = await resp.json() as RerankResponse;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`ApiLLM rerank request failed: ${detail}`);
    }

    const responseResults = Array.isArray(response.results)
      ? response.results
      : Array.isArray(response.data)
        ? response.data
        : null;

    if (!Array.isArray(responseResults)) {
      throw new Error("ApiLLM rerank error: invalid response (missing results/data array)");
    }

    const scoreByIndex = new Map<number, number>();
    for (const item of responseResults) {
      if (typeof item.index !== "number" || typeof item.relevance_score !== "number") continue;
      scoreByIndex.set(item.index, item.relevance_score);
    }

    const results = documents
      .map((doc, index) => ({
        file: doc.file,
        score: scoreByIndex.get(index) ?? 0,
        index,
      }))
      .sort((a, b) => b.score - a.score);

    return {
      results,
      model,
    };
  }

  async dispose(): Promise<void> {
    // No API client resources to dispose in this implementation.
  }
}
