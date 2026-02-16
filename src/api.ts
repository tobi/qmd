/**
 * api.ts - API-backed LLM implementation (incremental rollout)
 *
 * Current phase: embeddings (/v1/embeddings), query expansion (/v1/chat/completions),
 * and rerank (/v1/rerank).
 * Text generation can delegate to a fallback backend.
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

type CohereRerankResponse = {
  results?: Array<{ index?: number; relevance_score?: number }>;
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
  strictJsonOutput?: boolean;
  rerankBaseUrl?: string;
  rerankApiKey?: string;
  rerankModel?: string;
  fallbackLLM?: LLM;
};

/**
 * API-backed LLM implementation.
 * Embeddings/query-expansion/reranking are remote; text generation can fallback.
 */
export class ApiLLM implements LLM {
  private readonly embedBaseUrl: string;
  private readonly embedApiKey: string;
  private readonly embedModel: string;
  private readonly chatBaseUrl: string;
  private readonly chatApiKey: string;
  private readonly chatModel: string;
  private readonly strictJsonOutput: boolean;
  private readonly rerankBaseUrl: string;
  private readonly rerankApiKey: string;
  private readonly rerankModel: string;
  private readonly fallbackLLM?: LLM;

  constructor(config: ApiLLMConfig = {}) {
    const normalizedEmbedBaseUrl = (
      config.embedBaseUrl
      || process.env.QMD_EMBED_BASE_URL
      || DEFAULT_EMBED_BASE_URL
    ).replace(/\/+$/, "");
    this.embedBaseUrl = normalizedEmbedBaseUrl;

    this.embedApiKey =
      config.embedApiKey
      || process.env.QMD_EMBED_API_KEY
      || "";
    this.embedModel =
      config.embedModel
      || process.env.QMD_EMBED_MODEL
      || DEFAULT_EMBED_MODEL;
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
    this.strictJsonOutput = config.strictJsonOutput ?? this.parseBooleanEnv(
      process.env.QMD_CHAT_STRICT_JSON_OUTPUT,
      false
    );
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
    this.fallbackLLM = config.fallbackLLM;
  }

  private parseBooleanEnv(value: string | undefined, fallback: boolean): boolean {
    if (value === undefined) return fallback;
    const normalized = value.trim().toLowerCase();
    if (["1", "true", "yes", "on"].includes(normalized)) return true;
    if (["0", "false", "no", "off"].includes(normalized)) return false;
    return fallback;
  }

  private getHeaders(apiKey: string): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
    };
  }

  private getFallback(method: string): LLM {
    if (!this.fallbackLLM) {
      throw new Error(`ApiLLM.${method} is not implemented without fallback backend`);
    }
    return this.fallbackLLM;
  }

  private isLikelyLocalModel(model: string): boolean {
    const lower = model.toLowerCase();
    return (
      model.startsWith("hf:")
      || lower.includes(".gguf")
      || lower === "embeddinggemma"
      || lower.includes("qwen3-reranker")
      || lower.startsWith("expedientfalcon/")
    );
  }

  private resolveModel(modelOverride: string | undefined, configuredModel: string): string {
    if (!modelOverride) return configuredModel;
    return this.isLikelyLocalModel(modelOverride) ? configuredModel : modelOverride;
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

  private parseExpandedQueries(content: string, strictJson: boolean): Queryable[] {
    const trimmed = content.trim();
    if (!trimmed) {
      throw new Error("ApiLLM expandQuery error: empty model output");
    }

    // Try strict JSON shape first: [{ type, text }, ...] or { queries: [...] }
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      const asArray =
        Array.isArray(parsed) ? parsed : (
          typeof parsed === "object"
          && parsed !== null
          && Array.isArray((parsed as { queries?: unknown }).queries)
            ? (parsed as { queries: unknown[] }).queries
            : null
        );
      if (asArray) {
        const queries = asArray
          .map(item => {
            if (typeof item !== "object" || item === null) return null;
            const type = (item as { type?: unknown }).type;
            const text = (item as { text?: unknown }).text;
            if (
              (type === "lex" || type === "vec" || type === "hyde")
              && typeof text === "string"
              && text.trim().length > 0
            ) {
              return { type: type as QueryType, text: text.trim() };
            }
            return null;
          })
          .filter((q): q is Queryable => q !== null);
        if (queries.length > 0) return queries;
      }
    } catch {
      if (strictJson) {
        throw new Error("ApiLLM expandQuery error: strict JSON output is enabled, but response was not valid JSON");
      }
    }
    if (strictJson) {
      throw new Error("ApiLLM expandQuery error: strict JSON output is enabled, but response shape was invalid");
    }

    // Line format: "lex: ...", "vec: ...", "hyde: ..."
    const fromLines = trimmed
      .split("\n")
      .map(line => line.trim())
      .filter(Boolean)
      .map(line => {
        const match = line.match(/^(lex|vec|hyde)\s*:\s*(.+)$/i);
        if (!match) return null;
        const type = match[1]!.toLowerCase() as QueryType;
        const text = match[2]!.trim();
        if (!text) return null;
        return { type, text };
      })
      .filter((q): q is Queryable => q !== null);

    if (fromLines.length > 0) return fromLines;
    throw new Error("ApiLLM expandQuery error: could not parse query expansions");
  }

  private async requestChatCompletions(
    messages: Array<{ role: "system" | "user"; content: string }>,
    options?: { model?: string; strictJson?: boolean }
  ): Promise<string> {
    if (!this.chatApiKey) {
      throw new Error("ApiLLM chat error: missing API key (set QMD_CHAT_API_KEY or OPENAI_API_KEY)");
    }
    const model = options?.model || this.chatModel;
    const strictJson = options?.strictJson ?? this.strictJsonOutput;

    let response: OpenAIChatResponse;
    try {
      const payload: Record<string, unknown> = {
        model,
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
    if (!content.trim()) {
      throw new Error("ApiLLM chat error: empty response content");
    }
    return content;
  }

  private async requestEmbeddings(texts: string[], modelOverride?: string): Promise<OpenAIEmbeddingResponse | null> {
    if (!this.embedApiKey) {
      console.error("ApiLLM embedding error: missing API key (set QMD_EMBED_API_KEY or OPENAI_API_KEY)");
      return null;
    }

    const model = this.resolveModel(modelOverride, this.embedModel);
    try {
      const resp = await fetch(`${this.embedBaseUrl}/embeddings`, {
        method: "POST",
        headers: this.getHeaders(this.embedApiKey),
        body: JSON.stringify({
          model,
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
    const model = this.resolveModel(options.model, this.embedModel);
    const response = await this.requestEmbeddings([text], model);
    const vector = response?.data?.[0]?.embedding;
    if (!vector || !Array.isArray(vector)) return null;

    return {
      embedding: vector,
      model,
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
    return this.getFallback("generate").generate(prompt, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    return { name: model, exists: true };
  }

  async expandQuery(query: string, options?: { context?: string, includeLexical?: boolean }): Promise<Queryable[]> {
    const includeLexical = options?.includeLexical ?? true;
    const strictJson = this.strictJsonOutput;
    const formatInstruction = strictJson
      ? "Return ONLY valid JSON as an array of objects: [{\"type\":\"lex|vec|hyde\",\"text\":\"...\"}, ...]. No markdown."
      : "Return one query per line in format: type: text, where type is lex, vec, or hyde.";
    const lexicalInstruction = includeLexical
      ? "Include at least one lex query."
      : "Do not include any lex queries.";

    const systemPrompt = [
      "You expand search queries for hybrid retrieval.",
      "Produce useful variations for lexical and semantic search.",
      formatInstruction,
    ].join(" ");

    const userPrompt = [
      `Original query: ${query}`,
      options?.context ? `Context: ${options.context}` : "",
      lexicalInstruction,
      "Return 2-4 total items. Keep each text concise and relevant.",
      "Allowed types: lex, vec, hyde.",
    ].filter(Boolean).join("\n");

    const content = await this.requestChatCompletions(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      { model: this.chatModel, strictJson }
    );

    const parsed = this.parseExpandedQueries(content, strictJson);
    const filteredByLex = includeLexical ? parsed : parsed.filter(q => q.type !== "lex");
    const deduped = Array.from(new Map(
      filteredByLex
        .map(q => ({ ...q, text: q.text.trim() }))
        .filter(q => q.text.length > 0)
        .map(q => [`${q.type}|${q.text.toLowerCase()}`, q] as const)
    ).values());

    if (deduped.length === 0) {
      throw new Error("ApiLLM expandQuery error: no valid expansions produced");
    }
    return deduped;
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    if (!this.rerankApiKey) {
      throw new Error("ApiLLM rerank error: missing API key (set QMD_RERANK_API_KEY or COHERE_API_KEY)");
    }
    if (documents.length === 0) {
      return { results: [], model: this.resolveModel(options.model, this.rerankModel) };
    }

    const model = this.resolveModel(options.model, this.rerankModel);

    let response: CohereRerankResponse;
    try {
      const resp = await fetch(`${this.rerankBaseUrl}/rerank`, {
        method: "POST",
        headers: this.getHeaders(this.rerankApiKey),
        body: JSON.stringify({
          model,
          query,
          documents: documents.map((doc) => doc.text),
          top_n: documents.length,
        }),
      });
      if (!resp.ok) {
        const body = await resp.text().catch(() => "");
        throw new Error(`ApiLLM rerank error: ${resp.status} ${resp.statusText} ${body}`.trim());
      }
      response = await resp.json() as CohereRerankResponse;
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      throw new Error(`ApiLLM rerank request failed: ${detail}`);
    }

    if (!Array.isArray(response.results)) {
      throw new Error("ApiLLM rerank error: invalid response (missing results array)");
    }

    const scoreByIndex = new Map<number, number>();
    for (const item of response.results) {
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
