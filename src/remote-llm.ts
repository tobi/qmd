/**
 * remote-llm.ts - HTTP-based LLM backend for QMD
 *
 * Implements the LLM interface by delegating to:
 *   - embed/rerank: embed_reranker service (http://localhost:1235)
 *   - generate/expandQuery: LM Studio or any OpenAI-compatible endpoint (http://localhost:1234)
 *
 * Activated by setting QMD_REMOTE_EMBED_URL or QMD_REMOTE_LLM_URL.
 */

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  RerankDocument,
  RerankResult,
  RerankOptions,
  Queryable,
  ModelInfo,
} from "./llm.js";

const DEFAULT_EMBED_URL = "http://localhost:1235";
const DEFAULT_LLM_URL = "http://localhost:1234";
const DEFAULT_LLM_MODEL = "qwen3.5-2b-mlx";

export type RemoteLLMConfig = {
  embedUrl?: string;
  llmUrl?: string;
  llmModel?: string;
};

export class RemoteLLM implements LLM {
  private readonly embedUrl: string;
  private readonly llmUrl: string;
  private readonly llmModel: string;

  constructor(config: RemoteLLMConfig = {}) {
    this.embedUrl = config.embedUrl ?? process.env.QMD_REMOTE_EMBED_URL ?? DEFAULT_EMBED_URL;
    this.llmUrl = config.llmUrl ?? process.env.QMD_REMOTE_LLM_URL ?? DEFAULT_LLM_URL;
    this.llmModel = config.llmModel ?? process.env.QMD_REMOTE_LLM_MODEL ?? DEFAULT_LLM_MODEL;
  }

  get embedModelName(): string {
    return `remote:${this.embedUrl}`;
  }

  get generateModelName(): string {
    return this.llmModel;
  }

  get rerankModelName(): string {
    return `remote:${this.embedUrl}`;
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const body: any = { input: text };
      if (options.model) body.model = options.model;
      if (options.isQuery) body.input_type = "query";

      const res = await fetch(`${this.embedUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        process.stderr.write(`RemoteLLM embed error: ${res.status} ${res.statusText}\n`);
        return null;
      }
      const json = (await res.json()) as { data: { embedding: number[] }[]; model: string };
      const first = json.data[0];
      if (!first) return null;
      return { embedding: first.embedding, model: json.model };
    } catch (err) {
      process.stderr.write(`RemoteLLM embed failed: ${err}\n`);
      return null;
    }
  }

  async embedBatch(texts: string[], options: EmbedOptions = {}): Promise<(EmbeddingResult | null)[]> {
    try {
      const body: any = { input: texts };
      if (options.model) body.model = options.model;
      if (options.isQuery) body.input_type = "query";

      const res = await fetch(`${this.embedUrl}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const errBody = await res.text().catch(() => "");
        process.stderr.write(`RemoteLLM embedBatch error: ${res.status} ${res.statusText}\n  body: ${errBody}\n  sent: ${JSON.stringify(body).slice(0, 500)}\n`);
        return texts.map(() => null);
      }
      const json = (await res.json()) as { data: { index: number; embedding: number[] }[]; model: string };
      const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);
      for (const item of json.data) {
        results[item.index] = { embedding: item.embedding, model: json.model };
      }
      return results;
    } catch (err) {
      process.stderr.write(`RemoteLLM embedBatch failed: ${err}\n`);
      return texts.map(() => null);
    }
  }

  async rerank(query: string, documents: RerankDocument[], options?: RerankOptions): Promise<RerankResult> {
    const texts = documents.map((d) => d.text);
    try {
      const body: any = { query, documents: texts, top_k: texts.length };
      // Note: don't forward options.model — the remote server manages its own
      // rerank model selection (local model names like Ollama IDs are meaningless).

      const res = await fetch(`${this.embedUrl}/v1/rerank`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const errBody = await res.text().catch(() => "");
        process.stderr.write(`RemoteLLM rerank error: ${res.status} ${res.statusText}\n  body: ${errBody}\n`);
        return { results: documents.map((d, i) => ({ file: d.file, score: 0, index: i })), model: "remote" };
      }
      const json = (await res.json()) as { model: string; results: { index: number; relevance_score: number }[] };
      const results = json.results.map((r) => ({
        file: documents[r.index]?.file ?? "",
        score: r.relevance_score,
        index: r.index,
      }));
      return { results, model: json.model };
    } catch (err) {
      process.stderr.write(`RemoteLLM rerank failed: ${err}\n`);
      return { results: documents.map((d, i) => ({ file: d.file, score: 0, index: i })), model: "remote" };
    }
  }

  async expandQuery(query: string, options?: { context?: string; includeLexical?: boolean; intent?: string }): Promise<Queryable[]> {
    const fallback: Queryable[] = [
      { type: "lex", text: query },
      { type: "vec", text: query },
    ];
    const systemPrompt = `You are a search query expander. Return ONLY valid JSON with keys "lex", "vec", "hyde". No explanation, no markdown.`;
    const contextHint = [options?.context, options?.intent].filter(Boolean).join("\n");
    const userPrompt = `Expand this search query:\n"${query}"${contextHint ? `\nContext: ${contextHint}` : ""}\n\nReturn JSON: {"lex":"keyword terms for BM25","vec":"semantic search phrase","hyde":"short example document that answers the query"}`;
    try {
      const res = await fetch(`${this.llmUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: this.llmModel,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: userPrompt },
          ],
          temperature: 0.7,
          max_tokens: 300,
        }),
      });
      if (!res.ok) return fallback;
      const json = await res.json() as { choices: { message: { content: string } }[] };
      const text = json.choices[0]?.message?.content ?? "";
      const match = text.match(/\{[\s\S]*\}/);
      if (!match) return fallback;
      const parsed = JSON.parse(match[0]) as { lex?: string; vec?: string; hyde?: string };
      const result: Queryable[] = [];
      if (options?.includeLexical !== false && parsed.lex) result.push({ type: "lex", text: String(parsed.lex) });
      if (parsed.vec) result.push({ type: "vec", text: String(parsed.vec) });
      if (parsed.hyde) result.push({ type: "hyde", text: String(parsed.hyde) });
      return result.length > 0 ? result : fallback;
    } catch {
      return fallback;
    }
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    try {
      const res = await fetch(`${this.llmUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: this.llmModel,
          messages: [{ role: "user", content: prompt }],
          temperature: options.temperature ?? 0.7,
          max_tokens: options.maxTokens ?? 150,
        }),
      });
      if (!res.ok) return null;
      const json = await res.json() as { choices: { message: { content: string } }[]; model: string };
      return {
        text: json.choices[0]?.message?.content ?? "",
        model: json.model,
        done: true,
      };
    } catch {
      return null;
    }
  }

  async modelExists(_model: string): Promise<ModelInfo> {
    return { name: _model, exists: true };
  }

  async dispose(): Promise<void> {
    // no-op: remote APIs manage their own lifecycle
  }
}

/**
 * Returns true when environment variables indicate remote LLM should be used.
 */
export function shouldUseRemoteLLM(): boolean {
  return !!(process.env.QMD_REMOTE_EMBED_URL || process.env.QMD_REMOTE_LLM_URL);
}
