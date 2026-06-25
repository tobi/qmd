// =============================================================================
// remote-llm.ts - OpenAI-compatible remote LLM backend.
//
// Implements the LLM interface against any /v1/embeddings and /v1/chat/completions
// API. Used by HybridLLM when QMD_REMOTE_API_KEY is set; otherwise never
// constructed (HybridLLM.remote is undefined and routes fall back to local).
//
// Uses @mariozechner/pi-ai for chat completion. The /embeddings endpoint is
// hit directly via fetch — pi-ai does not have an embeddings abstraction.
// =============================================================================

import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  GenerateOptions,
  GenerateResult,
  ModelInfo,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
  QueryType,
} from "./llm-types.js";
import { getModel, complete, type Model, type Context } from "@mariozechner/pi-ai";

export type RemoteLLMConfig = {
  apiKey: string;
  baseURL?: string;
  embedModel?: string;
  generateModel?: string;
  rerankModel?: string;
  timeoutMs?: number;
};

const DEFAULT_BASE_URL = "https://openrouter.ai/api/v1";
const DEFAULT_TIMEOUT_MS = 60000;

export class RemoteLLM implements LLM {
  private apiKey: string;
  private baseURL: string;
  private embedModel: string;
  private generateModel: string;
  private rerankModel?: string;
  private timeoutMs: number;

  constructor(config: RemoteLLMConfig) {
    this.apiKey = config.apiKey;
    this.baseURL = config.baseURL || DEFAULT_BASE_URL;
    // Remove trailing slash if present
    if (this.baseURL.endsWith("/")) {
      this.baseURL = this.baseURL.slice(0, -1);
    }
    this.embedModel = config.embedModel || "text-embedding-3-small";
    this.generateModel = config.generateModel || "openai/gpt-3.5-turbo";
    this.rerankModel = config.rerankModel;
    this.timeoutMs = config.timeoutMs || DEFAULT_TIMEOUT_MS;
  }

  private async fetchAPI(endpoint: string, body: unknown): Promise<any> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${this.apiKey}`,
          // OpenRouter recommends these for app attribution
          "HTTP-Referer": "https://github.com/tobi/qmd",
          "X-Title": "QMD",
        },
        body: JSON.stringify(body),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API request failed: ${response.status} ${response.statusText} - ${errorText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    try {
      const model = options.model || this.embedModel;
      const response = await this.fetchAPI("/embeddings", {
        model,
        input: text,
      });

      if (!response.data || response.data.length === 0) {
        return null;
      }

      return {
        embedding: response.data[0].embedding,
        model,
      };
    } catch (error) {
      console.error("Remote embedding error:", error);
      return null;
    }
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    try {
      // OpenAI API supports batch embeddings
      const response = await this.fetchAPI("/embeddings", {
        model: this.embedModel,
        input: texts,
      });

      if (!response.data) {
        return texts.map(() => null);
      }

      // Sort by index to ensure order matches input
      const sortedData = response.data.sort((a: any, b: any) => a.index - b.index);

      return sortedData.map((item: any) => ({
        embedding: item.embedding,
        model: this.embedModel,
      }));
    } catch (error) {
      console.error("Remote batch embedding error:", error);
      return texts.map(() => null);
    }
  }

  /**
   * Resolve the pi-ai model from the configured string.
   * Handles "provider/modelId" or defaults to "openai/modelId".
   *
   * For any non-OpenAI base URL (OpenRouter, Ollama, etc.), skip pi-ai's
   * model registry entirely — it only knows about built-in providers.
   * Always return a custom OpenAI-compatible model object.
   */
  private resolvePiModel(modelStr: string): Model<any> {
    let provider = "openai";
    let modelId = modelStr;

    if (modelStr.includes("/")) {
      const parts = modelStr.split("/");
      provider = parts[0]!;
      modelId = parts.slice(1).join("/");
    }

    if (this.baseURL !== "https://api.openai.com/v1") {
      return {
        id: modelId,
        name: modelId,
        api: "openai-completions",
        provider: provider,
        baseUrl: this.baseURL,
        input: ["text"],
        contextWindow: 128000,
        maxTokens: 4096,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      } as any;
    }

    return getModel(provider as any, modelId);
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    const modelStr = options.model || this.generateModel;
    const debugLines: string[] = [];
    debugLines.push(`[generate] model: ${modelStr} baseURL: ${this.baseURL}`);
    try {
      const model = this.resolvePiModel(modelStr);
      debugLines.push(`[generate] resolved → api: ${model.api}  provider: ${model.provider}  reasoning: ${(model as any).reasoning ?? "?"}  baseUrl: ${model.baseUrl}`);

      const context: Context = {
        messages: [{ role: "user", content: prompt, timestamp: Date.now() }],
      };

      const response = await complete(model, context, {
        apiKey: this.apiKey,
        maxTokens: options.maxTokens,
        temperature: options.temperature,
      });

      debugLines.push(`[generate] stopReason: ${response.stopReason}`);
      debugLines.push(`[generate] errorMessage: ${response.errorMessage ?? "(none)"}`);
      debugLines.push(`[generate] provider: ${response.provider}  api: ${response.api}  model: ${response.model}`);
      debugLines.push(`[generate] usage: ${JSON.stringify(response.usage)}`);
      debugLines.push(`[generate] content block types: ${response.content.map((b: any) => b.type).join(", ") || "(none)"}`);

      // Extract text content — skip thinking blocks (reasoning model internal monologue)
      const text = response.content
        .filter((block: any): block is { type: "text"; text: string } => block.type === "text")
        .map((block: any) => block.text)
        .join("");

      if (!text) {
        debugLines.push(`[generate] warning: empty text output`);
        process.stderr.write(debugLines.join("\n") + "\n");
      }

      return {
        text,
        model: modelStr,
        done: true,
      };
    } catch (error) {
      process.stderr.write(debugLines.join("\n") + "\n");
      console.error("Remote generation error:", error);
      return null;
    }
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // For remote, we assume configured models exist or let the API fail
    return {
      name: model,
      exists: true,
    };
  }

  async expandQuery(query: string, options: { context?: string; includeLexical?: boolean } = {}): Promise<Queryable[]> {
    const includeLexical = options.includeLexical ?? true;

    const prompt = `You are a search query expansion assistant.
Expand the following search query into multiple variations for different search backends.
Return ONLY a JSON array of objects with "type" and "text" fields.
Types can be:
- "lex": Lexical/keyword search (exact matches)
- "vec": Vector/semantic search (meaning)
- "hyde": Hypothetical Document Embeddings (answer to the query)

Query: ${query}

JSON Response:`;

    let result: Awaited<ReturnType<typeof this.generate>> = null;
    try {
      result = await this.generate(prompt, {
        temperature: 0.7,
        maxTokens: 600,
      });

      if (!result) return [];

      let jsonStr = result.text.trim();

      // Remove markdown code blocks if present
      if (jsonStr.startsWith("```json")) {
        jsonStr = jsonStr.slice(7);
      }
      if (jsonStr.startsWith("```")) {
        jsonStr = jsonStr.slice(3);
      }
      if (jsonStr.endsWith("```")) {
        jsonStr = jsonStr.slice(0, -3);
      }
      jsonStr = jsonStr.trim();

      const parsed = JSON.parse(jsonStr);
      if (!Array.isArray(parsed)) return [];

      const queryables: Queryable[] = parsed
        .filter((item: any) =>
          item.type &&
          item.text &&
          ["lex", "vec", "hyde"].includes(item.type),
        )
        .map((item: any) => ({
          type: item.type as QueryType,
          text: item.text,
        }));

      return includeLexical ? queryables : queryables.filter((q) => q.type !== "lex");
    } catch (error) {
      console.error(`Remote query expansion failed — prompt:\n${prompt}\n\nraw response:\n${result?.text ?? "(null)"}\n\nerror:`, error);
      const fallback: Queryable[] = [{ type: "vec", text: query }];
      if (includeLexical) fallback.unshift({ type: "lex", text: query });
      return fallback;
    }
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    if (documents.length === 0) {
      return { results: [], model: this.rerankModel || this.generateModel };
    }

    const debugRerank = !!process.env.QMD_DEBUG_RERANK;
    const modelStr = options.model || this.rerankModel || this.generateModel;

    // Listwise reranking: score all candidates in batches with a single prompt per batch.
    // Batching avoids context window overflow on models with smaller windows.
    const BATCH_SIZE = 15;
    const chunkChars = options.chunkChars ?? parseInt(process.env.QMD_RERANK_CHUNK_CHARS || "1200", 10);
    const allScores: number[] = new Array(documents.length).fill(0.5);

    try {
      for (let batchStart = 0; batchStart < documents.length; batchStart += BATCH_SIZE) {
        const batch = documents.slice(batchStart, batchStart + BATCH_SIZE);

        const chunksSection = batch
          .map((doc, i) => {
            const path = doc.file.replace(/^qmd:\/\/[^/]+\//, "");
            return `[${batchStart + i + 1}] (${path})\n${doc.text.slice(0, chunkChars)}`;
          })
          .join("\n\n");

        const exampleScores = batch.length >= 3
          ? `[0.9, 0.1, 0.5${batch.length > 3 ? ", ..." : ""}]`
          : `[0.9${batch.length > 1 ? ", 0.1" : ""}]`;

        const prompt = `Score each document chunk's relevance to the query. Consider:
- Direct mentions of the query subject (names, keywords, synonyms)
- Topical relevance (is the chunk about the same subject?)
- File path (a file named after the query subject is likely relevant)

Scoring rubric:
  0.9-1.0  Directly about the query subject, answers it
  0.6-0.8  Clearly relevant, mentions the subject with context
  0.3-0.5  Tangentially related or only brief mention
  0.0-0.2  Unrelated to the query

Query: ${query}

${chunksSection}

Respond with ONLY a JSON array of exactly ${batch.length} scores, e.g. ${exampleScores}`;

        // Budget: each score is at most 6 chars ("0.99, "), plus brackets and a small buffer
        const maxTokens = batch.length * 10 + 50;

        if (debugRerank) {
          process.stderr.write(`[rerank:llm] batch ${batchStart / BATCH_SIZE + 1}: ${batch.length} docs, maxTokens=${maxTokens}\n`);
          process.stderr.write(`[rerank:llm] prompt:\n${prompt}\n`);
        }

        const result = await this.generate(prompt, {
          model: modelStr,
          temperature: 0,
          maxTokens,
        });

        if (debugRerank) {
          process.stderr.write(`[rerank:llm] raw response: ${JSON.stringify(result?.text ?? null)}\n`);
        }

        if (!result?.text) {
          if (debugRerank) process.stderr.write(`[rerank:llm] empty response — leaving batch at 0.5\n`);
          continue;
        }

        let jsonStr = result.text.trim();
        if (jsonStr.startsWith("```")) jsonStr = jsonStr.replace(/^```[a-z]*\n?/, "").replace(/```$/, "").trim();

        let parsed: any;
        try {
          parsed = JSON.parse(jsonStr);
        } catch (parseErr) {
          if (debugRerank) process.stderr.write(`[rerank:llm] JSON parse failed: ${parseErr}\n`);
          continue;
        }

        if (!Array.isArray(parsed) || parsed.length !== batch.length) {
          if (debugRerank) process.stderr.write(`[rerank:llm] array length mismatch: got ${Array.isArray(parsed) ? parsed.length : typeof parsed}, expected ${batch.length}\n`);
          continue;
        }

        // Normalize scores within this batch to 0–1 (handles models that use different scales)
        const nums = parsed.map((v: any) => typeof v === "number" ? v : parseFloat(v));
        const min = Math.min(...nums);
        const max = Math.max(...nums);
        const range = max - min;

        for (let i = 0; i < batch.length; i++) {
          const raw = nums[i] ?? 0.5;
          // If all scores are identical, range is 0 — keep raw value clamped to [0,1]
          const normalized = range > 0 ? (raw - min) / range : Math.min(1, Math.max(0, raw));
          allScores[batchStart + i] = normalized;
        }

        if (debugRerank) {
          process.stderr.write(`[rerank:llm] parsed scores: ${nums.map((n: number) => n.toFixed(3)).join(", ")}\n`);
        }
      }
    } catch (error) {
      console.error("Remote reranking failed, returning original order:", error);
      return {
        results: documents.map((doc, index) => ({ file: doc.file, score: 0.5, index })),
        model: modelStr,
      };
    }

    const results = documents
      .map((doc, index) => ({ file: doc.file, score: allScores[index] ?? 0.5, index }))
      .sort((a, b) => b.score - a.score);

    return { results, model: modelStr };
  }

  /**
   * getDeviceInfo is a CONCRETE method on RemoteLLM (not part of the LLM
   * interface — see docs/qmd-remote-llm-port.md §"Interface shape — final
   * decision"). HybridLLM proxies through to its local backend's
   * getDeviceInfo; RemoteLLM's own value is informational only.
   */
  async getDeviceInfo(): Promise<{
    gpu: string | false;
    gpuOffloading: boolean;
    gpuDevices: string[];
    vram?: { total: number; used: number; free: number };
    cpuCores: number;
  }> {
    return {
      gpu: "cloud",
      gpuOffloading: false,
      gpuDevices: ["remote"],
      cpuCores: 0,
    };
  }

  async tokenize(text: string): Promise<string[]> {
    // Simple regex to split by word boundaries but keep delimiters
    // This is an approximation for tokenization. Only reachable when
    // QMD_TOKENIZE_BACKEND=remote (defaults to local; see spec).
    return text.match(/[\w]+|[^\w\s]|\s+/g) || [];
  }

  async detokenize(tokens: string[]): Promise<string> {
    return tokens.join("");
  }

  async dispose(): Promise<void> {
    // Nothing to dispose for remote
  }
}
