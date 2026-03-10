import { readFileSync } from "node:fs";
import { extname } from "node:path";

export const GOOGLE_EMBED_MODEL = "gemini-embedding-2-preview";
export const GOOGLE_EMBED_MODEL_PATH = `models/${GOOGLE_EMBED_MODEL}`;
export const GOOGLE_EMBED_DEFAULT_DIMENSIONS = 3072;
export const GOOGLE_EMBED_BATCH_LIMIT = 100;
const GOOGLE_EMBED_MAX_RETRIES = 3;

export type GeminiTaskType =
  | "RETRIEVAL_DOCUMENT"
  | "RETRIEVAL_QUERY"
  | "SEMANTIC_SIMILARITY"
  | "CLASSIFICATION"
  | "CLUSTERING";

export type GeminiInlinePart = {
  inlineData: {
    mimeType: string;
    data: string;
  };
};

export type GeminiTextPart = { text: string };
export type GeminiPart = GeminiTextPart | GeminiInlinePart;

export type EmbedInput = string | {
  text?: string;
  filePath?: string;
  parts?: GeminiPart[];
};

type EmbedRequest = {
  model: string;
  content: { parts: GeminiPart[] };
  taskType: GeminiTaskType;
  outputDimensionality: number;
};

export type GeminiEmbedOptions = {
  taskType?: GeminiTaskType;
  outputDimensionality?: number;
};

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function parseGeminiDimensionsFromEnv(): number {
  const raw = process.env.QMD_EMBED_DIMENSIONS?.trim();
  if (!raw) return GOOGLE_EMBED_DEFAULT_DIMENSIONS;
  const parsed = Number.parseInt(raw, 10);
  if (parsed === 3072 || parsed === 1536 || parsed === 768) return parsed;
  return GOOGLE_EMBED_DEFAULT_DIMENSIONS;
}

function getMimeTypeForPath(path: string): string | null {
  const ext = extname(path).toLowerCase();
  if (ext === ".png") return "image/png";
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  if (ext === ".pdf") return "application/pdf";
  return null;
}

function fileToInlinePart(path: string): GeminiInlinePart {
  const mimeType = getMimeTypeForPath(path);
  if (!mimeType) throw new Error(`Unsupported file type for Gemini embedding: ${path}`);
  const data = readFileSync(path).toString("base64");
  return { inlineData: { mimeType, data } };
}

function normalizeInput(input: EmbedInput): GeminiPart[] {
  if (typeof input === "string") {
    return [{ text: input }];
  }
  if (Array.isArray(input.parts) && input.parts.length > 0) {
    return input.parts;
  }
  const parts: GeminiPart[] = [];
  if (typeof input.text === "string" && input.text.trim().length > 0) {
    parts.push({ text: input.text });
  }
  if (input.filePath) {
    parts.push(fileToInlinePart(input.filePath));
  }
  if (parts.length === 0) {
    throw new Error("Gemini embedding request needs at least one input part");
  }
  return parts;
}

export class GoogleAIEmbedder {
  private readonly apiKey: string;
  private readonly dimensions: number;

  constructor(apiKey: string, dimensionsOrOptions?: number | { dimensions?: number }) {
    this.apiKey = apiKey;
    if (typeof dimensionsOrOptions === "number") {
      this.dimensions = dimensionsOrOptions;
    } else {
      this.dimensions = dimensionsOrOptions?.dimensions ?? parseGeminiDimensionsFromEnv();
    }
  }

  private buildUrl(endpoint: "embedContent" | "batchEmbedContents"): string {
    return `https://generativelanguage.googleapis.com/v1beta/models/${GOOGLE_EMBED_MODEL}:${endpoint}?key=${encodeURIComponent(this.apiKey)}`;
  }

  private normalizeEmbedding(values: number[] | undefined, outputDimensionality: number): number[] | null {
    if (!Array.isArray(values) || values.length === 0) return null;
    if (values.length === outputDimensionality) return values;
    if (values.length > outputDimensionality) {
      // Matryoshka truncation keeps the leading dimensions.
      return values.slice(0, outputDimensionality);
    }
    return null;
  }

  private async postWithRetries(
    endpoint: "embedContent" | "batchEmbedContents",
    body: object
  ): Promise<Response> {
    let attempt = 0;
    let delayMs = 500;

    while (true) {
      const res = await fetch(this.buildUrl(endpoint), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (res.ok) return res;
      const retryable = res.status === 429 || res.status >= 500;
      if (!retryable || attempt >= GOOGLE_EMBED_MAX_RETRIES) return res;

      const retryAfter = Number.parseInt(res.headers.get("retry-after") ?? "", 10);
      await sleep(Number.isFinite(retryAfter) ? retryAfter * 1000 : delayMs);
      delayMs *= 2;
      attempt++;
    }
  }

  async embed(input: EmbedInput, options: GeminiEmbedOptions = {}): Promise<{ embedding: number[] } | null> {
    const outputDimensionality = options.outputDimensionality ?? this.dimensions;
    const taskType = options.taskType ?? "RETRIEVAL_DOCUMENT";
    const res = await this.postWithRetries("embedContent", {
      model: GOOGLE_EMBED_MODEL_PATH,
      content: { parts: normalizeInput(input) },
      taskType,
      outputDimensionality,
    });

    if (!res.ok) return null;
    const data = await res.json() as { embedding?: { values?: number[] } };
    const embedding = this.normalizeEmbedding(data.embedding?.values, outputDimensionality);
    if (!embedding) return null;
    return { embedding };
  }

  async embedBatch(inputs: EmbedInput[], options: GeminiEmbedOptions = {}): Promise<({ embedding: number[] } | null)[]> {
    if (inputs.length === 0) return [];
    const output: ({ embedding: number[] } | null)[] = Array(inputs.length).fill(null);

    for (let start = 0; start < inputs.length; start += GOOGLE_EMBED_BATCH_LIMIT) {
      const chunk = inputs.slice(start, start + GOOGLE_EMBED_BATCH_LIMIT);
      const outputDimensionality = options.outputDimensionality ?? this.dimensions;
      const taskType = options.taskType ?? "RETRIEVAL_DOCUMENT";
      const requests: EmbedRequest[] = chunk.map((input) => ({
        model: GOOGLE_EMBED_MODEL_PATH,
        content: { parts: normalizeInput(input) },
        taskType,
        outputDimensionality,
      }));

      const res = await this.postWithRetries("batchEmbedContents", { requests });
      if (!res.ok) continue;

      const data = await res.json() as { embeddings?: Array<{ values?: number[] }> };
      const embeddings = data.embeddings ?? [];
      for (let i = 0; i < requests.length; i++) {
        const values = embeddings[i]?.values;
        const embedding = this.normalizeEmbedding(values, requests[i]!.outputDimensionality);
        if (embedding) {
          output[start + i] = { embedding };
        }
      }
    }

    return output;
  }
}
