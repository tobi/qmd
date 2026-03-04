import type { EmbeddingResult, EmbedOptions } from "./llm.js";

export const GOOGLE_EMBED_MODEL = "gemini-embedding-001";
export const GOOGLE_EMBED_DIMENSIONS = 3072;

const MODEL_NAME = `models/${GOOGLE_EMBED_MODEL}`;
const BASE_URL = `https://generativelanguage.googleapis.com/v1beta/${MODEL_NAME}`;
const BATCH_LIMIT = 100;
const MAX_RETRIES = 3;

type TaskType = "RETRIEVAL_DOCUMENT" | "RETRIEVAL_QUERY";

interface EmbedContentRequest {
  model: string;
  content: { parts: { text: string }[] };
  taskType: TaskType;
  outputDimensionality: number;
}

export class GoogleAIEmbedder {
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  private get headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      "x-goog-api-key": this.apiKey,
    };
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    const taskType: TaskType = options.isQuery ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT";
    const url = `${BASE_URL}:embedContent`;

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({
          model: MODEL_NAME,
          content: { parts: [{ text }] },
          taskType,
          outputDimensionality: GOOGLE_EMBED_DIMENSIONS,
        }),
      });

      if (!res.ok) {
        const body = (await res.text()).slice(0, 200);
        console.warn(`Google embed API error ${res.status}: ${body}`);
        return null;
      }

      const data = await res.json() as { embedding?: { values?: number[] } };
      const vals = data?.embedding?.values;
      if (!Array.isArray(vals) || vals.length !== GOOGLE_EMBED_DIMENSIONS) {
        console.warn(`Google embed: unexpected response shape or dimensionality`);
        return null;
      }
      return { embedding: vals, model: GOOGLE_EMBED_MODEL };
    } catch (err) {
      console.warn("Google embed request failed:", err);
      return null;
    }
  }

  async embedBatch(texts: string[], isQuery = false): Promise<(EmbeddingResult | null)[]> {
    if (texts.length === 0) return [];

    const taskType: TaskType = isQuery ? "RETRIEVAL_QUERY" : "RETRIEVAL_DOCUMENT";
    const results: (EmbeddingResult | null)[] = new Array(texts.length).fill(null);

    for (let start = 0; start < texts.length; start += BATCH_LIMIT) {
      const chunk = texts.slice(start, start + BATCH_LIMIT);
      const requests: EmbedContentRequest[] = chunk.map((text) => ({
        model: MODEL_NAME,
        content: { parts: [{ text }] },
        taskType,
        outputDimensionality: GOOGLE_EMBED_DIMENSIONS,
      }));

      let retries = 0;
      while (retries < MAX_RETRIES) {
        try {
          const url = `${BASE_URL}:batchEmbedContents`;
          const res = await fetch(url, {
            method: "POST",
            headers: this.headers,
            body: JSON.stringify({ requests }),
          });

          if (res.status === 429) {
            const retryAfter = parseInt(res.headers.get("Retry-After") ?? "5", 10);
            console.warn(`Google embed rate limited, retrying in ${retryAfter}s...`);
            await new Promise(r => setTimeout(r, retryAfter * 1000));
            retries++;
            continue;
          }

          if (!res.ok) {
            const body = (await res.text()).slice(0, 200);
            console.warn(`Google batch embed API error ${res.status}: ${body}`);
            break;
          }

          const data = await res.json() as { embeddings?: { values: number[] }[] };
          const embeddings = data.embeddings ?? [];
          for (let i = 0; i < embeddings.length; i++) {
            const emb = embeddings[i];
            if (emb?.values && emb.values.length === GOOGLE_EMBED_DIMENSIONS) {
              results[start + i] = { embedding: emb.values, model: GOOGLE_EMBED_MODEL };
            }
          }
          break;
        } catch (err) {
          console.warn("Google batch embed request failed:", err);
          break;
        }
      }
    }

    return results;
  }
}
