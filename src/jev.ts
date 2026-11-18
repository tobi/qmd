/** Independent per-document TypeSafe System One relevance scores, selected explicitly. */
import { createHash } from "node:crypto";
import type { RerankDocument, RerankResult } from "./llm.js";
import { postModelJson } from "./remote-http.js";

export const JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone";
const INSTRUCTIONS = "Does the document contain information that answers or directly addresses the search query?";
const MAX_INPUT_BYTES = 2 * 1024 * 1024;

export type JevRerankOptions = {
  revision?: string;
  /** Environment variable NAME, never a credential value. Default: TYPESAFE_API_KEY. */
  apiKeyEnv?: string;
};

export function isJevModel(uri: string): boolean {
  return uri.startsWith("typesafe:");
}

function validIdentity(value: unknown): value is string {
  return typeof value === "string" && !!value.trim() && value.length <= 256
    && value.isWellFormed() && !/[\x00-\x1f\x7f-\x9f]/.test(value);
}

export function parseJevModel(uri: string): string {
  if (typeof uri !== "string" || !isJevModel(uri) || !validIdentity(uri.slice(9))) {
    throw new Error("Invalid TypeSafe rerank model (expected typesafe:model)");
  }
  return uri.slice(9);
}

export function copyJevRerankOptions(options: JevRerankOptions): Readonly<JevRerankOptions> {
  if (!options || typeof options !== "object" || Array.isArray(options)
    || Object.keys(options).some(key => !["revision", "apiKeyEnv"].includes(key))
    || (options.revision !== undefined && !validIdentity(options.revision))
    || (options.apiKeyEnv !== undefined && (typeof options.apiKeyEnv !== "string" || !/^[A-Za-z_][A-Za-z0-9_]{0,127}$/.test(options.apiKeyEnv)))) {
    throw new Error("Invalid TypeSafe rerank options");
  }
  return Object.freeze({ revision: options.revision, apiKeyEnv: options.apiKeyEnv });
}

function inputString(value: unknown, maxBytes: number, allowEmpty = false): value is string {
  return typeof value === "string" && (allowEmpty || value.length > 0)
    && value.isWellFormed() && Buffer.byteLength(value, "utf8") <= maxBytes;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export class JevReranker {
  private readonly model: string;
  private readonly options: Readonly<JevRerankOptions>;
  readonly modelName: string;

  constructor(uri: string, options: JevRerankOptions = {}) {
    this.model = parseJevModel(uri);
    this.options = copyJevRerankOptions(options);
    const identity = JSON.stringify([JEV_ENDPOINT, this.model, this.options.revision ?? "", "independent-document-noul/v1", INSTRUCTIONS]);
    this.modelName = `remote-typesafe:${createHash("sha256").update(identity).digest("hex")}`;
  }

  async rerank(query: string, documents: RerankDocument[], callerSignal?: AbortSignal): Promise<RerankResult> {
    callerSignal?.throwIfAborted();
    if (!inputString(query, 64 * 1024) || !Array.isArray(documents)) {
      throw new Error("Invalid TypeSafe rerank inputs");
    }
    if (documents.length > 64) {
      throw new Error("TypeSafe rerank limit is 64 documents per call; reduce candidateLimit (CLI -C)");
    }
    // Snapshot every field and validate the entire call before any await/auth.
    const docs = documents.map(doc => {
      if (!doc) throw new Error("Invalid TypeSafe rerank inputs");
      const { file, text, title } = doc;
      if (!inputString(file, 4096) || !inputString(text, MAX_INPUT_BYTES)
        || (title !== undefined && !inputString(title, 8192, true))) throw new Error("Invalid TypeSafe rerank inputs");
      return { file, text, title };
    });
    const requests = docs.map(doc => ({
      model: this.model,
      state: doc.title
        ? { query, document_title: doc.title, document: doc.text }
        : { query, document: doc.text },
      questions: { relevant: { type: "noul", instructions: INSTRUCTIONS } },
    }));
    let bytes = 0;
    for (const request of requests) {
      bytes += Buffer.byteLength(JSON.stringify(request), "utf8");
      if (bytes > MAX_INPUT_BYTES) throw new Error("TypeSafe rerank inputs exceed byte limit");
    }
    if (docs.length === 0) return { model: this.modelName, results: [] };

    const controller = new AbortController();
    const signal = callerSignal ? AbortSignal.any([callerSignal, controller.signal]) : controller.signal;
    const timer = setTimeout(() => controller.abort(), 30_000);
    timer.unref();
    const scores: number[] = new Array(docs.length);
    let next = 0;
    let failure: Error | undefined;
    const worker = async () => {
      try {
        while (next < docs.length) {
          signal.throwIfAborted();
          const index = next++;
          const body = await postModelJson(JEV_ENDPOINT, requests[index], this.options.apiKeyEnv ?? "TYPESAFE_API_KEY", signal);
          signal.throwIfAborted();
          const relevant = record(body) && record(body.answers) ? body.answers.relevant : undefined;
          const score = record(relevant) ? relevant.noul : undefined;
          if (!record(body) || (body.model !== undefined && body.model !== this.model)
            || typeof score !== "number" || !Number.isFinite(score) || score < 0 || score > 1) {
            throw new Error("Invalid TypeSafe rerank response");
          }
          scores[index] = score;
        }
      } catch (error) {
        if (!failure) {
          failure = signal.aborted ? new Error("TypeSafe rerank aborted")
            : error instanceof Error ? error : new Error("TypeSafe rerank failed");
        }
        controller.abort(); // Stop the queue and cancel the other in-flight call.
      }
    };
    try {
      // At most two independent requests, never Promise.all over documents.
      await Promise.all(Array.from({ length: Math.min(2, docs.length) }, worker));
      if (failure) throw failure;
      if (signal.aborted) throw new Error("TypeSafe rerank aborted");
      return {
        model: this.modelName,
        results: docs.map((doc, index) => ({ file: doc.file, index, score: scores[index]! }))
          .sort((a, b) => b.score - a.score || a.index - b.index),
      };
    } finally {
      clearTimeout(timer);
      controller.abort();
    }
  }
}
