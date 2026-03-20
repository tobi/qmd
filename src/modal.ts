/**
 * Modal inference backend client.
 *
 * Provides a ModalBackend class that calls deployed Modal functions
 * via the `modal` npm SDK (gRPC). All prompt formatting stays in JS;
 * this module handles only the raw RPC transport with retry logic.
 */

import { existsSync } from "fs";
import { join } from "path";
import { homedir } from "os";

// ============================================================================
// Types
// ============================================================================

/** Minimal shape of a Modal ClsInstance method handle. */
interface ModalMethodHandle {
  remote(args?: any[], kwargs?: Record<string, any>): Promise<any>;
}

/** Minimal shape of a Modal ClsInstance. */
interface ModalClsInstance {
  method(name: string): ModalMethodHandle;
}

// ============================================================================
// Connection error detection
// ============================================================================

const CONNECTION_ERROR_PATTERNS = [
  "ECONNREFUSED",
  "ETIMEDOUT",
  "ECONNRESET",
  "unavailable",
  "deadline exceeded",
] as const;

/**
 * Determine whether an error is a transient connection error worth retrying.
 * Checks error message and code for known network/gRPC patterns.
 */
export function isConnectionError(err: unknown): boolean {
  const message =
    err instanceof Error ? err.message : String(err);
  const code =
    err instanceof Error ? (err as any).code : undefined;
  const lower = message.toLowerCase();
  const codeLower = typeof code === "string" ? code.toLowerCase() : "";

  return CONNECTION_ERROR_PATTERNS.some(
    (pattern) =>
      lower.includes(pattern.toLowerCase()) ||
      codeLower.includes(pattern.toLowerCase()),
  );
}

// ============================================================================
// Retry helper
// ============================================================================

/**
 * Retry a function up to maxAttempts on connection errors.
 *
 * - Attempt 1: immediate
 * - Attempt 2: immediate retry after connection error
 * - Attempt 3+: wait retryDelayMs then retry
 * - Non-connection errors: throw immediately, no retry
 */
export async function withRetry<T>(
  fn: () => Promise<T>,
  maxAttempts: number = 3,
  retryDelayMs: number = 100,
): Promise<T> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (err) {
      if (!isConnectionError(err) || attempt === maxAttempts) {
        throw err;
      }
      // Wait before retry (skip delay on first retry for responsiveness)
      if (attempt > 1 && retryDelayMs > 0) {
        await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
      }
    }
  }
  // Unreachable, but TypeScript needs it
  throw new Error("withRetry: unreachable");
}

// ============================================================================
// ModalBackend
// ============================================================================

/**
 * Client for calling Modal-deployed QMDInference methods.
 *
 * Lazily initializes the Modal client and class reference on first use.
 * Requires ~/.modal.toml for authentication.
 */
/** Options for constructing a ModalBackend (primarily for testing). */
export interface ModalBackendOptions {
  /** Override the path to the Modal auth token file. */
  tomlPath?: string;
}

export class ModalBackend {
  private instance: ModalClsInstance | null = null;
  private initPromise: Promise<void> | null = null;
  private readonly tomlPath: string;

  constructor(options?: ModalBackendOptions) {
    this.tomlPath = options?.tomlPath ?? join(homedir(), ".modal.toml");
  }

  /**
   * Ensure the Modal client and cls instance are initialized.
   * Uses a cached promise to prevent concurrent initialization races.
   */
  private async ensureConnected(): Promise<void> {
    if (this.instance) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.connect();
    return this.initPromise;
  }

  private async connect(): Promise<void> {
    // Check auth file exists before attempting connection
    if (!existsSync(this.tomlPath)) {
      throw new Error(
        `Modal not authenticated. No ~/.modal.toml found.\n` +
          `Run: modal token set`,
      );
    }

    // Dynamic import — modal is an optional dependency
    let modalModule: typeof import("modal");
    try {
      modalModule = await import("modal");
    } catch {
      throw new Error(
        "Modal npm package not installed. Run: bun add modal",
      );
    }

    const client = new modalModule.ModalClient();
    const cls = await client.cls.fromName("qmd-inference", "QMDInference");
    this.instance = await cls.instance();
  }

  /**
   * Get a method handle from the cls instance.
   */
  private getMethod(name: string): ModalMethodHandle {
    if (!this.instance) {
      throw new Error("ModalBackend not connected");
    }
    return this.instance.method(name);
  }

  /**
   * Raw embedding. Caller is responsible for prompt formatting.
   */
  async embed(texts: string[]): Promise<number[][]> {
    await this.ensureConnected();
    return withRetry(async () => {
      const fn = this.getMethod("embed");
      return fn.remote([texts]);
    }).catch((err) => {
      throw isConnectionError(err)
        ? new Error(
            `Modal inference function not reachable after retries.\n` +
              `Run 'qmd modal status' to check deployment, or 'qmd modal deploy' to redeploy.\n` +
              `Original error: ${err instanceof Error ? err.message : err}`,
          )
        : err;
    });
  }

  /**
   * Tokenize texts using the embedding model's tokenizer on Modal.
   * Returns token IDs as nested number arrays.
   */
  async tokenize(texts: string[]): Promise<number[][]> {
    await this.ensureConnected();
    return withRetry(async () => {
      const fn = this.getMethod("tokenize");
      return fn.remote([texts]);
    }).catch((err) => {
      throw isConnectionError(err)
        ? new Error(
            `Modal inference function not reachable after retries.\n` +
              `Run 'qmd modal status' to check deployment, or 'qmd modal deploy' to redeploy.\n` +
              `Original error: ${err instanceof Error ? err.message : err}`,
          )
        : err;
    });
  }

  /**
   * Raw text generation with optional GBNF grammar.
   * Caller passes the fully formatted prompt including special tokens.
   */
  async generate(
    prompt: string,
    grammar: string | null,
    maxTokens: number,
    model: "expand" | "rerank" = "expand",
  ): Promise<string> {
    await this.ensureConnected();
    return withRetry(async () => {
      const fn = this.getMethod("generate");
      return fn.remote([prompt, grammar, maxTokens, model]);
    }).catch((err) => {
      throw isConnectionError(err)
        ? new Error(
            `Modal inference function not reachable after retries.\n` +
              `Run 'qmd modal status' to check deployment, or 'qmd modal deploy' to redeploy.\n` +
              `Original error: ${err instanceof Error ? err.message : err}`,
          )
        : err;
    });
  }

  /**
   * Cross-encoder reranking using Qwen3-Reranker.
   * Python side handles the chat template via create_chat_completion().
   */
  async rerank(query: string, texts: string[]): Promise<number[]> {
    await this.ensureConnected();
    return withRetry(async () => {
      const fn = this.getMethod("rerank");
      return fn.remote([query, texts]);
    }).catch((err) => {
      throw isConnectionError(err)
        ? new Error(
            `Modal inference function not reachable after retries.\n` +
              `Run 'qmd modal status' to check deployment, or 'qmd modal deploy' to redeploy.\n` +
              `Original error: ${err instanceof Error ? err.message : err}`,
          )
        : err;
    });
  }

  /**
   * Health check. Verifies the deployed function is reachable.
   */
  async ping(): Promise<boolean> {
    await this.ensureConnected();
    return withRetry(async () => {
      const fn = this.getMethod("ping");
      return fn.remote();
    }).catch((err) => {
      throw isConnectionError(err)
        ? new Error(
            `Modal inference function not reachable after retries.\n` +
              `Run 'qmd modal status' to check deployment, or 'qmd modal deploy' to redeploy.\n` +
              `Original error: ${err instanceof Error ? err.message : err}`,
          )
        : err;
    });
  }

  /**
   * No-op. Nothing local to clean up — Modal handles container lifecycle.
   */
  dispose(): void {
    this.instance = null;
    this.initPromise = null;
  }
}
