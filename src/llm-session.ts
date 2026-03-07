import type {
  LLM,
  EmbedOptions,
  EmbeddingResult,
  ILLMSession,
  LLMSessionOptions,
  Queryable,
  RerankDocument,
  RerankOptions,
  RerankResult,
} from "./llm.js";

/**
 * Scoped session wrapper for non-local backends.
 * Enforces release/abort semantics but delegates operations directly to the backend.
 */
export class PassthroughLLMSession implements ILLMSession {
  private llm: LLM;
  private released = false;
  private abortController: AbortController;
  private maxDurationTimer: ReturnType<typeof setTimeout> | null = null;
  private name: string;
  private createReleasedError: (message?: string) => Error;

  constructor(
    llm: LLM,
    options: LLMSessionOptions = {},
    createReleasedError: (message?: string) => Error = (message) =>
      new Error(message || "LLM session has been released or aborted")
  ) {
    this.llm = llm;
    this.name = options.name || "unnamed";
    this.abortController = new AbortController();
    this.createReleasedError = createReleasedError;

    // Link external abort signal if provided
    if (options.signal) {
      if (options.signal.aborted) {
        this.abortController.abort(options.signal.reason);
      } else {
        options.signal.addEventListener("abort", () => {
          this.abortController.abort(options.signal!.reason);
        }, { once: true });
      }
    }

    // Set up max duration timer
    const maxDuration = options.maxDuration ?? 10 * 60 * 1000; // Default 10 minutes
    if (maxDuration > 0) {
      this.maxDurationTimer = setTimeout(() => {
        this.abortController.abort(new Error(`Session "${this.name}" exceeded max duration of ${maxDuration}ms`));
      }, maxDuration);
      this.maxDurationTimer.unref(); // Don't keep process alive
    }
  }

  get isValid(): boolean {
    return !this.released && !this.abortController.signal.aborted;
  }

  get signal(): AbortSignal {
    return this.abortController.signal;
  }

  release(): void {
    if (this.released) return;
    this.released = true;

    if (this.maxDurationTimer) {
      clearTimeout(this.maxDurationTimer);
      this.maxDurationTimer = null;
    }

    this.abortController.abort(new Error("Session released"));
  }

  private async withOperation<T>(fn: () => Promise<T>): Promise<T> {
    if (!this.isValid) {
      throw this.createReleasedError();
    }

    if (this.abortController.signal.aborted) {
      throw this.createReleasedError(
        this.abortController.signal.reason?.message || "Session aborted"
      );
    }

    return await fn();
  }

  async embed(text: string, options?: EmbedOptions): Promise<EmbeddingResult | null> {
    return this.withOperation(() => this.llm.embed(text, options));
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.withOperation(() => this.llm.embedBatch(texts));
  }

  async expandQuery(
    query: string,
    options?: { context?: string; includeLexical?: boolean }
  ): Promise<Queryable[]> {
    return this.withOperation(() => this.llm.expandQuery(query, options));
  }

  async rerank(
    query: string,
    documents: RerankDocument[],
    options?: RerankOptions
  ): Promise<RerankResult> {
    return this.withOperation(() => this.llm.rerank(query, documents, options));
  }
}

