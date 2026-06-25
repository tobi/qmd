// =============================================================================
// hybrid-llm.ts - Routes per-operation between a local LlamaCpp backend and
// an optional remote OpenAI-compatible backend.
//
// Used by getDefaultLLM() in src/llm.ts. When `remote` is undefined
// (QMD_REMOTE_API_KEY not set), every operation falls back to local.
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
} from "./llm-types.js";
import { RemoteLLM } from "./remote-llm.js";

export type LLMBackend = "local" | "remote";

export type HybridLLMConfig = {
  embedBackend: LLMBackend;
  generateBackend: LLMBackend;
  rerankBackend: LLMBackend;
  tokenizeBackend: LLMBackend;
};

export type HybridLLMDeviceInfo = {
  gpu: string | false;
  gpuOffloading: boolean;
  gpuDevices: string[];
  vram?: { total: number; used: number; free: number };
  cpuCores: number;
};

export class HybridLLM implements LLM {
  private local: LLM;
  private remote?: RemoteLLM;
  private config: HybridLLMConfig;

  constructor(local: LLM, remote: RemoteLLM | undefined, config: HybridLLMConfig) {
    this.local = local;
    this.remote = remote;
    this.config = config;
  }

  /**
   * The concrete local backend (LlamaCpp or a test double). Exposed so the
   * CLI `qmd status` can call getDeviceInfo() on it directly, and so
   * downstream code that needs concrete-LlamaCpp-only features
   * (chunkDocumentByTokens, etc.) can downcast through getDefaultLlamaCpp().
   */
  getLocal(): LLM {
    return this.local;
  }

  private getBackend(preference: LLMBackend): LLM {
    if (preference === "remote") {
      if (!this.remote) {
        console.warn("Remote backend requested but not available (no API key). Falling back to local.");
        return this.local;
      }
      return this.remote;
    }
    return this.local;
  }

  async embed(text: string, options: EmbedOptions = {}): Promise<EmbeddingResult | null> {
    return this.getBackend(this.config.embedBackend).embed(text, options);
  }

  async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    return this.getBackend(this.config.embedBackend).embedBatch(texts);
  }

  async generate(prompt: string, options: GenerateOptions = {}): Promise<GenerateResult | null> {
    return this.getBackend(this.config.generateBackend).generate(prompt, options);
  }

  async modelExists(model: string): Promise<ModelInfo> {
    // We don't know the model type (embed/gen/rerank) here, so check both
    // backends. Local usually wins because the local registry is authoritative
    // for HF ggml-org URIs; remote is a permissive `{ exists: true }` stub.
    const localInfo = await this.local.modelExists(model);
    if (localInfo.exists) return localInfo;

    if (this.remote) {
      return this.remote.modelExists(model);
    }

    return localInfo;
  }

  async expandQuery(query: string, options: { context?: string; includeLexical?: boolean } = {}): Promise<Queryable[]> {
    // Query expansion is usually tied to generation
    return this.getBackend(this.config.generateBackend).expandQuery(query, options);
  }

  async rerank(query: string, documents: RerankDocument[], options: RerankOptions = {}): Promise<RerankResult> {
    return this.getBackend(this.config.rerankBackend).rerank(query, documents, options);
  }

  async tokenize(text: string): Promise<readonly any[]> {
    return this.getBackend(this.config.tokenizeBackend).tokenize(text);
  }

  async detokenize(tokens: readonly any[]): Promise<string> {
    return this.getBackend(this.config.tokenizeBackend).detokenize(tokens);
  }

  async dispose(): Promise<void> {
    await this.local.dispose();
    if (this.remote) {
      await this.remote.dispose();
    }
  }

  /**
   * Concrete-only getDeviceInfo — not part of the LLM interface. Always
   * proxies to the local backend because:
   *   - GPU/Metal/VRAM/CPU cores are hardware concepts the remote has no way
   *     to truthfully report.
   *   - If the user has any local operation routed through (e.g. rerank),
   *     they want to see GPU status.
   *
   * The `qmd status` command reads this from the HybridLLM (via
   * getDefaultLLM().getDeviceInfo) when the default LLM is HybridLLM, or
   * directly from LlamaCpp.getDeviceInfo when it is not.
   */
  async getDeviceInfo(): Promise<HybridLLMDeviceInfo> {
    const localAny = this.local as unknown as {
      getDeviceInfo: (options?: unknown) => Promise<HybridLLMDeviceInfo>;
    };
    if (typeof localAny.getDeviceInfo === "function") {
      // LlamaCpp.getDeviceInfo takes an options bag; others (RemoteLLM) take none.
      try {
        return await localAny.getDeviceInfo({ allowBuild: false });
      } catch {
        return await localAny.getDeviceInfo();
      }
    }
    return {
      gpu: false,
      gpuOffloading: false,
      gpuDevices: [],
      cpuCores: 0,
    };
  }
}
