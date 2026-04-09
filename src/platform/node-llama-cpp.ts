const NODE_LLAMA_CPP_MODULE_ID = "node-llama-cpp";

type LlamaContextSequence = unknown;
type LlamaGrammar = unknown;
type LlamaLogLevelValue = unknown;

export type LlamaToken = unknown;

export type LlamaVramState = {
  total: number;
  used: number;
  free: number;
};

export interface LlamaEmbeddingResult {
  vector: Iterable<number> | ArrayLike<number>;
}

export interface LlamaEmbeddingContext {
  getEmbeddingFor(text: string): Promise<LlamaEmbeddingResult>;
  dispose(): Promise<void>;
}

export interface LlamaRankingContext {
  rankAll(query: string, docs: string[]): Promise<number[]>;
  dispose(): Promise<void>;
}

export interface LlamaContext {
  getSequence(): LlamaContextSequence;
  dispose(): Promise<void>;
}

export interface LlamaModel {
  trainContextSize: number;
  tokenize(text: string): readonly LlamaToken[];
  detokenize(tokens: readonly LlamaToken[]): string;
  createEmbeddingContext(options: {
    contextSize: number;
    threads?: number;
  }): Promise<LlamaEmbeddingContext>;
  createContext(options?: {
    contextSize?: number;
  }): Promise<LlamaContext>;
  createRankingContext(options: {
    contextSize: number;
    flashAttention?: boolean;
    threads?: number;
  }): Promise<LlamaRankingContext>;
  dispose(): Promise<void>;
}

export interface Llama {
  gpu: string | false;
  supportsGpuOffloading: boolean;
  cpuMathCores: number;
  loadModel(options: { modelPath: string }): Promise<LlamaModel>;
  getGpuDeviceNames(): Promise<string[]>;
  getVramState(): Promise<LlamaVramState>;
  createGrammar(options: { grammar: string }): Promise<LlamaGrammar>;
  dispose(): Promise<void>;
}

export interface LlamaChatSession {
  prompt(prompt: string, options?: Record<string, unknown>): Promise<string>;
}

export interface NodeLlamaCppModule {
  getLlama(options: {
    build: "autoAttempt" | "never";
    logLevel: LlamaLogLevelValue;
    gpu: "auto" | false;
    skipDownload?: boolean;
  }): Promise<Llama>;
  resolveModelFile(modelUri: string, cacheDir: string): Promise<string>;
  LlamaChatSession: new (options: { contextSequence: LlamaContextSequence }) => LlamaChatSession;
  LlamaLogLevel: {
    error: LlamaLogLevelValue;
  };
}

let nodeLlamaCppModulePromise: Promise<NodeLlamaCppModule> | null = null;
let nodeLlamaCppLoader: (() => Promise<NodeLlamaCppModule>) | null = null;

function isNodeLlamaCppModule(value: unknown): value is NodeLlamaCppModule {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<NodeLlamaCppModule>;
  return (
    typeof candidate.getLlama === "function" &&
    typeof candidate.resolveModelFile === "function" &&
    typeof candidate.LlamaChatSession === "function" &&
    !!candidate.LlamaLogLevel &&
    "error" in candidate.LlamaLogLevel
  );
}

function normalizeNodeLlamaCppModule(moduleValue: unknown): NodeLlamaCppModule {
  if (isNodeLlamaCppModule(moduleValue)) {
    return moduleValue;
  }

  const defaultValue =
    moduleValue &&
    typeof moduleValue === "object" &&
    "default" in moduleValue
      ? (moduleValue as { default?: unknown }).default
      : undefined;

  if (isNodeLlamaCppModule(defaultValue)) {
    return defaultValue;
  }

  throw new Error("Loaded node-llama-cpp module has an unexpected shape.");
}

function defaultNodeLlamaCppLoader(): Promise<NodeLlamaCppModule> {
  const moduleId = NODE_LLAMA_CPP_MODULE_ID;
  return import(moduleId).then((moduleValue) => normalizeNodeLlamaCppModule(moduleValue));
}

export function formatNodeLlamaCppUnavailableMessage(cause?: unknown): string {
  const detail =
    cause instanceof Error
      ? cause.message
      : cause !== undefined && cause !== null
        ? String(cause)
        : null;

  const parts = [
    "node-llama-cpp is unavailable.",
    "QMD can still use BM25 and sqlite-vec features, but embeddings, query expansion, reranking, and model downloads require a working node-llama-cpp install.",
  ];

  if (process.platform === "freebsd") {
    parts.push(
      "On FreeBSD this usually means the optional dependency failed to build. Install the required C/C++ toolchain and node-llama-cpp build prerequisites, then reinstall qmd."
    );
  }

  if (detail) {
    parts.push(`Original error: ${detail}`);
  }

  return parts.join(" ");
}

export class NodeLlamaCppUnavailableError extends Error {
  override readonly cause: unknown;

  constructor(cause?: unknown) {
    super(formatNodeLlamaCppUnavailableMessage(cause));
    this.name = "NodeLlamaCppUnavailableError";
    this.cause = cause;
  }
}

export async function loadNodeLlamaCpp(): Promise<NodeLlamaCppModule> {
  if (!nodeLlamaCppModulePromise) {
    const loader = nodeLlamaCppLoader ?? defaultNodeLlamaCppLoader;
    nodeLlamaCppModulePromise = loader().catch((error) => {
      nodeLlamaCppModulePromise = null;
      throw new NodeLlamaCppUnavailableError(error);
    });
  }

  return await nodeLlamaCppModulePromise;
}

export function _setNodeLlamaCppLoaderForTesting(
  loader: (() => Promise<NodeLlamaCppModule>) | null
): void {
  nodeLlamaCppLoader = loader;
  nodeLlamaCppModulePromise = null;
}
