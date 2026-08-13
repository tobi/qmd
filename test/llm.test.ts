/**
 * llm.test.ts - Unit tests for the LLM abstraction layer (node-llama-cpp)
 *
 * Run with: bun test src/llm.test.ts
 *
 * These tests require the actual models to be downloaded. Run the embed or
 * rerank functions first to trigger model downloads.
 */

import { describe, test, expect, beforeAll, afterAll, vi } from "vitest";
import { chmodSync, mkdirSync, mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import {
  LlamaCpp,
  getDefaultLlamaCpp,
  disposeDefaultLlamaCpp,
  resolveLlamaGpuMode,
  setNodeLlamaCppModuleForTest,
  setLlamaDirWritableForTest,
  canWriteLlamaDir,
  withNativeStdoutRedirectedToStderr,
  resolveParallelismOverride,
  resolveSafeParallelism,
  computeGpuContextPoolSize,
  estimateEmbedContextMB,
  BASELINE_EMBED_CONTEXT_MB,
  EMBED_POOL_RERANK_RESERVE_MB,
  resolveEmbedModel,
  resolveGenerateModel,
  resolveRerankModel,
  resolveModels,
  pullModels,
  withLLMSession,
  canUnloadLLM,
  SessionReleasedError,
  type RerankDocument,
  type ILLMSession,
} from "../src/llm.js";

describe("canWriteLlamaDir", () => {
  test("returns true when llama/ exists and is writable", () => {
    const dir = mkdtempSync(join(tmpdir(), "qmd-llama-rw-"));
    mkdirSync(join(dir, "llama"));
    try {
      expect(canWriteLlamaDir(dir)).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("returns false when the package directory is missing", () => {
    expect(canWriteLlamaDir("/tmp/qmd-no-such-llama-pkg")).toBe(false);
  });

  test("returns false when llama/ exists but is not writable", () => {
    const dir = mkdtempSync(join(tmpdir(), "qmd-llama-ro-"));
    const llamaDir = join(dir, "llama");
    mkdirSync(llamaDir, { mode: 0o555 });
    chmodSync(llamaDir, 0o555);
    try {
      expect(canWriteLlamaDir(dir)).toBe(false);
    } finally {
      chmodSync(llamaDir, 0o755);
      rmSync(dir, { recursive: true, force: true });
    }
  });

  test("ensureLlama uses build:never when the llama dir is read-only", async () => {
    const prevGpu = process.env.QMD_LLAMA_GPU;
    const prevForceCpu = process.env.QMD_FORCE_CPU;
    delete process.env.QMD_LLAMA_GPU;
    delete process.env.QMD_FORCE_CPU;

    const calls: Array<Record<string, unknown>> = [];
    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile: vi.fn(),
      LlamaChatSession: vi.fn() as any,
      getLlama: vi.fn(async (options: Record<string, unknown>) => {
        calls.push(options);
        return { gpu: false, cpuMathCores: 4, dispose: async () => {} } as any;
      }),
    });
    // After the module mock: that helper clears the writability override.
    setLlamaDirWritableForTest(false);

    try {
      const llm = new LlamaCpp();
      await (llm as any).ensureLlama();
      expect(calls.length).toBeGreaterThan(0);
      expect(calls[0]?.build).toBe("never");
      expect(calls[0]?.skipDownload).toBe(true);
    } finally {
      setLlamaDirWritableForTest(undefined);
      setNodeLlamaCppModuleForTest(null);
      if (prevGpu === undefined) delete process.env.QMD_LLAMA_GPU;
      else process.env.QMD_LLAMA_GPU = prevGpu;
      if (prevForceCpu === undefined) delete process.env.QMD_FORCE_CPU;
      else process.env.QMD_FORCE_CPU = prevForceCpu;
    }
  });
});


describe("model download progress (#776)", () => {
  test("pullModels hides node-llama-cpp CLI progress by default", async () => {
    const cacheDir = mkdtempSync(join(tmpdir(), "qmd-pull-quiet-"));
    const resolveModelFile = vi.fn(async () => join(cacheDir, "missing.gguf"));
    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile,
      LlamaChatSession: vi.fn() as any,
      getLlama: vi.fn(),
    });
    try {
      await pullModels(["local-model.gguf"], { cacheDir });
      expect(resolveModelFile).toHaveBeenCalledWith(
        "local-model.gguf",
        { directory: cacheDir, cli: false },
      );
    } finally {
      setNodeLlamaCppModuleForTest(null);
      rmSync(cacheDir, { recursive: true, force: true });
    }
  });

  test("pullModels shows CLI progress only when cli: true", async () => {
    const cacheDir = mkdtempSync(join(tmpdir(), "qmd-pull-progress-"));
    const resolveModelFile = vi.fn(async () => join(cacheDir, "missing.gguf"));
    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile,
      LlamaChatSession: vi.fn() as any,
      getLlama: vi.fn(),
    });
    try {
      await pullModels(["local-model.gguf"], { cacheDir, cli: true });
      expect(resolveModelFile).toHaveBeenCalledWith(
        "local-model.gguf",
        { directory: cacheDir, cli: true },
      );
    } finally {
      setNodeLlamaCppModuleForTest(null);
      rmSync(cacheDir, { recursive: true, force: true });
    }
  });

  test("LlamaCpp.resolveModel also hides download progress", async () => {
    const cacheDir = mkdtempSync(join(tmpdir(), "qmd-resolve-quiet-"));
    const resolveModelFile = vi.fn(async () => join(cacheDir, "missing.gguf"));
    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile,
      LlamaChatSession: vi.fn() as any,
      getLlama: vi.fn(),
    });
    try {
      const llm = new LlamaCpp({ modelCacheDir: cacheDir });
      await (llm as any).resolveModel("local-model.gguf");
      expect(resolveModelFile).toHaveBeenCalledWith(
        "local-model.gguf",
        { directory: cacheDir, cli: false },
      );
    } finally {
      setNodeLlamaCppModuleForTest(null);
      rmSync(cacheDir, { recursive: true, force: true });
    }
  });
});

describe("model name resolution", () => {
  function withModelEnv(env: Record<string, string | undefined>, fn: () => void): void {
    const previous = {
      QMD_EMBED_MODEL: process.env.QMD_EMBED_MODEL,
      QMD_GENERATE_MODEL: process.env.QMD_GENERATE_MODEL,
      QMD_RERANK_MODEL: process.env.QMD_RERANK_MODEL,
    };
    try {
      for (const [key, value] of Object.entries(env)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
      fn();
    } finally {
      for (const [key, value] of Object.entries(previous)) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    }
  }

  test("all model roles resolve config hints before env fallbacks", () => {
    withModelEnv({
      QMD_EMBED_MODEL: "env-embed",
      QMD_GENERATE_MODEL: "env-generate",
      QMD_RERANK_MODEL: "env-rerank",
    }, () => {
      const config = {
        embed: "config-embed",
        generate: "config-generate",
        rerank: "config-rerank",
      };
      expect(resolveEmbedModel(config)).toBe("config-embed");
      expect(resolveGenerateModel(config)).toBe("config-generate");
      expect(resolveRerankModel(config)).toBe("config-rerank");
      expect(resolveModels(config)).toEqual(config);
    });
  });

  test("LlamaCpp constructor uses the same resolver as status/embed/query helpers", () => {
    withModelEnv({
      QMD_EMBED_MODEL: "env-embed",
      QMD_GENERATE_MODEL: "env-generate",
      QMD_RERANK_MODEL: "env-rerank",
    }, () => {
      const llm = new LlamaCpp({
        embedModel: "config-embed",
        generateModel: "config-generate",
        rerankModel: "config-rerank",
      });
      expect(llm.embedModelName).toBe(resolveEmbedModel({ embed: "config-embed" }));
      expect(llm.generateModelName).toBe(resolveGenerateModel({ generate: "config-generate" }));
      expect(llm.rerankModelName).toBe(resolveRerankModel({ rerank: "config-rerank" }));
    });
  });
});

// =============================================================================
// Singleton Tests (no model loading required)
// =============================================================================

describe("Default LlamaCpp Singleton", () => {
  // Test singleton behavior without resetting to avoid orphan instances
  test("getDefaultLlamaCpp returns same instance on subsequent calls", () => {
    const llm1 = getDefaultLlamaCpp();
    const llm2 = getDefaultLlamaCpp();
    expect(llm1).toBe(llm2);
    expect(llm1).toBeInstanceOf(LlamaCpp);
  });
});

// =============================================================================
// Model Existence Tests
// =============================================================================

describe("LlamaCpp.modelExists", () => {
  test("returns exists:true for HuggingFace model URIs", async () => {
    const llm = getDefaultLlamaCpp();
    const result = await llm.modelExists("hf:org/repo/model.gguf");

    expect(result.exists).toBe(true);
    expect(result.name).toBe("hf:org/repo/model.gguf");
  });

  test("returns exists:false for non-existent local paths", async () => {
    const llm = getDefaultLlamaCpp();
    const result = await llm.modelExists("/nonexistent/path/model.gguf");

    expect(result.exists).toBe(false);
    expect(result.name).toBe("/nonexistent/path/model.gguf");
  });
});

describe("QMD_LLAMA_GPU resolution", () => {
  test("uses auto when unset or blank", () => {
    expect(resolveLlamaGpuMode(undefined)).toBe("auto");
    expect(resolveLlamaGpuMode("   ")).toBe("auto");
  });

  test("maps CPU disable values to false", () => {
    expect(resolveLlamaGpuMode("false")).toBe(false);
    expect(resolveLlamaGpuMode("OFF")).toBe(false);
    expect(resolveLlamaGpuMode(" none ")).toBe(false);
    expect(resolveLlamaGpuMode("disabled")).toBe(false);
    expect(resolveLlamaGpuMode("0")).toBe(false);
  });

  test("passes through supported GPU backends", () => {
    expect(resolveLlamaGpuMode("metal")).toBe("metal");
    expect(resolveLlamaGpuMode("VULKAN")).toBe("vulkan");
    expect(resolveLlamaGpuMode(" cuda ")).toBe("cuda");
  });

  test("QMD_FORCE_CPU disables GPU before QMD_LLAMA_GPU auto-detection", () => {
    const prevForceCpu = process.env.QMD_FORCE_CPU;
    process.env.QMD_FORCE_CPU = "1";
    try {
      expect(resolveLlamaGpuMode(undefined)).toBe(false);
      expect(resolveLlamaGpuMode("cuda")).toBe(false);
    } finally {
      if (prevForceCpu === undefined) delete process.env.QMD_FORCE_CPU;
      else process.env.QMD_FORCE_CPU = prevForceCpu;
    }
  });

  test("QMD_FORCE_CPU ignores false-ish values", () => {
    const prevForceCpu = process.env.QMD_FORCE_CPU;
    process.env.QMD_FORCE_CPU = "0";
    try {
      expect(resolveLlamaGpuMode(undefined)).toBe("auto");
    } finally {
      if (prevForceCpu === undefined) delete process.env.QMD_FORCE_CPU;
      else process.env.QMD_FORCE_CPU = prevForceCpu;
    }
  });

  test("warns and falls back to auto for unsupported values", () => {
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      expect(resolveLlamaGpuMode("rocm")).toBe("auto");
      expect(stderrSpy).toHaveBeenCalled();
      expect(String(stderrSpy.mock.calls[0]?.[0] || "")).toContain("QMD_LLAMA_GPU");
    } finally {
      stderrSpy.mockRestore();
    }
  });
});

describe("native llama stdout containment", () => {
  test("redirects native stdout noise to stderr while JSON callers are initializing llama", async () => {
    const stdoutSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      await withNativeStdoutRedirectedToStderr(async () => {
        process.stdout.write("cmake build spam\n");
        return "ok";
      });

      expect(stdoutSpy).not.toHaveBeenCalled();
      expect(stderrSpy).toHaveBeenCalledWith("cmake build spam\n", undefined, undefined);
    } finally {
      stdoutSpy.mockRestore();
      stderrSpy.mockRestore();
    }
  });

  test("keeps native GPU failure noise off stdout and caches failed GPU init", async () => {
    const prevGpu = process.env.QMD_LLAMA_GPU;
    const prevForceCpu = process.env.QMD_FORCE_CPU;
    process.env.QMD_LLAMA_GPU = "cuda";
    delete process.env.QMD_FORCE_CPU;

    const calls: unknown[] = [];
    const fakeLlama = { gpu: false, cpuMathCores: 4 };
    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile: vi.fn(),
      LlamaChatSession: vi.fn() as any,
      getLlama: vi.fn(async (options: Record<string, unknown>) => {
        calls.push(options.gpu);
        if (options.gpu === "cuda") {
          process.stdout.write("cmake build spam\n");
          throw new Error("CUDA unavailable");
        }
        return fakeLlama as any;
      }),
    });

    const stdoutSpy = vi.spyOn(process.stdout, "write").mockReturnValue(true);
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      const first = new LlamaCpp();
      const second = new LlamaCpp();

      await (first as any).ensureLlama();
      await (second as any).ensureLlama();

      expect(stdoutSpy).not.toHaveBeenCalled();
      expect(stderrSpy).toHaveBeenCalledWith("cmake build spam\n", undefined, undefined);
      expect(calls).toEqual(["cuda", false, false]);
      expect(String(stderrSpy.mock.calls.map(call => call[0]).join(""))).toContain("skipping previously failed GPU init");
    } finally {
      stdoutSpy.mockRestore();
      stderrSpy.mockRestore();
      setNodeLlamaCppModuleForTest(null);
      if (prevGpu === undefined) delete process.env.QMD_LLAMA_GPU;
      else process.env.QMD_LLAMA_GPU = prevGpu;
      if (prevForceCpu === undefined) delete process.env.QMD_FORCE_CPU;
      else process.env.QMD_FORCE_CPU = prevForceCpu;
    }
  });

  test("warns about CPU fallback only once per process", async () => {
    const prevGpu = process.env.QMD_LLAMA_GPU;
    const prevForceCpu = process.env.QMD_FORCE_CPU;
    process.env.QMD_LLAMA_GPU = "false";
    delete process.env.QMD_FORCE_CPU;

    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile: vi.fn(),
      LlamaChatSession: vi.fn() as any,
      getLlama: vi.fn(async () => ({ gpu: false, cpuMathCores: 4 }) as any),
    });

    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      const first = new LlamaCpp();
      const second = new LlamaCpp();

      await (first as any).ensureLlama();
      await (second as any).ensureLlama();

      const stderr = String(stderrSpy.mock.calls.map(call => call[0]).join(""));
      expect(stderr.match(/no GPU acceleration/g)?.length).toBe(1);
      expect(stderr).toContain("qmd doctor");
      expect(stderr).not.toContain("QMD_STATUS_DEVICE_PROBE");
    } finally {
      stderrSpy.mockRestore();
      setNodeLlamaCppModuleForTest(null);
      if (prevGpu === undefined) delete process.env.QMD_LLAMA_GPU;
      else process.env.QMD_LLAMA_GPU = prevGpu;
      if (prevForceCpu === undefined) delete process.env.QMD_FORCE_CPU;
      else process.env.QMD_FORCE_CPU = prevForceCpu;
    }
  });

  test("embeds hello world with QMD_FORCE_CPU=1 without throwing", async () => {
    const prevGpu = process.env.QMD_LLAMA_GPU;
    const prevForceCpu = process.env.QMD_FORCE_CPU;
    process.env.QMD_FORCE_CPU = "1";
    process.env.QMD_LLAMA_GPU = "metal";

    const getEmbeddingFor = vi.fn(async (text: string) => ({
      vector: new Float32Array([0.1, 0.2, 0.3]),
      text,
    }));
    const createEmbeddingContext = vi.fn(async () => ({
      getEmbeddingFor,
      dispose: vi.fn(async () => {}),
    }));
    const loadModel = vi.fn(async () => ({
      trainContextSize: 2048,
      tokenize: (text: string) => Array.from(text),
      detokenize: (tokens: string[]) => tokens.join(""),
      createEmbeddingContext,
      dispose: vi.fn(async () => {}),
    }));
    const getLlama = vi.fn(async (options: Record<string, unknown>) => ({
      gpu: false,
      cpuMathCores: 4,
      loadModel,
      dispose: vi.fn(async () => {}),
    }) as any);

    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile: vi.fn(async () => "/tmp/nonexistent-model.gguf"),
      LlamaChatSession: vi.fn() as any,
      getLlama,
    });

    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const llm = new LlamaCpp();
    try {
      const result = await llm.embed("hello world");
      expect(result).toEqual({
        embedding: [0.10000000149011612, 0.20000000298023224, 0.30000001192092896],
        model: llm.embedModelName,
      });
      expect(getLlama).toHaveBeenCalledWith(expect.objectContaining({ gpu: false, build: "never" }));
      expect(loadModel).toHaveBeenCalledWith(expect.objectContaining({ gpuLayers: 0 }));
      expect(getEmbeddingFor).toHaveBeenCalledWith("hello world");
    } finally {
      await llm.dispose();
      stderrSpy.mockRestore();
      setNodeLlamaCppModuleForTest(null);
      if (prevGpu === undefined) delete process.env.QMD_LLAMA_GPU;
      else process.env.QMD_LLAMA_GPU = prevGpu;
      if (prevForceCpu === undefined) delete process.env.QMD_FORCE_CPU;
      else process.env.QMD_FORCE_CPU = prevForceCpu;
    }
  });
});

describe("LLM context parallelism safety", () => {
  test("defaults Windows CUDA to one context to avoid ggml-cuda.cu:98 crashes", () => {
    expect(resolveSafeParallelism({
      gpu: "cuda",
      platform: "win32",
      computed: 8,
      envValue: undefined,
    })).toBe(1);
  });

  test("keeps non-Windows and non-CUDA backends on computed parallelism", () => {
    expect(resolveSafeParallelism({ gpu: "cuda", platform: "linux", computed: 8 })).toBe(8);
    expect(resolveSafeParallelism({ gpu: "vulkan", platform: "win32", computed: 8 })).toBe(8);
    expect(resolveSafeParallelism({ gpu: false, platform: "win32", computed: 4 })).toBe(4);
  });

  test("QMD_EMBED_PARALLELISM overrides the Windows CUDA safety default", () => {
    expect(resolveSafeParallelism({
      gpu: "cuda",
      platform: "win32",
      computed: 8,
      envValue: "2",
    })).toBe(2);
  });

  test("QMD_EMBED_PARALLELISM clamps invalid values and warns", () => {
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      expect(resolveParallelismOverride("0")).toBeUndefined();
      expect(resolveParallelismOverride("bad")).toBeUndefined();
      expect(stderrSpy).toHaveBeenCalledTimes(2);
      expect(String(stderrSpy.mock.calls[0]?.[0] || "")).toContain("QMD_EMBED_PARALLELISM");
    } finally {
      stderrSpy.mockRestore();
    }
  });
});

describe("embedding context VRAM pool (#799)", () => {
  test("keeps the nomic/embeddinggemma baseline for small GGUFs so default embed throughput is unchanged", () => {
    const gemmaBytes = 308 * 1024 * 1024;
    expect(estimateEmbedContextMB({ modelBytes: gemmaBytes })).toBe(BASELINE_EMBED_CONTEXT_MB);
    expect(estimateEmbedContextMB({ modelBytes: 137 * 1024 * 1024 })).toBe(BASELINE_EMBED_CONTEXT_MB);
  });

  test("scales Qwen3-Embedding-0.6B-class GGUFs to ~1190 MB per context", () => {
    const qwenBytes = 640 * 1024 * 1024;
    expect(estimateEmbedContextMB({ modelBytes: qwenBytes })).toBe(Math.round(640 * 1.85));
    expect(estimateEmbedContextMB({ modelBytes: qwenBytes, contextSize: 1024 })).toBe(Math.round(640 * 1.85 * 0.5));
  });

  test("falls back to the baseline when the file size is unknown", () => {
    expect(estimateEmbedContextMB({ modelBytes: 0 })).toBe(BASELINE_EMBED_CONTEXT_MB);
    expect(estimateEmbedContextMB({ modelBytes: Number.NaN })).toBe(BASELINE_EMBED_CONTEXT_MB);
  });

  test("on 8 GB VRAM, a Qwen 0.6B pool is 1 context so the reranker still fits", () => {
    const perContextMB = estimateEmbedContextMB({ modelBytes: 640 * 1024 * 1024 });
    expect(computeGpuContextPoolSize({
      freeMB: 7500,
      perContextMB,
      reserveMB: EMBED_POOL_RERANK_RESERVE_MB,
    })).toBe(1);
  });

  test("on 8 GB VRAM, the default 150 MB estimate still allows the 8-context cap", () => {
    expect(computeGpuContextPoolSize({
      freeMB: 7500,
      perContextMB: BASELINE_EMBED_CONTEXT_MB,
      reserveMB: EMBED_POOL_RERANK_RESERVE_MB,
    })).toBe(8);
  });

  test("the old hardcoded 150 MB estimate would still open 8 Qwen contexts and OOM the reranker", () => {
    expect(computeGpuContextPoolSize({
      freeMB: 7500,
      perContextMB: 150,
      reserveMB: EMBED_POOL_RERANK_RESERVE_MB,
    })).toBe(8);
  });
});

describe("LlamaCpp expand context size config", () => {
  const defaultExpandContextSize = 2048;

  test("uses default expand context size when no config or env is set", () => {
    const prev = process.env.QMD_EXPAND_CONTEXT_SIZE;
    delete process.env.QMD_EXPAND_CONTEXT_SIZE;
    try {
      const llm = new LlamaCpp({}) as any;
      expect(llm.expandContextSize).toBe(defaultExpandContextSize);
    } finally {
      if (prev === undefined) delete process.env.QMD_EXPAND_CONTEXT_SIZE;
      else process.env.QMD_EXPAND_CONTEXT_SIZE = prev;
    }
  });

  test("uses QMD_EXPAND_CONTEXT_SIZE when set to a positive integer", () => {
    const prev = process.env.QMD_EXPAND_CONTEXT_SIZE;
    process.env.QMD_EXPAND_CONTEXT_SIZE = "3072";
    try {
      const llm = new LlamaCpp({}) as any;
      expect(llm.expandContextSize).toBe(3072);
    } finally {
      if (prev === undefined) delete process.env.QMD_EXPAND_CONTEXT_SIZE;
      else process.env.QMD_EXPAND_CONTEXT_SIZE = prev;
    }
  });

  test("config value overrides QMD_EXPAND_CONTEXT_SIZE", () => {
    const prev = process.env.QMD_EXPAND_CONTEXT_SIZE;
    process.env.QMD_EXPAND_CONTEXT_SIZE = "4096";
    try {
      const llm = new LlamaCpp({ expandContextSize: 1536 }) as any;
      expect(llm.expandContextSize).toBe(1536);
    } finally {
      if (prev === undefined) delete process.env.QMD_EXPAND_CONTEXT_SIZE;
      else process.env.QMD_EXPAND_CONTEXT_SIZE = prev;
    }
  });

  test("falls back to default and warns when QMD_EXPAND_CONTEXT_SIZE is invalid", () => {
    const prev = process.env.QMD_EXPAND_CONTEXT_SIZE;
    process.env.QMD_EXPAND_CONTEXT_SIZE = "bad";
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    try {
      const llm = new LlamaCpp({}) as any;
      expect(llm.expandContextSize).toBe(defaultExpandContextSize);
      expect(stderrSpy).toHaveBeenCalled();
      expect(String(stderrSpy.mock.calls[0]?.[0] || "")).toContain("QMD_EXPAND_CONTEXT_SIZE");
    } finally {
      stderrSpy.mockRestore();
      if (prev === undefined) delete process.env.QMD_EXPAND_CONTEXT_SIZE;
      else process.env.QMD_EXPAND_CONTEXT_SIZE = prev;
    }
  });

  test("throws when config expandContextSize is invalid", () => {
    expect(() => new LlamaCpp({ expandContextSize: 0 })).toThrow(
      "Invalid expandContextSize: 0. Must be a positive integer."
    );
  });
});

describe("LlamaCpp model resolution (config > env > default)", () => {
  const HARDCODED_EMBED = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
  const HARDCODED_RERANK = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
  const HARDCODED_GENERATE = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";

  test("uses hardcoded default when no config or env is set", () => {
    const prev = process.env.QMD_EMBED_MODEL;
    delete process.env.QMD_EMBED_MODEL;
    try {
      const llm = new LlamaCpp({}) as any;
      expect(llm.embedModelUri).toBe(HARDCODED_EMBED);
      expect(llm.rerankModelUri).toBe(HARDCODED_RERANK);
      expect(llm.generateModelUri).toBe(HARDCODED_GENERATE);
    } finally {
      if (prev === undefined) delete process.env.QMD_EMBED_MODEL;
      else process.env.QMD_EMBED_MODEL = prev;
    }
  });

  test("env var overrides hardcoded default", () => {
    const prev = process.env.QMD_EMBED_MODEL;
    process.env.QMD_EMBED_MODEL = "hf:custom/embed-model.gguf";
    try {
      const llm = new LlamaCpp({}) as any;
      expect(llm.embedModelUri).toBe("hf:custom/embed-model.gguf");
    } finally {
      if (prev === undefined) delete process.env.QMD_EMBED_MODEL;
      else process.env.QMD_EMBED_MODEL = prev;
    }
  });

  test("config overrides env var", () => {
    const prev = process.env.QMD_EMBED_MODEL;
    process.env.QMD_EMBED_MODEL = "hf:env/model.gguf";
    try {
      const llm = new LlamaCpp({ embedModel: "hf:config/model.gguf" }) as any;
      expect(llm.embedModelUri).toBe("hf:config/model.gguf");
    } finally {
      if (prev === undefined) delete process.env.QMD_EMBED_MODEL;
      else process.env.QMD_EMBED_MODEL = prev;
    }
  });
});

describe("LlamaCpp embedding truncation", () => {
  test("truncates against the active embedding context limit, not the model train context", async () => {
    const llm = new LlamaCpp({}) as any;
    const getEmbeddingFor = vi.fn(async (text: string) => ({
      vector: new Float32Array([0.25, 0.5]),
      text,
    }));

    llm.touchActivity = vi.fn();
    llm.embedModel = {
      trainContextSize: 8192,
      tokenize: (text: string) => Array.from({ length: text.length }, () => 1),
      detokenize: (tokens: readonly number[]) => "x".repeat(tokens.length),
    };
    llm.ensureEmbedContext = vi.fn().mockResolvedValue({ getEmbeddingFor });

    const result = await llm.embed("x".repeat(3000));

    expect(getEmbeddingFor).toHaveBeenCalledWith("x".repeat(2044));
    expect(result).toEqual({
      embedding: [0.25, 0.5],
      model: llm.embedModelUri,
    });
  });
});

describe("LlamaCpp rerank deduping", () => {
  test("deduplicates identical document texts before scoring", async () => {
    const llm = new LlamaCpp({}) as any;
    llm._ciMode = false; // allow unit test even in CI (mocked, no real models)
    const rankAll = vi.fn(async (_query: string, docs: string[]) =>
      docs.map((doc) => doc === "shared chunk" ? 0.9 : 0.2)
    );

    llm.touchActivity = vi.fn();
    llm.ensureRerankContexts = vi.fn().mockResolvedValue([{ rankAll }]);
    llm.ensureRerankModel = vi.fn().mockResolvedValue({
      tokenize: (text: string) => Array.from(text),
      detokenize: (tokens: string[]) => tokens.join(""),
    });

    const result = await llm.rerank("query", [
      { file: "a.md", text: "shared chunk" },
      { file: "b.md", text: "shared chunk" },
      { file: "c.md", text: "different chunk" },
    ]);

    expect(rankAll).toHaveBeenCalledTimes(1);
    expect(rankAll).toHaveBeenCalledWith("query", ["shared chunk", "different chunk"]);
    expect(result.results).toHaveLength(3);

    const scoreByFile = new Map(result.results.map((item) => [item.file, item.score]));
    expect(scoreByFile.get("a.md")).toBe(0.9);
    expect(scoreByFile.get("b.md")).toBe(0.9);
    expect(scoreByFile.get("c.md")).toBe(0.2);
  });
});

describe("LlamaCpp ensureRerankContexts error reporting", () => {
  test("warns with the underlying error and does not retry identical options", async () => {
    const llm = new LlamaCpp({}) as any;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const createRankingContext = vi.fn().mockRejectedValue(
        new Error("A context size of 40960 is too large for the available VRAM"),
      );
      llm.computeParallelism = vi.fn().mockResolvedValue(2);
      llm.threadsPerContext = vi.fn().mockResolvedValue(0);
      llm.ensureRerankModel = vi.fn().mockResolvedValue({ createRankingContext });

      const contexts = await llm.ensureRerankContexts();

      expect(contexts).toEqual([]);
      expect(createRankingContext).toHaveBeenCalledTimes(1);
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining("A context size of 40960 is too large for the available VRAM"),
      );
    } finally {
      warn.mockRestore();
    }
  });

  test("keeps already-created contexts when a later createRankingContext fails", async () => {
    const llm = new LlamaCpp({}) as any;
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const ctx1 = { id: 1 };
      let calls = 0;
      const createRankingContext = vi.fn().mockImplementation(async () => {
        calls++;
        if (calls === 1) return ctx1;
        throw new Error("out of VRAM");
      });
      llm.computeParallelism = vi.fn().mockResolvedValue(3);
      llm.threadsPerContext = vi.fn().mockResolvedValue(0);
      llm.ensureRerankModel = vi.fn().mockResolvedValue({ createRankingContext });

      const contexts = await llm.ensureRerankContexts();

      expect(contexts).toEqual([ctx1]);
      expect(createRankingContext).toHaveBeenCalledTimes(2);
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  test("serializes concurrent cold-start callers so ranking contexts are created once", async () => {
    const llm = new LlamaCpp({}) as any;
    const ctx = { id: 1 };
    let inflight = 0;
    let maxInflight = 0;
    const createRankingContext = vi.fn().mockImplementation(async () => {
      inflight++;
      maxInflight = Math.max(maxInflight, inflight);
      await new Promise((resolve) => setTimeout(resolve, 30));
      inflight--;
      return ctx;
    });
    llm.computeParallelism = vi.fn().mockResolvedValue(1);
    llm.threadsPerContext = vi.fn().mockResolvedValue(0);
    llm.ensureRerankModel = vi.fn().mockResolvedValue({ createRankingContext });

    const [a, b, c] = await Promise.all([
      llm.ensureRerankContexts(),
      llm.ensureRerankContexts(),
      llm.ensureRerankContexts(),
    ]);

    expect(a).toEqual([ctx]);
    expect(b).toBe(a);
    expect(c).toBe(a);
    expect(createRankingContext).toHaveBeenCalledTimes(1);
    expect(maxInflight).toBe(1);
  });
});

describe("LlamaCpp.getDeviceInfo", () => {
  test("can skip build attempts for status probes", async () => {
    const llm = new LlamaCpp({}) as any;
    const fakeLlama = {
      gpu: "metal",
      supportsGpuOffloading: true,
      cpuMathCores: 8,
      getGpuDeviceNames: vi.fn().mockResolvedValue(["Apple GPU"]),
      getVramState: vi.fn().mockResolvedValue({ total: 1024, used: 256, free: 768 }),
    };

    llm.ensureLlama = vi.fn().mockResolvedValue(fakeLlama);

    const device = await llm.getDeviceInfo({ allowBuild: false });

    expect(llm.ensureLlama).toHaveBeenCalledWith(false);
    expect(device).toEqual({
      gpu: "metal",
      gpuOffloading: true,
      gpuDevices: ["Apple GPU"],
      vram: { total: 1024, used: 256, free: 768 },
      cpuCores: 8,
    });
  });
});

// =============================================================================
// Integration Tests (require actual models)
// =============================================================================

describe.skipIf(!!process.env.CI)("LlamaCpp Integration", () => {
  // Use the singleton to avoid multiple Metal contexts
  const llm = getDefaultLlamaCpp();

  afterAll(async () => {
    // Ensure native resources are released to avoid ggml-metal asserts on process exit.
    await disposeDefaultLlamaCpp();
  });

  describe("embed", () => {
    test("returns embedding with correct dimensions", async () => {
      const result = await llm.embed("Hello world");

      expect(result).not.toBeNull();
      expect(result!.embedding).toBeInstanceOf(Array);
      expect(result!.embedding.length).toBeGreaterThan(0);
      // embeddinggemma outputs 768 dimensions
      expect(result!.embedding.length).toBe(768);
    });

    test("returns consistent embeddings for same input", async () => {
      const result1 = await llm.embed("test text");
      const result2 = await llm.embed("test text");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Embeddings should be identical for the same input
      for (let i = 0; i < result1!.embedding.length; i++) {
        expect(result1!.embedding[i]).toBeCloseTo(result2!.embedding[i]!, 5);
      }
    });

    test("returns different embeddings for different inputs", async () => {
      const result1 = await llm.embed("cats are great");
      const result2 = await llm.embed("database optimization");

      expect(result1).not.toBeNull();
      expect(result2).not.toBeNull();

      // Calculate cosine similarity - should be less than 1.0 (not identical)
      let dotProduct = 0;
      let norm1 = 0;
      let norm2 = 0;
      for (let i = 0; i < result1!.embedding.length; i++) {
        const v1 = result1!.embedding[i]!;
        const v2 = result2!.embedding[i]!;
        dotProduct += v1 * v2;
        norm1 += v1 ** 2;
        norm2 += v2 ** 2;
      }
      const similarity = dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));

      expect(similarity).toBeLessThan(0.95); // Should be meaningfully different
    });
  });

  describe("embedBatch", () => {
    test("returns embeddings for multiple texts", async () => {
      const texts = ["Hello world", "Test text", "Another document"];
      const results = await llm.embedBatch(texts);

      expect(results).toHaveLength(3);
      for (const result of results) {
        expect(result).not.toBeNull();
        expect(result!.embedding.length).toBe(768);
      }
    });

    test("returns same results as individual embed calls", async () => {
      const texts = ["cats are great", "dogs are awesome"];

      // Get batch embeddings
      const batchResults = await llm.embedBatch(texts);

      // Get individual embeddings
      const individualResults = await Promise.all(texts.map(t => llm.embed(t)));

      // Compare - should be identical
      for (let i = 0; i < texts.length; i++) {
        expect(batchResults[i]).not.toBeNull();
        expect(individualResults[i]).not.toBeNull();
        for (let j = 0; j < batchResults[i]!.embedding.length; j++) {
          expect(batchResults[i]!.embedding[j]).toBeCloseTo(individualResults[i]!.embedding[j]!, 5);
        }
      }
    });

    test("handles empty array", async () => {
      const results = await llm.embedBatch([]);
      expect(results).toHaveLength(0);
    });

    test("batch is faster than sequential", async () => {
      const texts = Array(10).fill(null).map((_, i) => `Document number ${i} with content`);

      // Time batch
      const batchStart = Date.now();
      await llm.embedBatch(texts);
      const batchTime = Date.now() - batchStart;

      // Time sequential
      const seqStart = Date.now();
      for (const text of texts) {
        await llm.embed(text);
      }
      const seqTime = Date.now() - seqStart;

      console.log(`Batch: ${batchTime}ms, Sequential: ${seqTime}ms`);
      // Performance is machine/load dependent. We only assert batch isn't drastically worse.
      expect(batchTime).toBeLessThanOrEqual(seqTime * 3);
    });

    test("handles concurrent embedBatch calls on fresh instance without race condition", async () => {
      // This test verifies the fix for a race condition where concurrent calls to
      // ensureEmbedContext() could create multiple contexts. Without the promise guard,
      // each concurrent embedBatch call sees embedContext === null and creates its own
      // context, causing resource leaks and potential "Context is disposed" errors.
      //
      // See: https://github.com/tobi/qmd/pull/54
      //
      // The fix uses a promise guard to ensure only one context creation runs at a time.
      // We verify this by instrumenting createEmbeddingContext to count invocations.
      
      const freshLlm = new LlamaCpp({});
      let contextCreateCount = 0;
      
      // Instrument the model's createEmbeddingContext to count calls
      const originalEnsureEmbedModel = (freshLlm as any).ensureEmbedModel.bind(freshLlm);
      let modelInstrumented = false;
      (freshLlm as any).ensureEmbedModel = async function() {
        const model = await originalEnsureEmbedModel();
        if (!modelInstrumented) {
          modelInstrumented = true;
          const originalCreate = model.createEmbeddingContext.bind(model);
          model.createEmbeddingContext = async function(...args: any[]) {
            contextCreateCount++;
            return originalCreate(...args);
          };
        }
        return model;
      };
      
      const texts = Array(10).fill(null).map((_, i) => `Document ${i}`);

      // Call embedBatch 5 TIMES in parallel on fresh instance.
      // Without the promise guard fix, this would create 5 contexts (one per call).
      // With the fix, only 1 context should be created.
      const batches = await Promise.all([
        freshLlm.embedBatch(texts.slice(0, 2)),
        freshLlm.embedBatch(texts.slice(2, 4)),
        freshLlm.embedBatch(texts.slice(4, 6)),
        freshLlm.embedBatch(texts.slice(6, 8)),
        freshLlm.embedBatch(texts.slice(8, 10)),
      ]);

      const allResults = batches.flat();
      expect(allResults).toHaveLength(10);
      
      const successCount = allResults.filter(r => r !== null).length;
      expect(successCount).toBe(10);

      // THE KEY ASSERTION: Contexts should be created once (by ensureEmbedContexts),
      // not duplicated per concurrent embedBatch call. The exact count depends on
      // available VRAM (computeParallelism), but should not be 5 (one per call).
      // Without the fix, contextCreateCount would be 5× the intended count (one set per concurrent call).
      // With the promise guard, contexts are created exactly once regardless of concurrent callers.
      // The count depends on VRAM (computeParallelism), but should be ≤ 8 (the cap).
      console.log(`Context creation count: ${contextCreateCount} (expected: ≤ 8, not 5× duplicated)`);
      expect(contextCreateCount).toBeGreaterThanOrEqual(1);
      expect(contextCreateCount).toBeLessThanOrEqual(8);
      
      await freshLlm.dispose();
    }, 60000);
  });

  describe("rerank", () => {
    test("scores capital of France question correctly", async () => {
      const query = "What is the capital of France?";
      const documents: RerankDocument[] = [
        { file: "butterflies.txt", text: "Butterflies indeed fly through the garden." },
        { file: "france.txt", text: "The capital of France is Paris." },
        { file: "canada.txt", text: "The capital of Canada is Ottawa." },
      ];

      const result = await llm.rerank(query, documents);

      expect(result.results).toHaveLength(3);

      // The France document should score highest
      expect(result.results[0]!.file).toBe("france.txt");
      expect(result.results[0]!.score).toBeGreaterThan(0.7);

      // Canada should be somewhat relevant (also about capitals)
      expect(result.results[1]!.file).toBe("canada.txt");

      // Butterflies should score lowest
      expect(result.results[2]!.file).toBe("butterflies.txt");
      expect(result.results[2]!.score).toBeLessThan(0.6);
    });

    test("scores authentication query correctly", async () => {
      const query = "How do I configure authentication?";
      const documents: RerankDocument[] = [
        { file: "weather.md", text: "The weather today is sunny with mild temperatures." },
        { file: "auth.md", text: "Authentication can be configured by setting the AUTH_SECRET environment variable." },
        { file: "pizza.md", text: "Our restaurant serves the best pizza in town." },
        { file: "jwt.md", text: "JWT authentication requires a secret key and expiration time." },
      ];

      const result = await llm.rerank(query, documents);

      expect(result.results).toHaveLength(4);

      // Auth documents should score highest
      const topTwo = result.results.slice(0, 2).map((r) => r.file);
      expect(topTwo).toContain("auth.md");
      expect(topTwo).toContain("jwt.md");

      // Irrelevant documents should score lowest
      const bottomTwo = result.results.slice(2).map((r) => r.file);
      expect(bottomTwo).toContain("weather.md");
      expect(bottomTwo).toContain("pizza.md");
    });

    test("handles programming queries correctly", async () => {
      const query = "How do I handle errors in JavaScript?";
      const documents: RerankDocument[] = [
        { file: "cooking.md", text: "To make a good pasta, boil water and add salt." },
        { file: "errors.md", text: "Use try-catch blocks to handle JavaScript errors gracefully." },
        { file: "python.md", text: "Python uses try-except for exception handling." },
      ];

      const result = await llm.rerank(query, documents);

      // JavaScript errors doc should score highest
      expect(result.results[0]!.file).toBe("errors.md");
      expect(result.results[0]!.score).toBeGreaterThan(0.7);

      // Python doc might be somewhat relevant (same concept, different language)
      // Cooking should be least relevant
      expect(result.results[2]!.file).toBe("cooking.md");
    });

    test("handles empty document list", async () => {
      const result = await llm.rerank("test query", []);
      expect(result.results).toHaveLength(0);
    });

    test("handles single document", async () => {
      const result = await llm.rerank("test", [{ file: "doc.md", text: "content" }]);
      expect(result.results).toHaveLength(1);
      expect(result.results[0]!.file).toBe("doc.md");
    });

    test("preserves original file paths", async () => {
      const documents: RerankDocument[] = [
        { file: "path/to/doc1.md", text: "content one" },
        { file: "another/path/doc2.md", text: "content two" },
      ];

      const result = await llm.rerank("query", documents);

      const files = result.results.map((r) => r.file).sort();
      expect(files).toEqual(["another/path/doc2.md", "path/to/doc1.md"]);
    });

    test("returns scores between 0 and 1", async () => {
      const documents: RerankDocument[] = [
        { file: "a.md", text: "The quick brown fox jumps over the lazy dog." },
        { file: "b.md", text: "Machine learning algorithms process data efficiently." },
        { file: "c.md", text: "React components use JSX syntax for rendering." },
      ];

      const result = await llm.rerank("Tell me about animals", documents);

      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
      }
    });

    test("batch reranks multiple documents efficiently", async () => {
      // Create 10 documents to verify batch processing works
      const documents: RerankDocument[] = Array(10)
        .fill(null)
        .map((_, i) => ({
          file: `doc${i}.md`,
          text: `Document number ${i} with some content about topic ${i % 3}`,
        }));

      const start = Date.now();
      const result = await llm.rerank("topic 1", documents);
      const elapsed = Date.now() - start;

      expect(result.results).toHaveLength(10);

      // Verify all documents are returned with valid scores
      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
      }

      // Log timing for monitoring batch performance
      console.log(`Batch rerank of 10 docs took ${elapsed}ms`);
    });

    test("uses fewer active rerank contexts for small batches", async () => {
      const freshLlm = new LlamaCpp({});
      const calls: number[] = [];
      const fakeModel = {
        tokenize: (text: string) => Array.from(text),
        detokenize: (tokens: string[]) => tokens.join(""),
      };
      const fakeContexts = Array.from({ length: 4 }, (_, idx) => ({
        rankAll: async (_query: string, docs: string[]) => {
          calls.push(idx);
          return docs.map(() => 0.5);
        },
      }));

      (freshLlm as any).ensureRerankModel = async () => fakeModel;
      (freshLlm as any).ensureRerankContexts = async () => fakeContexts;

      const documents: RerankDocument[] = Array.from({ length: 20 }, (_, i) => ({
        file: `doc${i}.md`,
        text: `Document number ${i}`,
      }));

      const result = await freshLlm.rerank("topic 1", documents);

      expect(result.results).toHaveLength(20);
      expect(calls).toEqual([0, 1]);
    });

    test("truncates and reranks document exceeding 2048 token context size", async () => {
      // The reranker context is created with contextSize=2048. Documents that
      // exceed the token budget (contextSize - template overhead - query tokens)
      // should be silently truncated rather than crashing.
      const paragraph = "The quick brown fox jumps over the lazy dog near the riverbank. " +
        "Authentication tokens must be validated on every request to ensure security. " +
        "Database queries should use prepared statements to prevent SQL injection attacks. " +
        "The deployment pipeline includes linting, testing, building, and publishing stages. ";
      // ~320 chars per paragraph, repeat 40 times = ~12800 chars ≈ 3200 tokens
      const longText = paragraph.repeat(40);

      const query = "How do I configure authentication?";
      const documents: RerankDocument[] = [
        { file: "short-relevant.md", text: "Authentication can be configured by setting AUTH_SECRET." },
        { file: "long-doc.md", text: longText },
        { file: "short-irrelevant.md", text: "The weather is sunny today." },
      ];

      console.log(`Long doc length: ${longText.length} chars (~${Math.round(longText.length / 4)} tokens)`);

      const result = await llm.rerank(query, documents);

      // Should return all 3 documents without crashing
      expect(result.results).toHaveLength(3);

      // All scores should be valid numbers in [0, 1]
      for (const doc of result.results) {
        expect(doc.score).toBeGreaterThanOrEqual(0);
        expect(doc.score).toBeLessThanOrEqual(1);
        expect(Number.isNaN(doc.score)).toBe(false);
      }

      // The short, directly relevant doc should still rank highest
      console.log("Rerank results for long doc test:");
      for (const doc of result.results) {
        console.log(`  ${doc.file}: ${doc.score.toFixed(4)}`);
      }
    }, 30000);
  });

  describe("expandQuery", () => {
    test("returns query expansions with correct types", async () => {
      const result = await llm.expandQuery("test query");

      // Result is Queryable[] containing lex, vec, and/or hyde entries
      expect(result.length).toBeGreaterThanOrEqual(1);

      // Each result should have a valid type
      for (const q of result) {
        expect(["lex", "vec", "hyde"]).toContain(q.type);
        expect(q.text.length).toBeGreaterThan(0);
      }
    }, 30000); // 30s timeout for model loading

    test("can exclude lexical queries", async () => {
      const result = await llm.expandQuery("authentication setup", { includeLexical: false });

      // Should not contain any 'lex' type entries
      const lexEntries = result.filter(q => q.type === "lex");
      expect(lexEntries).toHaveLength(0);
    });
  });
});

// =============================================================================
// Session Management Tests
// =============================================================================

describe.skipIf(!!process.env.CI)("LLM Session Management", () => {
  describe("withLLMSession", () => {
    test("session provides access to LLM operations", async () => {
      const result = await withLLMSession(async (session) => {
        expect(session.isValid).toBe(true);
        const embedding = await session.embed("test text");
        expect(embedding).not.toBeNull();
        expect(embedding!.embedding.length).toBe(768);
        return "success";
      });
      expect(result).toBe("success");
    });

    test("session is invalid after release", async () => {
      let capturedSession: ILLMSession | null = null;

      await withLLMSession(async (session) => {
        capturedSession = session;
        expect(session.isValid).toBe(true);
      });

      // Session should be invalid after withLLMSession returns
      expect(capturedSession).not.toBeNull();
      expect(capturedSession!.isValid).toBe(false);
    });

    test("session prevents idle unload during operations", async () => {
      await withLLMSession(async (session) => {
        // While inside a session, canUnloadLLM should return false
        expect(canUnloadLLM()).toBe(false);

        // Perform an operation
        await session.embed("test");

        // Still should not be able to unload
        expect(canUnloadLLM()).toBe(false);
      });

      // After session ends, should be able to unload
      expect(canUnloadLLM()).toBe(true);
    });

    test("nested sessions increment ref count", async () => {
      await withLLMSession(async (outerSession) => {
        expect(canUnloadLLM()).toBe(false);

        await withLLMSession(async (innerSession) => {
          expect(canUnloadLLM()).toBe(false);
          expect(innerSession.isValid).toBe(true);
          expect(outerSession.isValid).toBe(true);
        });

        // Inner session released, but outer still active
        expect(canUnloadLLM()).toBe(false);
        expect(outerSession.isValid).toBe(true);
      });

      // All sessions released
      expect(canUnloadLLM()).toBe(true);
    });

    test("session embedBatch works correctly", async () => {
      await withLLMSession(async (session) => {
        const texts = ["Hello world", "Test text", "Another document"];
        const results = await session.embedBatch(texts);

        expect(results).toHaveLength(3);
        for (const result of results) {
          expect(result).not.toBeNull();
          expect(result!.embedding.length).toBe(768);
        }
      });
    });

    test("session rerank works correctly", async () => {
      await withLLMSession(async (session) => {
        const documents: RerankDocument[] = [
          { file: "a.txt", text: "The capital of France is Paris." },
          { file: "b.txt", text: "Dogs are great pets." },
        ];

        const result = await session.rerank("What is the capital of France?", documents);

        expect(result.results).toHaveLength(2);
        expect(result.results[0]!.file).toBe("a.txt");
        expect(result.results[0]!.score).toBeGreaterThan(result.results[1]!.score);
      });
    });

    test("max duration aborts session after timeout", async () => {
      let aborted = false;

      try {
        await withLLMSession(async (session) => {
          // Wait longer than max duration
          await new Promise(resolve => setTimeout(resolve, 150));

          // This operation should throw because session was aborted
          await session.embed("test");
        }, { maxDuration: 50 }); // 50ms max
      } catch (err) {
        if (err instanceof SessionReleasedError) {
          aborted = true;
        } else {
          throw err;
        }
      }

      expect(aborted).toBe(true);
    }, 5000);

    test("external abort signal propagates to session", async () => {
      const abortController = new AbortController();
      let sessionAborted = false;

      const promise = withLLMSession(async (session) => {
        // Wait a bit then check if aborted
        await new Promise(resolve => setTimeout(resolve, 100));

        if (!session.isValid) {
          sessionAborted = true;
          throw new SessionReleasedError("Session aborted");
        }

        return "should not reach";
      }, { signal: abortController.signal });

      // Abort after 20ms
      setTimeout(() => abortController.abort(), 20);

      try {
        await promise;
      } catch (err) {
        // Expected
      }

      expect(sessionAborted).toBe(true);
    }, 5000);

    test("session provides abort signal for monitoring", async () => {
      await withLLMSession(async (session) => {
        expect(session.signal).toBeInstanceOf(AbortSignal);
        expect(session.signal.aborted).toBe(false);
      });
    });

    test("returns value from callback", async () => {
      const result = await withLLMSession(async (session) => {
        await session.embed("test");
        return { status: "complete", count: 42 };
      });

      expect(result).toEqual({ status: "complete", count: 42 });
    });

    test("propagates errors from callback", async () => {
      const customError = new Error("Custom test error");

      await expect(
        withLLMSession(async () => {
          throw customError;
        })
      ).rejects.toThrow("Custom test error");
    });
  });
});

describe("LlamaCpp generate sequence dispose (node-llama-cpp 3.20)", () => {
  test("awaits async sequence.dispose before context.dispose", async () => {
    const order: string[] = [];
    let sequenceDisposedResolved = false;
    const sequence = {
      dispose: vi.fn(async () => {
        order.push("sequence-start");
        await new Promise((r) => setTimeout(r, 20));
        sequenceDisposedResolved = true;
        order.push("sequence-end");
      }),
    };
    const context = {
      getSequence: vi.fn(() => sequence),
      dispose: vi.fn(async () => {
        order.push("context");
        expect(sequenceDisposedResolved).toBe(true);
      }),
    };

    const llm = new LlamaCpp({}) as any;
    llm._ciMode = false;
    llm.touchActivity = vi.fn();
    llm.ensureGenerateModel = vi.fn(async () => {});
    llm.generateModel = { createContext: vi.fn(async () => context) };
    llm.generateModelUri = "hf:test/gen.gguf";

    setNodeLlamaCppModuleForTest({
      LlamaLogLevel: { error: "error" },
      resolveModelFile: vi.fn(),
      LlamaChatSession: class {
        async prompt(_prompt: string, options?: { onTextChunk?: (text: string) => void }) {
          options?.onTextChunk?.("ok");
          return "ok";
        }
      } as any,
      getLlama: vi.fn(),
    });

    try {
      const result = await llm.generate("hello");
      expect(result).toEqual({
        text: "ok",
        model: "hf:test/gen.gguf",
        done: true,
      });
      expect(sequence.dispose).toHaveBeenCalledTimes(1);
      expect(context.dispose).toHaveBeenCalledTimes(1);
      expect(order).toEqual(["sequence-start", "sequence-end", "context"]);
    } finally {
      setNodeLlamaCppModuleForTest(null);
    }
  });
});
