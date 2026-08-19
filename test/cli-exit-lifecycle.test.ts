import { describe, expect, test } from "vitest";
import { finishSuccessfulCliCommand } from "../src/cli/qmd.ts";
import { LlamaCpp, describeMetalResidencyPolicy, shouldKeepMetalResidencySets, withLLMSessionForLlm } from "../src/llm.ts";

describe("CLI successful-exit lifecycle", () => {
  test("does not exit 0 after successful output when post-output LLM cleanup fails", async () => {
    const exitCodes: number[] = [];
    const stderr: string[] = [];
    const flushed: string[] = [];

    await expect(finishSuccessfulCliCommand({
      command: "query",
      format: "json",
      cleanup: async () => {
        throw new Error("ggml_metal_device_free abort simulation");
      },
      exit: (code) => {
        exitCodes.push(code);
      },
      stdout: { write: (chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { flushed.push(String(chunk)); cb?.(); return true; } },
      stderr: { write: (chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { stderr.push(String(chunk)); cb?.(); return true; } },
    })).rejects.toThrow("ggml_metal_device_free abort simulation");

    expect(exitCodes).toEqual([]);
    expect(stderr.join("")).toContain("QMD Error: shutdown after output failed");
    expect(flushed).toEqual([""]);
  });

  test("flushes stdout, runs cleanup, flushes stderr, then exits (when exit is provided)", async () => {
    // The legacy lifecycle order is preserved for callers that pass an
    // explicit `exit` function — primarily this test, which needs an
    // observable terminating step.
    const calls: string[] = [];

    await finishSuccessfulCliCommand({
      command: "query",
      format: "json",
      cleanup: async () => { calls.push("cleanup"); },
      exit: (code) => { calls.push(`exit:${code}`); },
      stdout: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stdout-flush"); cb?.(); return true; } },
      stderr: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stderr-flush"); cb?.(); return true; } },
    });

    expect(calls).toEqual(["stdout-flush", "cleanup", "stderr-flush", "exit:0"]);
  });

  test("production path: sets process.exitCode=0 and returns instead of calling process.exit", async () => {
    // The real CLI does NOT pass `exit` — finishSuccessfulCliCommand should set
    // process.exitCode and return, letting Node's `beforeExit` fire so
    // node-llama-cpp's auto-dispose runs BEFORE libc's static destructor.
    // process.exit() skips `beforeExit`, which is what trips the libggml-metal
    // assertion (ggml-org/llama.cpp#22593) even with explicit dispose.
    const prevCode = process.exitCode;
    process.exitCode = undefined;
    try {
      const calls: string[] = [];
      await finishSuccessfulCliCommand({
        command: "query",
        format: "json",
        cleanup: async () => { calls.push("cleanup"); },
        stdout: { write: (_c: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stdout-flush"); cb?.(); return true; } },
        stderr: { write: (_c: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stderr-flush"); cb?.(); return true; } },
      });

      expect(calls).toEqual(["stdout-flush", "cleanup", "stderr-flush"]);
      expect(process.exitCode).toBe(0);
    } finally {
      process.exitCode = prevCode;
    }
  });

  test("residency predicate: default, opt-in, opt-out, and explicit GGML env", () => {
    const darwin = { platform: "darwin" as const, gpuMode: "auto" as const };

    // Default: follow node-llama-cpp, which disables residency sets.
    expect(shouldKeepMetalResidencySets({ ...darwin, env: {} })).toBe(false);
    // Explicit opt-in.
    expect(shouldKeepMetalResidencySets({ ...darwin, env: { QMD_METAL_KEEP_RESIDENCY: "1" } })).toBe(true);
    // Explicit opt-out.
    expect(shouldKeepMetalResidencySets({ ...darwin, env: { QMD_METAL_KEEP_RESIDENCY: "0" } })).toBe(false);

    // Any set GGML_METAL_NO_RESIDENCY wins: node-llama-cpp reads the variable
    // directly, so QMD must not force the skip flag on top of it.
    for (const value of ["1", "0", ""]) {
      expect(shouldKeepMetalResidencySets({ ...darwin, env: { GGML_METAL_NO_RESIDENCY: value } })).toBe(false);
      expect(shouldKeepMetalResidencySets({
        ...darwin,
        env: { GGML_METAL_NO_RESIDENCY: value, QMD_METAL_KEEP_RESIDENCY: "1" },
      })).toBe(false);
    }

    // Not a Darwin/Metal concern otherwise.
    expect(shouldKeepMetalResidencySets({ platform: "linux", gpuMode: "auto", env: { QMD_METAL_KEEP_RESIDENCY: "1" } })).toBe(false);
    expect(shouldKeepMetalResidencySets({ ...darwin, gpuMode: false, env: { QMD_METAL_KEEP_RESIDENCY: "1" } })).toBe(false);
  });

  test("doctor description reports the effective residency state", () => {
    const darwin = { platform: "darwin" as const, gpuMode: "auto" as const };
    expect(describeMetalResidencyPolicy({ ...darwin, env: {} })).toContain("node-llama-cpp default");
    expect(describeMetalResidencyPolicy({ ...darwin, env: { QMD_METAL_KEEP_RESIDENCY: "1" } }))
      .toContain("QMD_METAL_KEEP_RESIDENCY=1");
    expect(describeMetalResidencyPolicy({ ...darwin, env: { GGML_METAL_NO_RESIDENCY: "1" } }))
      .toContain("GGML_METAL_NO_RESIDENCY=1");
  });

  test("disposes Llama resources in dependency order before CLI exit", async () => {
    const calls: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const disposable = (name: string) => ({
      dispose: async () => {
        calls.push(name);
      },
    });

    Object.assign(llm, {
      embedContexts: [disposable("embed-context")],
      rerankContexts: [disposable("rerank-context")],
      embedModel: disposable("embed-model"),
      generateModel: disposable("generate-model"),
      rerankModel: disposable("rerank-model"),
      llama: disposable("llama"),
    });

    await llm.dispose();

    expect(calls).toEqual([
      "embed-context",
      "rerank-context",
      "embed-model",
      "generate-model",
      "rerank-model",
      "llama",
    ]);
  });

  test("waits for an active instance-scoped session before native teardown", async () => {
    const calls: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    Object.assign(llm as Record<string, unknown>, {
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    let markStarted!: () => void;
    let releaseSession!: () => void;
    const started = new Promise<void>((resolve) => { markStarted = resolve; });
    const gate = new Promise<void>((resolve) => { releaseSession = resolve; });
    const activeSession = withLLMSessionForLlm(llm, async () => {
      markStarted();
      await gate;
    });

    await started;
    const disposing = llm.dispose();
    await Promise.resolve();
    expect(calls).toEqual([]);

    releaseSession();
    await activeSession;
    await disposing;
    expect(calls).toEqual(["llama"]);
  });

  test("memoizes concurrent disposal and tears down native resources once", async () => {
    const calls: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    Object.assign(llm as Record<string, unknown>, {
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    const first = llm.dispose();
    const second = llm.dispose();
    expect(first).toBe(second);
    await Promise.all([first, second]);
    expect(calls).toEqual(["llama"]);
  });

  test("does not dispose parent models when child context disposal fails", async () => {
    const calls: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    Object.assign(llm as Record<string, unknown>, {
      embedContexts: [{ dispose: async () => { calls.push("context"); throw new Error("context busy"); } }],
      embedModel: { dispose: async () => { calls.push("model"); } },
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    await expect(llm.dispose()).rejects.toThrow("context busy");
    expect(calls).toEqual(["context"]);
  });

  test("waits for slow child disposal instead of racing into its parent", async () => {
    const calls: string[] = [];
    let finishContext!: () => void;
    const contextGate = new Promise<void>((resolve) => { finishContext = resolve; });
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    Object.assign(llm as Record<string, unknown>, {
      embedContexts: [{
        dispose: async () => {
          calls.push("context-start");
          await contextGate;
          calls.push("context-end");
        },
      }],
      embedModel: { dispose: async () => { calls.push("model"); } },
      llama: { dispose: async () => { calls.push("llama"); } },
    });

    const disposing = llm.dispose();
    const deadline = Date.now() + 1000;
    while (calls.length === 0 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    expect(calls).toEqual(["context-start"]);
    finishContext();
    await disposing;
    expect(calls).toEqual(["context-start", "context-end", "model", "llama"]);
  });
});
