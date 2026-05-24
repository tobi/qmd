import { describe, expect, test } from "vitest";
import { finishSuccessfulCliCommand } from "../src/cli/qmd.ts";
import { LlamaCpp } from "../src/llm.ts";

describe("CLI successful-exit lifecycle", () => {
  test("exits 0 after successful JSON output when post-output LLM cleanup fails", async () => {
    const exitCodes: number[] = [];
    const stderr: string[] = [];
    const flushed: string[] = [];

    await finishSuccessfulCliCommand({
      command: "query",
      format: "json",
      platform: "linux",
      cleanup: async () => {
        throw new Error("ggml_metal_device_free abort simulation");
      },
      exit: (code) => {
        exitCodes.push(code);
      },
      stdout: { write: (chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { flushed.push(String(chunk)); cb?.(); return true; } },
      stderr: { write: (chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { stderr.push(String(chunk)); cb?.(); return true; } },
    });

    expect(exitCodes).toEqual([0]);
    expect(stderr.join("")).toContain("QMD Warning: cleanup after successful output failed");
    expect(flushed).toEqual([""]);
  });

  test("uses immediate exit for successful macOS JSON query after stdout flush", async () => {
    const calls: string[] = [];

    await finishSuccessfulCliCommand({
      command: "query",
      format: "json",
      platform: "darwin",
      cleanup: async () => {
        calls.push("cleanup");
      },
      exit: (code) => {
        calls.push(`exit:${code}`);
      },
      immediateExit: (code) => {
        calls.push(`immediate-exit:${code}`);
      },
      stdout: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stdout-flush"); cb?.(); return true; } },
      stderr: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stderr-flush"); cb?.(); return true; } },
    });

    expect(calls).toEqual(["stdout-flush", "stderr-flush", "immediate-exit:0"]);
  });

  test("disposes Llama resources in dependency order before CLI exit", async () => {
    const calls: string[] = [];
    const llm = new LlamaCpp({ inactivityTimeoutMs: 0 });
    const disposable = (name: string) => ({
      dispose: async () => {
        calls.push(name);
      },
    });

    Object.assign(llm as unknown as Record<string, unknown>, {
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
});
