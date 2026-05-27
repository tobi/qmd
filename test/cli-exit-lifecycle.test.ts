import { join as pathJoin } from "path";
import { describe, expect, test } from "vitest";
import {
  finishSuccessfulCliCommand,
  getSelfSpawnArgs,
  isSuccessfulJsonSupervisorOutput,
  shouldSuperviseDarwinJsonQuery,
  writeSupervisorOutput,
} from "../src/cli/qmd.ts";
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

  test("uses normal cleanup for unsupervised macOS JSON query runs", async () => {
    const calls: string[] = [];
    const previousChild = process.env.QMD_DARWIN_QUERY_JSON_CHILD;
    const previousSuccessFd = process.env.QMD_DARWIN_QUERY_JSON_SUCCESS_FD;
    delete process.env.QMD_DARWIN_QUERY_JSON_CHILD;
    process.env.QMD_DARWIN_QUERY_JSON_SUCCESS_FD = "3";

    try {
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
        writeSync: (fd, data) => {
          calls.push(`sentinel:${fd}:${String(data)}`);
          return String(data).length;
        },
        terminateImmediately: () => {
          calls.push("terminate");
        },
        stdout: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stdout-flush"); cb?.(); return true; } },
        stderr: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stderr-flush"); cb?.(); return true; } },
      });
    } finally {
      if (previousChild === undefined) {
        delete process.env.QMD_DARWIN_QUERY_JSON_CHILD;
      } else {
        process.env.QMD_DARWIN_QUERY_JSON_CHILD = previousChild;
      }
      if (previousSuccessFd === undefined) {
        delete process.env.QMD_DARWIN_QUERY_JSON_SUCCESS_FD;
      } else {
        process.env.QMD_DARWIN_QUERY_JSON_SUCCESS_FD = previousSuccessFd;
      }
    }

    expect(calls).toEqual(["stdout-flush", "cleanup", "stderr-flush", "exit:0"]);
  });

  test("supervised macOS JSON query commands write success sentinel and terminate before native cleanup", async () => {
    for (const command of ["query", "deep-search"]) {
      const calls: string[] = [];

      await finishSuccessfulCliCommand({
        command,
        format: "json",
        platform: "darwin",
        successFd: 3,
        cleanup: async () => {
          calls.push("cleanup");
        },
        writeSync: (fd, data) => {
          calls.push(`sentinel:${fd}:${String(data)}`);
          return String(data).length;
        },
        terminateImmediately: () => {
          calls.push("terminate");
        },
        stdout: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stdout-flush"); cb?.(); return true; } },
        stderr: { write: (_chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => { calls.push("stderr-flush"); cb?.(); return true; } },
      });

      expect(calls).toEqual([
        "stdout-flush",
        "stderr-flush",
        "sentinel:3:qmd:query-json-success\n",
        "terminate",
      ]);
    }
  });

  test("supervisor is limited to GPU-capable macOS query JSON runs", () => {
    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: {},
      values: {},
    })).toBe(true);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "deep-search",
      query: "test",
      format: "json",
      platform: "darwin",
      env: {},
      values: {},
    })).toBe(true);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: { QMD_FORCE_CPU: "1" },
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: { QMD_FORCE_CPU: "on" },
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: { QMD_LLAMA_GPU: "off" },
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: { QMD_LLAMA_GPU: "disabled" },
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: {},
      values: { "no-gpu": true },
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: { QMD_DARWIN_QUERY_JSON_CHILD: "1" },
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "linux",
      env: {},
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "search",
      query: "test",
      format: "json",
      platform: "darwin",
      env: {},
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "cli",
      platform: "darwin",
      env: {},
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "   ",
      format: "json",
      platform: "darwin",
      env: {},
      values: {},
    })).toBe(false);

    expect(shouldSuperviseDarwinJsonQuery({
      command: "query",
      query: "test",
      format: "json",
      platform: "darwin",
      env: { QMD_DISABLE_DARWIN_QUERY_JSON_SAFE_EXIT: "1" },
      values: {},
    })).toBe(false);
  });

  test("supervisor requires both valid JSON and the side-channel success sentinel", () => {
    expect(isSuccessfulJsonSupervisorOutput("[]\n", "qmd:query-json-success\n")).toBe(true);
    expect(isSuccessfulJsonSupervisorOutput("[]\n", "")).toBe(false);
    expect(isSuccessfulJsonSupervisorOutput("not json\n", "qmd:query-json-success\n")).toBe(false);
  });

  test("forwards supervised stderr before stdout and waits for callbacks", async () => {
    const calls: string[] = [];
    const makeStream = (name: string) => ({
      write: (chunk: string | Uint8Array, cb?: (error?: Error | null) => void) => {
        calls.push(`${name}:start:${String(chunk)}`);
        queueMicrotask(() => {
          calls.push(`${name}:flushed`);
          cb?.();
        });
        return true;
      },
    });

    await writeSupervisorOutput("json\n", "status\n", makeStream("stdout"), makeStream("stderr"));

    expect(calls).toEqual([
      "stderr:start:status\n",
      "stderr:flushed",
      "stdout:start:json\n",
      "stdout:flushed",
    ]);
  });

  test("builds self-spawn args for compiled, Node source, and Bun source entrypoints", () => {
    const compiledPath = pathJoin("repo", "dist", "cli", "qmd.js");
    const sourcePath = pathJoin("repo", "src", "cli", "qmd.ts");

    expect(getSelfSpawnArgs(compiledPath, ["query"], false)).toEqual([compiledPath, "query"]);
    expect(getSelfSpawnArgs(sourcePath, ["query"], true)).toEqual([sourcePath, "query"]);

    const nodeSourceArgs = getSelfSpawnArgs(sourcePath, ["query"], false);
    expect(nodeSourceArgs).toEqual([
      "--import",
      pathJoin("repo", "node_modules", "tsx", "dist", "esm", "index.mjs"),
      sourcePath,
      "query",
    ]);
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
