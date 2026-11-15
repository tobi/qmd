import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import { createStore, JevReranker, RemoteEmbedder, type CollectionConfig } from "../src/index.js";
import { findLocalConfigPath, loadConfig, setConfigSource } from "../src/collections.js";
import {
  DEFAULT_EMBED_MODEL_URI,
  DEFAULT_GENERATE_MODEL_URI,
  DEFAULT_RERANK_MODEL_URI,
  LlamaCpp,
  setNodeLlamaCppModuleForTest,
} from "../src/llm.js";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.js";
import { recordTrust, sensitiveDigest, type SensitiveSnapshot } from "../src/trust.js";

const builtins = {
  embed: DEFAULT_EMBED_MODEL_URI,
  rerank: DEFAULT_RERANK_MODEL_URI,
  generate: DEFAULT_GENERATE_MODEL_URI,
};
const envNames = [
  "QMD_CONFIG_DIR", "QMD_TRUST_LOCAL_CONFIG", "QMD_TRUST_UPDATE_HOOKS",
  "QMD_EMBED_MODEL", "QMD_RERANK_MODEL", "QMD_GENERATE_MODEL",
  "QMD_TEST_MCP_CREDENTIAL", "QMD_TEST_MCP_OTHER_CREDENTIAL",
  "QMD_ALLOWED_ORIGINS", "QMD_ALLOWED_HOSTS",
] as const;
let savedEnv: Array<[string, string | undefined]>;
let root: string;
let configPath: string;
let dbPath: string;
let config: CollectionConfig;
let handle: HttpServerHandle | undefined;
let selected: typeof builtins[];

function snapshot(): SensitiveSnapshot {
  return {
    processors: Object.entries(config.collections)
      .filter(([, col]) => col.processor !== undefined)
      .map(([collection, col]) => ({ collection, spec: col.processor! })),
    hooks: Object.entries(config.collections)
      .filter(([, col]) => !!col.update)
      .map(([collection, col]) => ({ collection, command: col.update! })),
    paths: Object.entries(config.collections).map(([collection, col]) => ({ collection, path: col.path })),
    models: config.models ?? {},
  };
}

function writeConfig(): void {
  writeFileSync(configPath, YAML.stringify(config));
  // Match CLI discovery without changing the test runner's working directory.
  setConfigSource({ configPath: findLocalConfigPath(root)! });
}

async function startAndStop(): Promise<typeof builtins> {
  handle = await startMcpHttpServer(0, { dbPath, host: "127.0.0.1", quiet: true });
  await handle.stop();
  handle = undefined;
  return selected[selected.length - 1]!;
}

beforeEach(async () => {
  savedEnv = envNames.map(name => [name, process.env[name]]);
  for (const name of envNames) delete process.env[name];
  root = mkdtempSync(join(tmpdir(), "qmd-mcp-trust-"));
  process.env.QMD_CONFIG_DIR = join(root, "global-config");
  // These are made-up fixture values, never inherited credentials.
  process.env.QMD_TEST_MCP_CREDENTIAL = "synthetic-mcp-fixture-only";
  process.env.QMD_TEST_MCP_OTHER_CREDENTIAL = "synthetic-mcp-other-fixture-only";
  mkdirSync(join(root, ".qmd"));
  configPath = join(root, ".qmd", "index.yml");
  dbPath = join(root, "index.sqlite");
  const docs = join(root, "docs");
  mkdirSync(docs);
  writeFileSync(join(docs, "nebula.md"), "# Nebula\nA luminous cloud among distant stars.\n");
  config = {
    collections: { docs: { path: docs, pattern: "**/*.md" } },
    models: {
      embed: "https://mcp-embedding.invalid/v1#synthetic-model",
      embedOptions: { apiKeyEnv: "QMD_TEST_MCP_CREDENTIAL", dimensions: 2, revision: "fixture-v1" },
      rerank: "typesafe:synthetic-mcp-reranker",
      rerankOptions: { apiKeyEnv: "QMD_TEST_MCP_CREDENTIAL", revision: "fixture-v1" },
      generate: "hf:synthetic/mcp-generator/model.gguf",
    },
  };
  // Seed through the public SDK, without embeddings or model inference.
  const seed = await createStore({ dbPath, config: { collections: config.collections } });
  try { expect((await seed.update()).indexed).toBe(1); }
  finally { await seed.close(); }
  writeConfig();
  selected = [];
  const dispose = LlamaCpp.prototype.dispose;
  vi.spyOn(LlamaCpp.prototype, "dispose").mockImplementation(async function (this: LlamaCpp) {
    selected.push({ embed: this.embedModelName, rerank: this.rerankModelName, generate: this.generateModelName });
    await dispose.call(this);
  });
  vi.spyOn(globalThis, "fetch").mockImplementation(() => { throw new Error("Provider fetch forbidden"); });
  setNodeLlamaCppModuleForTest({
    getLlama: () => { throw new Error("Native inference forbidden"); },
    resolveModelFile: () => { throw new Error("Model downloads forbidden"); },
    LlamaChatSession: class { async prompt(): Promise<string> { throw new Error("Inference forbidden"); } },
    LlamaLogLevel: { error: 0 },
  });
});

afterEach(async () => {
  await handle?.stop();
  handle = undefined;
  vi.restoreAllMocks();
  setNodeLlamaCppModuleForTest(null);
  setConfigSource();
  for (const [name, value] of savedEnv) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  rmSync(root, { recursive: true, force: true });
});

describe("MCP startup config trust (also runs in CI)", () => {
  test("untrusted local config drops every model and option; real HTTP lexical query needs no provider", async () => {
    const original = readFileSync(configPath, "utf8");
    handle = await startMcpHttpServer(0, { dbPath, host: "127.0.0.1", quiet: true });
    // Drive loopback via node:http so every provider fetch remains forbidden.
    const response = await new Promise<{ status: number | undefined; body: string }>((resolve, reject) => {
      const req = request({ hostname: "127.0.0.1", port: handle!.port, path: "/query", method: "POST",
        headers: { "Content-Type": "application/json" } }, res => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", chunk => { body += chunk; });
        res.on("end", () => resolve({ status: res.statusCode, body }));
      });
      req.on("error", reject);
      req.end(JSON.stringify({ searches: [{ type: "lex", query: "nebula" }], rerank: false }));
    });
    expect(response.status).toBe(200);
    expect(JSON.parse(response.body).results).toEqual([expect.objectContaining({ title: "Nebula" })]);
    await handle.stop();
    handle = undefined;
    expect(selected).toEqual([builtins]);
    expect(loadConfig().models).toBeUndefined();
    expect(readFileSync(configPath, "utf8")).toBe(original);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  test("matching trust preserves remote identity; changing only apiKeyEnv invalidates it", async () => {
    recordTrust(configPath, sensitiveDigest(snapshot(), configPath, builtins));
    expect(await startAndStop()).toEqual({
      embed: new RemoteEmbedder(config.models!.embed!, config.models!.embedOptions).modelName,
      rerank: new JevReranker(config.models!.rerank!, config.models!.rerankOptions).modelName,
      generate: config.models!.generate,
    });
    config.models!.embedOptions!.apiKeyEnv = "QMD_TEST_MCP_OTHER_CREDENTIAL";
    writeConfig();
    expect(await startAndStop()).toEqual(builtins);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  test("changing only rerank credential options invalidates approval", async () => {
    recordTrust(configPath, sensitiveDigest(snapshot(), configPath, builtins));
    expect((await startAndStop()).rerank).toBe(new JevReranker(config.models!.rerank!, config.models!.rerankOptions).modelName);
    config.models!.rerankOptions!.apiKeyEnv = "QMD_TEST_MCP_OTHER_CREDENTIAL";
    writeConfig();
    expect(await startAndStop()).toEqual(builtins);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  test("changing only a processor invalidates model approval", async () => {
    config.collections.docs!.processor = { command: ["synthetic-not-executed"], version: "fixture-v1" };
    recordTrust(configPath, sensitiveDigest(snapshot(), configPath, builtins));
    writeConfig();
    expect((await startAndStop()).embed).toBe(new RemoteEmbedder(config.models!.embed!, config.models!.embedOptions).modelName);
    config.collections.docs!.processor!.version = "fixture-v2";
    writeConfig();
    expect(await startAndStop()).toEqual(builtins);
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  test.each(["hook", "path"] as const)("approval includes %s, not just models", async field => {
    config.collections.docs!.update = "echo synthetic-hook-not-executed";
    config.collections.outside = { path: join(root, "..", "synthetic-outside"), pattern: "**/*.md" };
    recordTrust(configPath, sensitiveDigest(snapshot(), configPath, builtins));
    writeConfig();
    expect((await startAndStop()).embed).toBe(new RemoteEmbedder(config.models!.embed!, config.models!.embedOptions).modelName);
    if (field === "hook") config.collections.docs!.update = "echo changed-synthetic-hook";
    else config.collections.outside.path = join(root, "..", "changed-synthetic-outside");
    writeConfig();
    expect(await startAndStop()).toEqual(builtins);
  });

  test("options alone are gated even with a deliberate process-env remote model", async () => {
    process.env.QMD_EMBED_MODEL = config.models!.embed;
    config.models = { embedOptions: config.models!.embedOptions };
    writeConfig();
    expect(await startAndStop()).toEqual({
      ...builtins,
      embed: new RemoteEmbedder(process.env.QMD_EMBED_MODEL!).modelName,
    });
    expect(loadConfig().models).toBeUndefined();
  });

  test("global config remains deliberate without project approval", async () => {
    configPath = join(process.env.QMD_CONFIG_DIR!, "index.yml");
    mkdirSync(process.env.QMD_CONFIG_DIR!, { recursive: true });
    writeFileSync(configPath, YAML.stringify(config));
    setConfigSource({ configPath });
    expect(await startAndStop()).toEqual({
      embed: new RemoteEmbedder(config.models!.embed!, config.models!.embedOptions).modelName,
      rerank: new JevReranker(config.models!.rerank!, config.models!.rerankOptions).modelName,
      generate: config.models!.generate,
    });
  });

  test("stdio and HTTP both use the same gated startup helper", () => {
    const source = readFileSync(new URL("../src/mcp/server.ts", import.meta.url), "utf8");
    const stdio = source.slice(source.indexOf("export async function startMcpServer("), source.indexOf("export type HttpServerHandle"));
    const http = source.slice(source.indexOf("export async function startMcpHttpServer("));
    for (const transport of [stdio, http]) {
      expect(transport).toContain("await createMcpStore(options)");
      expect(transport).not.toContain("await createStore(");
    }
  });
});
