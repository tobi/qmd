import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { createServer, type ServerResponse } from "node:http";
import { once } from "node:events";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, JevReranker, type CollectionConfig, type QMDStore } from "../src/index.js";
import { JEV_ENDPOINT } from "../src/jev.js";
import { setConfigSource } from "../src/collections.js";
import { LlamaCpp, DEFAULT_RERANK_MODEL_URI, pullModels, setNodeLlamaCppModuleForTest, withLLMSessionForLlm } from "../src/llm.js";
import { sensitiveDigest, type SensitiveSnapshot } from "../src/trust.js";

const nativeFetch = globalThis.fetch;
const stores: QMDStore[] = [];
const cleanup: (() => Promise<void>)[] = [];
beforeEach(() => { process.env.TYPESAFE_API_KEY = "synthetic-jev-only"; });
afterEach(async () => {
  for (const store of stores.splice(0)) await store.close();
  for (const close of cleanup.splice(0)) await close();
  vi.restoreAllMocks();
  setNodeLlamaCppModuleForTest(null);
  setConfigSource();
  delete process.env.TYPESAFE_API_KEY;
  delete process.env.QMD_TEST_JEV_MISSING;
  delete process.env.QMD_TEST_JEV_CUSTOM;
});

function forbidNative() {
  const getLlama = vi.fn(() => { throw new Error("native loading forbidden"); });
  const resolveModelFile = vi.fn(() => { throw new Error("model download forbidden"); });
  setNodeLlamaCppModuleForTest({ getLlama, resolveModelFile, LlamaChatSession: class { async prompt(): Promise<string> { throw new Error("forbidden"); } }, LlamaLogLevel: { error: 0 } });
  const tokenize = vi.spyOn(LlamaCpp.prototype, "tokenize").mockImplementation(() => { throw new Error("tokenizer forbidden"); });
  return { getLlama, resolveModelFile, tokenize };
}

type Body = { model: string; state: { query: string; document: string; document_title?: string }; questions: unknown };
type Call = { body: Body; authorization?: string; path: string };
function reply(response: ServerResponse, score: number) {
  response.setHeader("Content-Type", "application/json");
  response.end(JSON.stringify({ answers: { relevant: { noul: score } } }));
}

async function fixture() {
  const root = mkdtempSync(join(tmpdir(), "qmd-jev-"));
  const docs = join(root, "docs"); mkdirSync(docs);
  writeFileSync(join(docs, "stars.md"), "# Nebula\nA cloud among distant stars.\n");
  writeFileSync(join(docs, "garden.md"), "# Garden\nGrowing carrots and cabbages.\n");
  const calls: Call[] = [];
  const signals: AbortSignal[] = [];
  const live = new Set<ServerResponse>();
  let active = 0, maxActive = 0;
  let respond = (body: Body, response: ServerResponse) => reply(response, /Garden|low/.test(body.state.document) ? 0.2 : body.state.document === "zero" ? 0 : 0.8);
  const server = createServer(async (request, response) => {
    live.add(response);
    active++; maxActive = Math.max(maxActive, active);
    response.on("close", () => { live.delete(response); active--; });
    try {
      const chunks: Buffer[] = [];
      for await (const chunk of request) chunks.push(Buffer.from(chunk));
      const body = JSON.parse(Buffer.concat(chunks).toString());
      if (request.url === "/embeddings") {
        response.setHeader("Content-Type", "application/json");
        response.end(JSON.stringify({ model: body.model, data: body.input.map((_text: string, index: number) => ({ index, embedding: [1, 0] })) }));
      } else if (request.url === "/systemone") {
        calls.push({ body, authorization: request.headers.authorization, path: request.url });
        respond(body, response);
      } else response.writeHead(404).end();
    } catch { if (!response.destroyed) response.writeHead(400).end(); }
  });
  server.listen(0, "127.0.0.1"); await once(server, "listening");
  const address = server.address();
  if (!address || typeof address === "string") throw new Error("No loopback port");
  const origin = `http://127.0.0.1:${address.port}`;
  const transport = { ignoreAbort: false };
  vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
    const url = String(input);
    if (url === JEV_ENDPOINT) {
      signals.push(init!.signal as AbortSignal);
      // Unit adaptation only: the production descriptor remains the fixed URL,
      // but genuine native fetch/Response/streams travel exclusively on loopback.
      return nativeFetch(`${origin}/systemone`, transport.ignoreAbort ? { ...init, signal: undefined } : init);
    }
    if (url === `${origin}/embeddings`) return nativeFetch(url, init);
    throw new Error("Only synthetic loopback requests are permitted");
  });
  cleanup.push(async () => {
    for (const response of live) response.destroy();
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    rmSync(root, { recursive: true, force: true });
  });
  const config: CollectionConfig = {
    collections: { docs: { path: docs, pattern: "*.md" } },
    models: { embed: `${origin}#synthetic`, embedOptions: { dimensions: 2 }, rerank: "typesafe:jev-test", rerankOptions: { revision: "v1" } },
  };
  // The embedding API base is origin, which appends /embeddings.
  async function open(selected = config) {
    const store = await createStore({ dbPath: join(root, "index.sqlite"), config: selected });
    stores.push(store); return store;
  }
  return { root, config, open, calls, signals, transport, maxActive: () => maxActive,
    respond: (handler: typeof respond) => { respond = handler; } };
}

const query = { queries: [{ type: "vec" as const, query: "unmentionedcosmos" }], explain: true };
const cacheCount = (store: QMDStore) => (store.internal.db.prepare("SELECT COUNT(*) AS n FROM llm_cache").get() as { n: number }).n;

test("public SDK typed-vector search uses independent System One scores and revision/model-scoped SQLite caches", async () => {
  const native = forbidNative();
  const f = await fixture(); const store = await f.open();
  const original = store.internal.llm!.rerankModelName;
  expect(original).toMatch(/^remote-typesafe:[0-9a-f]{64}$/);
  f.config.models!.rerankOptions!.revision = "caller-mutated";
  f.config.models!.rerankOptions!.apiKeyEnv = "QMD_TEST_JEV_MISSING";
  expect(store.internal.llm!.rerankModelName).toBe(original);
  await store.update(); await store.embed();
  expect(await store.searchLex("unmentionedcosmos")).toHaveLength(0);
  const results = await store.search(query);
  expect(results).toHaveLength(2);
  expect(results.find(r => r.title === "Nebula")?.explain?.rerankScore).toBe(0.8);
  expect(results.find(r => r.title === "Garden")?.explain?.rerankScore).toBe(0.2);
  expect(f.calls).toHaveLength(2);
  expect(cacheCount(store)).toBe(2);
  expect((await store.search(query)).map(r => r.score)).toEqual(results.map(r => r.score));
  expect(f.calls).toHaveLength(2); // Query embedding still runs; rerank HTTP does not.
  expect(f.calls.every(c => c.authorization === "Bearer synthetic-jev-only")).toBe(true);

  f.respond((_body, response) => reply(response, 0.4));
  const revised = await f.open({ ...f.config, models: { ...f.config.models, rerankOptions: { revision: "v2" } } });
  expect(revised.internal.llm!.rerankModelName).not.toBe(original);
  expect((await revised.search(query)).every(r => r.explain?.rerankScore === 0.4)).toBe(true);
  expect(f.calls).toHaveLength(4);
  const another = await f.open({ ...f.config, models: { ...f.config.models, rerank: "typesafe:jev-other", rerankOptions: { revision: "v2" } } });
  expect(another.internal.llm!.rerankModelName).not.toBe(revised.internal.llm!.rerankModelName);
  await another.search(query);
  expect(f.calls).toHaveLength(6);
  expect(f.calls.slice(4).every(c => c.body.model === "jev-other")).toBe(true);
  expect(native.getLlama).not.toHaveBeenCalled();
  expect(native.resolveModelFile).not.toHaveBeenCalled();
  expect(native.tokenize).not.toHaveBeenCalled();
});

test("fixed vendor wire format scores each document separately, accepts zero, and sorts stable original indexes", async () => {
  const f = await fixture();
  process.env.QMD_TEST_JEV_CUSTOM = "synthetic-custom";
  const client = new JevReranker("typesafe:jev-test", { apiKeyEnv: "QMD_TEST_JEV_CUSTOM" });
  f.respond((body, response) => response.end(JSON.stringify({ model: body.model, answers: { relevant: { noul: body.state.document === "zero" ? 0 : 0.8 } } })));
  const result = await client.rerank("query", [{ file: "logical/private.md", text: "zero", title: "Zero" }, { file: "high.md", text: "high" }, { file: "tie.md", text: "high too" }]);
  expect(result.model).toBe(client.modelName);
  expect(result.results).toEqual([{ file: "high.md", score: 0.8, index: 1 }, { file: "tie.md", score: 0.8, index: 2 }, { file: "logical/private.md", score: 0, index: 0 }]);
  expect(f.calls.find(c => c.body.state.document === "zero")?.body).toEqual({
    model: "jev-test", state: { query: "query", document_title: "Zero", document: "zero" },
    questions: { relevant: { type: "noul", instructions: "Does the document contain information that answers or directly addresses the search query?" } },
  });
  expect(f.calls.every(c => !JSON.stringify(c.body).includes("logical/private.md"))).toBe(true);
  expect(f.calls.every(c => c.authorization === "Bearer synthetic-custom")).toBe(true);
  process.env.QMD_TEST_JEV_CUSTOM = "synthetic-custom-rotated";
  await client.rerank("query", [{ file: "a.md", text: "high" }]);
  expect(f.calls[3]?.authorization).toBe("Bearer synthetic-custom-rotated");
  expect(f.maxActive()).toBeLessThanOrEqual(2);
});

test("two chunks with one file name cache their own indexed scores; pre-aborted cache hits make no HTTP calls", async () => {
  const f = await fixture(); const store = await f.open();
  const docs = [{ file: "same.md", text: "high" }, { file: "same.md", text: "low" }];
  expect((await store.internal.rerank("chunks", docs)).map(r => r.score)).toEqual([0.8, 0.2]);
  expect((await store.internal.rerank("chunks", [docs[0]!]))[0]?.score).toBe(0.8);
  expect((await store.internal.rerank("chunks", [docs[1]!]))[0]?.score).toBe(0.2);
  expect(f.calls).toHaveLength(2);
  const controller = new AbortController(); controller.abort();
  await expect(store.internal.rerank("chunks", docs, undefined, undefined, controller.signal)).rejects.toThrow();
  expect(f.calls).toHaveLength(2);
});

test("two-worker queue stops on session cancellation, drains late native HTTP, and returns no scores", async () => {
  const native = forbidNative(); const f = await fixture();
  f.transport.ignoreAbort = true; // Simulate a provider transport delivering late responses.
  const pending: ServerResponse[] = [];
  let ready!: () => void; const held = new Promise<void>(resolve => { ready = resolve; });
  f.respond((_body, response) => { pending.push(response); if (pending.length === 2) ready(); });
  const llm = new LlamaCpp({ rerankModel: "typesafe:jev-test" });
  const caller = new AbortController(), sessionController = new AbortController();
  const docs = Array.from({ length: 5 }, (_, index) => ({ file: `${index}.md`, text: `doc ${index}` }));
  try {
    const work = withLLMSessionForLlm(llm, session => session.rerank("held", docs, { signal: caller.signal }), { signal: sessionController.signal });
    const settled = work.then(() => null, error => error);
    await held;
    sessionController.abort();
    expect(f.signals.every(signal => signal.aborted)).toBe(true);
    pending.forEach(response => reply(response, 0.8));
    const error = await settled;
    expect(error).toBeInstanceOf(Error);
    expect(error.message).toBe("TypeSafe rerank aborted");
    expect(f.calls).toHaveLength(2);
    expect(f.maxActive()).toBe(2);
    expect(native.getLlama).not.toHaveBeenCalled();
  } finally { await llm.dispose(); }
});

test("public search caller cancellation reaches held native HTTP and writes no partial scores", async () => {
  const f = await fixture(); const store = await f.open();
  await store.update(); await store.embed();
  const closed: Promise<unknown>[] = [];
  let ready!: () => void; const held = new Promise<void>(resolve => { ready = resolve; });
  f.respond((_body, response) => { closed.push(once(response, "close")); if (closed.length === 2) ready(); });
  const controller = new AbortController();
  const settled = store.search({ ...query, signal: controller.signal }).then(() => null, error => error);
  await held; controller.abort();
  expect(f.signals.every(signal => signal.aborted)).toBe(true);
  const error = await settled;
  expect(error).toBeInstanceOf(Error);
  expect(error.message).toBe("TypeSafe rerank aborted");
  await Promise.all(closed);
  expect(f.calls).toHaveLength(2);
  expect(cacheCount(store)).toBe(0);
});

test("declared credentials fail before HTTP; provider errors and bad scores are fixed errors with no new cache rows", async () => {
  const native = forbidNative(); const f = await fixture();
  const missing = await f.open({ ...f.config, models: { ...f.config.models, rerankOptions: { apiKeyEnv: "QMD_TEST_JEV_MISSING" } } });
  delete process.env.QMD_TEST_JEV_MISSING;
  await expect(missing.internal.rerank("q", [{ file: "a", text: "high" }])).rejects.toThrow("Remote model credential unavailable");
  expect(f.calls).toHaveLength(0);
  const store = await f.open();
  f.respond((_body, response) => response.writeHead(401).end("private provider body synthetic-key"));
  await expect(store.internal.rerank("q", [{ file: "a", text: "high" }])).rejects.toThrow("Remote model request failed");
  expect(cacheCount(store)).toBe(0);
  f.respond((body, response) => reply(response, body.state.document === "bad" ? 1.1 : 0.8));
  await expect(store.internal.rerank("q", [{ file: "a", text: "high" }, { file: "b", text: "bad" }])).rejects.toThrow("Invalid TypeSafe rerank response");
  expect(cacheCount(store)).toBe(0);
  f.respond((_body, response) => response.end(JSON.stringify({ model: "other-model", answers: { relevant: { noul: 0.8 } } })));
  await expect(store.internal.rerank("q", [{ file: "a", text: "high" }])).rejects.toThrow("Invalid TypeSafe rerank response");
  expect(cacheCount(store)).toBe(0);
  expect(native.getLlama).not.toHaveBeenCalled();
  expect(native.resolveModelFile).not.toHaveBeenCalled();
});

test("rerank options re-arm trust without changing absent/default digests; invalid options fail before opening SQLite", async () => {
  const f = await fixture();
  const builtins = { embed: "builtin-embed", generate: "builtin-gen", rerank: "builtin-rerank" };
  const snapshot: SensitiveSnapshot = { hooks: [], paths: [], models: {} };
  expect(sensitiveDigest(snapshot, "/project/.qmd/index.yml", builtins)).toBe("c19f165773ffb5de8eb51275cb1b5408d51c78c9b7f0137cd7c31bf28c0ac37f");
  const base: SensitiveSnapshot = { ...snapshot, models: { rerank: "typesafe:jev-test", rerankOptions: { revision: "v1", apiKeyEnv: "KEY_ONE" } } };
  const approved = sensitiveDigest(base, "/project/.qmd/index.yml", builtins);
  for (const rerankOptions of [{ revision: "v2", apiKeyEnv: "KEY_ONE" }, { revision: "v1", apiKeyEnv: "KEY_TWO" }]) {
    expect(sensitiveDigest({ ...base, models: { ...base.models, rerankOptions } }, "/project/.qmd/index.yml", builtins)).not.toBe(approved);
  }
  expect(new JevReranker("typesafe:jev-test", { revision: "v1", apiKeyEnv: "KEY_ONE" }).modelName).toBe(new JevReranker("typesafe:jev-test", { revision: "v1", apiKeyEnv: "KEY_TWO" }).modelName);
  const dbPath = join(f.root, "invalid.sqlite");
  await expect(createStore({ dbPath, config: { ...f.config, models: { rerank: "typesafe:jev-test", rerankOptions: { apiKey: "not-allowed" } as never } } })).rejects.toThrow("Invalid TypeSafe rerank options");
  expect(existsSync(dbPath)).toBe(false);
});

test("explicit remote selection skips pull/model discovery without weakening the local CI guard", async () => {
  const native = forbidNative(); const f = await fixture();
  const cacheDir = join(f.root, "models");
  expect(await pullModels(["typesafe:jev-test"], { cacheDir })).toEqual([]);
  const previous = process.env.CI; process.env.CI = "true";
  const local = new LlamaCpp({ rerankModel: DEFAULT_RERANK_MODEL_URI });
  try {
    expect(local.rerankModelName).toBe(DEFAULT_RERANK_MODEL_URI);
    await expect(local.rerank("q", [{ file: "a", text: "body" }])).rejects.toThrow("LLM operations are disabled in CI");
    expect(await local.modelExists("typesafe:jev-test")).toEqual({ name: "typesafe:jev-test", exists: false });
    expect(f.calls).toHaveLength(0);
    expect(native.getLlama).not.toHaveBeenCalled(); expect(native.resolveModelFile).not.toHaveBeenCalled();
  } finally {
    await local.dispose();
    if (previous === undefined) delete process.env.CI; else process.env.CI = previous;
  }
});
