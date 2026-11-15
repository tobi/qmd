import { afterEach, describe, expect, test, vi } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, RemoteEmbedder, type CollectionConfig, type QMDStore } from "../src/index.js";
import { LlamaCpp, DEFAULT_EMBED_MODEL_URI as DEFAULT_EMBED_MODEL, formatDocForEmbedding, formatQueryForEmbedding, pullModels, setNodeLlamaCppModuleForTest, withLLMSessionForLlm } from "../src/llm.js";
import { getEmbeddingFingerprint, createStore as openInternal } from "../src/store.js";
import { sensitiveDigest, type SensitiveSnapshot } from "../src/trust.js";
import { VEC_TABLE } from "../src/vec-layout.js";

const uri = "http://localhost:1234/v1#synthetic";
const roots: string[] = [];
const stores: QMDStore[] = [];

afterEach(async () => {
  vi.restoreAllMocks();
  delete process.env.QMD_TEST_SYNTHETIC_AUTH;
  delete process.env.OPENAI_API_KEY;
  setNodeLlamaCppModuleForTest(null);
  for (const store of stores.splice(0)) await store.close();
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function fixture() {
  const root = mkdtempSync(join(tmpdir(), "qmd-remote-"));
  roots.push(root);
  const docs = join(root, "docs");
  mkdirSync(docs);
  writeFileSync(join(docs, "stars.md"), "---\nqmd:\n  metadata:\n    category: sky\n---\n# Nebula\nA luminous cloud among distant stars.\n");
  writeFileSync(join(docs, "garden.md"), "---\nqmd:\n  metadata:\n    category: land\n---\n# Garden\nGrowing carrots and cabbages.\n");
  const config: CollectionConfig = {
    collections: { docs: { path: docs, pattern: "**/*.md" } },
    models: { embed: uri, embedOptions: { dimensions: 2, revision: "v1" } },
  };
  return { root, docs, config, dbPath: join(root, "index.sqlite") };
}

async function open(f: ReturnType<typeof fixture>, config = f.config) {
  const store = await createStore({ dbPath: f.dbPath, config });
  stores.push(store);
  return store;
}

function mockEmbeddings() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
    const body = JSON.parse(String(init?.body));
    return Response.json({ model: body.model, data: body.input.map((text: string, index: number) => ({
      index, embedding: /Garden|carrots|Galaxy|🌌/.test(text) ? [0, 1] : [1, 0],
    })).reverse() });
  });
}

function forbidNative() {
  const getLlama = vi.fn(() => { throw new Error("native model loading forbidden"); });
  const resolveModelFile = vi.fn(() => { throw new Error("model download forbidden"); });
  setNodeLlamaCppModuleForTest({ getLlama, resolveModelFile, LlamaChatSession: class { async prompt(): Promise<string> { throw new Error("forbidden"); } }, LlamaLogLevel: { error: 0 } });
  const tokenize = vi.spyOn(LlamaCpp.prototype, "tokenize").mockImplementation(() => { throw new Error("local tokenizer forbidden"); });
  return { getLlama, resolveModelFile, tokenize };
}

describe("explicit remote embeddings", () => {
  test("public SDK update/embed/vector search uses SQLite and never loads a model or tokenizer", async () => {
    const native = forbidNative();
    const http = mockEmbeddings();
    const f = fixture();
    // Astral characters straddle the fixed character chunk boundary.
    writeFileSync(join(f.docs, "emoji.md"), "# Galaxy\n" + "🌌".repeat(4000));
    const store = await open(f);
    expect((await store.update()).indexed).toBe(3);
    const result = await store.embed();
    expect(result.errors).toBe(0);
    expect(result.chunksEmbedded).toBeGreaterThan(3);
    expect(await store.searchLex("unmentionedcosmos")).toHaveLength(0);
    const found = await store.searchVector("unmentionedcosmos");
    expect(found[0]?.title).toBe("Nebula");
    expect(found[0]?.metadata?.category).toBe("sky");
    const inputs = http.mock.calls.flatMap(([, init]) => JSON.parse(String(init?.body)).input as string[]);
    expect(inputs.every(input => input.isWellFormed())).toBe(true);
    expect(inputs).toContain("unmentionedcosmos");
    expect(inputs.some(input => input.startsWith("Nebula\n"))).toBe(true);
    expect(inputs.some(input => input.includes("task: search"))).toBe(false);
    expect(http.mock.calls.every(([, init]) => JSON.parse(String(init?.body)).dimensions === 2)).toBe(true);
    expect(native.getLlama).not.toHaveBeenCalled();
    expect(native.resolveModelFile).not.toHaveBeenCalled();
    expect(native.tokenize).not.toHaveBeenCalled();
  });

  test("space identity pins endpoint, dimensions, revision and copies per-store options", async () => {
    const http = mockEmbeddings();
    const f = fixture();
    const first = await open(f);
    const name = first.internal.llm!.embedModelName;
    expect(name).toMatch(/^remote-openai:[0-9a-f]{64}$/);
    f.config.models!.embedOptions!.revision = "changed";
    f.config.models!.embedOptions!.dimensions = 3;
    f.config.models!.embed = "http://other.invalid/v1#synthetic";
    expect(first.internal.llm!.embedModelName).toBe(name);
    await first.update();
    await first.embed();
    expect(http.mock.calls.every(([url, init]) => String(url) === "http://localhost:1234/v1/embeddings" && JSON.parse(String(init?.body)).dimensions === 2)).toBe(true);
    expect(new RemoteEmbedder(uri, { dimensions: 2, revision: "v1", apiKeyEnv: "SYNTHETIC_KEY" }).modelName).toBe(name);
    for (const [endpoint, options] of [
      ["http://other.invalid/v1#synthetic", { dimensions: 2, revision: "v1" }],
      [uri, { dimensions: 3, revision: "v1" }],
      [uri, { dimensions: 2, revision: "v2" }],
    ] as const) expect(new RemoteEmbedder(endpoint, options).modelName).not.toBe(name);
    expect(new LlamaCpp({ embedModel: DEFAULT_EMBED_MODEL }).embedModelName).toBe(DEFAULT_EMBED_MODEL);
    expect(formatQueryForEmbedding("x", DEFAULT_EMBED_MODEL)).toBe("task: search result | query: x");
    expect(formatDocForEmbedding("x", "t", DEFAULT_EMBED_MODEL)).toBe("title: t | text: x");
    expect(getEmbeddingFingerprint(DEFAULT_EMBED_MODEL)).toBe("c37385");
  });

  test("explicit remote embeddings do not relax the local CI guard", async () => {
    const previous = process.env.CI;
    process.env.CI = "true";
    const native = forbidNative();
    const llm = new LlamaCpp({ embedModel: DEFAULT_EMBED_MODEL });
    try {
      await expect(llm.embedBatch(["never load a local model"])).rejects.toThrow("LLM operations are disabled in CI");
      expect(native.getLlama).not.toHaveBeenCalled();
    } finally {
      await llm.dispose();
      if (previous === undefined) delete process.env.CI;
      else process.env.CI = previous;
    }
  });

  test("local legacy fingerprints and local model labels keep working, but a remote row blocks a local query before ML", async () => {
    const native = forbidNative();
    const http = vi.spyOn(globalThis, "fetch").mockImplementation(() => { throw new Error("HTTP forbidden"); });
    const f = fixture();
    const store = await open(f, { collections: f.config.collections, models: { embed: DEFAULT_EMBED_MODEL } });
    await store.update();
    const row = store.internal.db.prepare("SELECT hash FROM documents WHERE path = 'stars.md' AND active = 1").get() as { hash: string };
    store.internal.ensureVecTable(2);
    store.internal.insertEmbedding(row.hash, 0, 0, new Float32Array([1, 0]), DEFAULT_EMBED_MODEL, "2026-01-01T00:00:00Z", 1, "");
    const legacy = await store.internal.searchVec("unused", DEFAULT_EMBED_MODEL, 5, "docs", undefined, [1, 0]);
    expect(legacy).toHaveLength(1);
    const otherLocal = await store.internal.searchVec("unused", "other-local-model", 5, "docs", undefined, [1, 0]);
    expect(otherLocal).toHaveLength(1);
    store.internal.db.prepare("UPDATE content_vectors SET model = ?").run(new RemoteEmbedder(uri).modelName);
    await expect(store.searchVector("must not load a local query model")).rejects.toThrow("Vector space mismatch");
    expect(native.getLlama).not.toHaveBeenCalled();
    expect(native.resolveModelFile).not.toHaveBeenCalled();
    expect(native.tokenize).not.toHaveBeenCalled();
    expect(http).not.toHaveBeenCalled();
  });

  test("wrong endpoint/model/revision/fingerprint refuses BEFORE HTTP; re-embed repairs the same database", async () => {
    const http = mockEmbeddings();
    const f = fixture();
    const original = await open(f);
    await original.update();
    await original.embed();
    for (const [embed, revision] of [["http://other.invalid/v1#synthetic", "v1"], ["http://localhost:1234/v1#other", "v1"], [uri, "v2"]]) {
      const other = await open(f, { ...f.config, models: { embed, embedOptions: { dimensions: 2, revision } } });
      http.mockClear();
      await expect(other.searchVector("unmentionedcosmos")).rejects.toThrow("Vector space mismatch");
      expect(http).not.toHaveBeenCalled();
    }
    const changed = stores[stores.length - 1]!;
    expect((await changed.embed()).errors).toBe(0);
    expect((await changed.searchVector("unmentionedcosmos"))[0]?.title).toBe("Nebula");
    changed.internal.db.exec("UPDATE content_vectors SET embed_fingerprint = 'wrong'");
    http.mockClear();
    await expect(changed.searchVector("unmentionedcosmos")).rejects.toThrow("Vector space mismatch");
    expect(http).not.toHaveBeenCalled();
  });

  test("metadata scope excludes foreign rows; recheck catches writes during HTTP", async () => {
    const http = mockEmbeddings();
    const f = fixture();
    const store = await open(f);
    await store.update();
    await store.embed();
    store.internal.db.exec("UPDATE content_vectors SET model = 'foreign' WHERE hash IN (SELECT hash FROM documents WHERE path = 'garden.md')");
    const filter = { field: "category", operator: "eq", value: "sky" } as const;
    expect((await store.searchVector("unmentionedcosmos", { filter }))[0]?.title).toBe("Nebula");
    http.mockImplementationOnce(async (_url, init) => {
      expect(store.internal.db.inTransaction).toBe(false);
      store.internal.db.exec("UPDATE content_vectors SET model = 'foreign'");
      return Response.json({ model: "synthetic", data: [{ index: 0, embedding: [1, 0] }] });
    });
    await expect(store.searchVector("unmentionedcosmos", { filter })).rejects.toThrow("Vector space mismatch");
  });

  test("post-HTTP guard, KNN, resolver and bodies share a SQLite read snapshot", async () => {
    mockEmbeddings();
    const f = fixture();
    const store = await open(f);
    await store.update();
    await store.embed();
    const writer = openInternal(f.dbPath);
    const db = store.internal.db;
    const prepare = db.prepare.bind(db);
    let raced = false;
    const spy = vi.spyOn(db, "prepare").mockImplementation(sql => {
      const statement = prepare(sql);
      if (sql.includes(`FROM ${VEC_TABLE}`) && sql.includes("embedding MATCH")) {
        const all = statement.all.bind(statement);
        statement.all = (...params) => {
          expect(db.inTransaction).toBe(true);
          if (!raced) {
            raced = true;
            writer.db.exec("UPDATE content_vectors SET model = 'foreign'; UPDATE document_metadata SET metadata_json = '{}' ");
          }
          return all(...params);
        };
      }
      return statement;
    });
    try {
      const results = await store.searchVector("unmentionedcosmos");
      expect(raced).toBe(true);
      expect(results[0]?.metadata?.category).toBe("sky");
      expect(results[0]?.body).toContain("luminous");
    } finally { spy.mockRestore(); writer.close(); }
    await expect(store.searchVector("unmentionedcosmos")).rejects.toThrow("Vector space mismatch");
  });

  test("validates the entire batch, including index order, before returning any vectors", async () => {
    const client = new RemoteEmbedder(uri, { dimensions: 2 });
    const http = vi.spyOn(globalThis, "fetch");
    const good = [{ index: 1, embedding: [0, 2] }, { index: 0, embedding: [3, 4] }];
    http.mockResolvedValueOnce(Response.json({ model: "synthetic", data: good }));
    expect(await client.embedBatch(["one", "two"])).toEqual([[3, 4], [0, 2]]);
    for (const body of [
      { model: "other", data: good },
      { model: "synthetic", data: good.slice(0, 1) },
      { model: "synthetic", data: [good[0], good[0]] },
      { model: "synthetic", data: [good[0], { index: 0, embedding: [1] }] },
      { model: "synthetic", data: [good[0], { index: 0, embedding: [0, 0] }] },
      { model: "synthetic", data: [good[0], { index: 0, embedding: [1e100, 1] }] },
    ]) {
      http.mockResolvedValueOnce(Response.json(body));
      await expect(client.embedBatch(["one", "two"])).rejects.toThrow("Invalid remote embedding response");
    }
    // A replacement-character model ID makes lenient UTF-8 decoding unsafe:
    // invalid 0xff would decode to the expected model and admit the vectors.
    const replacementModel = new RemoteEmbedder("http://localhost:1234/v1#%EF%BF%BD");
    const malformed = Buffer.concat([
      Buffer.from('{"model":"'), Uint8Array.of(0xff),
      Buffer.from('","data":[{"index":0,"embedding":[1,0]}]}'),
    ]);
    http.mockResolvedValueOnce(new Response(malformed));
    await expect(replacementModel.embedBatch(["one"])).rejects.toThrow("Remote model request failed");
    expect(http).toHaveBeenCalledTimes(8);
  });

  test("malformed indexing batch writes no vectors and never retries individual inputs", async () => {
    const http = mockEmbeddings();
    const f = fixture();
    const store = await open(f);
    await store.update();
    http.mockResolvedValueOnce(Response.json({ model: "synthetic", data: [{ index: 0, embedding: [1, 0] }] }));
    http.mockResolvedValueOnce(Response.json({ model: "synthetic", data: [{ index: 0, embedding: [1, 0] }] }));
    const result = await store.embed();
    expect(result.chunksEmbedded).toBe(0);
    expect(result.errors).toBe(2);
    expect(http).toHaveBeenCalledTimes(2); // dimension probe + one batch
    expect(store.internal.db.prepare("SELECT COUNT(*) AS n FROM content_vectors").get()).toEqual({ n: 0 });
  });

  test("public embed forwards its caller signal before accepting a late response", async () => {
    forbidNative();
    const f = fixture();
    const store = await open(f);
    await store.update();
    const controller = new AbortController();
    const http = vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
      controller.abort();
      expect(init!.signal!.aborted).toBe(true);
      return Response.json({ model: "synthetic", data: [{ index: 0, embedding: [1, 0] }] });
    });
    await expect(store.embed({ signal: controller.signal })).rejects.toThrow("Remote model request aborted");
    expect(http).toHaveBeenCalledTimes(1);
    expect(store.internal.db.prepare("SELECT COUNT(*) AS n FROM content_vectors").get()).toEqual({ n: 0 });
  });

  test("session and caller cancellation reach held HTTP; late responses never return data", async () => {
    forbidNative();
    const llm = new LlamaCpp({ embedModel: uri });
    for (const abortSession of [false, true]) {
      const sessionController = new AbortController();
      const caller = new AbortController();
      let observed: AbortSignal | undefined;
      const http = vi.spyOn(globalThis, "fetch").mockImplementation(async (_url, init) => {
        observed = init!.signal as AbortSignal;
        (abortSession ? sessionController : caller).abort();
        expect(observed.aborted).toBe(true);
        // Deliberately ignore abort and deliver a late provider result.
        return Response.json({ model: "synthetic", data: [{ index: 0, embedding: [1, 0] }] });
      });
      await expect(withLLMSessionForLlm(llm, session => session.embed("x", { signal: caller.signal }), { signal: sessionController.signal })).rejects.toThrow("Remote model request aborted");
      expect(observed?.aborted).toBe(true);
      http.mockRestore();
    }
    await llm.dispose();
  });

  test("explicit auth only, dispatch-time env lookup, fixed errors and remote pull skip", async () => {
    const native = forbidNative();
    const http = mockEmbeddings();
    process.env.OPENAI_API_KEY = "ambient-never-send";
    expect(() => new RemoteEmbedder("http://user:synthetic-secret@localhost/v1#model")).toThrow("Invalid remote embedding URI");
    expect(() => new RemoteEmbedder("http://@localhost/v1#model")).toThrow("Invalid remote embedding URI");
    expect(() => new RemoteEmbedder("http:localhost/v1#model")).toThrow("Invalid remote embedding URI");
    const client = new RemoteEmbedder(uri, { apiKeyEnv: "QMD_TEST_SYNTHETIC_AUTH" });
    process.env.QMD_TEST_SYNTHETIC_AUTH = "";
    await expect(client.embedBatch(["x"])).rejects.toThrow("Remote model credential unavailable");
    expect(http).not.toHaveBeenCalled();
    process.env.QMD_TEST_SYNTHETIC_AUTH = "synthetic-only";
    await client.embedBatch(["x"]);
    expect(http.mock.calls[0]?.[1]?.headers).toEqual({ "Content-Type": "application/json", Authorization: "Bearer synthetic-only" });
    await new RemoteEmbedder(uri).embedBatch(["x"]);
    expect(http.mock.calls[1]?.[1]?.headers).toEqual({ "Content-Type": "application/json" });
    const cache = mkdtempSync(join(tmpdir(), "qmd-remote-cache-")); roots.push(cache);
    expect(await pullModels([uri], { cacheDir: cache })).toEqual([]);
    expect(native.resolveModelFile).not.toHaveBeenCalled();
  });

  test("remote options participate in local trust; absent options preserve the prior digest", () => {
    const builtins = { embed: "builtin-embed", generate: "builtin-gen", rerank: "builtin-rerank" };
    const snapshot: SensitiveSnapshot = { hooks: [], paths: [], models: {} };
    const digest = sensitiveDigest(snapshot, "/project/.qmd/index.yml", builtins);
    // sha256 of the existing canonical {hooks:[],paths:[],models:[]} representation.
    expect(digest).toBe("c19f165773ffb5de8eb51275cb1b5408d51c78c9b7f0137cd7c31bf28c0ac37f");
    const configured = { ...snapshot, models: { embed: uri, embedOptions: { apiKeyEnv: "KEY_ONE", dimensions: 2, revision: "v1" } } };
    const approved = sensitiveDigest(configured, "/project/.qmd/index.yml", builtins);
    for (const embedOptions of [{ apiKeyEnv: "KEY_TWO", dimensions: 2, revision: "v1" }, { apiKeyEnv: "KEY_ONE", dimensions: 3, revision: "v1" }, { apiKeyEnv: "KEY_ONE", dimensions: 2, revision: "v2" }]) {
      expect(sensitiveDigest({ ...configured, models: { embed: uri, embedOptions } }, "/project/.qmd/index.yml", builtins)).not.toBe(approved);
    }
  });
});
