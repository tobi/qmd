/**
 * Remote rerank (rerank_api_* / rerank_score).
 *
 * Covers: score normalization knob, config resolution (env always trusted,
 * project-local YAML only when trusted), the trust gate over remote endpoints
 * (the #705 exfiltration fix), the Cohere-shaped /v1/rerank client against an
 * in-process fake server, graceful fallback/degradation, and rerank cache
 * identity (scores are cached under the backend that actually produced them).
 *
 * Run with CI=true (like the repo's test:unit) so the local GGUF path stays
 * disabled; every test stubs or expects the degrade path explicitly.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { mkdtempSync, rmSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  normalizeRerankScores,
  remoteRerank,
  resolveRemoteRerankConfig,
  type RemoteRerankConfig,
} from "../src/remote-rerank.js";
import { builtinModels, LlamaCpp, type RerankResult } from "../src/llm.js";
import {
  gatedItems,
  hasGatedItems,
  recordTrust,
  revokeTrust,
  sensitiveDigest,
  snapshotFromConfig,
  snapshotIsTrusted,
} from "../src/trust.js";
import { createStore, rerank as rerankWithLlm, type Store } from "../src/store.js";
import type { ModelsConfig } from "../src/collections.js";

// =============================================================================
// Fake /v1/rerank server (in-process, same pattern as the trial harness)
// =============================================================================

type RecordedRequest = { body: unknown; headers: Record<string, string | undefined> };

type FakeRerankServer = {
  server: Server;
  url: string;
  requests: RecordedRequest[];
  close: () => Promise<void>;
};

function startRerankServer(
  respond: (req: IncomingMessage, body: unknown, res: ServerResponse) => void,
): Promise<FakeRerankServer> {
  const requests: RecordedRequest[] = [];
  const server = createServer((req, res) => {
    let raw = "";
    req.on("data", (chunk: Buffer) => { raw += chunk.toString(); });
    req.on("end", () => {
      let parsed: unknown = {};
      try { parsed = JSON.parse(raw); } catch { /* leave {} */ }
      requests.push({ body: parsed, headers: req.headers });
      respond(req, parsed, res);
    });
  });
  const started = Promise.withResolvers<FakeRerankServer>();
  server.on("error", (err: Error) => started.reject(err));
  server.listen(0, "127.0.0.1", () => {
    const addr = server.address();
    const port = typeof addr === "object" && addr ? addr.port : 0;
    started.resolve({
      server,
      url: `http://127.0.0.1:${port}/v1`,
      requests,
      close: () => new Promise<void>((done) => server.close(() => done())),
    });
  });
  return started.promise;
}

function rerankResults(results: Array<{ index: number; relevance_score: number }>) {
  return (_req: IncomingMessage, _body: unknown, res: ServerResponse) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ results }));
  };
}

const DOCS = [
  { file: "a.md", text: "alpha" },
  { file: "b.md", text: "beta" },
  { file: "c.md", text: "gamma" },
];

function remoteConfig(url: string, overrides: Partial<ModelsConfig> = {}): RemoteRerankConfig {
  const cfg = resolveRemoteRerankConfig(
    { rerank_api_url: url, rerank_api_model: "bge-test", ...overrides },
    { trusted: true },
  );
  if (!cfg) throw new Error("expected remote rerank config to resolve");
  return cfg;
}

/** LlamaCpp test double: remote config + injectable local rerank result. */
class StubbedLocalLlamaCpp extends LlamaCpp {
  localResult: { results: Array<{ file: string; score: number }>; model: string } | null = null;

  protected override localRerank(): Promise<RerankResult> {
    if (!this.localResult) throw new Error("no local stub configured");
    return Promise.resolve({
      results: this.localResult.results.map((r, i) => ({ ...r, index: i })),
      model: this.localResult.model,
    });
  }
}

function remoteLlm(cfg: RemoteRerankConfig | null, local?: { results: Array<{ file: string; score: number }>; model: string }): LlamaCpp {
  const llm = new StubbedLocalLlamaCpp({ remoteRerank: cfg });
  if (local) llm.localResult = local;
  return llm;
}

let warnings: string[];
let consoleWarn: typeof console.warn;

function captureWarnings(): void {
  warnings = [];
  consoleWarn = console.warn;
  console.warn = (...args: unknown[]) => { warnings.push(args.join(" ")); };
}

function restoreWarnings(): void {
  console.warn = consoleWarn;
}

// =============================================================================
// Score normalization knob
// =============================================================================

describe("normalizeRerankScores", () => {
  test("logits mode applies sigmoid to every score", () => {
    const out = normalizeRerankScores([0, -2, 2], "logits");
    expect(out[0]).toBeCloseTo(0.5, 6);
    expect(out[1]).toBeLessThan(0.5);
    expect(out[2]).toBeGreaterThan(0.5);
    expect(out.every((s) => s >= 0 && s <= 1)).toBe(true);
  });

  test("probability mode passes scores through untouched", () => {
    expect(normalizeRerankScores([0.9, 0.2], "probability")).toEqual([0.9, 0.2]);
    expect(normalizeRerankScores([1.7], "probability")).toEqual([1.7]);
  });

  test("auto keeps an in-range batch unchanged", () => {
    expect(normalizeRerankScores([0.9, 0.2, 0], "auto")).toEqual([0.9, 0.2, 0]);
  });

  test("auto applies sigmoid to the whole batch when any score is outside [0,1]", () => {
    const out = normalizeRerankScores([0.5, -2], "auto");
    expect(out[0]).toBeCloseTo(1 / (1 + Math.exp(-0.5)), 6);
    expect(out.every((s) => s >= 0 && s <= 1)).toBe(true);
  });
});

// =============================================================================
// Config resolution — env is trusted, YAML only when trusted
// =============================================================================

describe("resolveRemoteRerankConfig", () => {
  const ENV_KEYS = ["QMD_RERANK_API_URL", "QMD_RERANK_API_MODEL", "QMD_RERANK_API_KEY", "QMD_RERANK_SCORE", "QMD_RERANK_API_TIMEOUT"];
  let savedEnv: Record<string, string | undefined>;

  beforeEach(() => {
    savedEnv = {};
    for (const key of ENV_KEYS) { savedEnv[key] = process.env[key]; delete process.env[key]; }
  });

  afterEach(() => {
    for (const key of ENV_KEYS) {
      if (savedEnv[key] === undefined) delete process.env[key];
      else process.env[key] = savedEnv[key];
    }
  });

  test("returns null when nothing is configured", () => {
    expect(resolveRemoteRerankConfig({}, { trusted: true })).toBeNull();
    expect(resolveRemoteRerankConfig(undefined, { trusted: true })).toBeNull();
  });

  test("requires the url+model pair", () => {
    expect(resolveRemoteRerankConfig({ rerank_api_url: "http://x:8082/v1" }, { trusted: true })).toBeNull();
    expect(resolveRemoteRerankConfig({ rerank_api_model: "bge" }, { trusted: true })).toBeNull();
  });

  test("uses YAML fields only when trusted", () => {
    const models: ModelsConfig = { rerank_api_url: "http://10.0.0.9:8082/v1", rerank_api_model: "bge" };
    expect(resolveRemoteRerankConfig(models, { trusted: true })?.url).toBe("http://10.0.0.9:8082/v1");
    expect(resolveRemoteRerankConfig(models, { trusted: false })).toBeNull();
  });

  test("env vars enable remote even when YAML is untrusted", () => {
    process.env.QMD_RERANK_API_URL = "http://10.0.0.9:8082/v1";
    process.env.QMD_RERANK_API_MODEL = "bge-env";
    const cfg = resolveRemoteRerankConfig({ rerank_api_url: "http://evil:1/v1", rerank_api_model: "evil" }, { trusted: false });
    expect(cfg?.url).toBe("http://10.0.0.9:8082/v1");
    expect(cfg?.model).toBe("bge-env");
  });

  test("trusted YAML wins over env; key/score/timeout flow through", () => {
    process.env.QMD_RERANK_API_URL = "http://env:1/v1";
    process.env.QMD_RERANK_API_MODEL = "bge-env";
    const cfg = resolveRemoteRerankConfig(
      { rerank_api_url: "http://yaml:2/v1", rerank_api_model: "bge-yaml", rerank_api_key: "sk-test", rerank_score: "logits", rerank_api_timeout: 2500 },
      { trusted: true },
    );
    expect(cfg?.url).toBe("http://yaml:2/v1");
    expect(cfg?.model).toBe("bge-yaml");
    expect(cfg?.apiKey).toBe("sk-test");
    expect(cfg?.scoreMode).toBe("logits");
    expect(cfg?.timeoutMs).toBe(2500);
    expect(cfg?.identity).toBe("remote:bge-yaml@http://yaml:2/v1@logits");
  });

  test("score mode defaults to auto and timeout to 10000ms", () => {
    const cfg = resolveRemoteRerankConfig({ rerank_api_url: "http://x:1/v1", rerank_api_model: "bge" }, { trusted: true });
    expect(cfg?.scoreMode).toBe("auto");
    expect(cfg?.timeoutMs).toBe(10000);
  });

  test("score mode participates in the cache identity", () => {
    // The identity doubles as the store's rerank cache key, and cached scores
    // are already normalized — switching rerank_score must not serve old
    // normalized scores under the new mode.
    const base: ModelsConfig = { rerank_api_url: "http://host:8082/v1", rerank_api_model: "bge" };
    const auto = resolveRemoteRerankConfig(base, { trusted: true });
    const logits = resolveRemoteRerankConfig({ ...base, rerank_score: "logits" }, { trusted: true });
    const probability = resolveRemoteRerankConfig({ ...base, rerank_score: "probability" }, { trusted: true });
    expect(auto?.identity).not.toBe(logits?.identity);
    expect(logits?.identity).not.toBe(probability?.identity);
    expect(new LlamaCpp({ remoteRerank: auto! }).rerankModelName)
      .not.toBe(new LlamaCpp({ remoteRerank: logits! }).rerankModelName);
  });

  test("rejects invalid score mode and timeout", () => {
    expect(() => resolveRemoteRerankConfig({ rerank_api_url: "http://x:1/v1", rerank_api_model: "bge", rerank_score: "banana" }, { trusted: true })).toThrow(/rerank_score/);
    expect(() => resolveRemoteRerankConfig({ rerank_api_url: "http://x:1/v1", rerank_api_model: "bge", rerank_api_timeout: -5 }, { trusted: true })).toThrow(/rerank_api_timeout/);
    expect(() => resolveRemoteRerankConfig({ rerank_api_url: "http://x:1/v1", rerank_api_model: "bge", rerank_api_timeout: Number.NaN }, { trusted: true })).toThrow(/rerank_api_timeout/);
  });
});

// =============================================================================
// Trust gate over remote endpoints (the #705 exfiltration fix)
// =============================================================================

describe("trust gate over remote rerank endpoints", () => {
  let configDir: string;
  let projectDir: string;
  let savedConfigDir: string | undefined;

  const localYml = join(".qmd", "index.yml");
  const globalYml = join(".config", "qmd", "index.yml");

  beforeEach(() => {
    configDir = mkdtempSync(join(tmpdir(), "qmd-remote-rerank-cfg-"));
    projectDir = mkdtempSync(join(tmpdir(), "qmd-remote-rerank-proj-"));
    mkdirSync(join(projectDir, ".qmd"), { recursive: true });
    savedConfigDir = process.env.QMD_CONFIG_DIR;
    process.env.QMD_CONFIG_DIR = configDir;
  });

  afterEach(() => {
    if (savedConfigDir === undefined) delete process.env.QMD_CONFIG_DIR;
    else process.env.QMD_CONFIG_DIR = savedConfigDir;
    for (const dir of [configDir, projectDir]) {
      try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
    }
  });

  function remoteOnlySnapshot() {
    return snapshotFromConfig({
      collections: {},
      models: { rerank_api_url: "http://evil.example:8082/v1", rerank_api_model: "evil" },
    });
  }

  test("a remote-only local config has gated items naming the endpoint", () => {
    const snapshot = remoteOnlySnapshot();
    const gated = gatedItems(join(projectDir, localYml), snapshot, builtinModels());
    expect(hasGatedItems(gated)).toBe(true);
    expect(gated.models.some((m) => m.uri.includes("evil.example:8082"))).toBe(true);
  });

  test("changing the endpoint URL re-arms the gate (digest changes)", () => {
    const localPath = join(projectDir, localYml);
    const before = sensitiveDigest(remoteOnlySnapshot(), localPath, builtinModels());
    const after = sensitiveDigest(
      snapshotFromConfig({ collections: {}, models: { rerank_api_url: "http://other.example:1/v1", rerank_api_model: "evil" } }),
      localPath,
      builtinModels(),
    );
    expect(before).not.toBe(after);
  });

  test("global config path is trusted; local path needs approval; URL change revokes", () => {
    const globalPath = join(configDir, globalYml);
    expect(snapshotIsTrusted(globalPath, remoteOnlySnapshot(), builtinModels())).toBe(true);

    const localPath = join(projectDir, localYml);
    const snapshot = remoteOnlySnapshot();
    expect(snapshotIsTrusted(localPath, snapshot, builtinModels())).toBe(false);

    recordTrust(localPath, sensitiveDigest(snapshot, localPath, builtinModels()));
    expect(snapshotIsTrusted(localPath, snapshot, builtinModels())).toBe(true);

    const moved = snapshotFromConfig({ collections: {}, models: { rerank_api_url: "http://evil.example:9999/v1", rerank_api_model: "evil" } });
    expect(snapshotIsTrusted(localPath, moved, builtinModels())).toBe(false);

    expect(revokeTrust(localPath)).toBe(true);
    expect(snapshotIsTrusted(localPath, snapshot, builtinModels())).toBe(false);
  });

  test("absent and built-in local rerank approve identically (ensureModels fill-in keeps trust)", () => {
    // ensureModelsConfiguredForCli() writes the built-in rerank URI into the
    // YAML on the first query; that fill-in must not re-arm the remote gate.
    const localPath = join(projectDir, localYml);
    const endpoint = { rerank_api_url: "http://trusted.example:8082/v1", rerank_api_model: "bge" };
    const absent = snapshotFromConfig({ collections: {}, models: { ...endpoint } });
    const builtinFilled = snapshotFromConfig({ collections: {}, models: { rerank: builtinModels().rerank, ...endpoint } });
    recordTrust(localPath, sensitiveDigest(absent, localPath, builtinModels()));
    try {
      expect(snapshotIsTrusted(localPath, builtinFilled, builtinModels())).toBe(true);
      // A genuinely custom local rerank still re-arms the gate.
      const custom = snapshotFromConfig({ collections: {}, models: { rerank: "hf:custom/reranker", ...endpoint } });
      expect(snapshotIsTrusted(localPath, custom, builtinModels())).toBe(false);
    } finally {
      revokeTrust(localPath);
    }
  });

  test("EXFIL: rerank with an untrusted project-local endpoint never contacts the server", async () => {
    captureWarnings();
    const server = await startRerankServer(rerankResults([{ index: 0, relevance_score: 0.9 }]));
    try {
      const yamlModels: ModelsConfig = { rerank_api_url: server.url, rerank_api_model: "evil" };
      // getStore() passes trusted=false for an untrusted project-local config.
      const cfg = resolveRemoteRerankConfig(yamlModels, { trusted: false });
      expect(cfg).toBeNull();
      const llm = remoteLlm(cfg, { results: [{ file: "a.md", score: 0.5 }], model: "local:default" });
      const out = await llm.rerank("q", DOCS);
      expect(server.requests).toHaveLength(0);
      expect(out.model).toBe("local:default");
      // The trusted config would have contacted it — proves the server was reachable.
      const allowed = remoteLlm(remoteConfig(server.url));
      await allowed.rerank("q", DOCS);
      expect(server.requests.length).toBeGreaterThan(0);
    } finally {
      await server.close();
      restoreWarnings();
    }
  });
});

// =============================================================================
// /v1/rerank client against a fake server
// =============================================================================

describe("remote rerank client", () => {
  let server: FakeRerankServer | undefined;

  afterEach(async () => {
    if (server) { await server.close(); server = undefined; }
  });

  test("POSTs a Cohere-shaped body to <url>/rerank", async () => {
    server = await startRerankServer(rerankResults([{ index: 0, relevance_score: 0.5 }]));
    const llm = remoteLlm(remoteConfig(server.url));
    const out = await llm.rerank("what is qmd", DOCS.slice(0, 1));
    expect(server.requests).toHaveLength(1);
    expect(server.requests[0]!.body).toEqual({
      model: "bge-test",
      query: "what is qmd",
      documents: ["alpha"],
      top_n: 1,
    });
    expect(out.results[0]!.file).toBe("a.md");
    expect(out.model).toMatch(/^remote:bge-test@http:\/\/127\.0\.0\.1:\d+\/v1@auto$/);
  });

  test("reverse-ordered logits come back sorted with normalized scores (auto)", async () => {
    server = await startRerankServer(rerankResults([
      { index: 2, relevance_score: 2.5 },
      { index: 0, relevance_score: 0.1 },
      { index: 1, relevance_score: -1.5 },
    ]));
    const llm = remoteLlm(remoteConfig(server.url));
    const out = await llm.rerank("q", DOCS);
    expect(out.results.map((r) => r.file)).toEqual(["c.md", "a.md", "b.md"]);
    const scores = out.results.map((r) => r.score);
    expect(scores[0]).toBeCloseTo(1 / (1 + Math.exp(-2.5)), 12);
    expect(scores[1]).toBeCloseTo(1 / (1 + Math.exp(-0.1)), 12);
    expect(scores[2]).toBeCloseTo(1 / (1 + Math.exp(1.5)), 12);
    expect(out.results.every((r) => r.score >= 0 && r.score <= 1)).toBe(true);
  });

  test("probability mode preserves raw scores", async () => {
    server = await startRerankServer(rerankResults([
      { index: 1, relevance_score: 0.9 },
      { index: 0, relevance_score: 0.2 },
    ]));
    const llm = remoteLlm(remoteConfig(server.url, { rerank_score: "probability" }));
    const out = await llm.rerank("q", DOCS.slice(0, 2));
    expect(out.results.map((r) => r.score)).toEqual([0.9, 0.2]);
  });

  test("key auth: bearer header sent, wrong key rejected gracefully", async () => {
    server = await startRerankServer((req, _body, res) => {
      if (req.headers.authorization !== "Bearer sk-right") {
        res.writeHead(401, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "invalid api key" }));
        return;
      }
      rerankResults([{ index: 0, relevance_score: 0.8 }])(req, _body, res);
    });
    captureWarnings();
    try {
      const bad = remoteLlm(remoteConfig(server.url, { rerank_api_key: "sk-wrong" }));
      const badOut = await bad.rerank("q", DOCS.slice(0, 1));
      expect(badOut.model).toBe("fallback");
      expect(badOut.results[0]!.score).toBe(0.5);
      expect(warnings.some((w) => w.includes("Remote rerank failed"))).toBe(true);

      const good = remoteLlm(remoteConfig(server.url, { rerank_api_key: "sk-right" }));
      const goodOut = await good.rerank("q", DOCS.slice(0, 1));
      expect(goodOut.model).toMatch(/^remote:bge-test@/);
      // auto keeps an in-range batch raw — no sigmoid, proving the header
      // succeeded (a 401 would have degraded to "fallback").
      expect(goodOut.results[0]!.score).toBe(0.8);
    } finally {
      restoreWarnings();
    }
  });

  test("HTTP 500 falls back to the local model when available", async () => {
    server = await startRerankServer((_req, _body, res) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "boom" }));
    });
    captureWarnings();
    try {
      const llm = remoteLlm(remoteConfig(server.url), { results: [{ file: "a.md", score: 0.7 }], model: "local:gguf" });
      const out = await llm.rerank("q", DOCS);
      expect(server.requests.length).toBeGreaterThan(0);
      expect(out.model).toBe("local:gguf");
      expect(out.results[0]!.score).toBe(0.7);
      expect(warnings.some((w) => w.includes("Remote rerank failed"))).toBe(true);
      expect(warnings.every((w) => !w.includes("unhandled"))).toBe(true);
    } finally {
      restoreWarnings();
    }
  });

  test("HTTP 500 with no local model degrades to neutral scores with a warning, never throws", async () => {
    server = await startRerankServer((_req, _body, res) => {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end("{}");
    });
    captureWarnings();
    try {
      const llm = remoteLlm(remoteConfig(server.url));
      const out = await llm.rerank("q", DOCS);
      expect(out.model).toBe("fallback");
      expect(out.results.map((r) => r.score)).toEqual([0.5, 0.5, 0.5]);
      expect(out.results.map((r) => r.file)).toEqual(["a.md", "b.md", "c.md"]);
      expect(warnings.some((w) => w.includes("Local rerank unavailable"))).toBe(true);
    } finally {
      restoreWarnings();
    }
  });

  test("ECONNREFUSED degrades gracefully with a warning", async () => {
    const dead = await startRerankServer(rerankResults([]));
    const deadUrl = dead.url;
    await dead.close();

    captureWarnings();
    try {
      const llm = remoteLlm(remoteConfig(deadUrl));
      const out = await llm.rerank("q", DOCS);
      expect(out.model).toBe("fallback");
      expect(warnings.some((w) => w.includes("Remote rerank failed"))).toBe(true);
    } finally {
      restoreWarnings();
    }
  });

  test("timeout aborts the request and degrades gracefully", async () => {
    server = await startRerankServer((_req, _body, res) => {
      // Real abort semantics need a real server-side stall; the client aborts
      // at 100ms, well before this 400ms response. unref'd so vitest can exit.
      const t = setTimeout(() => { try { rerankResults([{ index: 0, relevance_score: 1 }])(_req, _body, res); } catch { /* closed */ } }, 400);
      if (typeof t.unref === "function") t.unref();
    });
    captureWarnings();
    try {
      const llm = remoteLlm(remoteConfig(server.url, { rerank_api_timeout: 100 }));
      const out = await llm.rerank("q", DOCS.slice(0, 1));
      expect(out.model).toBe("fallback");
      expect(warnings.some((w) => w.includes("Remote rerank failed"))).toBe(true);
    } finally {
      restoreWarnings();
    }
  });

  test("malformed responses reject: missing results, out-of-range/duplicate index, non-finite score", async () => {
    const cfg = remoteConfig("http://127.0.0.1:9/v1");
    const badPayloads: unknown[] = [
      {},
      { results: "nope" },
      { results: [{ index: 3, relevance_score: 0.5 }] },
      { results: [{ index: 0, relevance_score: 0.5 }, { index: 0, relevance_score: 0.6 }] },
      { results: [{ index: 0, relevance_score: "high" }] },
      { results: [{ index: 0, relevance_score: Number.NaN }] },
      { results: [{ index: 0, relevance_score: 0.5 }] }, // missing indexes 1,2
    ];
    for (const payload of badPayloads) {
      await expect(remoteRerank("q", DOCS, cfg)).rejects.toThrow();
    }
  });

  test("empty document list skips the network call", async () => {
    server = await startRerankServer(rerankResults([]));
    const llm = remoteLlm(remoteConfig(server.url));
    const out = await llm.rerank("q", []);
    expect(server.requests).toHaveLength(0);
    expect(out.results).toEqual([]);
  });

  test("rerankModelName reflects the remote identity for cache keys", () => {
    const llm = remoteLlm(remoteConfig("http://host:8082/v1"));
    expect(llm.rerankModelName).toBe("remote:bge-test@http://host:8082/v1@auto");
    expect(new LlamaCpp({}).rerankModelName).not.toMatch(/^remote:/);
  });
});

// =============================================================================
// Rerank cache identity — scores cached under the backend that produced them
// =============================================================================

describe("rerank cache identity", () => {
  let store: Store;
  let dbDir: string;

  beforeEach(() => {
    dbDir = mkdtempSync(join(tmpdir(), "qmd-remote-rerank-db-"));
    store = createStore(join(dbDir, "index.sqlite"));
  });

  afterEach(() => {
    store.close();
    try { rmSync(dbDir, { recursive: true, force: true }); } catch { /* ignore */ }
  });

  /** Cache-identity double: rerankModelName is what the caller configured,
   *  producedBy is the backend that actually answered. */
  class FakeRerankLlm extends LlamaCpp {
    constructor(private readonly configuredAs: string, private readonly producedBy: string, private readonly score: number) {
      super();
    }
    override get rerankModelName(): string {
      return this.configuredAs;
    }
    override async rerank(): Promise<RerankResult> {
      return { results: [{ file: "a.md", text: "alpha", score: this.score, index: 0 }], model: this.producedBy };
    }
  }

  test("remote success caches under the remote identity", async () => {
    await rerankWithLlm("q", [DOCS[0]!], undefined, store.db, undefined, new FakeRerankLlm("remote:bge@http://s/v1", "remote:bge@http://s/v1", 0.9));
    const remoteKey = store.getCacheKey("rerank", { query: "q", model: "remote:bge@http://s/v1", chunk: "alpha" });
    expect(store.getCachedResult(remoteKey)).toBe("0.9");
  });

  test("local fallback during a remote outage caches under the local identity, not the remote key", async () => {
    await rerankWithLlm("q", [DOCS[0]!], undefined, store.db, undefined, new FakeRerankLlm("remote:bge@http://s/v1", "local:gguf-uri", 0.4));
    const remoteKey = store.getCacheKey("rerank", { query: "q", model: "remote:bge@http://s/v1", chunk: "alpha" });
    const localKey = store.getCacheKey("rerank", { query: "q", model: "local:gguf-uri", chunk: "alpha" });
    expect(store.getCachedResult(remoteKey)).toBeNull();
    expect(store.getCachedResult(localKey)).toBe("0.4");
  });

  test("neutral fallback results are not cached at all", async () => {
    await rerankWithLlm("q", [DOCS[0]!], undefined, store.db, undefined, new FakeRerankLlm("remote:bge@http://s/v1", "fallback", 0.5));
    const remoteKey = store.getCacheKey("rerank", { query: "q", model: "remote:bge@http://s/v1", chunk: "alpha" });
    const fallbackKey = store.getCacheKey("rerank", { query: "q", model: "fallback", chunk: "alpha" });
    expect(store.getCachedResult(remoteKey)).toBeNull();
    expect(store.getCachedResult(fallbackKey)).toBeNull();
  });

  test("neutral fallback scores flow to the per-call results instead of zeros", async () => {
    class NeutralFallbackLlm extends FakeRerankLlm {
      constructor() { super("remote:bge@http://s/v1", "fallback", 0.5); }
      override async rerank(): Promise<RerankResult> {
        return { results: DOCS.map((d, i) => ({ ...d, score: 0.5, index: i })), model: "fallback" };
      }
    }
    const out = await rerankWithLlm("q", DOCS, undefined, store.db, undefined, new NeutralFallbackLlm());
    expect(out.map((r) => r.score)).toEqual([0.5, 0.5, 0.5]);
    // ...while staying out of the persistent cache.
    const remoteKey = store.getCacheKey("rerank", { query: "q", model: "remote:bge@http://s/v1", chunk: "beta" });
    expect(store.getCachedResult(remoteKey)).toBeNull();
  });
});
