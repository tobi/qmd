/**
 * embed-lock-domain.test.ts — one embed domain per index.
 *
 * `generateEmbeddings()` is the single entry point every embed run funnels
 * through (CLI, SDK `store.embed()`, direct library calls), and it now owns the
 * existing crash-safe sidecar embed lock from `src/cli/embed-lock.ts` for the
 * whole run. Without that, two runs sharing an index interleave their partial
 * chunk cleanup (`removeIncompleteEmbeddings`) and their vector publication, so
 * each can delete chunks the other just completed (#825).
 *
 * These tests assert what callers observe: a second run is rejected with
 * `EmbedLockBusyError` instead of embedding concurrently, the "...already
 * embedded" classification is answered while the lock is held, ownership is
 * released on failure, and databases no other process can open are not put
 * behind an on-disk lock.
 */
import { describe, test, expect, beforeAll, afterAll, beforeEach, afterEach } from "vitest";
import { realpathSync } from "node:fs";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import YAML from "yaml";
import type { Token as LlamaToken } from "node-llama-cpp";
import { LlamaCpp, type EmbeddingResult } from "../src/llm.js";
import type { Database } from "../src/db.js";
import {
  createStore,
  generateEmbeddings,
  hashContentSync,
  extractTitle,
  insertContent,
  insertDocument,
  EmbedLockBusyError,
  type EmbedResult,
  type Store,
} from "../src/store.js";
import {
  createStore as createSdkStore,
  EmbedLockBusyError as EmbedLockBusyErrorFromSdk,
  type QMDStore,
} from "../src/index.js";
import { acquireEmbedLock, embedLockPathForDb } from "../src/cli/embed-lock.js";
import type { CollectionConfig } from "../src/collections.js";

let testDir: string;
let dbPath: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-embed-lock-domain-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

beforeEach(async () => {
  dbPath = join(testDir, `embed-${Date.now()}-${Math.random().toString(36).slice(2)}.sqlite`);
  const configDir = await mkdtemp(join(testDir, "config-"));
  process.env.QMD_CONFIG_DIR = configDir;
  const emptyConfig: CollectionConfig = { collections: {} };
  await writeFile(join(configDir, "index.yml"), YAML.stringify(emptyConfig));
});

afterEach(() => {
  delete process.env.QMD_CONFIG_DIR;
});

/**
 * A fake store-local LLM: a real `LlamaCpp` whose model loading is replaced by
 * deterministic, in-memory work. Subclassing keeps the double exactly as typed
 * as the field it stands in for (`Store["llm"]`), so `withLLMSessionForLlm()`
 * opens a session on a genuine instance, while the pinned local model names
 * mean no model file and no network are ever touched — the run is CI-safe.
 */
class FakeEmbedLlm extends LlamaCpp {
  readonly embedCalls: string[] = [];
  private readonly onBatch?: () => Promise<void>;
  private readonly tokenizeFails: boolean;

  constructor(hooks: { onBatch?: () => Promise<void>; tokenizeThrows?: boolean } = {}) {
    // Pinned model names: the shipped defaults would inherit QMD_*_MODEL from
    // the environment, and a remote or JEV selection would bypass the local
    // tokenizer path these tests exercise.
    super({
      embedModel: "fake-embed",
      generateModel: "fake-generate",
      rerankModel: "fake-rerank",
      inactivityTimeoutMs: 0,
    });
    this.onBatch = hooks.onBatch;
    this.tokenizeFails = hooks.tokenizeThrows === true;
  }

  override async tokenize(text: string): Promise<readonly LlamaToken[]> {
    if (this.tokenizeFails) throw new Error("fake tokenizer failure");
    // A token id is `number & { __token: never }`, a brand node-llama-cpp only
    // mints inside its own tokenizer; the double's ids are plain numbers, so
    // the brand is applied here, once.
    const token = 1 as LlamaToken;
    return Array.from({ length: Math.max(1, Math.ceil(text.length / 16)) }, () => token);
  }

  override async detokenize(tokens: readonly LlamaToken[]): Promise<string> {
    return "x".repeat(tokens.length * 16);
  }

  override async embed(text: string): Promise<EmbeddingResult | null> {
    this.embedCalls.push(text);
    return { embedding: [0.1, 0.2, 0.3], model: "fake-embed" };
  }

  override async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    this.embedCalls.push(...texts);
    await this.onBatch?.();
    return texts.map((_text, index) => ({
      embedding: [index + 1, index + 2, index + 3],
      model: "fake-embed",
    }));
  }
}

function insertDoc(db: Database, collection: string, path: string, body: string): void {
  const now = new Date().toISOString();
  const hash = hashContentSync(body);
  insertContent(db, hash, body, now);
  insertDocument(db, collection, path, extractTitle(body, path), hash, now, now);
}

function countVectors(db: Database): number {
  const row = db.prepare(`SELECT COUNT(*) AS n FROM content_vectors`).get() as { n: number };
  return Number(row.n);
}

/**
 * The lock path the store computes for `dbPath`. The store canonicalises the
 * file behind its connection (macOS tmpdirs are reached through /private), so
 * the expectation must canonicalise the same way. Four call sites share it.
 */
function indexLockPath(dbFile: string): string {
  return embedLockPathForDb(realpathSync(dbFile));
}

function isLockFree(store: Store): boolean {
  const attempt = acquireEmbedLock(indexLockPath(store.dbPath));
  if (attempt.status !== "acquired") return false;
  attempt.handle.release();
  return true;
}

describe("generateEmbeddings — exclusive per index", () => {
  test("a run that cannot take the lock rejects before touching any vector", async () => {
    const store = createStore(dbPath);
    const llm = new FakeEmbedLlm();
    store.llm = llm;
    try {
      insertDoc(store.db, "docs", "a.md", "# A\n\nalpha body for embedding");

      // Another process is embedding this index right now.
      const other = acquireEmbedLock(indexLockPath(store.dbPath));
      expect(other.status).toBe("acquired");
      if (other.status !== "acquired") throw new Error("unreachable");

      try {
        await expect(generateEmbeddings(store)).rejects.toBeInstanceOf(EmbedLockBusyError);
        // Rejected before the run cleared, copied or read anything.
        expect(llm.embedCalls).toHaveLength(0);
        expect(countVectors(store.db)).toBe(0);
        expect(store.getHashesNeedingEmbedding()).toBe(1);
      } finally {
        other.handle.release();
      }

      // Once the other run is gone, this index embeds normally.
      const result = await generateEmbeddings(store);
      expect(result.errors).toBe(0);
      expect(result.chunksEmbedded).toBeGreaterThan(0);
      expect(result.nothingToDo).toBe(false);
      expect(countVectors(store.db)).toBeGreaterThan(0);
      expect(isLockFree(store)).toBe(true);
    } finally {
      store.close();
    }
  });

  test("the no-work answer is only given while holding the lock", async () => {
    const store = createStore(dbPath);
    store.llm = new FakeEmbedLlm();
    try {
      insertDoc(store.db, "docs", "a.md", "# A\n\nalready embedded body");
      const first = await generateEmbeddings(store);
      expect(first.nothingToDo).toBe(false);

      // Nothing is pending, but another run owns the index: a caller must not
      // report "already embedded" from outside the lock, because that other run
      // may still be publishing.
      const other = acquireEmbedLock(indexLockPath(store.dbPath));
      expect(other.status).toBe("acquired");
      if (other.status !== "acquired") throw new Error("unreachable");
      try {
        await expect(generateEmbeddings(store)).rejects.toBeInstanceOf(EmbedLockBusyError);
      } finally {
        other.handle.release();
      }

      // Holding the lock, the run classifies itself as no-work and copies
      // nothing.
      const second: EmbedResult = await generateEmbeddings(store);
      expect(second.nothingToDo).toBe(true);
      expect(second.docsProcessed).toBe(0);
      expect(second.chunksEmbedded).toBe(0);
    } finally {
      store.close();
    }
  });

  test("two stores on one index serialize instead of embedding together", async () => {
    const first = createStore(dbPath);
    const second = createStore(dbPath);
    const started = Promise.withResolvers<void>();
    const gate = Promise.withResolvers<void>();
    const firstLlm = new FakeEmbedLlm({
      onBatch: async () => {
        started.resolve();
        await gate.promise;
      },
    });
    first.llm = firstLlm;
    second.llm = new FakeEmbedLlm();

    try {
      insertDoc(first.db, "docs", "a.md", "# A\n\nbody embedded by the first run");

      const running = generateEmbeddings(first);
      await started.promise;

      // The second store is a different connection to the same index.
      const busy = await generateEmbeddings(second).then(
        () => null,
        (error: unknown) => error
      );
      expect(busy).toBeInstanceOf(EmbedLockBusyError);
      expect((busy as EmbedLockBusyError).lockPath).toBe(indexLockPath(dbPath));

      gate.resolve();
      const result = await running;
      expect(result.errors).toBe(0);
      expect(result.chunksEmbedded).toBeGreaterThan(0);

      // Ownership was released: the same index is embeddable again.
      const after = await generateEmbeddings(second);
      expect(after.nothingToDo).toBe(true);
    } finally {
      gate.resolve();
      first.close();
      second.close();
    }
  });

  test("the SDK embed path reports the same busy error", async () => {
    const stores: QMDStore[] = [];
    // Declared outside the try so the finally below can always release a run
    // parked on the gate, even when opening a store throws.
    const started = Promise.withResolvers<void>();
    const gate = Promise.withResolvers<void>();
    try {
      const owner = await createSdkStore({ dbPath });
      const contender = await createSdkStore({ dbPath });
      stores.push(owner, contender);

      owner.internal.llm = new FakeEmbedLlm({
        onBatch: async () => {
          started.resolve();
          await gate.promise;
        },
      });
      contender.internal.llm = new FakeEmbedLlm();

      insertDoc(owner.internal.db, "docs", "a.md", "# A\n\nsdk embedded body");

      const running = owner.embed();
      await started.promise;

      const busy = await contender.embed().then(
        () => null,
        (error: unknown) => error
      );
      expect(busy).toBeInstanceOf(EmbedLockBusyErrorFromSdk);
      expect(busy).toBeInstanceOf(EmbedLockBusyError);

      gate.resolve();
      const result = await running;
      expect(result.errors).toBe(0);
      expect(result.chunksEmbedded).toBeGreaterThan(0);
    } finally {
      gate.resolve();
      for (const store of stores) await store.close();
    }
  });

  test("failing runs release the lock", async () => {
    const store = createStore(dbPath);
    store.llm = new FakeEmbedLlm({ tokenizeThrows: true });
    try {
      insertDoc(store.db, "docs", "a.md", "# A\n\nnever chunked body");

      await expect(generateEmbeddings(store)).rejects.toThrow(/fake tokenizer failure/);
      expect(countVectors(store.db)).toBe(0);
      expect(isLockFree(store)).toBe(true);
    } finally {
      store.close();
    }
  });

  test("databases no other process can open are not put behind the on-disk lock", async () => {
    const alpha = createStore(":memory:");
    const beta = createStore(":memory:");
    alpha.llm = new FakeEmbedLlm();
    beta.llm = new FakeEmbedLlm();

    try {
      insertDoc(alpha.db, "docs", "a.md", "# A\n\nin-memory alpha body");
      insertDoc(beta.db, "docs", "b.md", "# B\n\nin-memory beta body");

      // Concurrent on purpose: a shared sidecar lock would reject one of them.
      const [alphaResult, betaResult] = await Promise.all([
        generateEmbeddings(alpha),
        generateEmbeddings(beta),
      ]);
      expect(alphaResult.chunksEmbedded).toBeGreaterThan(0);
      expect(betaResult.chunksEmbedded).toBeGreaterThan(0);
      expect(alphaResult.errors).toBe(0);
      expect(betaResult.errors).toBe(0);
    } finally {
      alpha.close();
      beta.close();
    }
  });

  test("a second spelling of one index file shares the same lock", async () => {
    // The lock is derived from the file behind the connection (PRAGMA
    // database_list, canonicalised), not from the string the store was opened
    // with, so a relative alias of the same index cannot bypass mutual
    // exclusion. A relative path is used rather than a symlink because SQLite
    // derives its -wal/-shm names from the spelling, and two spellings of one
    // inode are not a supported way to open one database.
    const plain = createStore(dbPath);
    const aliasPath = relative(process.cwd(), dbPath);
    expect(aliasPath).not.toBe(dbPath);
    const viaAlias = createStore(aliasPath);
    expect(viaAlias.dbPath).toBe(aliasPath);

    const started = Promise.withResolvers<void>();
    const gate = Promise.withResolvers<void>();
    plain.llm = new FakeEmbedLlm({
      onBatch: async () => {
        started.resolve();
        await gate.promise;
      },
    });
    viaAlias.llm = new FakeEmbedLlm();

    try {
      insertDoc(plain.db, "docs", "a.md", "# A\n\naliased index body");

      const running = generateEmbeddings(plain);
      await started.promise;

      const busy = await generateEmbeddings(viaAlias).then(
        () => null,
        (error: unknown) => error
      );
      expect(busy).toBeInstanceOf(EmbedLockBusyError);

      gate.resolve();
      await running;
    } finally {
      gate.resolve();
      plain.close();
      viaAlias.close();
    }
  });
});
