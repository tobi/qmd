/**
 * Regression tests for keeping one collection configuration file and the
 * SQLite `store_collections` mirror of it in step across store opens and SDK
 * mutations.
 *
 * The reported hazard: a store read the YAML (snapshot S), another process added
 * a collection to the YAML *and* its SQLite mirror, and the first store then
 * synced S — deleting the other process's collection and stamping S's hash over
 * the result. A second hazard sat in the same code: SDK mutations wrote SQLite
 * before taking the config lock, so a refused config write left a
 * database-only edit that the next sync silently threw away.
 *
 * The cross-process tests freeze one store's snapshot behind a barrier, land a
 * real SDK mutation from this process, and then let the store sync. The
 * `snapshot` worker models the pre-fix store open, the `locked` worker the
 * current one; a control runs the same pre-fix worker *after* the mutation, so
 * the assertion cannot pass because the sync became a no-op.
 *
 * Crash consistency: the file publish and the SQLite commit are separate
 * resources. A crash between them leaves SQLite behind the file; the stored
 * `config_hash` does not certify that state, so the next sync re-derives the
 * rows from the file. No test here claims a single atomic commit across both.
 *
 * A rename is the one mutation whose index half is keyed by a name pair, so its
 * mirror only runs against durable evidence of that specific rename: the
 * `.pending-rename.json` marker written before the publication (or the scope's
 * in-memory marker for an inline config). Repeating the rename finishes the
 * index half; without evidence nothing moves, because "the file no longer lists
 * the old name and lists another one" is not a rename.
 */
import { describe, test, expect, afterEach } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";
import { createStore } from "../src/index.js";
import {
  CollectionConfigScope,
  ConfigLockTimeoutError,
  CorruptPendingRenameError,
  PendingRenameError,
  type CollectionConfig,
} from "../src/collections.js";
import { openDatabase, type Database } from "../src/db.js";
import {
  createStore as createInternalStore,
  extractTitle,
  getStoreCollections,
  hashContentSync,
  insertContent,
  insertDocument,
  syncConfigToDb,
  upsertStoreCollection,
  type Store as InternalStore,
} from "../src/store.js";

const thisDir = dirname(fileURLToPath(import.meta.url));
const workerScript = join(thisDir, "_helpers", "config-sync-worker.ts");
const holderScript = join(thisDir, "_helpers", "config-mutation-child.ts");
const tsxCli = join(thisDir, "..", "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function makeTempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "qmd-config-sync-"));
  tempDirs.push(dir);
  return dir;
}

function writeConfig(configPath: string, config: CollectionConfig): void {
  writeFileSync(configPath, YAML.stringify(config), "utf-8");
}

/** Parse the file the way a store does, so hashes compare like for like. */
function readConfig(configPath: string): CollectionConfig {
  return YAML.parse(readFileSync(configPath, "utf-8")) as CollectionConfig;
}

/** The hash `syncConfigToDb` stores and certifies a document with. */
function configHash(config: CollectionConfig): string {
  return createHash("sha256").update(JSON.stringify(config)).digest("hex");
}

type InspectedDb = {
  names: string[];
  hash: string | undefined;
};

/** Read what a later process would see: the rows, and the hash certifying them. */
function inspectDb(dbPath: string): InspectedDb {
  const db = openDatabase(dbPath);
  try {
    const hashRow = db
      .prepare(`SELECT value FROM store_config WHERE key = 'config_hash'`)
      .get() as { value: string } | null | undefined;
    return {
      names: getStoreCollections(db).map(collection => collection.name).sort(),
      hash: hashRow?.value,
    };
  } finally {
    db.close();
  }
}

/** Seed one active document under `collection`, as an index run would. */
function indexDocument(internal: InternalStore, collection: string, path: string, body: string): void {
  const now = "2026-01-01T00:00:00.000Z";
  const hash = hashContentSync(body);
  insertContent(internal.db, hash, body, now);
  insertDocument(internal.db, collection, path, extractTitle(body, path), hash, now, now);
}

function documentCount(db: Database, collection: string): number {
  const row = db.prepare(`SELECT COUNT(*) AS n FROM documents WHERE collection = ?`).get(collection) as { n: number };
  return Number(row.n);
}

/** Config-side sidecars a finished mutation must not leave behind. */
function pendingRenameMarkers(dir: string): string[] {
  return readdirSync(dir).filter(name => name.endsWith(".pending-rename.json"));
}

type Child = {
  proc: ChildProcess;
  output: () => string;
  send: (line: string) => void;
  waitFor: (needle: string, timeoutMs?: number) => Promise<void>;
  done: Promise<number | null>;
};

function spawnChild(scriptArgs: string[], env: Record<string, string> = {}): Child {
  const proc = spawn(
    process.execPath,
    isBunRuntime ? scriptArgs : [tsxCli, ...scriptArgs],
    { stdio: ["pipe", "pipe", "pipe"], env: { ...process.env, ...env } },
  );

  let output = "";
  const waiters = new Set<{ needle: string; resolve: () => void }>();
  const publish = () => {
    for (const waiter of waiters) {
      if (!output.includes(waiter.needle)) continue;
      waiters.delete(waiter);
      waiter.resolve();
    }
  };
  proc.stdout?.on("data", (chunk: Buffer) => {
    output += chunk.toString();
    publish();
  });
  proc.stderr?.on("data", (chunk: Buffer) => {
    output += chunk.toString();
    publish();
  });

  const waitFor = (needle: string, timeoutMs = 30_000): Promise<void> => {
    if (output.includes(needle)) return Promise.resolve();
    const { promise, resolve, reject } = Promise.withResolvers<void>();
    const waiter = { needle, resolve };
    waiters.add(waiter);
    // Failure guard only: bounds a hung child, never decides the outcome.
    const timer = setTimeout(() => {
      if (!waiters.delete(waiter)) return;
      reject(new Error(`child never printed ${needle}: ${output}`));
    }, timeoutMs);
    void promise.then(
      () => clearTimeout(timer),
      () => clearTimeout(timer),
    );
    return promise;
  };

  const { promise: done, resolve: onClose } = Promise.withResolvers<number | null>();
  proc.on("close", (code) => onClose(code));

  return {
    proc,
    output: () => output,
    send: (line: string) => proc.stdin?.write(`${line}\n`),
    waitFor,
    done,
  };
}

/** Spawn the store-open worker for one index. */
function spawnWorker(configPath: string, dbPath: string, mode: "snapshot" | "locked"): Child {
  return spawnChild([workerScript, configPath, dbPath, mode]);
}

/**
 * One current-code SDK mutation: add `name` to the file and its SQLite mirror.
 * This is the other process of the race, not a test fixture.
 */
async function addFromAnotherStore(dbPath: string, configPath: string, name: string): Promise<void> {
  const store = await createStore({ dbPath, configPath });
  try {
    await store.addCollection(name, { path: `/${name}` });
  } finally {
    await store.close();
  }
}

describe("store open mirrors an authoritative config", () => {
  test("a stale snapshot deletes a collection another process published", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });
    // The pre-fix store's snapshot, and the hash it will stamp.
    const staleSnapshot = readConfig(configPath);

    const child = spawnWorker(configPath, dbPath, "snapshot");
    try {
      await child.waitFor("READY");
      await addFromAnotherStore(dbPath, configPath, "notes");

      child.send("GO");
      await child.done;
      expect(child.output()).toContain("SYNCED");
    } finally {
      child.proc.kill("SIGKILL");
    }

    expect(readConfig(configPath).collections).toHaveProperty("notes");
    const db = inspectDb(dbPath);
    // The lost update: the file has `notes`, the mirror does not.
    expect(db.names).toEqual(["docs"]);
    // And the hash now certifies the stale document the rows came from.
    expect(db.hash).toBe(configHash(staleSnapshot));
  });

  test("the locked sync keeps a collection another process published", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const child = spawnWorker(configPath, dbPath, "locked");
    try {
      await child.waitFor("READY");
      await addFromAnotherStore(dbPath, configPath, "notes");

      child.send("GO");
      await child.done;
      expect(child.output()).toContain("SYNCED");
    } finally {
      child.proc.kill("SIGKILL");
    }

    const db = inspectDb(dbPath);
    expect(db.names).toEqual(["docs", "notes"]);
    expect(db.hash).toBe(configHash(readConfig(configPath)));
  });

  test("control: the snapshot sync keeps a collection it already saw", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    // Mutation before the snapshot: same pre-fix worker, same sync, but there
    // is nothing stale to apply. If this lost `notes`, the race test above
    // would prove nothing about the interleaving.
    await addFromAnotherStore(dbPath, configPath, "notes");

    const child = spawnWorker(configPath, dbPath, "snapshot");
    try {
      await child.waitFor("READY");
      child.send("GO");
      await child.done;
      expect(child.output()).toContain("SYNCED");
    } finally {
      child.proc.kill("SIGKILL");
    }

    expect(inspectDb(dbPath).names).toEqual(["docs", "notes"]);
  });
});

describe("SDK mutations publish the config before mirroring it", () => {
  test("a refused config write leaves no database-only edit", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const store = await createStore({ dbPath, configPath });
    // Another process holds this config's write lock across a barrier.
    const holder = spawnChild([holderScript, configPath, "hold-update", "held", "/held"]);
    const savedTimeout = process.env.QMD_CONFIG_LOCK_TIMEOUT_MS;
    try {
      await holder.waitFor("HOLDING");
      process.env.QMD_CONFIG_LOCK_TIMEOUT_MS = "0";

      await expect(
        store.addCollection("notes", { path: "/notes" }),
      ).rejects.toBeInstanceOf(ConfigLockTimeoutError);

      // Pretending to have written is what the pre-fix order did: the row
      // landed while the config write was still refused.
      expect(getStoreCollections(store.internal.db).map(c => c.name)).toEqual(["docs"]);
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["docs"]);
    } finally {
      if (savedTimeout === undefined) delete process.env.QMD_CONFIG_LOCK_TIMEOUT_MS;
      else process.env.QMD_CONFIG_LOCK_TIMEOUT_MS = savedTimeout;
      holder.send("GO");
      await holder.done;
      await store.close();
    }
  });

  test("a mutation retries successfully once the config lock is free", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const store = await createStore({ dbPath, configPath });
    try {
      await store.addCollection("notes", { path: "/notes" });
      expect(getStoreCollections(store.internal.db).map(c => c.name).sort()).toEqual(["docs", "notes"]);
      expect(Object.keys(readConfig(configPath).collections).sort()).toEqual(["docs", "notes"]);
    } finally {
      await store.close();
    }
  });

  test("a mirror that fails leaves the file authoritative, and the next open converges", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    // Seed the index and its hash, so the recovery below cannot pass because
    // nothing had ever been synced.
    const seeded = await createStore({ dbPath, configPath });
    await seeded.close();

    const scope = new CollectionConfigScope({ configPath });
    const db = openDatabase(dbPath);
    try {
      expect(() =>
        scope.addCollection("notes", "/notes", "**/*.md", () => {
          throw new Error("mirror failed");
        }),
      ).toThrow("mirror failed");

      // The publish happened first: the change is in the file, not the mirror.
      expect(Object.keys(scope.load().collections).sort()).toEqual(["docs", "notes"]);
      expect(getStoreCollections(db).map(c => c.name)).toEqual(["docs"]);
    } finally {
      scope.close();
      db.close();
    }

    const reopened = await createStore({ dbPath, configPath });
    try {
      expect((await reopened.listCollections()).map(c => c.name).sort()).toEqual(["docs", "notes"]);
    } finally {
      await reopened.close();
    }
  });

  test("a direct row write invalidates the stored hash, so a reverted file is re-synced", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    const original: CollectionConfig = {
      collections: { docs: { path: "/docs", pattern: "**/*.md" } },
    };
    writeConfig(configPath, original);

    const store = await createStore({ dbPath, configPath });
    try {
      await store.addCollection("notes", { path: "/notes" });
    } finally {
      await store.close();
    }

    // Someone reverts the file (a git checkout, an editor undo). The hash the
    // open above stamped would now match again while the rows still hold a
    // collection the file never had, so a skip here would keep a phantom.
    writeConfig(configPath, original);

    const reopened = await createStore({ dbPath, configPath });
    try {
      expect((await reopened.listCollections()).map(c => c.name)).toEqual(["docs"]);
    } finally {
      await reopened.close();
    }
  });

  test("an unguarded snapshot sync certifies a document the rows no longer match", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    // Snapshot S, then a change another process publishes to the same file.
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });
    const stale = readConfig(configPath);
    writeConfig(configPath, {
      collections: {
        docs: { path: "/docs", pattern: "**/*.md" },
        notes: { path: "/notes", pattern: "**/*.md" },
      },
    });

    // A bare connection has no schema; the internal store creates the tables a
    // later process would see (store_collections, store_config).
    const internal = createInternalStore(dbPath);
    try {
      // The unguarded shape: read earlier, sync later. `notes` never lands, and
      // the hash certifies the stale document that produced the rows.
      syncConfigToDb(internal.db, stale);
      const hashRow = internal.db
        .prepare(`SELECT value FROM store_config WHERE key = 'config_hash'`)
        .get() as { value: string };
      expect(getStoreCollections(internal.db).map(c => c.name)).toEqual(["docs"]);
      expect(hashRow.value).toBe(configHash(stale));
      expect(hashRow.value).not.toBe(configHash(readConfig(configPath)));
    } finally {
      internal.close();
    }
  });
});

describe("single-config scope synchronization", () => {
  test("withLockedConfig sees a publish made after the last load", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const reader = new CollectionConfigScope({ configPath });
    const writer = new CollectionConfigScope({ configPath });
    try {
      expect(Object.keys(reader.load().collections)).toEqual(["docs"]);
      writer.addCollection("notes", "/notes");
      expect(reader.withLockedConfig(config => Object.keys(config.collections).sort())).toEqual([
        "docs",
        "notes",
      ]);
    } finally {
      reader.close();
      writer.close();
    }
  });

  test("a closed scope still rejects the locked read", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeConfig(configPath, { collections: {} });

    const scope = new CollectionConfigScope({ configPath });
    scope.close();
    expect(() => scope.withLockedConfig(config => config)).toThrow();
  });
});

describe("interrupted remove/rename finish on retry", () => {
  test("a removal whose index half failed is finished by re-running it", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    // The user removes `docs`: the file is published first, then the index
    // commit fails.
    const scope = new CollectionConfigScope({ configPath });
    try {
      expect(() =>
        scope.removeCollection("docs", () => {
          throw new Error("index commit failed");
        }),
      ).toThrow("index commit failed");
    } finally {
      scope.close();
    }
    expect(readConfig(configPath).collections).toEqual({});

    // A later open syncs the rows from the published file: the row goes, the
    // document stays. The sync never garbage-collects index data.
    const store = await createStore({ dbPath, configPath });
    try {
      expect(getStoreCollections(store.internal.db)).toEqual([]);
      expect(documentCount(store.internal.db, "docs")).toBe(1);

      // Re-running the same removal finishes it; the configuration has not
      // changed, so only the index move is left.
      expect(await store.removeCollection("docs")).toBe(true);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
    } finally {
      await store.close();
    }
  });

  test("a rename whose index half failed is finished by re-running it", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    const scope = new CollectionConfigScope({ configPath });
    try {
      expect(() =>
        scope.renameCollection("docs", "guides", () => {
          throw new Error("index commit failed");
        }),
      ).toThrow("index commit failed");
    } finally {
      scope.close();
    }
    expect(Object.keys(readConfig(configPath).collections)).toEqual(["guides"]);

    // The intervening open syncs the target row from the published file while
    // the document is still under the old name: exactly the state a retry must
    // be able to finish, instead of refusing with "already exists".
    const store = await createStore({ dbPath, configPath });
    try {
      expect(getStoreCollections(store.internal.db).map(c => c.name)).toEqual(["guides"]);
      expect(documentCount(store.internal.db, "docs")).toBe(1);
      expect(documentCount(store.internal.db, "guides")).toBe(0);

      expect(await store.renameCollection("docs", "guides")).toBe(true);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
      expect(documentCount(store.internal.db, "guides")).toBe(1);
    } finally {
      await store.close();
    }
  });

  test("a populated target refuses before the configuration is published", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    // `other` owns a document in the index but is not in the file (a leftover
    // the config no longer lists). Renaming onto it would merge two
    // collections, so it must be refused before `docs` is renamed in the file.
    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "other", "leftover.md", "# Leftover\n\nbody");
    } finally {
      seed.close();
    }

    const store = await createStore({ dbPath, configPath });
    try {
      await expect(store.renameCollection("docs", "other")).rejects.toThrow("already exists");
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["docs"]);
      expect(documentCount(store.internal.db, "other")).toBe(1);
    } finally {
      await store.close();
    }
  });

  test("a rename without evidence of itself never moves another collection's state", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    // `target` is an unrelated, never-indexed collection. `leftover` owns a
    // document but is in no configuration: exactly the shape that must not be
    // mistaken for an interrupted rename.
    writeConfig(configPath, { collections: { target: { path: "/target", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "leftover", "doc.md", "# Leftover\n\nbody");
    } finally {
      seed.close();
    }

    const store = await createStore({ dbPath, configPath });
    try {
      // No marker and no publication: fail closed, moving nothing.
      expect(await store.renameCollection("leftover", "target")).toBe(false);
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["target"]);
      expect(documentCount(store.internal.db, "leftover")).toBe(1);
      expect(documentCount(store.internal.db, "target")).toBe(0);
    } finally {
      await store.close();
    }
  });

  test("a completed rename moves the index state and leaves no marker", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    const store = await createStore({ dbPath, configPath });
    try {
      expect(await store.renameCollection("docs", "guides")).toBe(true);
      expect(documentCount(store.internal.db, "guides")).toBe(1);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
    } finally {
      await store.close();
    }

    expect(Object.keys(readConfig(configPath).collections)).toEqual(["guides"]);
    expect(pendingRenameMarkers(dir)).toEqual([]);
  });

  test("a rename retried after a crash-equivalent publish still carries its marker", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    // Publish the rename and then "crash" before the index half: the mirror is
    // never called, which leaves exactly the state a killed process leaves.
    const scope = new CollectionConfigScope({ configPath });
    try {
      scope.renameCollection("docs", "guides", () => {
        throw new Error("crash");
      });
    } catch {
      // The publication already happened; only the index half is missing.
    } finally {
      scope.close();
    }
    expect(pendingRenameMarkers(dir)).toHaveLength(1);

    const store = await createStore({ dbPath, configPath });
    try {
      expect(await store.renameCollection("docs", "guides")).toBe(true);
      expect(documentCount(store.internal.db, "guides")).toBe(1);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
    } finally {
      await store.close();
    }
    expect(pendingRenameMarkers(dir)).toEqual([]);
  });
});

describe("pending rename evidence", () => {
  test("a live marker refuses a second rename and any change to its names", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    // The publication happens, then the index move fails: the marker is live.
    const scope = new CollectionConfigScope({ configPath });
    try {
      expect(() =>
        scope.renameCollection("docs", "guides", () => {
          throw new Error("index commit failed");
        }),
      ).toThrow("index commit failed");
    } finally {
      scope.close();
    }
    expect(pendingRenameMarkers(dir)).toHaveLength(1);

    const store = await createStore({ dbPath, configPath });
    try {
      // A second rename, and any add/remove of a name the marker speaks about,
      // would destroy the only evidence of the unfinished index move.
      await expect(store.renameCollection("guides", "notes")).rejects.toBeInstanceOf(PendingRenameError);
      await expect(store.addCollection("guides", { path: "/guides" })).rejects.toBeInstanceOf(PendingRenameError);
      await expect(store.removeCollection("guides")).rejects.toBeInstanceOf(PendingRenameError);
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["guides"]);
      expect(documentCount(store.internal.db, "docs")).toBe(1);

      // An unrelated name is not blocked, and must leave the marker alone.
      await store.addCollection("unrelated", { path: "/unrelated" });
      expect(pendingRenameMarkers(dir)).toHaveLength(1);
      expect(Object.keys(readConfig(configPath).collections).sort()).toEqual(["guides", "unrelated"]);

      // Repeating the published rename finishes it and clears the evidence.
      expect(await store.renameCollection("docs", "guides")).toBe(true);
      expect(documentCount(store.internal.db, "guides")).toBe(1);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
      expect(pendingRenameMarkers(dir)).toEqual([]);
    } finally {
      await store.close();
    }
  });

  test("a marker whose publication never landed neither blocks nor moves", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });
    // Exactly what a failed publish leaves: the marker is written, while the
    // file still lists the old name and no new one.
    writeFileSync(
      join(dir, "index.yml.pending-rename.json"),
      JSON.stringify({ from: "docs", to: "guides" }),
    );

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    const store = await createStore({ dbPath, configPath });
    try {
      // The stale marker is not evidence: a different target is not blocked,
      // and nothing is moved into `guides`.
      expect(await store.renameCollection("docs", "other")).toBe(true);
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["other"]);
      expect(documentCount(store.internal.db, "other")).toBe(1);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
      expect(pendingRenameMarkers(dir)).toEqual([]);
    } finally {
      await store.close();
    }
  });

  test("a malformed journal fails closed and writes nothing", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    const store = await createStore({ dbPath, configPath });
    const markerPath = join(dir, "index.yml.pending-rename.json");
    try {
      const before = readFileSync(configPath, "utf-8");
      const malformed = [
        "{ not json",
        "[]",
        '{"from":"docs"}',
        '{"from":"docs","to":"docs"}',
        '{"from":"../escape","to":"guides"}',
        '{"from":42,"to":"guides"}',
      ];
      for (const payload of malformed) {
        writeFileSync(markerPath, payload);
        await expect(store.addCollection("notes", { path: "/notes" })).rejects.toBeInstanceOf(
          CorruptPendingRenameError,
        );
        // A rename is refused too, so no index move can act on a journal that
        // was not understood.
        await expect(store.renameCollection("docs", "guides")).rejects.toBeInstanceOf(
          CorruptPendingRenameError,
        );
      }

      // Nothing was published, nothing was mirrored, and the journal is kept
      // for a deliberate repair instead of being deleted as "no journal".
      expect(readFileSync(configPath, "utf-8")).toBe(before);
      expect(getStoreCollections(store.internal.db).map(c => c.name)).toEqual(["docs"]);
      expect(documentCount(store.internal.db, "docs")).toBe(1);
      expect(documentCount(store.internal.db, "guides")).toBe(0);
      expect(existsSync(markerPath)).toBe(true);
    } finally {
      await store.close();
    }
  });

  test("an unreadable journal is not treated as absent", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const store = await createStore({ dbPath, configPath });
    try {
      // A directory at the journal's path is an existing-but-unreadable file
      // (EISDIR), not ENOENT: the mutation must not assume there is no journal.
      mkdirSync(join(dir, "index.yml.pending-rename.json"));
      await expect(store.addCollection("notes", { path: "/notes" })).rejects.toBeInstanceOf(
        CorruptPendingRenameError,
      );
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["docs"]);
      expect(getStoreCollections(store.internal.db).map(c => c.name)).toEqual(["docs"]);
    } finally {
      await store.close();
    }
  });
});

describe("rename target ownership", () => {
  test("an existing collection row with no documents is a valid target", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    const store = await createStore({ dbPath, configPath });
    try {
      // A row that owns no documents is an empty collection, not a populated
      // one. Driver row results are null (bun:sqlite) or undefined
      // (better-sqlite3), so "no row" must be tested loosely or every name
      // looks populated and the rename is refused with "already exists".
      upsertStoreCollection(store.internal.db, "empty", { path: "/empty" });

      expect(await store.renameCollection("docs", "empty")).toBe(true);
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["empty"]);
      expect(getStoreCollections(store.internal.db).map(c => c.name)).toEqual(["empty"]);
      expect(documentCount(store.internal.db, "empty")).toBe(1);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
    } finally {
      await store.close();
    }
  });
});

describe("pending rename inspection", () => {
  test("reads the exact pair without mutating and fails closed when corrupt", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });

    const scope = new CollectionConfigScope({ configPath });
    try {
      expect(scope.pendingRename()).toBeUndefined();

      expect(() =>
        scope.renameCollection("docs", "guides", () => {
          throw new Error("index commit failed");
        }),
      ).toThrow("index commit failed");
      expect(scope.pendingRename()).toEqual({ from: "docs", to: "guides" });

      // A caller deciding whether an operation has anything to finish gets the
      // journal's verdict, never a silent "none".
      writeFileSync(join(dir, "index.yml.pending-rename.json"), "{ broken");
      expect(() => scope.pendingRename()).toThrow(CorruptPendingRenameError);
    } finally {
      scope.close();
    }
  });
});

describe("pending rename recovery before publication", () => {
  test("a journal whose publish never landed is finished by publishing first", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const dbPath = join(dir, "index.sqlite");
    writeConfig(configPath, { collections: { docs: { path: "/docs", pattern: "**/*.md" } } });
    // The journal was written, then the publication failed: the file still
    // lists the old name, which is what distinguishes this from a published
    // rename awaiting its index half.
    writeFileSync(
      join(dir, "index.yml.pending-rename.json"),
      JSON.stringify({ from: "docs", to: "guides" }),
    );

    const seed = createInternalStore(dbPath);
    try {
      indexDocument(seed, "docs", "guide.md", "# Guide\n\nbody");
    } finally {
      seed.close();
    }

    const store = await createStore({ dbPath, configPath });
    try {
      // Repeating the rename publishes it and only then moves the index state.
      expect(await store.renameCollection("docs", "guides")).toBe(true);
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["guides"]);
      expect(documentCount(store.internal.db, "guides")).toBe(1);
      expect(documentCount(store.internal.db, "docs")).toBe(0);
      expect(pendingRenameMarkers(dir)).toEqual([]);
    } finally {
      await store.close();
    }
  });
});