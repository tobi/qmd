/**
 * Consumer-contract tests for cross-process collection-config writes.
 *
 * Real user-visible mutations (adding collections, adding contexts, setting
 * global context through the public scope API) are executed by separate OS
 * processes against one config file, and the tests assert what a user observes:
 *
 *   - no mutation is lost when two qmd processes write at once (the YAML holds
 *     every change);
 *   - a writer's publish never exposes a truncated or half-written document to
 *     a reader, an editor, or an open reader fd (atomic replacement);
 *   - a crashed writer's lock is recovered, and a live foreign writer's lock is
 *     waited on or reported — never silently bypassed;
 *   - file permissions and symlink targets survive a mutation.
 *
 * Child processes run test/_helpers/config-mutation-child.ts so the lock is
 * exercised across real process boundaries, not simulated in one process.
 */
import { describe, test, expect, afterEach } from "vitest";
import { spawn, type ChildProcess } from "node:child_process";
import {
  chmodSync,
  closeSync,
  lstatSync,
  mkdtempSync,
  openSync,
  readFileSync,
  readSync,
  readdirSync,
  readlinkSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";
import {
  addCollection,
  addContext,
  CollectionConfigScope,
  ConfigLockTimeoutError,
  getConfigLockPath,
  loadConfig,
  NestedConfigMutationError,
  saveConfig,
  setConfigSource,
  setGlobalContext,
  type CollectionConfig,
} from "../src/collections.js";

const thisDir = dirname(fileURLToPath(import.meta.url));
const childScript = join(thisDir, "_helpers", "config-mutation-child.ts");
const tsxCli = join(thisDir, "..", "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function makeTempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "qmd-config-lock-"));
  tempDirs.push(dir);
  return dir;
}

function readConfig(configPath: string): CollectionConfig {
  const scope = new CollectionConfigScope({ configPath });
  try {
    return scope.load();
  } finally {
    scope.close();
  }
}

/**
 * Directory entries that are not the lock sidecar: the sidecar (and its WAL
 * siblings) is deliberately left on disk, so anything else here — a temp file,
 * a second config copy — is a leaked publish.
 */
function entriesBesidesSidecar(dir: string): string[] {
  return readdirSync(dir)
    .filter((name) => !name.endsWith(".lock.sqlite"))
    .filter((name) => !name.endsWith(".lock.sqlite-wal") && !name.endsWith(".lock.sqlite-shm"))
    .sort();
}

const LOCK_ENV_KEYS = [
  "QMD_CONFIG_LOCK_TIMEOUT_MS",
  "QMD_CONFIG_LOCK_STALE_MS",
] as const;

function saveLockEnv(): Record<string, string | undefined> {
  return Object.fromEntries(LOCK_ENV_KEYS.map((key) => [key, process.env[key]]));
}

function restoreLockEnv(saved: Record<string, string | undefined>): void {
  for (const key of LOCK_ENV_KEYS) {
    if (saved[key] === undefined) delete process.env[key];
    else process.env[key] = saved[key];
  }
}

/**
 * A mutation must fail closed — never proceed unlocked — while another process
 * holds the config's write lock. Uses a short budget so the assertion is quick.
 */
function expectMutationLockedOut(configPath: string, collectionName: string): void {
  const saved = saveLockEnv();
  process.env.QMD_CONFIG_LOCK_TIMEOUT_MS = "250";
  try {
    const scope = new CollectionConfigScope({ configPath });
    try {
      expect(() => scope.addCollection(collectionName, `/tmp/${collectionName}`)).toThrow(
        ConfigLockTimeoutError,
      );
    } finally {
      scope.close();
    }
  } finally {
    restoreLockEnv(saved);
  }
}

type Child = {
  proc: ChildProcess;
  output: () => string;
  send: (line: string) => void;
  /** Resolves once the child's combined output contains `needle`. */
  waitFor: (needle: string, timeoutMs?: number) => Promise<void>;
  done: Promise<number | null>;
};

/** Spawn the config mutator helper; trailing args are op-specific. */
function spawnChild(
  configPath: string,
  args: string[],
  env: Record<string, string> = {},
): Child {
  const scriptArgs = isBunRuntime
    ? [childScript, configPath, ...args]
    : [tsxCli, childScript, configPath, ...args];
  const proc = spawn(process.execPath, scriptArgs, {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, ...env },
  });

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

  const waitFor = (needle: string, timeoutMs = 20_000): Promise<void> => {
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

async function runChild(configPath: string, args: string[]): Promise<void> {
  const child = spawnChild(configPath, args);
  const code = await child.done;
  expect(child.output()).toContain("DONE");
  expect(code).toBe(0);
}

/** Spawn barrier-synchronised children and fire them together. */
async function runTogether(
  jobs: Array<{ configPath: string; args: string[]; display: string }>,
): Promise<void> {
  const children = jobs.map((job) =>
    spawnChild(job.configPath, job.args, { CONFIG_CHILD_BARRIER: "1" }),
  );
  try {
    await Promise.all(children.map((child) => child.waitFor("READY")));
    for (const child of children) child.send("GO");

    const codes = await Promise.all(children.map((child) => child.done));
    children.forEach((child, index) => {
      expect(child.output(), jobs[index]!.display).toContain("DONE");
      expect(codes[index], jobs[index]!.display).toBe(0);
    });
  } finally {
    // A failed assertion must not leave writers holding the lock for later tests.
    for (const child of children) child.proc.kill("SIGKILL");
  }
}

describe("CollectionConfigScope writes across processes", () => {
  test("keeps every collection added by a separate process", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n");

    const names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"];
    await runTogether(
      names.map((name) => ({
        configPath,
        args: ["add-collection", name, `/tmp/${name}`],
        display: `add ${name}`,
      })),
    );

    const config = readConfig(configPath);
    expect(Object.keys(config.collections).sort()).toEqual(["docs", ...names].sort());
    for (const name of names) {
      expect(config.collections[name]!.path).toBe(`/tmp/${name}`);
    }
    // The pre-existing collection and its file are intact, and no lock or
    // temp file outlived the writers.
    expect(config.collections.docs!.path).toBe("/docs");
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml"]);
  });

  test("keeps every context added by a separate process", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n");

    const prefixes = ["/api", "/guides", "/guides/admin", "/notes", "/specs"];
    await runTogether(
      prefixes.map((prefix) => ({
        configPath,
        args: ["add-context", "docs", prefix, `context for ${prefix}`],
        display: `context ${prefix}`,
      })),
    );

    const config = readConfig(configPath);
    const contexts = config.collections.docs!.context ?? {};
    for (const prefix of prefixes) {
      expect(contexts[prefix]).toBe(`context for ${prefix}`);
    }
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml"]);
  });

  test("holds the lock across another process's mutation instead of losing it", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n");

    // This child holds the config's sidecar write transaction inside update(),
    // reports HOLDING, and only then writes. A second process must wait for it:
    // without the lock it would read the pre-hold config, publish its own
    // collection, and the holder would publish a stale base that drops it.
    const holder = spawnChild(configPath, [
      "hold-update",
      "held",
      "/tmp/held",
      "after-hold",
    ]);
    try {
      await holder.waitFor("HOLDING");
      expect(lstatSync(getConfigLockPath(configPath)).isFile()).toBe(true);
      // Nothing is published while the mutator itself is still running.
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["docs"]);

      const contender = spawnChild(configPath, [
        "add-collection",
        "contender",
        "/tmp/contender",
      ]);
      // ATTEMPT means the contender is inside addCollection, queued on the
      // sidecar's write lock. It cannot report DONE or publish before release.
      await contender.waitFor("ATTEMPT");
      expect(contender.output()).not.toContain("DONE");
      expect(Object.keys(readConfig(configPath).collections)).toEqual(["docs"]);

      holder.send("GO");
      expect(await holder.done).toBe(0);
      expect(holder.output()).toContain("DONE");

      expect(await contender.done).toBe(0);
      expect(contender.output()).toContain("DONE");
    } finally {
      holder.proc.kill("SIGKILL");
    }

    const config = readConfig(configPath);
    expect(Object.keys(config.collections).sort()).toEqual([
      "after-hold",
      "contender",
      "docs",
      "held",
    ]);
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml"]);
  });

  test("publishes atomically: an open reader keeps the previous document", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(
      configPath,
      "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n    context:\n      /: documentation\n",
    );
    const original = readFileSync(configPath, "utf-8");
    const originalSize = statSync(configPath).size;
    const fd = openSync(configPath, "r");
    try {
      await runChild(configPath, ["add-collection", "fresh", "/tmp/fresh"]);

      // The reader's fd still refers to the previous inode: an in-place write
      // would have truncated that inode, so this returns the complete old YAML.
      const buffer = Buffer.alloc(originalSize);
      let total = 0;
      while (total < originalSize) {
        const read = readSync(fd, buffer, total, originalSize - total, total);
        if (read === 0) break;
        total += read;
      }
      expect(total).toBe(originalSize);
      expect(buffer.toString("utf-8")).toBe(original);
    } finally {
      closeSync(fd);
    }

    const published = readFileSync(configPath, "utf-8");
    expect(published).toContain("fresh");
    const publishedConfig = YAML.parse(published) as CollectionConfig;
    expect(publishedConfig.collections.docs!.path).toBe("/docs");
  });

  test("readers never observe a partial file while a process rewrites it", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n");

    const writer = spawnChild(configPath, ["stress", "stress", "120"]);
    await writer.waitFor("ATTEMPT");

    // Read concurrently with the writer process (a separate OS process, so it
    // keeps publishing while this loop runs). An in-place writer would expose a
    // truncated file here; an atomic replace exposes only complete documents.
    for (let reads = 0; reads < 400; reads++) {
      const content = readFileSync(configPath, "utf-8");
      expect(content.length).toBeGreaterThan(0);
      const parsed = YAML.parse(content) as CollectionConfig | null;
      expect(typeof parsed?.collections).toBe("object");
    }

    expect(await writer.done).toBe(0);
    expect(writer.output()).toContain("DONE");
    expect(Object.keys(readConfig(configPath).collections)).toEqual(["docs"]);
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml"]);
  });

  test("a live holder stays exclusive past any stale threshold and is never unlinked", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections: {}\n");
    const lockPath = getConfigLockPath(configPath);

    const holder = spawnChild(configPath, ["hold-update", "held", "/tmp/held"]);
    const saved = saveLockEnv();
    try {
      await holder.waitFor("HOLDING");
      const before = statSync(lockPath);

      // `QMD_CONFIG_LOCK_STALE_MS` is the knob the former pidfile lock honoured.
      // No age may retire a live holder, so with it set to 1 ms the contender
      // still fails closed instead of stealing the lock and writing over it.
      process.env.QMD_CONFIG_LOCK_STALE_MS = "1";

      expectMutationLockedOut(configPath, "blocked");

      // Nothing was published, and the sidecar was neither unlinked nor
      // replaced by the would-be reclaimer.
      expect(readFileSync(configPath, "utf-8")).toBe("collections: {}\n");
      expect(statSync(lockPath).ino).toBe(before.ino);

      holder.send("GO");
      expect(await holder.done).toBe(0);
      expect(holder.output()).toContain("DONE");

      process.env.QMD_CONFIG_LOCK_TIMEOUT_MS = "5000";
      const scope = new CollectionConfigScope({ configPath });
      try {
        scope.addCollection("after", "/tmp/after");
      } finally {
        scope.close();
      }
      // Publishing and releasing do not remove the sidecar either.
      expect(statSync(lockPath).ino).toBe(before.ino);
    } finally {
      restoreLockEnv(saved);
      holder.proc.kill("SIGKILL");
    }

    expect(Object.keys(readConfig(configPath).collections).sort()).toEqual(["after", "held"]);
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml"]);
  });

  test("releases the lock when a writer crashes mid-transaction", async () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections: {}\n");
    const lockPath = getConfigLockPath(configPath);

    const holder = spawnChild(configPath, ["hold-update", "never", "/tmp/never"]);
    const saved = saveLockEnv();
    try {
      await holder.waitFor("HOLDING");
      const before = statSync(lockPath);
      expectMutationLockedOut(configPath, "blocked");

      // SIGKILL: the kernel drops the SQLite write transaction with the
      // process. A short budget proves the release came from the kernel, not
      // from a stale-lock timeout.
      holder.proc.kill("SIGKILL");
      await holder.done;
      process.env.QMD_CONFIG_LOCK_TIMEOUT_MS = "2000";
      const scope = new CollectionConfigScope({ configPath });
      try {
        scope.addCollection("recovered", "/tmp/recovered");
      } finally {
        scope.close();
      }
      expect(statSync(lockPath).ino).toBe(before.ino);
    } finally {
      restoreLockEnv(saved);
      holder.proc.kill("SIGKILL");
    }

    expect(Object.keys(readConfig(configPath).collections)).toEqual(["recovered"]);
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml"]);
  });

  test("preserves file permissions across a mutation", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections: {}\n");
    chmodSync(configPath, 0o640);

    const scope = new CollectionConfigScope({ configPath });
    try {
      scope.addCollection("docs", "/docs");
    } finally {
      scope.close();
    }

    expect(statSync(configPath).mode & 0o777).toBe(0o640);
  });

  test("writes through a symlinked config instead of replacing the link", () => {
    const dir = makeTempDir();
    const realPath = join(dir, "real.yml");
    const linkPath = join(dir, "index.yml");
    writeFileSync(realPath, "collections: {}\n");
    symlinkSync("real.yml", linkPath);

    const scope = new CollectionConfigScope({ configPath: linkPath });
    try {
      scope.addCollection("linked", "/tmp/linked");
    } finally {
      scope.close();
    }

    expect(lstatSync(linkPath).isSymbolicLink()).toBe(true);
    expect(readlinkSync(linkPath)).toBe("real.yml");
    expect(readConfig(realPath).collections.linked!.path).toBe("/tmp/linked");
    expect(readConfig(linkPath).collections.linked!.path).toBe("/tmp/linked");
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml", "real.yml"]);
  });

  test("a symlink and its target share one lock", async () => {
    const dir = makeTempDir();
    const realPath = join(dir, "real.yml");
    const linkPath = join(dir, "index.yml");
    writeFileSync(realPath, "collections: {}\n");
    symlinkSync("real.yml", linkPath);

    expect(getConfigLockPath(linkPath)).toBe(getConfigLockPath(realPath));

    // One writer addresses the symlink, the other its target: the shared lock
    // (not a per-path-name lock) is what keeps both changes.
    await runTogether([
      {
        configPath: realPath,
        args: ["add-collection", "via-real", "/tmp/via-real"],
        display: "add via real path",
      },
      {
        configPath: linkPath,
        args: ["add-collection", "via-link", "/tmp/via-link"],
        display: "add via symlink",
      },
    ]);

    expect(Object.keys(readConfig(realPath).collections).sort()).toEqual(["via-link", "via-real"]);
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml", "real.yml"]);
  });

  test("leaves inline configs in memory and touches no file", () => {
    const dir = makeTempDir();
    const savedConfigDir = process.env.QMD_CONFIG_DIR;
    process.env.QMD_CONFIG_DIR = dir;
    const inline: CollectionConfig = { collections: {} };
    try {
      // The CLI's default scope bound to an SDK-style inline config: mutations
      // must stay in the caller's object, never lock or write a YAML file.
      setConfigSource({ config: inline });
      addCollection("memory", "/tmp/memory");
      addContext("memory", "/", "in memory");
      setGlobalContext("global");
      expect(inline.collections.memory!.path).toBe("/tmp/memory");
      expect(inline.collections.memory!.context).toEqual({ "/": "in memory" });
      expect(inline.global_context).toBe("global");
      expect(loadConfig()).toBe(inline);

      const replacement: CollectionConfig = { collections: {}, global_context: "next" };
      saveConfig(replacement);
      expect(loadConfig()).toBe(replacement);
    } finally {
      setConfigSource();
      if (savedConfigDir === undefined) delete process.env.QMD_CONFIG_DIR;
      else process.env.QMD_CONFIG_DIR = savedConfigDir;
    }

    expect(readdirSync(dir)).toEqual([]);
  });

  test("update() reads the current file and publishes the caller's change", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    writeFileSync(configPath, "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n");

    const scope = new CollectionConfigScope({ configPath });
    try {
      const published = scope.update((config) => {
        config.collections.docs!.includeByDefault = false;
        config.models = { embed: "hf:test/embed" };
      });
      expect(published.collections.docs!.includeByDefault).toBe(false);
    } finally {
      scope.close();
    }

    const config = readConfig(configPath);
    expect(config.collections.docs!.includeByDefault).toBe(false);
    expect(config.models?.embed).toBe("hf:test/embed");
    expect(entriesBesidesSidecar(dir)).toEqual(["index.yml"]);
  });

  test("rejects a nested mutation of the same config before any write", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const original = "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n";
    writeFileSync(configPath, original);

    const scope = new CollectionConfigScope({ configPath });
    // Any other scope bound to the same file is the same transaction target.
    const otherScope = new CollectionConfigScope({ configPath });
    try {
      // Same scope: update() inside update(), and a helper inside update().
      expect(() => scope.update(() => scope.update(() => {}))).toThrow(NestedConfigMutationError);
      expect(() => scope.update(() => scope.addCollection("inner", "/inner"))).toThrow(
        NestedConfigMutationError,
      );
      // Another scope on the same file is the same target.
      expect(() =>
        scope.update((config) => {
          config.collections.outer = { path: "/outer", pattern: "**/*.md" };
          // A nested mutation would publish its own snapshot; the enclosing
          // update would then overwrite that publish with its older one.
          otherScope.addCollection("inner", "/inner");
        }),
      ).toThrow(NestedConfigMutationError);

      // Rejected before writing: neither the outer edit nor the nested one landed.
      expect(readFileSync(configPath, "utf-8")).toBe(original);

      // The failed transaction released its state; the scope still works.
      scope.addCollection("after", "/after");
    } finally {
      scope.close();
      otherScope.close();
    }

    expect(Object.keys(readConfig(configPath).collections).sort()).toEqual(["after", "docs"]);
  });

  test("rejects save() from inside an update instead of overwriting it", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const original = "collections:\n  docs:\n    path: /docs\n    pattern: \"**/*.md\"\n";
    writeFileSync(configPath, original);

    const scope = new CollectionConfigScope({ configPath });
    const otherScope = new CollectionConfigScope({ configPath });
    try {
      expect(() =>
        scope.update((config) => {
          config.collections.outer = { path: "/outer", pattern: "**/*.md" };
          otherScope.save({ collections: {} });
        }),
      ).toThrow(NestedConfigMutationError);
      expect(readFileSync(configPath, "utf-8")).toBe(original);

      scope.addCollection("after", "/after");
    } finally {
      scope.close();
      otherScope.close();
    }

    expect(Object.keys(readConfig(configPath).collections).sort()).toEqual(["after", "docs"]);
  });

  test("allows a nested mutation of a different config file", () => {
    const dir = makeTempDir();
    const configPath = join(dir, "index.yml");
    const otherPath = join(dir, "other.yml");
    writeFileSync(configPath, "collections: {}\n");
    writeFileSync(otherPath, "collections: {}\n");

    const scope = new CollectionConfigScope({ configPath });
    const otherScope = new CollectionConfigScope({ configPath: otherPath });
    try {
      scope.update((config) => {
        config.collections.outer = { path: "/outer", pattern: "**/*.md" };
        // Different lock, different transaction: independent and allowed.
        otherScope.addCollection("elsewhere", "/elsewhere");
      });
    } finally {
      scope.close();
      otherScope.close();
    }

    expect(Object.keys(readConfig(configPath).collections)).toEqual(["outer"]);
    expect(Object.keys(readConfig(otherPath).collections)).toEqual(["elsewhere"]);
  });
});
