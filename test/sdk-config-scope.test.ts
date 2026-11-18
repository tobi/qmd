/**
 * Regression tests for per-store collection configuration ownership.
 *
 * Each SDK store owns one collection config scope. A store must write
 * collection/context mutations through to the config it was created with —
 * never to another store's config, and never to the process-global config the
 * CLI uses. Closing a store must release only its own scope.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import YAML from "yaml";
import {
  createStore,
  CollectionConfigScope,
  ConfigScopeClosedError,
  type CollectionConfig,
} from "../src/index.js";
import {
  getConfigPath,
  loadConfig,
  setConfigIndexName,
  setConfigSource,
} from "../src/collections.js";

let testDir: string;
let configDir: string;
let docsDir: string;
let notesDir: string;
let savedConfigDir: string | undefined;

beforeEach(async () => {
  savedConfigDir = process.env.QMD_CONFIG_DIR;
  testDir = await mkdtemp(join(tmpdir(), "qmd-config-scope-"));
  configDir = join(testDir, "global-config");
  docsDir = join(testDir, "docs");
  notesDir = join(testDir, "notes");
  await mkdir(docsDir, { recursive: true });
  await mkdir(notesDir, { recursive: true });

  // Keep the CLI-style default scope fully inside the test directory.
  process.env.QMD_CONFIG_DIR = configDir;
  setConfigIndexName("index");
  setConfigSource();
});

afterEach(async () => {
  setConfigSource();
  setConfigIndexName("index");
  if (savedConfigDir === undefined) {
    delete process.env.QMD_CONFIG_DIR;
  } else {
    process.env.QMD_CONFIG_DIR = savedConfigDir;
  }
  await rm(testDir, { recursive: true, force: true });
});

function dbPath(name: string): string {
  return join(testDir, `${name}.sqlite`);
}

function inlineConfig(): CollectionConfig {
  return { collections: {} };
}

async function writeYamlConfig(path: string, config: CollectionConfig): Promise<void> {
  await writeFile(path, YAML.stringify(config), "utf-8");
}

async function readYamlConfig(path: string): Promise<CollectionConfig> {
  return YAML.parse(await readFile(path, "utf-8")) as CollectionConfig;
}

describe("per-store collection config ownership", () => {
  test("two inline-config stores write through to their own config only", async () => {
    const configA = inlineConfig();
    const configB = inlineConfig();
    const storeA = await createStore({ dbPath: dbPath("inline-a"), config: configA });
    const storeB = await createStore({ dbPath: dbPath("inline-b"), config: configB });

    try {
      await storeA.addCollection("docs", { path: docsDir });
      expect(Object.keys(configA.collections)).toEqual(["docs"]);
      expect(Object.keys(configB.collections)).toEqual([]);
      expect((await storeB.listCollections()).map(c => c.name)).toEqual([]);

      await storeB.addCollection("notes", { path: notesDir });
      expect(Object.keys(configA.collections)).toEqual(["docs"]);
      expect(Object.keys(configB.collections)).toEqual(["notes"]);

      await storeA.addContext("docs", "/", "Docs context");
      await storeB.setGlobalContext("Notes context");
      expect(configA.collections.docs?.context).toEqual({ "/": "Docs context" });
      expect(configA.global_context).toBeUndefined();
      expect(configB.collections.notes?.context).toBeUndefined();
      expect(configB.global_context).toBe("Notes context");

      await storeA.renameCollection("docs", "guides");
      expect(Object.keys(configA.collections)).toEqual(["guides"]);
      expect(Object.keys(configB.collections)).toEqual(["notes"]);

      await storeB.removeCollection("notes");
      expect(Object.keys(configA.collections)).toEqual(["guides"]);
      expect(Object.keys(configB.collections)).toEqual([]);
    } finally {
      await storeA.close();
      await storeB.close();
    }
  });

  test("closing one store does not redirect another store's writes", async () => {
    const closedConfig = inlineConfig();
    const liveConfig = inlineConfig();
    const first = await createStore({ dbPath: dbPath("close-first"), config: closedConfig });
    const second = await createStore({ dbPath: dbPath("close-second"), config: liveConfig });

    try {
      await first.close();

      // With a shared global config source this write would land in the default
      // YAML file (or another store's config) instead of liveConfig.
      await second.addCollection("notes", { path: notesDir });

      expect(Object.keys(liveConfig.collections)).toEqual(["notes"]);
      expect(Object.keys(closedConfig.collections)).toEqual([]);
      expect(existsSync(join(configDir, "index.yml"))).toBe(false);
    } finally {
      await second.close();
    }
  });

  test("two YAML config stores keep their write-through files separate", async () => {
    const configPathA = join(testDir, "a.yml");
    const configPathB = join(testDir, "b.yml");
    await writeYamlConfig(configPathA, {
      collections: { docs: { path: docsDir, pattern: "**/*.md" } },
    });
    await writeYamlConfig(configPathB, {
      collections: { notes: { path: notesDir, pattern: "**/*.md" } },
    });

    const storeA = await createStore({ dbPath: dbPath("yaml-a"), configPath: configPathA });
    const storeB = await createStore({ dbPath: dbPath("yaml-b"), configPath: configPathB });

    try {
      await storeA.addCollection("guides", { path: docsDir });
      await storeA.addContext("guides", "/setup", "Setup docs");

      const fileA = await readYamlConfig(configPathA);
      const fileB = await readYamlConfig(configPathB);
      expect(Object.keys(fileA.collections).sort()).toEqual(["docs", "guides"]);
      expect(fileA.collections.guides?.context).toEqual({ "/setup": "Setup docs" });
      expect(Object.keys(fileB.collections)).toEqual(["notes"]);
      expect(fileB.collections.notes?.context).toBeUndefined();

      await storeB.removeCollection("notes");
      expect(Object.keys((await readYamlConfig(configPathA)).collections).sort()).toEqual([
        "docs",
        "guides",
      ]);
      expect(Object.keys((await readYamlConfig(configPathB)).collections)).toEqual([]);
    } finally {
      await storeA.close();
      await storeB.close();
    }
  });

  test("SDK stores never rebind the process-global config source", async () => {
    const globalPath = getConfigPath();
    expect(globalPath).toBe(join(configDir, "index.yml"));

    const storeConfigPath = join(testDir, "store.yml");
    await writeYamlConfig(storeConfigPath, { collections: {} });
    const store = await createStore({
      dbPath: dbPath("global-isolation"),
      configPath: storeConfigPath,
    });
    let closed = false;
    try {
      expect(getConfigPath()).toBe(globalPath);

      await store.addCollection("docs", { path: docsDir });
      expect(getConfigPath()).toBe(globalPath);
      expect(existsSync(globalPath)).toBe(false);
      expect(Object.keys((await readYamlConfig(storeConfigPath)).collections)).toEqual(["docs"]);

      await store.close();
      closed = true;

      // Closing a store must not reset the CLI's config source either.
      expect(getConfigPath()).toBe(globalPath);
      expect(loadConfig().collections).toEqual({});
    } finally {
      if (!closed) await store.close();
    }

    // The CLI-style default scope still binds explicitly, as before.
    setConfigSource({ configPath: storeConfigPath });
    expect(getConfigPath()).toBe(storeConfigPath);
    expect(Object.keys(loadConfig().collections)).toEqual(["docs"]);
  });

  test("a closed scope rejects every read and write", () => {
    const scope = new CollectionConfigScope({ config: { collections: {} } });
    scope.addCollection("docs", docsDir);
    scope.close();
    expect(scope.isClosed).toBe(true);

    expect(() => scope.load()).toThrow(ConfigScopeClosedError);
    expect(() => scope.addCollection("notes", notesDir)).toThrow(ConfigScopeClosedError);
    expect(() => scope.setGlobalContext("context")).toThrow(ConfigScopeClosedError);
    expect(() => scope.getConfigPath()).toThrow(ConfigScopeClosedError);

    // Closing again is a no-op; rebinding reopens the scope explicitly.
    scope.close();
    scope.setSource({ config: { collections: {} } });
    expect(scope.isClosed).toBe(false);
    expect(scope.listCollections()).toEqual([]);
  });

  test("a closed store never mutates its former config", async () => {
    const config: CollectionConfig = {
      collections: { docs: { path: docsDir, pattern: "**/*.md" } },
    };
    const store = await createStore({ dbPath: dbPath("closed-store"), config });

    await store.addContext("docs", "/", "Before close");
    expect(config.collections.docs?.context).toEqual({ "/": "Before close" });

    await store.close();
    const snapshot = JSON.stringify(config);

    await expect(store.addCollection("notes", { path: notesDir })).rejects.toThrow();
    expect(JSON.stringify(config)).toBe(snapshot);
  });
});
