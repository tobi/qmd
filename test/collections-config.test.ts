/**
 * Unit tests for collection config path resolution (PR #190).
 *
 * Tests that getConfigDir() respects XDG_CONFIG_HOME, QMD_CONFIG_DIR,
 * and falls back to ~/.config/qmd.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, readFile, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import { qmdHomedir } from "../src/paths.js";
import { getConfigPath, loadConfig, saveModelsConfig, setConfigIndexName, setConfigSource } from "../src/collections.js";

// Save/restore env vars around each test
let savedEnv: Record<string, string | undefined>;

beforeEach(() => {
  savedEnv = {
    HOME: process.env.HOME,
    USERPROFILE: process.env.USERPROFILE,
    QMD_CONFIG_DIR: process.env.QMD_CONFIG_DIR,
    XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME,
  };
  // Reset index name to default
  setConfigIndexName("index");
});

afterEach(() => {
  // Reset index name to default (prevents leaking into other test files under bun test)
  setConfigIndexName("index");
  for (const [key, val] of Object.entries(savedEnv)) {
    if (val === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = val;
    }
  }
});

describe("getConfigDir via getConfigPath", () => {
  test("defaults to ~/.config/qmd when no env vars are set", () => {
    delete process.env.QMD_CONFIG_DIR;
    delete process.env.XDG_CONFIG_HOME;
    expect(getConfigPath()).toBe(join(qmdHomedir(), ".config", "qmd", "index.yml"));
  });

  test("uses the same USERPROFILE fallback as default DB path when HOME is unset", () => {
    delete process.env.HOME;
    delete process.env.QMD_CONFIG_DIR;
    delete process.env.XDG_CONFIG_HOME;
    process.env.USERPROFILE = "/Users/windows-user";

    expect(getConfigPath()).toBe(join("/Users/windows-user", ".config", "qmd", "index.yml"));
  });

  test("QMD_CONFIG_DIR takes highest priority", () => {
    process.env.QMD_CONFIG_DIR = "/custom/qmd-config";
    process.env.XDG_CONFIG_HOME = "/xdg/config";
    expect(getConfigPath()).toBe(join("/custom/qmd-config", "index.yml"));
  });

  test("XDG_CONFIG_HOME is used when QMD_CONFIG_DIR is not set", () => {
    delete process.env.QMD_CONFIG_DIR;
    process.env.XDG_CONFIG_HOME = "/xdg/config";
    expect(getConfigPath()).toBe(join("/xdg/config", "qmd", "index.yml"));
  });

  test("XDG_CONFIG_HOME appends qmd subdirectory", () => {
    delete process.env.QMD_CONFIG_DIR;
    process.env.XDG_CONFIG_HOME = "/home/agent/.config";
    expect(getConfigPath()).toBe(join("/home/agent/.config", "qmd", "index.yml"));
  });

  test("QMD_CONFIG_DIR overrides XDG_CONFIG_HOME", () => {
    process.env.QMD_CONFIG_DIR = "/override";
    process.env.XDG_CONFIG_HOME = "/should-not-use";
    expect(getConfigPath()).toBe(join("/override", "index.yml"));
  });

  test("respects custom index name", () => {
    delete process.env.QMD_CONFIG_DIR;
    process.env.XDG_CONFIG_HOME = "/xdg/config";
    setConfigIndexName("myindex");
    expect(getConfigPath()).toBe(join("/xdg/config", "qmd", "myindex.yml"));
  });

  test("sanitizes a Windows absolute index name into a safe filename", () => {
    delete process.env.QMD_CONFIG_DIR;
    process.env.XDG_CONFIG_HOME = "/xdg/config";
    setConfigIndexName("C:\\Users\\axulo\\Documents\\ppttest");
    expect(getConfigPath()).toBe(
      join("/xdg/config", "qmd", "C_Users_axulo_Documents_ppttest.yml")
    );
  });

  test("loadConfig treats an empty YAML file as an empty config", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-empty-config-"));
    try {
      process.env.QMD_CONFIG_DIR = dir;
      await writeFile(join(dir, "index.yml"), "");
      expect(loadConfig()).toEqual({ collections: {} });
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("saveModelsConfig keeps comments and quoting in an existing file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-models-config-"));
    const configPath = join(dir, "index.yml");
    try {
      await writeFile(configPath, [
        "# Load-bearing comment",
        'global_context: "demo project"',
        "collections:",
        "  docs:",
        "    path: ./docs  # relative to .qmd",
        '    pattern: "**/*.md"',
        "",
      ].join("\n"));
      setConfigSource({ configPath });

      saveModelsConfig({ embed: "hf:e", generate: "hf:g", rerank: "hf:r" });

      const written = await readFile(configPath, "utf-8");
      expect(written).toContain("# Load-bearing comment\n");
      expect(written).toContain('global_context: "demo project"');
      expect(written).toContain("# relative to .qmd");
      expect(loadConfig().models).toEqual({ embed: "hf:e", generate: "hf:g", rerank: "hf:r" });
    } finally {
      setConfigSource();
      await rm(dir, { recursive: true, force: true });
    }
  });

  test.each([
    ["an existing models block", "models:\n  # pinned for reproducible embeddings\n  embed: hf:mine\n", "# pinned for reproducible embeddings"],
    ["a bare models key", "models:\n  # embed: hf:later\n", "# embed: hf:later"],
  ])("saveModelsConfig fills in %s", async (_name, models, kept) => {
    const dir = await mkdtemp(join(tmpdir(), "qmd-models-config-"));
    const configPath = join(dir, "index.yml");
    try {
      await writeFile(configPath, `collections: {}\n${models}`);
      setConfigSource({ configPath });

      saveModelsConfig({ embed: "hf:e", generate: "hf:g", rerank: "hf:r" });

      expect(await readFile(configPath, "utf-8")).toContain(kept);
      expect(loadConfig().models).toEqual({ embed: "hf:e", generate: "hf:g", rerank: "hf:r" });
    } finally {
      setConfigSource();
      await rm(dir, { recursive: true, force: true });
    }
  });
});
