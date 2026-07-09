/**
 * Unit tests for collection config path resolution (PR #190).
 *
 * Tests that getConfigDir() respects XDG_CONFIG_HOME, QMD_CONFIG_DIR,
 * and falls back to ~/.config/qmd.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtemp, rm, writeFile } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";
import { qmdHomedir } from "../src/paths.js";
import {
  getConfigPath,
  loadConfig,
  saveConfig,
  setConfigIndexName,
  validateEmbedConfig,
  type CollectionConfig,
} from "../src/collections.js";

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
});

describe("models.embed config round-trip", () => {
  async function withTempConfigDir(fn: () => void): Promise<void> {
    const dir = await mkdtemp(join(tmpdir(), "qmd-embed-config-"));
    try {
      process.env.QMD_CONFIG_DIR = dir;
      fn();
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  }

  test("array embed config round-trips through saveConfig/loadConfig preserving order", async () => {
    await withTempConfigDir(() => {
      const config: CollectionConfig = {
        collections: {},
        models: {
          embed: [
            "http://host-a:1234/v1#test-embed-model",
            "http://host-b:1234/v1#test-embed-model",
            "hf:user/repo/model.gguf",
          ],
        },
      };
      saveConfig(config);
      const loaded = loadConfig();
      expect(loaded.models?.embed).toEqual([
        "http://host-a:1234/v1#test-embed-model",
        "http://host-b:1234/v1#test-embed-model",
        "hf:user/repo/model.gguf",
      ]);
    });
  });

  test("single string embed config still round-trips exactly as before", async () => {
    await withTempConfigDir(() => {
      const config: CollectionConfig = {
        collections: {},
        models: {
          embed: "hf:user/repo/model.gguf",
          rerank: "hf:user/repo/rerank.gguf",
          generate: "hf:user/repo/generate.gguf",
        },
      };
      saveConfig(config);
      const loaded = loadConfig();
      expect(loaded.models?.embed).toBe("hf:user/repo/model.gguf");
      expect(loaded.models?.rerank).toBe("hf:user/repo/rerank.gguf");
      expect(loaded.models?.generate).toBe("hf:user/repo/generate.gguf");
    });
  });

  test("loadConfig rejects an on-disk empty embed array instead of silently defaulting", async () => {
    // Regression guard: validateEmbedConfig must actually run inside loadConfig,
    // not just exist as an unwired helper — an empty array is otherwise a
    // silent no-op (resolveEmbedEndpoints would fall through to the default
    // model without ever surfacing the malformed config to the user).
    await withTempConfigDir(() => {
      saveConfig({ collections: {}, models: { embed: [] } });
      expect(() => loadConfig()).toThrow(/must not be empty/);
    });
  });

  test("loadConfig rejects mismatched #model-id fragments in an on-disk embed array", async () => {
    await withTempConfigDir(() => {
      saveConfig({
        collections: {},
        models: { embed: ["http://host-a:1234/v1#test-embed-model", "http://host-b:1234/v1#other-model"] },
      });
      expect(() => loadConfig()).toThrow(/must share the same/);
    });
  });
});

describe("validateEmbedConfig", () => {
  test("passes for a single string", () => {
    expect(() => validateEmbedConfig("hf:user/repo/model.gguf")).not.toThrow();
  });

  test("passes for undefined", () => {
    expect(() => validateEmbedConfig(undefined)).not.toThrow();
  });

  test("passes for an array of remote URIs sharing the same #model-id", () => {
    expect(() =>
      validateEmbedConfig([
        "http://host-a:1234/v1#test-embed-model",
        "https://host-b:1234/v1#test-embed-model",
      ])
    ).not.toThrow();
  });

  test("throws for an empty array", () => {
    expect(() => validateEmbedConfig([])).toThrow(/must not be empty/);
  });

  test("throws for an array of remote URIs with mismatched #model-id fragments", () => {
    expect(() =>
      validateEmbedConfig([
        "http://host-a:1234/v1#test-embed-model",
        "http://host-b:1234/v1#other-model",
      ])
    ).toThrow(/must share the same/);
  });

  test("throws for a malformed http entry with no #fragment", () => {
    expect(() => validateEmbedConfig(["http://host-a:1234/v1"])).toThrow(/Malformed remote embed URI/);
  });
});
