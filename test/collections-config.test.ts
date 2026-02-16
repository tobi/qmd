/**
 * Unit tests for collection config path resolution (PR #190).
 *
 * Tests that getConfigDir() respects XDG_CONFIG_HOME, QMD_CONFIG_DIR,
 * and falls back to ~/.config/qmd.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { join } from "path";
import { homedir } from "os";
import { getConfigPath, setConfigIndexName } from "../src/collections.js";

// Save/restore env vars around each test
let savedEnv: Record<string, string | undefined>;

beforeEach(() => {
  savedEnv = {
    QMD_CONFIG_DIR: process.env.QMD_CONFIG_DIR,
    XDG_CONFIG_HOME: process.env.XDG_CONFIG_HOME,
  };
  // Reset index name to default
  setConfigIndexName("index");
});

afterEach(() => {
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
    expect(getConfigPath()).toBe(join(homedir(), ".config", "qmd", "index.yml"));
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
});
