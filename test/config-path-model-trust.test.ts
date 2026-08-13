/**
 * Trust gate for project-local collection paths and model URIs (#889).
 *
 * A `.qmd/index.yml` arrives with a `git clone`. Paths that resolve outside
 * the project, and model URIs that are not the built-in defaults, must not
 * apply unattended just because someone typed `qmd update` or `qmd embed`.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync, rmSync, mkdirSync, symlinkSync } from "node:fs";
import {
  collectGatedSurfaces,
  configDigest,
  decideConfigGate,
  hookDigest,
  isCollectionPathInsideProject,
  isLocalConfigPath,
  projectRootFromConfigPath,
  resolveConfigCollectionPath,
  type GatedConfig,
} from "../src/trust.js";

let configDir: string;
const origConfigDir = process.env.QMD_CONFIG_DIR;

beforeEach(() => {
  configDir = mkdtempSync(join(tmpdir(), "qmd-trust-cfg-"));
  process.env.QMD_CONFIG_DIR = configDir;
});

afterEach(() => {
  if (origConfigDir !== undefined) process.env.QMD_CONFIG_DIR = origConfigDir;
  else delete process.env.QMD_CONFIG_DIR;
  try { rmSync(configDir, { recursive: true, force: true }); } catch { /* ignore */ }
});

function localConfigPath(): string {
  const project = mkdtempSync(join(tmpdir(), "qmd-trust-proj-"));
  mkdirSync(join(project, ".qmd"), { recursive: true });
  return join(project, ".qmd", "index.yml");
}

const defaults = {
  embed: "hf:default/embed.gguf",
  generate: "hf:default/generate.gguf",
  rerank: "hf:default/rerank.gguf",
};

describe("project root and collection path resolution", () => {
  test("project root is the directory that contains .qmd", () => {
    expect(projectRootFromConfigPath("/home/me/app/.qmd/index.yml")).toBe("/home/me/app");
  });

  test("relative paths resolve against the project root, not cwd", () => {
    const configPath = "/home/me/app/.qmd/index.yml";
    expect(resolveConfigCollectionPath(configPath, "./docs")).toBe("/home/me/app/docs");
    expect(resolveConfigCollectionPath(configPath, "notes")).toBe("/home/me/app/notes");
    expect(resolveConfigCollectionPath(configPath, ".")).toBe("/home/me/app");
  });

  test("absolute and ~ paths stay outside a typical project", () => {
    const configPath = localConfigPath();
    expect(isCollectionPathInsideProject(configPath, "./docs")).toBe(true);
    expect(isCollectionPathInsideProject(configPath, ".")).toBe(true);
    expect(isCollectionPathInsideProject(configPath, "/etc")).toBe(false);
    expect(isCollectionPathInsideProject(configPath, "../outside")).toBe(false);
    expect(isCollectionPathInsideProject(configPath, "~/notes")).toBe(false);
  });

  test("a symlink that escapes the project is treated as outside", () => {
    const configPath = localConfigPath();
    const project = projectRootFromConfigPath(configPath);
    const outside = mkdtempSync(join(tmpdir(), "qmd-trust-outside-"));
    const link = join(project, "escape");
    symlinkSync(outside, link);
    expect(isCollectionPathInsideProject(configPath, "./escape")).toBe(false);
    try { rmSync(outside, { recursive: true, force: true }); } catch { /* ignore */ }
  });
});

describe("collectGatedSurfaces", () => {
  test("in-project paths and default models are not gated", () => {
    const configPath = localConfigPath();
    const gated = collectGatedSurfaces(configPath, {
      collections: { docs: { path: "./docs", pattern: "**/*.md" } },
      models: { ...defaults },
    }, defaults);
    expect(gated.paths).toEqual([]);
    expect(gated.models).toEqual({});
    expect(gated.hooks).toEqual([]);
  });

  test("an out-of-project path is gated", () => {
    const configPath = localConfigPath();
    const gated = collectGatedSurfaces(configPath, {
      collections: {
        docs: { path: "./docs" },
        secrets: { path: "/etc" },
      },
    }, defaults);
    expect(gated.paths).toEqual([{ collection: "secrets", path: "/etc" }]);
  });

  test("custom model URIs are gated; defaults are not", () => {
    const configPath = localConfigPath();
    const gated = collectGatedSurfaces(configPath, {
      collections: {},
      models: {
        embed: "hf:evil/malware.gguf",
        generate: defaults.generate,
        rerank: "/tmp/evil.gguf",
      },
    }, defaults);
    expect(gated.models).toEqual({
      embed: "hf:evil/malware.gguf",
      rerank: "/tmp/evil.gguf",
    });
  });

  test("the user's own config is not gated for paths or models", () => {
    const global = "/home/me/.config/qmd/index.yml";
    expect(isLocalConfigPath(global)).toBe(false);
    const gated = collectGatedSurfaces(global, {
      collections: { notes: { path: "/etc", update: "git pull" } },
      models: { embed: "hf:evil/malware.gguf" },
    }, defaults);
    expect(gated.paths).toEqual([]);
    expect(gated.models).toEqual({});
    expect(gated.hooks).toEqual([{ collection: "notes", command: "git pull" }]);
  });
});

describe("configDigest", () => {
  test("equals hookDigest when only hooks are gated", () => {
    const gated: GatedConfig = {
      hooks: [{ collection: "docs", command: "git pull" }],
      paths: [],
      models: {},
    };
    expect(configDigest(gated)).toBe(hookDigest(gated.hooks));
  });

  test("changes when an out-of-project path is added", () => {
    const hooks = [{ collection: "docs", command: "git pull" }];
    const without: GatedConfig = { hooks, paths: [], models: {} };
    const withPath: GatedConfig = {
      hooks,
      paths: [{ collection: "secrets", path: "/etc" }],
      models: {},
    };
    expect(configDigest(withPath)).not.toBe(configDigest(without));
  });

  test("changes when a custom model URI is edited", () => {
    const a: GatedConfig = { hooks: [], paths: [], models: { embed: "hf:a/a.gguf" } };
    const b: GatedConfig = { hooks: [], paths: [], models: { embed: "hf:b/b.gguf" } };
    expect(configDigest(a)).not.toBe(configDigest(b));
  });

  test("is stable across collection ordering", () => {
    const a: GatedConfig = {
      hooks: [],
      paths: [
        { collection: "a", path: "/tmp/a" },
        { collection: "b", path: "/tmp/b" },
      ],
      models: {},
    };
    const b: GatedConfig = {
      hooks: [],
      paths: [
        { collection: "b", path: "/tmp/b" },
        { collection: "a", path: "/tmp/a" },
      ],
      models: {},
    };
    expect(configDigest(a)).toBe(configDigest(b));
  });
});

describe("decideConfigGate", () => {
  const project = "/home/me/project/.qmd/index.yml";
  const global = "/home/me/.config/qmd/index.yml";
  const noEnv = {} as NodeJS.ProcessEnv;
  const untrusted = () => false;
  const gated: GatedConfig = {
    hooks: [],
    paths: [{ collection: "secrets", path: "/etc" }],
    models: { embed: "hf:evil/malware.gguf" },
  };

  test("nothing gated needs no decision", () => {
    const decision = decideConfigGate({
      configPath: project,
      gated: { hooks: [], paths: [], models: {} },
      isInteractive: false,
      env: noEnv,
      trustedCheck: untrusted,
    });
    expect(decision.action).toBe("run");
  });

  test("the user's own config always applies", () => {
    const decision = decideConfigGate({
      configPath: global,
      gated,
      isInteractive: false,
      env: noEnv,
      trustedCheck: untrusted,
    });
    expect(decision.action).toBe("run");
  });

  test("an untrusted project config prompts on a terminal", () => {
    const decision = decideConfigGate({
      configPath: project,
      gated,
      isInteractive: true,
      env: noEnv,
      trustedCheck: untrusted,
    });
    expect(decision.action).toBe("prompt");
  });

  test("an untrusted project config is skipped with nobody to ask", () => {
    const decision = decideConfigGate({
      configPath: project,
      gated,
      isInteractive: false,
      env: noEnv,
      trustedCheck: untrusted,
    });
    expect(decision.action).toBe("skip");
  });

  test("a previously approved set runs unattended", () => {
    const decision = decideConfigGate({
      configPath: project,
      gated,
      isInteractive: false,
      env: noEnv,
      trustedCheck: (path, digest) => path === project && digest === configDigest(gated),
    });
    expect(decision.action).toBe("run");
  });

  test("editing an approved path re-arms the gate", () => {
    const approved = configDigest(gated);
    const decision = decideConfigGate({
      configPath: project,
      gated: {
        ...gated,
        paths: [{ collection: "secrets", path: "/var" }],
      },
      isInteractive: false,
      env: noEnv,
      trustedCheck: (_path, digest) => digest === approved,
    });
    expect(decision.action).toBe("skip");
  });

  test("QMD_TRUST_UPDATE_HOOKS opts unattended runs back in", () => {
    const decision = decideConfigGate({
      configPath: project,
      gated,
      isInteractive: false,
      env: { QMD_TRUST_UPDATE_HOOKS: "1" } as NodeJS.ProcessEnv,
      trustedCheck: untrusted,
    });
    expect(decision.action).toBe("run");
  });
});
