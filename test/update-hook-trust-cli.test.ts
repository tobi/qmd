/**
 * End-to-end trust gate for project-local `update:` hooks (#886).
 *
 * Spawns real `qmd update` processes against a fixture that plays the part of a
 * freshly cloned repository shipping its own `.qmd/index.yml`.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const runnerArgs = isBunRuntime ? [qmdScript] : [tsxCli, qmdScript];

let projectDir: string;
let configDir: string;
/** Written by the hook if — and only if — it was allowed to run. */
let markerPath: string;

function runQmd(
  args: string[],
  env: Record<string, string> = {},
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [...runnerArgs, ...args], {
      cwd: projectDir,
      env: {
        ...process.env,
        QMD_CONFIG_DIR: configDir,
        PWD: projectDir,
        QMD_DOCTOR_DEVICE_PROBE: "0",
        ...env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });
    proc.on("error", reject);
    proc.on("close", (code) => resolve({ stdout, stderr, exitCode: code ?? 1 }));
  });
}

function writeLocalConfig(command: string): void {
  writeFileSync(
    join(projectDir, ".qmd", "index.yml"),
    [
      "collections:",
      "  docs:",
      "    path: ./docs",
      '    pattern: "**/*.md"',
      `    update: ${JSON.stringify(command)}`,
      "",
    ].join("\n"),
    "utf-8",
  );
}

beforeEach(() => {
  projectDir = mkdtempSync(join(tmpdir(), "qmd-hostile-repo-"));
  configDir = mkdtempSync(join(tmpdir(), "qmd-hostile-cfg-"));
  mkdirSync(join(projectDir, ".qmd"), { recursive: true });
  mkdirSync(join(projectDir, "docs"), { recursive: true });
  writeFileSync(join(projectDir, "docs", "readme.md"), "# Readme\n\nSome indexable content.\n", "utf-8");
  // Hooks run with the collection directory as cwd, so a relative redirect
  // keeps the fixture free of path-quoting concerns.
  markerPath = join(projectDir, "docs", "hook-ran.txt");
  writeLocalConfig("echo ran > hook-ran.txt");
});

afterEach(() => {
  for (const dir of [projectDir, configDir]) {
    // Best effort: Windows holds the SQLite WAL handles briefly after exit.
    try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
  }
});

describe("qmd update with a checked-in .qmd config", () => {
  test("does not run the repo's update command unattended, but still indexes", async () => {
    const result = await runQmd(["update"]);

    expect(existsSync(markerPath)).toBe(false);
    expect(result.stdout).toContain("defines update commands");
    expect(result.stdout).toContain("qmd trust");
    // The caller asked for an index refresh; they get one.
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.exitCode).toBe(0);
  }, 120_000);

  test("runs it after `qmd trust`", async () => {
    const trust = await runQmd(["trust"]);
    expect(trust.stdout).toContain("Trusted");

    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("Running update command");
    expect(existsSync(markerPath)).toBe(true);
  }, 120_000);

  test("re-arms the gate when the command changes after approval", async () => {
    await runQmd(["trust"]);
    // Stand-in for a `git pull` that rewrites the hook under an approval.
    writeLocalConfig("echo pwned > hook-ran.txt && echo extra");

    const result = await runQmd(["update"]);
    expect(existsSync(markerPath)).toBe(false);
    expect(result.stdout).toContain("defines update commands");
  }, 120_000);

  test("`qmd trust revoke` puts the gate back", async () => {
    await runQmd(["trust"]);
    const revoke = await runQmd(["trust", "revoke"]);
    expect(revoke.stdout).toContain("Revoked trust");

    const result = await runQmd(["update"]);
    expect(existsSync(markerPath)).toBe(false);
  }, 120_000);

  test("QMD_TRUST_UPDATE_HOOKS=1 opts unattended runs back in", async () => {
    const result = await runQmd(["update"], { QMD_TRUST_UPDATE_HOOKS: "1" });
    expect(result.stdout).toContain("Running update command");
    expect(existsSync(markerPath)).toBe(true);
  }, 120_000);

  test("`qmd trust list` reports the approved config", async () => {
    await runQmd(["trust"]);
    const list = await runQmd(["trust", "list"]);
    expect(list.stdout).toContain(join(projectDir, ".qmd", "index.yml"));
  }, 120_000);
});
