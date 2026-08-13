/**
 * End-to-end trust gate for project-local collection paths and model URIs (#889).
 *
 * Spawns real `qmd update` processes against a fixture that plays the part of a
 * freshly cloned repository shipping its own `.qmd/index.yml`.
 */

import { describe, test, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, rmSync, readFileSync } from "node:fs";
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
let outsideDir: string;
let outsideFile: string;

function runQmd(
  args: string[],
  env: Record<string, string> = {},
  cwd: string = projectDir,
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve, reject) => {
    const proc = spawn(process.execPath, [...runnerArgs, ...args], {
      cwd,
      env: {
        ...process.env,
        QMD_CONFIG_DIR: configDir,
        PWD: cwd,
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

function writeLocalConfig(opts: { path: string; embed?: string }): void {
  const models = opts.embed
    ? ["models:", `  embed: ${JSON.stringify(opts.embed)}`, ""]
    : [""];
  writeFileSync(
    join(projectDir, ".qmd", "index.yml"),
    [
      "collections:",
      "  docs:",
      `    path: ${JSON.stringify(opts.path)}`,
      '    pattern: "**/*.md"',
      "",
      ...models,
    ].join("\n"),
    "utf-8",
  );
}

beforeEach(() => {
  projectDir = mkdtempSync(join(tmpdir(), "qmd-hostile-repo-"));
  configDir = mkdtempSync(join(tmpdir(), "qmd-hostile-cfg-"));
  outsideDir = mkdtempSync(join(tmpdir(), "qmd-hostile-outside-"));
  mkdirSync(join(projectDir, ".qmd"), { recursive: true });
  mkdirSync(join(projectDir, "docs"), { recursive: true });
  writeFileSync(join(projectDir, "docs", "readme.md"), "# Readme\n\nSome indexable content.\n", "utf-8");
  outsideFile = join(outsideDir, "secret.md");
  writeFileSync(outsideFile, "# Secret\n\nThis must not be indexed unattended.\n", "utf-8");
  writeLocalConfig({ path: outsideDir });
});

afterEach(() => {
  for (const dir of [projectDir, configDir, outsideDir]) {
    try { rmSync(dir, { recursive: true, force: true }); } catch { /* ignore */ }
  }
});

describe("qmd update with a checked-in collection path outside the project", () => {
  test("does not index the outside path unattended", async () => {
    const result = await runQmd(["update"]);

    expect(result.stdout).toContain("names collection paths outside the project");
    expect(result.stdout).toContain("qmd trust");
    expect(result.stdout).toContain("Skipping");
    expect(result.exitCode).toBe(0);

    const dbPath = join(projectDir, ".qmd", "index.sqlite");
    expect(existsSync(dbPath)).toBe(true);
    const db = readFileSync(dbPath);
    expect(db.includes("This must not be indexed unattended")).toBe(false);
  }, 120_000);

  test("indexes it after `qmd trust`", async () => {
    const trust = await runQmd(["trust"]);
    expect(trust.stdout).toContain("Trusted");

    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.stdout).not.toContain("Skipping — path");
  }, 120_000);

  test("re-arms the gate when the path changes after approval", async () => {
    await runQmd(["trust"]);
    const otherOutside = mkdtempSync(join(tmpdir(), "qmd-hostile-other-"));
    writeFileSync(join(otherOutside, "other.md"), "# Other\n\nAlso secret.\n", "utf-8");
    writeLocalConfig({ path: otherOutside });

    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("names collection paths outside the project");
    expect(result.stdout).toContain("Skipping");
    try { rmSync(otherOutside, { recursive: true, force: true }); } catch { /* ignore */ }
  }, 120_000);

  test("in-project paths still index without trust", async () => {
    writeLocalConfig({ path: "./docs" });
    const result = await runQmd(["update"]);
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.stdout).not.toContain("names collection paths outside the project");
    expect(result.exitCode).toBe(0);
  }, 120_000);

  test("QMD_TRUST_UPDATE_HOOKS=1 opts unattended path indexing back in", async () => {
    const result = await runQmd(["update"], { QMD_TRUST_UPDATE_HOOKS: "1" });
    expect(result.stdout).toContain("Indexed: 1 new");
    expect(result.stdout).not.toContain("Skipping — path");
  }, 120_000);
});

describe("qmd with checked-in custom model URIs", () => {
  test("does not apply an untrusted model URI", async () => {
    writeLocalConfig({ path: "./docs", embed: "hf:evil/malware.gguf" });
    const result = await runQmd(["status"]);
    expect(result.stdout).not.toContain("hf:evil/malware.gguf");
    expect(result.exitCode).toBe(0);
  }, 120_000);

  test("applies it after `qmd trust`", async () => {
    writeLocalConfig({ path: "./docs", embed: "hf:evil/malware.gguf" });
    const trust = await runQmd(["trust"]);
    expect(trust.stdout).toContain("Trusted");
    expect(trust.stdout).toContain("hf:evil/malware.gguf");

    const result = await runQmd(["status"]);
    expect(result.stdout).toContain("hf:evil/malware.gguf");
  }, 120_000);

});
