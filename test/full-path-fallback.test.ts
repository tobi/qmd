/**
 * `--full-path` fallback tests.
 *
 * `--full-path` swaps the `qmd://` URI + docid for the file's on-disk path.
 * When a result can't be resolved on disk — the file moved or was deleted
 * since the last index — it falls back to the URI. That fallback must:
 *   1. keep the docid, so the row is still addressable (search/query used to
 *      drop it, unlike get/multi-get), and
 *   2. say so on stderr, so the stale index is visible rather than silent.
 *
 * stdout must stay machine-clean in every format.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdir, mkdtemp, rename, rm, writeFile } from "fs/promises";
import { realpathSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");

async function runQmd(
  args: string[],
  opts: { cwd: string; dbPath: string; configDir: string }
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const runner = isBunRuntime
    ? { command: process.execPath, args: [qmdScript, ...args] }
    : { command: process.execPath, args: [tsxCli, qmdScript, ...args] };

  const proc = spawn(runner.command, runner.args, {
    cwd: opts.cwd,
    env: {
      ...process.env,
      INDEX_PATH: opts.dbPath,
      QMD_CONFIG_DIR: opts.configDir,
      PWD: opts.cwd,
      QMD_DOCTOR_DEVICE_PROBE: "0",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  let stdout = "";
  let stderr = "";
  proc.stdout?.on("data", (c: Buffer) => { stdout += c.toString(); });
  proc.stderr?.on("data", (c: Buffer) => { stderr += c.toString(); });
  const exitCode = await new Promise<number>((res, rej) => {
    proc.once("error", rej);
    proc.on("close", (code) => res(code ?? 1));
  });
  return { stdout, stderr, exitCode };
}

// The runtime prints unrelated Node deprecation notices on some versions;
// assert on our own warning rather than on stderr being empty.
const hasFullPathWarning = (stderr: string) =>
  /--full-path could not resolve/.test(stderr);

let testDir: string;
let collectionDir: string;
let dbPath: string;
let configDir: string;

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-full-path-"));
  const envDir = join(testDir, "env");
  collectionDir = join(envDir, "corpus");
  dbPath = join(envDir, "test.sqlite");
  configDir = join(envDir, "config");

  await mkdir(collectionDir, { recursive: true });
  await mkdir(configDir, { recursive: true });
  await writeFile(join(configDir, "index.yml"), "collections: {}\n");
  await writeFile(join(collectionDir, "alpha.md"), "# Alpha\n\nsearchterm-stale alpha\n");
  await writeFile(join(collectionDir, "beta.md"), "# Beta\n\nsearchterm-stale beta\n");
  collectionDir = realpathSync(collectionDir);

  const add = await runQmd(
    ["collection", "add", collectionDir, "--name", "stale"],
    { cwd: collectionDir, dbPath, configDir }
  );
  expect(add.exitCode, `collection add failed: ${add.stderr}`).toBe(0);

  // beta.md moves out of the collection: its row is now stale.
  await rename(join(collectionDir, "beta.md"), join(testDir, "beta-moved.md"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

describe("--full-path fallback for unresolvable results", () => {
  test("search --json keeps the docid on the row it could not resolve", async () => {
    const { stdout, stderr, exitCode } = await runQmd(
      ["search", "searchterm-stale", "--full-path", "--json"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);

    const results = JSON.parse(stdout) as Array<{ file: string; docid?: string }>;
    expect(results.length).toBe(2);

    const resolved = results.find((r) => !r.file.startsWith("qmd://"));
    const unresolved = results.find((r) => r.file.startsWith("qmd://"));
    expect(resolved, "alpha.md should resolve on disk").toBeDefined();
    expect(unresolved, "beta.md should fall back to its qmd:// URI").toBeDefined();

    // Resolved row: the path is the identifier, so no docid.
    expect(resolved!.file).toContain("alpha.md");
    expect(resolved!.docid).toBeUndefined();
    // Unresolved row: the docid is all that is left to address it by.
    expect(unresolved!.docid).toMatch(/^#[a-f0-9]{6}$/);

    expect(hasFullPathWarning(stderr)).toBe(true);
    expect(stderr).toContain("qmd update");
  });

  test("search --format csv always emits the docid column", async () => {
    const { stdout, exitCode } = await runQmd(
      ["search", "searchterm-stale", "--full-path", "--format", "csv"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);

    const lines = stdout.trim().split("\n");
    expect(lines[0]).toBe("docid,score,file,title,context,line,snippet");

    // Resolved rows leave the column empty; the unresolved row fills it. The
    // column count is the same either way, so positional parsing still works.
    const resolvedRow = lines.find((l) => l.includes("alpha.md"));
    const unresolvedRow = lines.find((l) => l.includes("qmd://stale/beta.md"));
    expect(resolvedRow).toBeDefined();
    expect(unresolvedRow).toBeDefined();
    expect(resolvedRow!.startsWith(",")).toBe(true);
    expect(unresolvedRow).toMatch(/^#[a-f0-9]{6},/);
  });

  test("search default CLI format keeps the docid next to the fallback URI", async () => {
    const { stdout, stderr, exitCode } = await runQmd(
      ["search", "searchterm-stale", "--full-path"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);
    // eslint-disable-next-line no-control-regex
    const plain = stdout.replace(/\x1b\[[0-9;]*m/g, "").replace(/\x1b\]8;;[^\x07]*\x07/g, "");

    const betaLine = plain.split("\n").find((l) => l.includes("qmd://stale/beta.md"));
    expect(betaLine, "beta should fall back to its qmd:// URI").toBeDefined();
    expect(betaLine).toMatch(/#[a-f0-9]{6}\s*$/);

    const alphaLine = plain.split("\n").find((l) => l.includes("alpha.md") && !l.startsWith("Title"));
    expect(alphaLine).toBeDefined();
    expect(alphaLine).not.toMatch(/#[a-f0-9]{6}/);

    expect(hasFullPathWarning(stderr)).toBe(true);
  });

  test("get warns and falls back to qmd:// + docid", async () => {
    const { stdout, stderr, exitCode } = await runQmd(
      ["get", "beta.md", "--full-path"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);
    expect(stdout.split("\n")[0]).toMatch(/^qmd:\/\/stale\/beta\.md {2}#[a-f0-9]{6}$/);
    expect(hasFullPathWarning(stderr)).toBe(true);
  });

  test("multi-get warns when a requested file is gone from disk", async () => {
    const { stdout, stderr, exitCode } = await runQmd(
      ["multi-get", "alpha.md,beta.md", "--full-path", "--format", "files"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);
    expect(stdout).toMatch(/#[a-f0-9]{6},qmd:\/\/stale\/beta\.md/);
    expect(hasFullPathWarning(stderr)).toBe(true);
  });

  test("no warning when every result resolves", async () => {
    const { stdout, stderr, exitCode } = await runQmd(
      ["search", "alpha", "--full-path", "--json"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);
    const results = JSON.parse(stdout) as Array<{ file: string; docid?: string }>;
    expect(results.length).toBe(1);
    expect(results[0]!.file).not.toMatch(/^qmd:\/\//);
    expect(results[0]!.docid).toBeUndefined();
    expect(hasFullPathWarning(stderr)).toBe(false);
  });

  test("without --full-path nothing changes and nothing is warned", async () => {
    const { stdout, stderr, exitCode } = await runQmd(
      ["search", "searchterm-stale", "--json"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);
    const results = JSON.parse(stdout) as Array<{ file: string; docid?: string }>;
    expect(results.length).toBe(2);
    for (const r of results) {
      expect(r.file).toMatch(/^qmd:\/\/stale\//);
      expect(r.docid).toMatch(/^#[a-f0-9]{6}$/);
    }
    expect(hasFullPathWarning(stderr)).toBe(false);
  });
});
