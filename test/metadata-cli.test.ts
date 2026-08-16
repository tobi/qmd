/**
 * metadata-cli.test.ts - CLI --filter integration: parsing, propagation,
 * JSON output metadata, and error behavior. Spawns real qmd processes.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdtemp, rm, writeFile, mkdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const qmdCommand = isBunRuntime
  ? { command: process.execPath, args: [qmdScript] }
  : { command: process.execPath, args: [tsxCli, qmdScript] };

let testDir: string;
let fixturesDir: string;
let dbPath: string;
let configDir: string;

async function runQmd(args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const proc = spawn(qmdCommand.command, [...qmdCommand.args, ...args], {
    cwd: fixturesDir,
    env: {
      ...process.env,
      INDEX_PATH: dbPath,
      QMD_CONFIG_DIR: configDir,
      PWD: fixturesDir,
    },
  });

  let stdout = "";
  let stderr = "";
  proc.stdout.on("data", chunk => { stdout += chunk; });
  proc.stderr.on("data", chunk => { stderr += chunk; });

  const exitCode = await new Promise<number>(resolve => {
    proc.on("close", code => resolve(code ?? -1));
  });
  return { stdout, stderr, exitCode };
}

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-metadata-cli-"));
  fixturesDir = join(testDir, "fixtures");
  configDir = join(testDir, "config");
  dbPath = join(testDir, "index.sqlite");
  await mkdir(fixturesDir, { recursive: true });
  await mkdir(configDir, { recursive: true });

  await writeFile(join(fixturesDir, "published.md"), [
    "---",
    "qmd:",
    "  metadata:",
    "    status: published",
    "    topics: [typescript, programming]",
    "---",
    "",
    "# Published doc",
    "",
    "cli filter keyword body",
    "",
  ].join("\n"));
  await writeFile(join(fixturesDir, "draft.md"), [
    "---",
    "qmd:",
    "  metadata:",
    "    status: draft",
    "---",
    "",
    "# Draft doc",
    "",
    "cli filter keyword body",
    "",
  ].join("\n"));

  const addResult = await runQmd(["collection", "add", ".", "--name", "notes"]);
  expect(addResult.exitCode).toBe(0);
}, 60000);

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

describe("qmd search --filter", () => {
  test("returns only matching documents and includes metadata in JSON output", async () => {
    const { stdout, exitCode } = await runQmd([
      "search", "cli filter keyword",
      "--format", "json",
      "--filter", '{"key":"status","operator":"eq","value":"published"}',
    ]);
    expect(exitCode).toBe(0);

    const results = JSON.parse(stdout);
    expect(results.length).toBe(1);
    expect(results[0].file).toBe("qmd://notes/published.md");
    expect(results[0].metadata).toEqual({ status: "published", topics: ["typescript", "programming"] });
  }, 30000);

  test("supports nested filters", async () => {
    const filter = JSON.stringify({
      operator: "and",
      operands: [
        { key: "topics", operator: "all", value: ["typescript", "programming"] },
        { operator: "not", operand: { key: "status", operator: "eq", value: "draft" } },
      ],
    });
    const { stdout, exitCode } = await runQmd([
      "search", "cli filter keyword", "--format", "json", "--filter", filter,
    ]);
    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout).map((r: { file: string }) => r.file)).toEqual(["qmd://notes/published.md"]);
  }, 30000);

  test("returns format-safe empty output when nothing matches", async () => {
    const { stdout, exitCode } = await runQmd([
      "search", "cli filter keyword",
      "--format", "json",
      "--filter", '{"key":"status","operator":"eq","value":"missing"}',
    ]);
    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout)).toEqual([]);
  }, 30000);

  test("omits metadata from JSON output when a document has none", async () => {
    await writeFile(join(fixturesDir, "plain.md"), "# Plain doc\n\ncli filter keyword body\n");
    const updateResult = await runQmd(["update"]);
    expect(updateResult.exitCode).toBe(0);

    const { stdout, exitCode } = await runQmd(["search", "cli filter keyword", "--format", "json"]);
    expect(exitCode).toBe(0);

    const results = JSON.parse(stdout);
    const plainResult = results.find((r: { file: string }) => r.file === "qmd://notes/plain.md");
    expect(plainResult).toBeDefined();
    expect(plainResult.metadata).toBeUndefined();
  }, 60000);

  test("rejects malformed --filter JSON with a non-zero exit", async () => {
    const { stderr, exitCode } = await runQmd([
      "search", "cli filter keyword", "--filter", "{not json",
    ]);
    expect(exitCode).toBe(1);
    expect(stderr).toMatch(/Invalid --filter JSON/);
  }, 30000);

  test("rejects valid JSON with an invalid filter AST", async () => {
    const { stderr, exitCode } = await runQmd([
      "search", "cli filter keyword", "--filter", '{"key":"status","operator":"equal","value":"x"}',
    ]);
    expect(exitCode).toBe(1);
    expect(stderr).toMatch(/Invalid metadata filter at \$/);
    expect(stderr).toMatch(/unknown operator 'equal'/);
  }, 30000);
});
