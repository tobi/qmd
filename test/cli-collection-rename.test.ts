/**
 * Isolated CLI regression for `qmd collection rename` / `collection remove`.
 *
 * Seeds only a temporary YAML config and a directory: no `qmd collection add`,
 * `qmd update` or `qmd embed` runs, so the commands under test operate on an
 * empty index plus whatever the config names. `INDEX_PATH` and `QMD_CONFIG_DIR`
 * keep the whole run inside the test's temp directory.
 *
 * Asserts what a user observes: the exit code, the printed diagnostics, and the
 * YAML the commands leave behind — including the two refusals (an existing
 * target, a name nothing knows) that must not modify the config.
 */
import { describe, test, expect, afterEach } from "vitest";
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import YAML from "yaml";
import type { CollectionConfig } from "../src/collections.js";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function makeTempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "qmd-cli-collection-"));
  tempDirs.push(dir);
  return dir;
}

function writeConfig(configPath: string, config: CollectionConfig): void {
  writeFileSync(configPath, YAML.stringify(config), "utf-8");
}

function readConfig(configPath: string): CollectionConfig {
  return YAML.parse(readFileSync(configPath, "utf-8")) as CollectionConfig;
}

type Layout = {
  configDir: string;
  configPath: string;
  dbPath: string;
  cwd: string;
  /** Source directory seeded for each collection name. */
  paths: Record<string, string>;
};

type Run = { stdout: string; stderr: string; exitCode: number };

/** A temp config dir, index path and one source directory per collection. */
function seedLayout(names: string[]): Layout {
  const dir = makeTempDir();
  const configDir = join(dir, "config");
  mkdirSync(configDir, { recursive: true });

  const paths: Record<string, string> = {};
  const collections: CollectionConfig["collections"] = {};
  for (const name of names) {
    const source = join(dir, "collections", name);
    mkdirSync(source, { recursive: true });
    paths[name] = source;
    collections[name] = { path: source, pattern: "**/*.md" };
  }

  const configPath = join(configDir, "index.yml");
  writeConfig(configPath, { collections });
  return { configDir, configPath, dbPath: join(dir, "index.sqlite"), cwd: dir, paths };
}

/** Run one real qmd invocation against a throwaway config and index. */
function runQmd(args: string[], layout: Layout): Promise<Run> {
  const scriptArgs = isBunRuntime ? [qmdScript, ...args] : [tsxCli, qmdScript, ...args];
  const proc = spawn(process.execPath, scriptArgs, {
    cwd: layout.cwd,
    env: {
      ...process.env,
      INDEX_PATH: layout.dbPath,
      QMD_CONFIG_DIR: layout.configDir,
      PWD: layout.cwd,
      QMD_DOCTOR_DEVICE_PROBE: "0",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  const stdout: string[] = [];
  const stderr: string[] = [];
  proc.stdout?.on("data", (chunk: Buffer) => stdout.push(chunk.toString()));
  proc.stderr?.on("data", (chunk: Buffer) => stderr.push(chunk.toString()));

  const { promise, resolve, reject } = Promise.withResolvers<Run>();
  proc.once("error", reject);
  proc.on("close", (code) => resolve({
    stdout: stdout.join(""),
    stderr: stderr.join(""),
    exitCode: code ?? 1,
  }));
  return promise;
}

describe("qmd collection rename/remove", () => {
  test("rename then remove update the YAML and report each step", async () => {
    const layout = seedLayout(["docs"]);

    const renamed = await runQmd(["collection", "rename", "docs", "guides"], layout);
    expect(renamed.exitCode, renamed.stderr).toBe(0);
    expect(renamed.stdout).toContain("Renamed collection 'docs' to 'guides'");

    const afterRename = readConfig(layout.configPath);
    expect(Object.keys(afterRename.collections)).toEqual(["guides"]);
    expect(afterRename.collections.guides!.path).toBe(layout.paths.docs);
    expect(afterRename.collections.guides!.pattern).toBe("**/*.md");

    const removed = await runQmd(["collection", "remove", "guides"], layout);
    expect(removed.exitCode, removed.stderr).toBe(0);
    expect(removed.stdout).toContain("Removed collection 'guides'");
    expect(readConfig(layout.configPath).collections).toEqual({});
  });

  test("an existing target is refused before the index is opened", async () => {
    const layout = seedLayout(["docs", "notes"]);
    const before = readFileSync(layout.configPath, "utf-8");

    const conflict = await runQmd(["collection", "rename", "docs", "notes"], layout);
    expect(conflict.exitCode, conflict.stderr).not.toBe(0);
    expect(conflict.stderr).toContain("Collection name already exists: notes");
    // The refusal is decided from the YAML alone, so opening the index (and the
    // model defaults it materializes) must not happen.
    expect(readFileSync(layout.configPath, "utf-8")).toBe(before);
  });

  test("a name the config and index do not know is reported and changes nothing", async () => {
    const layout = seedLayout(["docs"]);
    const before = readFileSync(layout.configPath, "utf-8");

    // Rename: refused from the YAML plus the pending-rename journal alone.
    const rename = await runQmd(["collection", "rename", "absent", "guides"], layout);
    expect(rename.exitCode, rename.stderr).not.toBe(0);
    expect(rename.stderr).toContain("Collection not found: absent");
    expect(readFileSync(layout.configPath, "utf-8")).toBe(before);

    // Remove: refused after probing the index for state under the name only, so
    // no collection or model default is written to the YAML.
    const remove = await runQmd(["collection", "remove", "absent"], layout);
    expect(remove.exitCode, remove.stderr).not.toBe(0);
    expect(remove.stderr).toContain("Collection not found: absent");
    expect(readFileSync(layout.configPath, "utf-8")).toBe(before);
  });
});
