/**
 * Path Fidelity Tests
 *
 * Verifies that QMD stores literal filesystem paths (not handalized slugs) so
 * that paths with special characters — spaces, #, &, @, [], (), etc. — round-
 * trip correctly through index → search → get → full-path.
 *
 * This covers the five breakage points found before the literal-path fix:
 *   1. search --json `file` field shows handalized slug instead of real path
 *   2. `qmd get --full-path` silently falls back (resolveVirtualPath built
 *      a non-existent path from the slug, existsSync returned false)
 *   3. `qmd get <actual-fs-path>` returns "Document not found"
 *   4. `qmd ls` shows handalized slugs
 *   5. `toVirtualPath(db, absPath)` returns null
 *
 * Also covers backward-compat migration: an index created with the old
 * handalize-at-index-time code can be updated with `qmd update` and the paths
 * are renamed to their literal forms in-place.
 */

import { describe, test, expect, beforeAll, afterAll } from "vitest";
import { mkdir, mkdtemp, rm, writeFile } from "fs/promises";
import { existsSync, realpathSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { dirname } from "path";
import YAML from "yaml";
import { openDatabase } from "../src/db.js";
import type { Database } from "../src/db.js";
import {
  createStore,
  toVirtualPath,
  insertDocument,
  insertContent,
  hashContent,
  handelize,
  normalizePathSeparators,
  syncConfigToDb,
} from "../src/store.js";
import type { CollectionConfig } from "../src/collections.js";

const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");

async function runQmd(
  args: string[],
  opts: { cwd: string; dbPath: string; configDir: string; env?: Record<string, string> }
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
      ...(opts.env ?? {}),
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

// ---------------------------------------------------------------------------
// Test environment setup
// ---------------------------------------------------------------------------

let testDir: string;

// Files with names that previously broke due to handalize() at index time.
const crazyFiles: Array<{ name: string; content: string }> = [
  {
    name: "# Meeting - 234232 3432 __ 5.md",
    content: "# Meeting - 234232 3432 // 5\n\nSome meeting content with searchterm-alpha.\n",
  },
  {
    name: "Budget & Revenue (Q4) [2024].md",
    content: "# Budget & Revenue Q4 2024\n\nFinancial overview searchterm-beta.\n",
  },
  {
    name: "normal-file.md",
    content: "# Normal File\n\nPlain filename, should always work.\n",
  },
];

const crazySubFiles: Array<{ name: string; content: string }> = [
  {
    name: "Notes #42 - foo@bar.md",
    content: "# Notes #42\n\nSubdir file with searchterm-gamma.\n",
  },
];

beforeAll(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-path-fidelity-"));
});

afterAll(async () => {
  await rm(testDir, { recursive: true, force: true });
});

// Helper: create a fresh isolated test environment with a corpus of crazy filenames.
async function createCrazyCollection(prefix: string): Promise<{
  collectionDir: string;
  dbPath: string;
  configDir: string;
}> {
  const envDir = join(testDir, prefix);
  const collectionDir = join(envDir, "corpus");
  const dbPath = join(envDir, "test.sqlite");
  const configDir = join(envDir, "config");

  await mkdir(collectionDir, { recursive: true });
  await mkdir(join(collectionDir, "subdir"), { recursive: true });
  await mkdir(configDir, { recursive: true });

  // Resolve symlinks so the path matches what getRealPath() stores in the DB.
  // On macOS /tmp is a symlink to /private/tmp; without this normalisation
  // toVirtualPath() and --full-path resolution fail.
  const realCollectionDir = realpathSync(collectionDir);

  for (const f of crazyFiles) {
    await writeFile(join(collectionDir, f.name), f.content);
  }
  for (const f of crazySubFiles) {
    await writeFile(join(collectionDir, "subdir", f.name), f.content);
  }

  // Write empty YAML config — `collection add` will populate it
  await writeFile(join(configDir, "index.yml"), "collections: {}\n");

  return { collectionDir: realCollectionDir, dbPath, configDir };
}

// ---------------------------------------------------------------------------
// Unit tests: store-level path storage
// ---------------------------------------------------------------------------

describe("Path fidelity — store level", () => {
  test("reindexCollection stores literal relative paths, not handalized slugs", async () => {
    const { collectionDir, dbPath, configDir } = await createCrazyCollection("store-unit");

    // Run `collection add` to index
    const add = await runQmd(
      ["collection", "add", collectionDir, "--name", "crazytest"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(add.exitCode, `collection add failed: ${add.stderr}`).toBe(0);

    // Inspect the DB directly
    const db = openDatabase(dbPath);
    const rows = db.prepare(
      "SELECT path FROM documents WHERE active = 1 ORDER BY path"
    ).all() as { path: string }[];
    db.close();

    const paths = rows.map((r) => r.path);

    // Must contain literal filenames — not handalized slugs
    expect(paths).toContain("# Meeting - 234232 3432 __ 5.md");
    expect(paths).toContain("Budget & Revenue (Q4) [2024].md");
    expect(paths).toContain("normal-file.md");
    expect(paths).toContain("subdir/Notes #42 - foo@bar.md");

    // Must NOT contain handalized versions
    expect(paths).not.toContain("Meeting-234232-3432-5.md");
    expect(paths).not.toContain("Budget-Revenue-Q4-2024.md");
    expect(paths).not.toContain("subdir/Notes-42-foo-bar.md");
  });

  test("toVirtualPath returns non-null for crazy-named files", async () => {
    const { collectionDir, dbPath, configDir } = await createCrazyCollection("store-to-virtual");
    const add = await runQmd(
      ["collection", "add", collectionDir, "--name", "crazytest"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(add.exitCode).toBe(0);

    const rawDb = openDatabase(dbPath);
    const result = toVirtualPath(rawDb, join(collectionDir, "Budget & Revenue (Q4) [2024].md"));
    rawDb.close();

    expect(result).not.toBeNull();
    expect(result).toBe(`qmd://crazytest/Budget & Revenue (Q4) [2024].md`);
  });
});

// ---------------------------------------------------------------------------
// CLI integration tests — the five original breakage points
// ---------------------------------------------------------------------------

describe("Path fidelity — CLI integration", () => {
  let collectionDir: string;
  let dbPath: string;
  let configDir: string;

  // Index once for the whole describe block (read-only tests share it)
  beforeAll(async () => {
    ({ collectionDir, dbPath, configDir } = await createCrazyCollection("cli-shared"));
    const add = await runQmd(
      ["collection", "add", collectionDir, "--name", "crazytest"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(add.exitCode, `collection add failed: ${add.stderr}`).toBe(0);
  });

  test("(1) search --json file field contains literal path, not handalized slug", async () => {
    const { stdout, exitCode } = await runQmd(
      ["search", "searchterm-alpha", "--json"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);

    const results = JSON.parse(stdout) as Array<{ file: string }>;
    expect(results.length).toBeGreaterThan(0);

    const meetingResult = results.find((r) => r.file.includes("Meeting"));
    expect(meetingResult).toBeDefined();
    // Must contain the literal filename fragment
    expect(meetingResult!.file).toContain("# Meeting - 234232 3432 __ 5.md");
    // Must not contain the handalized version
    expect(meetingResult!.file).not.toContain("Meeting-234232-3432-5.md");
  });

  test("(2) get --full-path resolves to real filesystem path for crazy-named file", async () => {
    const virtualPath = `qmd://crazytest/Budget & Revenue (Q4) [2024].md`;
    const { stdout, exitCode } = await runQmd(
      ["get", virtualPath, "--full-path"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode, `get failed: ${stdout}`).toBe(0);

    const header = stdout.split("\n")[0]!;
    // Should show a real filesystem path, not a qmd:// virtual path
    expect(header).not.toMatch(/^qmd:\/\//);
    // Should include the literal filename
    expect(header).toContain("Budget & Revenue (Q4) [2024].md");
    // The resolved filesystem path should exist — strip the trailing docid (#abc123)
    const fsPath = header.trim().replace(/\s+#[a-f0-9]{6}$/, "");
    // Path may be absolute or relative-to-collectionDir; resolve against collectionDir
    const absPath = fsPath.startsWith("/") ? fsPath : join(collectionDir, fsPath.replace(/^\.\//, ""));
    expect(existsSync(absPath), `resolved path does not exist: ${absPath}`).toBe(true);
  });
  test("(3) get <actual-fs-path> finds the document", async () => {
    const fsPath = join(collectionDir, "Budget & Revenue (Q4) [2024].md");
    const { stdout, exitCode, stderr } = await runQmd(
      ["get", fsPath],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode, `get by fs path failed: ${stderr}`).toBe(0);
    // Header should contain the document identifier
    expect(stdout).toContain("Budget & Revenue (Q4) [2024].md");
  });

  test("(3b) get <actual-fs-path> finds subdir file with crazy name", async () => {
    const fsPath = join(collectionDir, "subdir", "Notes #42 - foo@bar.md");
    const { stdout, exitCode, stderr } = await runQmd(
      ["get", fsPath],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode, `get subdir file failed: ${stderr}`).toBe(0);
    expect(stdout).toContain("Notes #42 - foo@bar.md");
  });

  test("(4) ls shows literal paths, not handalized slugs", async () => {
    const { stdout, exitCode } = await runQmd(
      ["ls", "crazytest"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);

    // Literal paths must appear
    expect(stdout).toContain("# Meeting - 234232 3432 __ 5.md");
    expect(stdout).toContain("Budget & Revenue (Q4) [2024].md");
    expect(stdout).toContain("Notes #42 - foo@bar.md");

    // Handalized slugs must NOT appear
    expect(stdout).not.toContain("Meeting-234232-3432-5.md");
    expect(stdout).not.toContain("Budget-Revenue-Q4-2024.md");
    expect(stdout).not.toContain("Notes-42-foo-bar.md");
  });

  test("(5) search --json returns docid that can be fetched back", async () => {
    const { stdout: searchOut, exitCode: searchExit } = await runQmd(
      ["search", "searchterm-beta", "--json"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(searchExit).toBe(0);

    const results = JSON.parse(searchOut) as Array<{ docid: string; file: string }>;
    expect(results.length).toBeGreaterThan(0);

    const hit = results[0]!;
    expect(hit.docid).toMatch(/^#[a-f0-9]{6}$/);

    // Fetch by docid — must work
    const { stdout: getOut, exitCode: getExit } = await runQmd(
      ["get", hit.docid],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(getExit, `get by docid failed`).toBe(0);
    expect(getOut).toContain("Budget & Revenue (Q4) [2024].md");
  });

  test("normal filenames are still stored correctly (regression)", async () => {
    const { stdout, exitCode } = await runQmd(
      ["search", "Plain filename", "--json"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(exitCode).toBe(0);
    const results = JSON.parse(stdout) as Array<{ file: string }>;
    const hit = results.find((r) => r.file.includes("normal-file"));
    expect(hit).toBeDefined();
    expect(hit!.file).toContain("normal-file.md");
  });
});

// ---------------------------------------------------------------------------
// Migration test: old handalized DB upgraded by `qmd update`
// ---------------------------------------------------------------------------

describe("Path fidelity — migration from handalized index", () => {
  test("qmd update migrates handalized paths to literal paths in existing index", async () => {
    const { collectionDir, dbPath, configDir } = await createCrazyCollection("migration");

    // Manually build an old-style DB using handalize() (simulates pre-fix index)
    const store = createStore(dbPath);
    const now = new Date().toISOString();
    // Write and sync a config that points at the collection so `qmd update` knows where it is
    const migrationYaml = `collections:\n  crazytest:\n    path: "${collectionDir}"\n    mask: "**/*.md"\n`;
    await writeFile(join(configDir, "index.yml"), migrationYaml);
    const config = YAML.parse(migrationYaml) as CollectionConfig;
    syncConfigToDb(store.db, config);

    // Insert documents with handalized paths (old behavior)
    for (const f of crazyFiles) {
      const relPath = normalizePathSeparators(f.name);
      const handleized = handelize(relPath);
      const hash = await hashContent(f.content);
      insertContent(store.db, hash, f.content, now);
      insertDocument(store.db, "crazytest", handleized, `Title ${f.name}`, hash, now, now);
    }
    const subFile = crazySubFiles[0]!;
    const subRel = `subdir/${subFile.name}`;
    const subHandelized = handelize(subRel);
    const subHash = await hashContent(subFile.content);
    insertContent(store.db, subHash, subFile.content, now);
    insertDocument(store.db, "crazytest", subHandelized, "Sub title", subHash, now, now);
    store.close();

    // Verify the old DB has handalized paths
    const dbBefore = openDatabase(dbPath);
    const pathsBefore = (dbBefore.prepare(
      "SELECT path FROM documents WHERE active = 1 ORDER BY path"
    ).all() as { path: string }[]).map((r) => r.path);
    dbBefore.close();

    expect(pathsBefore).toContain("Meeting-234232-3432-5.md");
    expect(pathsBefore).toContain("Budget-Revenue-Q4-2024.md");
    expect(pathsBefore).not.toContain("# Meeting - 234232 3432 __ 5.md");

    // Run `qmd update` with the new code — should migrate paths in-place
    const update = await runQmd(
      ["update"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(update.exitCode, `qmd update failed: ${update.stderr}`).toBe(0);

    // Verify the DB now has literal paths
    const dbAfter = openDatabase(dbPath);
    const pathsAfter = (dbAfter.prepare(
      "SELECT path FROM documents WHERE active = 1 ORDER BY path"
    ).all() as { path: string }[]).map((r) => r.path);
    dbAfter.close();

    expect(pathsAfter).toContain("# Meeting - 234232 3432 __ 5.md");
    expect(pathsAfter).toContain("Budget & Revenue (Q4) [2024].md");
    expect(pathsAfter).toContain("normal-file.md");
    expect(pathsAfter).toContain("subdir/Notes #42 - foo@bar.md");

    // Handalized slugs must be gone
    expect(pathsAfter).not.toContain("Meeting-234232-3432-5.md");
    expect(pathsAfter).not.toContain("Budget-Revenue-Q4-2024.md");

    // Search must work after migration
    const { stdout: searchOut, exitCode: searchExit } = await runQmd(
      ["search", "searchterm-alpha", "--json"],
      { cwd: collectionDir, dbPath, configDir }
    );
    expect(searchExit).toBe(0);
    const results = JSON.parse(searchOut) as Array<{ file: string }>;
    expect(results.length).toBeGreaterThan(0);
    const meetingResult = results.find((r) => r.file.includes("Meeting"));
    expect(meetingResult).toBeDefined();
    expect(meetingResult!.file).toContain("# Meeting - 234232 3432 __ 5.md");
  });
});
