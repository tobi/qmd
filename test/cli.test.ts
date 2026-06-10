/**
 * CLI Integration Tests
 *
 * Tests all qmd CLI commands using a temporary test database via INDEX_PATH.
 * These tests spawn actual qmd processes to verify end-to-end functionality.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { chmod, copyFile, mkdtemp, rm, writeFile, mkdir } from "fs/promises";
import { existsSync, lstatSync, readFileSync, symlinkSync, writeFileSync, unlinkSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { spawn } from "child_process";
import { setTimeout as sleep } from "timers/promises";
import { buildEditorUri, termLink, resolveEmbedModelForCli } from "../src/cli/qmd.ts";
import { openDatabase } from "../src/db.ts";
import { DEFAULT_EMBED_MODEL_URI, DEFAULT_GENERATE_MODEL_URI, DEFAULT_RERANK_MODEL_URI } from "../src/llm.ts";
import { setConfigSource } from "../src/collections.ts";

// Test fixtures directory and database path
let testDir: string;
let testDbPath: string;
let testConfigDir: string;
let fixturesDir: string;
let testCounter = 0; // Unique counter for each test run

// Get the directory where this test file lives
const thisDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(thisDir, "..");
const qmdScript = join(projectRoot, "src", "cli", "qmd.ts");
const isBunRuntime = typeof (globalThis as { Bun?: unknown }).Bun !== "undefined";
const tsxCli = join(projectRoot, "node_modules", "tsx", "dist", "cli.mjs");
const qmdCommand = isBunRuntime
  ? { command: process.execPath, args: [qmdScript] }
  : { command: process.execPath, args: [tsxCli, qmdScript] };

function qmdRunnerArgs(args: string[]): { command: string; args: string[] } {
  return { command: qmdCommand.command, args: [...qmdCommand.args, ...args] };
}

// Helper to run qmd command with test database
async function runQmd(
  args: string[],
  options: { cwd?: string; env?: Record<string, string>; dbPath?: string; configDir?: string } = {}
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const workingDir = options.cwd || fixturesDir;
  const dbPath = options.dbPath || testDbPath;
  const configDir = options.configDir || testConfigDir;
  const runner = qmdRunnerArgs(args);
  const proc = spawn(runner.command, runner.args, {
    cwd: workingDir,
    env: {
      ...process.env,
      INDEX_PATH: dbPath,
      QMD_CONFIG_DIR: configDir, // Use test config directory
      PWD: workingDir, // Must explicitly set PWD since getPwd() checks this
      QMD_DOCTOR_DEVICE_PROBE: "0", // Keep integration tests deterministic on CI hosts without usable GPU backends.
      ...options.env,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  const stdoutPromise = new Promise<string>((resolve, reject) => {
    let data = "";
    proc.stdout?.on("data", (chunk: Buffer) => { data += chunk.toString(); });
    proc.once("error", reject);
    proc.stdout?.once("end", () => resolve(data));
  });
  const stderrPromise = new Promise<string>((resolve, reject) => {
    let data = "";
    proc.stderr?.on("data", (chunk: Buffer) => { data += chunk.toString(); });
    proc.once("error", reject);
    proc.stderr?.once("end", () => resolve(data));
  });
  const exitCode = await new Promise<number>((resolve, reject) => {
    proc.once("error", reject);
    proc.on("close", (code) => resolve(code ?? 1));
  });
  const stdout = await stdoutPromise;
  const stderr = await stderrPromise;

  return { stdout, stderr, exitCode };
}

// Get a fresh database path for isolated tests
function getFreshDbPath(): string {
  testCounter++;
  return join(testDir, `test-${testCounter}.sqlite`);
}

// Create an isolated test environment (db + config dir)
async function createIsolatedTestEnv(prefix: string): Promise<{ dbPath: string; configDir: string }> {
  testCounter++;
  const dbPath = join(testDir, `${prefix}-${testCounter}.sqlite`);
  const configDir = join(testDir, `${prefix}-config-${testCounter}`);
  await mkdir(configDir, { recursive: true });
  await writeFile(join(configDir, "index.yml"), "collections: {}\n");
  return { dbPath, configDir };
}

// Setup test fixtures
beforeAll(async () => {
  // Create temp directory structure
  testDir = await mkdtemp(join(tmpdir(), "qmd-test-"));
  testDbPath = join(testDir, "test.sqlite");
  testConfigDir = join(testDir, "config");
  fixturesDir = join(testDir, "fixtures");

  await mkdir(testConfigDir, { recursive: true });
  await mkdir(fixturesDir, { recursive: true });
  await mkdir(join(fixturesDir, "notes"), { recursive: true });
  await mkdir(join(fixturesDir, "docs"), { recursive: true });

  // Create empty YAML config for tests
  await writeFile(
    join(testConfigDir, "index.yml"),
    "collections: {}\n"
  );

  // Create test markdown files
  await writeFile(
    join(fixturesDir, "README.md"),
    `# Test Project

This is a test project for QMD CLI testing.

## Features

- Full-text search with BM25
- Vector similarity search
- Hybrid search with reranking
`
  );

  await writeFile(
    join(fixturesDir, "notes", "meeting.md"),
    `# Team Meeting Notes

Date: 2024-01-15

## Attendees
- Alice
- Bob
- Charlie

## Discussion Topics
- Project timeline review
- Resource allocation
- Technical debt prioritization

## Action Items
1. Alice to update documentation
2. Bob to fix authentication bug
3. Charlie to review pull requests
`
  );

  await writeFile(
    join(fixturesDir, "notes", "ideas.md"),
    `# Product Ideas

## Feature Requests
- Dark mode support
- Keyboard shortcuts
- Export to PDF

## Technical Improvements
- Improve search performance
- Add caching layer
- Optimize database queries
`
  );

  await writeFile(
    join(fixturesDir, "docs", "api.md"),
    `# API Documentation

## Endpoints

### GET /search
Search for documents.

Parameters:
- q: Search query (required)
- limit: Max results (default: 10)

### GET /document/:id
Retrieve a specific document.

### POST /index
Index new documents.
`
  );

  // Create test files for path normalization tests
  await writeFile(
    join(fixturesDir, "test1.md"),
    `# Test Document 1

This is the first test document.

It has multiple lines for testing line numbers.
Line 6 is here.
Line 7 is here.
`
  );

  await writeFile(
    join(fixturesDir, "test2.md"),
    `# Test Document 2

This is the second test document.
`
  );
});

// Cleanup after all tests
afterAll(async () => {
  if (testDir) {
    await rm(testDir, { recursive: true, force: true });
  }
});

// Reset YAML config before each test to ensure isolation
beforeEach(async () => {
  // Reset to empty collections config
  await writeFile(
    join(testConfigDir, "index.yml"),
    "collections: {}\n"
  );
});

describe("CLI Help", () => {
  test("shows help with --help flag", async () => {
    const { stdout, exitCode } = await runQmd(["--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Usage:");
    expect(stdout).toContain("qmd collection add");
    expect(stdout).toContain("qmd search");
    expect(stdout).toContain("--no-gpu");
    expect(stdout).toContain("qmd skill show/install");
  });

  test("shows help with no arguments", async () => {
    const { stdout, exitCode } = await runQmd([]);
    expect(exitCode).toBe(1);
    expect(stdout).toContain("Usage:");
  });
});



describe("CLI Skills", () => {
  test("lists bundled runtime skills", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["skills", "list"]);
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain("qmd");
    expect(stdout).toContain("Search local markdown knowledge bases");
  });

  test("gets version-matched runtime skill content", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["skills", "get", "qmd"]);
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain("# QMD - Query Markdown Documents");
    expect(stdout).toContain("## MCP Tool: `query`");
    expect(stdout).not.toContain("This file is a discovery stub");
  });

  test("gets runtime skill with supplementary references", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["skills", "get", "qmd", "--full"]);
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain("# QMD - Query Markdown Documents");
    expect(stdout).toContain("--- references/mcp-setup.md ---");
    expect(stdout).toContain("# QMD MCP Server Setup");
  });

  test("prints canonical repository skill path", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["skills", "path", "qmd"]);
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout.trim()).toMatch(/skills\/qmd$/);
  });

  test("legacy skill show prints the canonical skill", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["skill", "show"]);
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain("# QMD - Query Markdown Documents");
    expect(stdout).toContain("## MCP Tool: `query`");
    expect(stdout).not.toContain("This file is a discovery stub");
  });

  test("legacy skill install writes a qmd skill show bootstrap", async () => {
    const installDir = join(testDir, "skill-install-target");
    await mkdir(installDir, { recursive: true });

    const { stdout, stderr, exitCode } = await runQmd(["skill", "install", "--yes"], { cwd: installDir });
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Installed QMD skill");

    const installedSkillDir = join(installDir, ".agents", "skills", "qmd");
    const installed = readFileSync(join(installedSkillDir, "SKILL.md"), "utf8");
    expect(installed).toContain("# QMD - Query Markdown Documents");
    expect(installed).toContain("!`qmd skill show`");
    expect(installed).toContain("qmd get");
    expect(installed).not.toContain("## MCP Tool: `query`");
    expect(readFileSync(join(installedSkillDir, "references", "mcp-setup.md"), "utf8")).toContain("# QMD MCP Server Setup");
  });
});

describe("CLI Embed", () => {
  test("prefers QMD_EMBED_MODEL for qmd embed when the index has no model pin", () => {
    const prev = process.env.QMD_EMBED_MODEL;
    process.env.QMD_EMBED_MODEL = "hf:env/embed-model.gguf";
    setConfigSource({ config: { collections: {} } });

    try {
      expect(resolveEmbedModelForCli()).toBe("hf:env/embed-model.gguf");
    } finally {
      setConfigSource();
      if (prev === undefined) delete process.env.QMD_EMBED_MODEL;
      else process.env.QMD_EMBED_MODEL = prev;
    }
  });

  test("falls back to the default embed model when QMD_EMBED_MODEL is unset", () => {
    const prev = process.env.QMD_EMBED_MODEL;
    delete process.env.QMD_EMBED_MODEL;
    setConfigSource({ config: { collections: {} } });

    try {
      expect(resolveEmbedModelForCli()).toBe(DEFAULT_EMBED_MODEL_URI);
    } finally {
      setConfigSource();
      if (prev === undefined) delete process.env.QMD_EMBED_MODEL;
      else process.env.QMD_EMBED_MODEL = prev;
    }
  });

  test("rejects invalid --max-docs-per-batch", async () => {
    const { stderr, exitCode } = await runQmd(["embed", "--max-docs-per-batch", "0"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("maxDocsPerBatch");
  });

  test("rejects invalid --max-batch-mb", async () => {
    const { stderr, exitCode } = await runQmd(["embed", "--max-batch-mb", "0"]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("maxBatchBytes");
  });
});

describe("CLI Skill Commands", () => {
  test("shows embedded skill with --skill alias", async () => {
    const { stdout, exitCode } = await runQmd(["--skill"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("QMD Skill");
    expect(stdout).toContain("name: qmd");
    expect(stdout).toContain("allowed-tools: Bash(qmd:*), mcp__qmd__*");
  });

  test("shows skill help with -h", async () => {
    const { stdout, exitCode } = await runQmd(["skill", "-h"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Usage: qmd skill <show|install> [options]");
    expect(stdout).toContain("install");
    expect(stdout).toContain("--global");
  });

  test("installs the skill into the current project", async () => {
    const projectDir = join(testDir, "skill-project");
    await mkdir(projectDir, { recursive: true });

    const { stdout, exitCode } = await runQmd(["skill", "install"], { cwd: projectDir });
    expect(exitCode).toBe(0);

    const skillDir = join(projectDir, ".agents", "skills", "qmd");
    const installed = readFileSync(join(skillDir, "SKILL.md"), "utf-8");
    expect(installed).toContain("# QMD - Query Markdown Documents");
    expect(installed).toContain("!`qmd skill show`");
    expect(existsSync(join(projectDir, ".claude", "skills", "qmd"))).toBe(false);
    expect(stdout).toContain(`✓ Installed QMD skill to ${skillDir}`);
    expect(stdout).toContain("Tip: create a Claude symlink manually");
  });

  test("installs globally and creates the Claude symlink with --yes", async () => {
    const fakeHome = join(testDir, "skill-home");
    await mkdir(fakeHome, { recursive: true });

    const { stdout, exitCode } = await runQmd(["skill", "install", "--global", "--yes"], {
      env: { HOME: fakeHome },
    });
    expect(exitCode).toBe(0);

    const skillDir = join(fakeHome, ".agents", "skills", "qmd");
    const claudeLink = join(fakeHome, ".claude", "skills", "qmd");

    expect(readFileSync(join(skillDir, "SKILL.md"), "utf-8")).toContain("!`qmd skill show`");
    expect(lstatSync(claudeLink).isSymbolicLink()).toBe(true);
    expect(readFileSync(join(claudeLink, "SKILL.md"), "utf-8")).toContain("!`qmd skill show`");
    expect(stdout).toContain(`✓ Installed QMD skill to ${skillDir}`);
    expect(stdout).toContain(`✓ Linked Claude skill at ${claudeLink}`);
  });

  test("skips Claude qmd symlink when .claude/skills already points to .agents/skills", async () => {
    const fakeHome = join(testDir, "skill-home-shared");
    await mkdir(join(fakeHome, ".agents"), { recursive: true });
    await mkdir(join(fakeHome, ".claude"), { recursive: true });
    symlinkSync(join(fakeHome, ".agents", "skills"), join(fakeHome, ".claude", "skills"), "dir");

    const { stdout, exitCode } = await runQmd(["skill", "install", "--global", "--yes"], {
      env: { HOME: fakeHome },
    });
    expect(exitCode).toBe(0);

    const skillDir = join(fakeHome, ".agents", "skills", "qmd");
    expect(lstatSync(skillDir).isSymbolicLink()).toBe(false);
    expect(readFileSync(join(skillDir, "SKILL.md"), "utf-8")).toContain("!`qmd skill show`");
    expect(stdout).toContain(`✓ Claude already sees the skill via ${join(fakeHome, ".claude", "skills")}`);
  });

  test("refuses to overwrite an existing install without --force", async () => {
    const projectDir = join(testDir, "skill-project-force");
    await mkdir(projectDir, { recursive: true });

    const first = await runQmd(["skill", "install"], { cwd: projectDir });
    expect(first.exitCode).toBe(0);

    const second = await runQmd(["skill", "install"], { cwd: projectDir });
    expect(second.exitCode).toBe(1);
    expect(second.stderr).toContain("Skill already exists");
    expect(second.stderr).toContain("--force");
  });
});

describe("CLI Init Command", () => {
  test("creates a project-local .qmd index", async () => {
    const projectDir = join(testDir, "init-project");
    await mkdir(projectDir, { recursive: true });

    const { stdout, exitCode } = await runQmd(["init"], { cwd: projectDir });
    expect(exitCode).toBe(0);
    expect(stdout.trim()).toBe("ready to go with new local index");
    expect(existsSync(join(projectDir, ".qmd", "index.yml"))).toBe(true);
    expect(existsSync(join(projectDir, ".qmd", "index.sqlite"))).toBe(true);
    const configText = readFileSync(join(projectDir, ".qmd", "index.yml"), "utf-8");
    expect(configText).toContain("collections: {}");
    expect(configText).toContain("models:");
  });

  test("refuses to initialize in HOME", async () => {
    const fakeHome = join(testDir, "init-home");
    await mkdir(fakeHome, { recursive: true });

    const { stderr, exitCode } = await runQmd(["init"], {
      cwd: fakeHome,
      env: { HOME: fakeHome },
    });
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Refusing to initialize a local index in $HOME");
    expect(stderr).toContain("global index is automatically created");
    expect(existsSync(join(fakeHome, ".qmd", "index.yml"))).toBe(false);
  });
});

describe("CLI Add Command", () => {
  test("adds files from current directory", async () => {
    const { stdout, exitCode } = await runQmd(["collection", "add", "."]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Collection:");
    expect(stdout).toContain("Indexed:");
  });

  test("adds files with custom glob pattern", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["collection", "add", ".", "--mask", "notes/*.md"]);
    if (exitCode !== 0) {
      console.error("Command failed:", stderr);
    }
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Collection:");
    // Should find meeting.md and ideas.md in notes/
    expect(stdout).toContain("notes/*.md");
  });

  test("can recreate collection with remove and add", async () => {
    // First add
    await runQmd(["collection", "add", "."]);
    // Remove it
    await runQmd(["collection", "remove", "fixtures"]);
    // Re-add
    const { stdout, exitCode } = await runQmd(["collection", "add", "."]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Collection 'fixtures' created successfully");
  });
});

describe("CLI Status Command", () => {
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["collection", "add", "."]);
  });

  test("qmd doctor reports core index health checks", async () => {
    const { stdout, exitCode } = await runQmd(["doctor"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("QMD Doctor");
    expect(stdout).toContain("SQLite runtime");
    expect(stdout).toContain("sqlite-vec");
    expect(stdout).toContain("environment overrides");
    expect(stdout).toContain("INDEX_PATH");
    expect(stdout).toContain("overrides the SQLite index path");
    expect(stdout).toContain("QMD_CONFIG_DIR");
    expect(stdout).toContain("overrides the QMD config directory");
    expect(stdout).toContain("model defaults");
    expect(stdout).toContain("model cache");
    expect(stdout).toContain("device mode");
    expect(stdout).toContain("device probe");
    expect(stdout).toContain("embedding freshness");
    expect(stdout).toContain("embedding fingerprints");
    expect(stdout).toContain("embedding vector sample");
    expect(stdout).toContain("please run qmd embed again");

    const configText = readFileSync(join(testConfigDir, "index.yml"), "utf-8");
    expect(configText).toContain("models:");
    expect(configText).toContain(DEFAULT_EMBED_MODEL_URI);
    expect(configText).toContain(DEFAULT_GENERATE_MODEL_URI);
    expect(configText).toContain(DEFAULT_RERANK_MODEL_URI);
  }, 20000);

  test("qmd doctor warns when no collections are configured", async () => {
    const env = await createIsolatedTestEnv("doctor-no-collections");
    const { stdout, exitCode } = await runQmd(["doctor"], { dbPath: env.dbPath, configDir: env.configDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("index config");
    expect(stdout).toContain("no collections configured");
    expect(stdout).toContain("qmd collection add .");
  }, 20000);

  test("qmd doctor reports invalid index.yml without crashing", async () => {
    const env = await createIsolatedTestEnv("doctor-invalid-config");
    await writeFile(join(env.configDir, "index.yml"), "collections:\n  bad: [unterminated\n");

    const { stdout, exitCode } = await runQmd(["doctor"], { dbPath: env.dbPath, configDir: env.configDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("index config");
    expect(stdout).toContain("invalid index.yml at");
    expect(stdout).toContain(join(env.configDir, "index.yml"));
    expect(stdout).toContain("fix the YAML");
  }, 20000);

  test("qmd doctor warns when configured models differ from code defaults", async () => {
    const env = await createIsolatedTestEnv("doctor-custom-models");
    await writeFile(join(env.configDir, "index.yml"), `collections: {}\nmodels:\n  embed: hf:example/custom-embed/custom.gguf\n  generate: ${DEFAULT_GENERATE_MODEL_URI}\n  rerank: ${DEFAULT_RERANK_MODEL_URI}\n`);

    const { stdout, exitCode } = await runQmd(["doctor"], { dbPath: env.dbPath, configDir: env.configDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("model defaults");
    expect(stdout).toContain("non-default model configuration");
    expect(stdout).toContain("index hf:example/custom-embed/custom.gguf");
    expect(stdout).toContain("might be ok");
    expect(stdout).toContain("qmd pull");
  }, 20000);

  test("qmd doctor identifies cached non-GGUF model files", async () => {
    const env = await createIsolatedTestEnv("doctor-invalid-model-cache");
    const model = "hf:example/custom-model/custom.gguf";
    await writeFile(join(env.configDir, "index.yml"), `collections: {}\nmodels:\n  embed: ${model}\n  generate: ${model}\n  rerank: ${model}\n`);
    const cacheRoot = join(env.configDir, "cache");
    const modelCacheDir = join(cacheRoot, "qmd", "models");
    await mkdir(modelCacheDir, { recursive: true });
    const badModelPath = join(modelCacheDir, "custom.gguf");
    await writeFile(badModelPath, "<!doctype html><html>blocked</html>");

    const { stdout, exitCode } = await runQmd(["doctor"], {
      dbPath: env.dbPath,
      configDir: env.configDir,
      env: {
        XDG_CACHE_HOME: cacheRoot,
        QMD_DOCTOR_DEVICE_PROBE: "0",
      },
    });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("model cache");
    expect(stdout).toContain("invalid 1");
    expect(stdout).toContain("HTML page, not a GGUF model");
    expect(stdout).toContain("qmd pull --refresh");
  }, 20000);

  test("qmd doctor says when models are overridden by env", async () => {
    const env = await createIsolatedTestEnv("doctor-env-models");
    await writeFile(join(env.configDir, "index.yml"), "collections: {}\n");

    const customEmbed = "hf:example/env-embed/custom.gguf";
    const { stdout, exitCode } = await runQmd(["doctor"], {
      dbPath: env.dbPath,
      configDir: env.configDir,
      env: { QMD_EMBED_MODEL: customEmbed },
    });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("model defaults");
    expect(stdout).toContain(`env QMD_EMBED_MODEL=${customEmbed}`);
    expect(stdout).toContain("might be ok");
    expect(stdout).toContain("environment overrides");
    expect(stdout).toContain(`QMD_EMBED_MODEL=${customEmbed}`);
    expect(stdout).toContain("sets the active embed model");
  }, 20000);

  test("qmd doctor shows CPU-forced device mode with QMD_FORCE_CPU=1", async () => {
    const env = await createIsolatedTestEnv("doctor-force-cpu");
    const { stdout, exitCode } = await runQmd(["doctor"], {
      dbPath: env.dbPath,
      configDir: env.configDir,
      env: {
        QMD_FORCE_CPU: "1",
        QMD_DOCTOR_DEVICE_PROBE: "0",
      },
    });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("QMD_FORCE_CPU=1");
    expect(stdout).toContain("forces llama.cpp to bypass GPU backends");
    expect(stdout).toContain("device mode: CPU forced (QMD_FORCE_CPU)");
  }, 20000);

  test("qmd doctor lists known environment overrides and consequences", async () => {
    const env = await createIsolatedTestEnv("doctor-env-overrides");
    const overrides = {
      XDG_CACHE_HOME: join(env.configDir, "cache"),
      QMD_DOCTOR_DEVICE_PROBE: "0",
      QMD_FORCE_CPU: "1",
      QMD_LLAMA_GPU: "metal",
      QMD_EMBED_PARALLELISM: "2",
      QMD_EXPAND_CONTEXT_SIZE: "4096",
      QMD_RERANK_CONTEXT_SIZE: "8192",
      QMD_EMBED_CONTEXT_SIZE: "1024",
      QMD_EDITOR_URI: "vscode://file/{file}:{line}:{col}",
      QMD_SKILLS_DIR: "/tmp/qmd-skills",
      QMD_METAL_KEEP_RESIDENCY: "1",
      NO_COLOR: "1",
      CI: "1",
      HF_ENDPOINT: "https://hf-mirror.com",
      WSL_DISTRO_NAME: "Ubuntu",
      WSL_INTEROP: "1",
    };

    const { stdout, exitCode } = await runQmd(["doctor"], {
      dbPath: env.dbPath,
      configDir: env.configDir,
      env: overrides,
    });
    expect(exitCode).toBe(0);
    for (const name of Object.keys(overrides)) {
      expect(stdout).toContain(name);
    }
    expect(stdout).toContain("forces llama.cpp to bypass GPU backends");
    expect(stdout).toContain("moves the default index cache");
    expect(stdout).toContain("disables real LLM operations");
    expect(stdout).toContain("changes Hugging Face download endpoint");
  }, 20000);

  test("qmd doctor flags mixed embedding fingerprints", async () => {
    const db = openDatabase(testDbPath);
    const doc = db.prepare(`SELECT hash FROM documents WHERE active = 1 LIMIT 1`).get() as { hash: string };
    const now = new Date().toISOString();
    db.prepare(`
      INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      VALUES (?, 0, 0, ?, 'stale1', 2, ?)
    `).run(doc.hash, resolveEmbedModelForCli(), now);
    db.prepare(`
      INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embed_fingerprint, total_chunks, embedded_at)
      VALUES (?, 1, 1, ?, 'stale2', 2, ?)
    `).run(doc.hash, resolveEmbedModelForCli(), now);
    db.close();

    const { stdout, exitCode } = await runQmd(["doctor"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("embedding fingerprints");
    expect(stdout).toContain("mixed named embedding fingerprints");
    expect(stdout).toContain("stale1");
  }, 20000);

  test("shows index status", async () => {
    const { stdout, exitCode } = await runQmd(["status"]);
    expect(exitCode).toBe(0);
    // Should show collection info
    expect(stdout).toContain("Collection");
  });

  test("status omits device probing details; doctor owns GPU diagnostics", async () => {
    const { stdout, exitCode } = await runQmd(["status"]);
    expect(exitCode).toBe(0);
    expect(stdout).not.toContain("Device");
    expect(stdout).not.toContain("QMD_STATUS_DEVICE_PROBE");
    expect(stdout).not.toContain("not probed");
  });
});

describe("CLI Search Command", () => {
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["collection", "add", "."]);
  });

  test("searches for documents with BM25", async () => {
    const { stdout, exitCode } = await runQmd(["search", "meeting"]);
    expect(exitCode).toBe(0);
    // Should find meeting.md
    expect(stdout.toLowerCase()).toContain("meeting");
  });

  test("searches with limit option", async () => {
    const { stdout, exitCode } = await runQmd(["search", "-n", "1", "test"]);
    expect(exitCode).toBe(0);
  });

  test("searches with all results option", async () => {
    const { stdout, exitCode } = await runQmd(["search", "--all", "the"]);
    expect(exitCode).toBe(0);
  });

  test("returns no results message for non-matching query", async () => {
    const { stdout, exitCode } = await runQmd(["search", "xyznonexistent123"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("No results");
  });

  test("returns empty JSON array for non-matching query with --json", async () => {
    const { stdout, exitCode } = await runQmd(["search", "xyznonexistent123", "--json"]);
    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout)).toEqual([]);
  });

  test("returns CSV header only for non-matching query with --csv", async () => {
    const { stdout, exitCode } = await runQmd(["search", "xyznonexistent123", "--csv"]);
    expect(exitCode).toBe(0);
    expect(stdout.trim()).toBe("docid,score,file,title,context,line,snippet");
  });

  test("returns empty XML container for non-matching query with --xml", async () => {
    const { stdout, exitCode } = await runQmd(["search", "xyznonexistent123", "--xml"]);
    expect(exitCode).toBe(0);
    expect(stdout.trim()).toBe("<results></results>");
  });

  test("returns empty output for non-matching query with --md", async () => {
    const { stdout, exitCode } = await runQmd(["search", "xyznonexistent123", "--md"]);
    expect(exitCode).toBe(0);
    expect(stdout.trim()).toBe("");
  });

  test("returns empty output for non-matching query with --files", async () => {
    const { stdout, exitCode } = await runQmd(["search", "xyznonexistent123", "--files"]);
    expect(exitCode).toBe(0);
    expect(stdout.trim()).toBe("");
  });

  test("returns min-score threshold message for default CLI output", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "--min-score", "2"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("No results found above minimum score threshold.");
  });

  test("returns format-safe empty output when --min-score filters all results", async () => {
    const json = await runQmd(["search", "test", "--json", "--min-score", "2"]);
    expect(json.exitCode).toBe(0);
    expect(JSON.parse(json.stdout)).toEqual([]);

    const csv = await runQmd(["search", "test", "--csv", "--min-score", "2"]);
    expect(csv.exitCode).toBe(0);
    expect(csv.stdout.trim()).toBe("docid,score,file,title,context,line,snippet");

    const xml = await runQmd(["search", "test", "--xml", "--min-score", "2"]);
    expect(xml.exitCode).toBe(0);
    expect(xml.stdout.trim()).toBe("<results></results>");

    const md = await runQmd(["search", "test", "--md", "--min-score", "2"]);
    expect(md.exitCode).toBe(0);
    expect(md.stdout.trim()).toBe("");

    const files = await runQmd(["search", "test", "--files", "--min-score", "2"]);
    expect(files.exitCode).toBe(0);
    expect(files.stdout.trim()).toBe("");
  });

  test("requires query argument", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["search"]);
    expect(exitCode).toBe(1);
    // Error message goes to stderr
    expect(stderr).toContain("Usage:");
  });

  test("--json --full includes line field for round-tripping to qmd get", async () => {
    const { stdout, exitCode } = await runQmd(["search", "meeting", "--json", "--full", "-n", "1"]);
    expect(exitCode).toBe(0);
    const results = JSON.parse(stdout);
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].line).toBeTypeOf("number");
    expect(results[0].line).toBeGreaterThan(0);
    expect(results[0].body).toBeTypeOf("string");
  });
});

describe("CLI Get Command", () => {
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["collection", "add", "."]);
  });

  test("retrieves document content by path", async () => {
    const { stdout, exitCode } = await runQmd(["get", "README.md"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Test Project");
  });

  test("retrieves document from subdirectory", async () => {
    const { stdout, exitCode } = await runQmd(["get", "notes/meeting.md"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Team Meeting");
  });

  test("handles non-existent file", async () => {
    const { stdout, exitCode } = await runQmd(["get", "nonexistent.md"]);
    // Should indicate file not found
    expect(exitCode).toBe(1);
  });

  test("clamps negative --from to top of file (no silent tail content)", async () => {
    const baseline = await runQmd(["get", "README.md"]);
    const negative = await runQmd(["get", "README.md", "--from", "-19"]);
    expect(negative.exitCode).toBe(0);
    expect(negative.stdout).toBe(baseline.stdout);
  });
});

describe("CLI Multi-Get Command", () => {
  let localDbPath: string;

  beforeEach(async () => {
    // Use fresh database for each test
    localDbPath = getFreshDbPath();
    // Ensure we have indexed files
    const addResult = await runQmd(["collection", "add", ".", "--name", "fixtures"], { dbPath: localDbPath });
    if (addResult.exitCode !== 0) {
      throw new Error(`Failed to add collection: ${addResult.stderr}`);
    }
  });

  test("retrieves multiple documents by pattern", async () => {
    // Test glob pattern matching
    const { stdout, stderr, exitCode } = await runQmd(["multi-get", "notes/*.md"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    // Should contain content from both notes files
    expect(stdout).toContain("Meeting");
    expect(stdout).toContain("Ideas");
  });

  test("retrieves documents by comma-separated paths", async () => {
    const { stdout, exitCode } = await runQmd([
      "multi-get",
      "README.md,notes/meeting.md",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Test Project");
    expect(stdout).toContain("Team Meeting");
  });

  test("--md output includes a #docid for each file", async () => {
    const { stdout, exitCode } = await runQmd(["multi-get", "notes/*.md", "--md"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    // Every result carries a docid line, consistent with `search --md`.
    expect(stdout).toMatch(/\*\*docid:\*\* `#[a-f0-9]{6}`/);
  });

  test("--json output includes a #docid for each file", async () => {
    const { stdout, exitCode } = await runQmd(["multi-get", "notes/*.md", "--json"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.length).toBeGreaterThan(0);
    for (const entry of parsed) {
      expect(entry.docid).toMatch(/^#[a-f0-9]{6}$/);
    }
  });

  test("shows line numbers by default and --no-line-numbers disables them", async () => {
    const withNums = await runQmd(["multi-get", "README.md"], { dbPath: localDbPath });
    expect(withNums.exitCode).toBe(0);
    expect(withNums.stdout).toMatch(/^1: /m);

    const raw = await runQmd(["multi-get", "README.md", "--no-line-numbers"], { dbPath: localDbPath });
    expect(raw.exitCode).toBe(0);
    expect(raw.stdout).not.toMatch(/^1: /m);
  });

  test("--full-path --md shows ./-prefixed on-disk paths and drops the docid", async () => {
    // Default runQmd cwd is fixturesDir, so notes/*.md files are subpaths.
    const { stdout, exitCode } = await runQmd(["multi-get", "notes/*.md", "--md", "--full-path"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    // Headings are ./-prefixed relative paths under fixturesDir.
    expect(stdout).toMatch(/^## \.\/notes\/[^\s]+\.md$/m);
    expect(stdout).not.toContain("qmd://");
    expect(stdout).not.toMatch(/\*\*docid:\*\*/);
  });

  test("--full-path --json puts the ./-prefixed path in `file` and omits docid", async () => {
    const { stdout, exitCode } = await runQmd(["multi-get", "notes/*.md", "--json", "--full-path"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.length).toBeGreaterThan(0);
    for (const entry of parsed) {
      expect(entry.file.startsWith("./notes/")).toBe(true);
      expect(entry.docid).toBeUndefined();
    }
  });

  test("--full-path --json uses absolute path when files are outside $PWD", async () => {
    const { stdout, exitCode } = await runQmd(
      ["multi-get", "notes/*.md", "--json", "--full-path"],
      { dbPath: localDbPath, cwd: "/" }
    );
    expect(exitCode).toBe(0);
    const parsed = JSON.parse(stdout);
    expect(parsed.length).toBeGreaterThan(0);
    for (const entry of parsed) {
      expect(entry.file.startsWith("/")).toBe(true);
      expect(entry.file).not.toMatch(/^\.\//);
      expect(entry.docid).toBeUndefined();
    }
  });
});

describe("CLI Update Command", () => {
  let localDbPath: string;

  beforeEach(async () => {
    // Use a fresh database for this test suite
    localDbPath = getFreshDbPath();
    // Ensure we have indexed files
    await runQmd(["collection", "add", "."], { dbPath: localDbPath });
  });

  test("updates all collections", async () => {
    const { stdout, exitCode } = await runQmd(["update"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Updating");
  });

  test("deactivates stale docs when collection has zero matching files", async () => {
    const { dbPath, configDir } = await createIsolatedTestEnv("update-empty");
    const collectionDir = join(testDir, `update-empty-${Date.now()}`);
    await mkdir(collectionDir, { recursive: true });

    const docPath = join(collectionDir, "only.md");
    const token = `stale-proof-${Date.now()}`;
    await writeFile(
      docPath,
      `---
date: 2026-03-06
---
# Empty Collection Deactivation
${token}
`
    );

    const add = await runQmd(
      ["collection", "add", collectionDir, "--name", "empty-check"],
      { dbPath, configDir }
    );
    expect(add.exitCode).toBe(0);

    const before = await runQmd(["get", "qmd://empty-check/only.md"], { dbPath, configDir });
    expect(before.exitCode).toBe(0);
    expect(before.stdout).toContain(token);

    unlinkSync(docPath);

    const update = await runQmd(["update"], { dbPath, configDir });
    expect(update.exitCode).toBe(0);
    expect(update.stdout).toContain("0 new, 0 updated, 0 unchanged, 1 removed");

    const after = await runQmd(["get", "qmd://empty-check/only.md"], { dbPath, configDir });
    expect(after.exitCode).toBe(1);
  });
});

describe("CLI Add-Context Command", () => {
  let localDbPath: string;
  let localConfigDir: string;
  const collName = "fixtures";

  beforeAll(async () => {
    const env = await createIsolatedTestEnv("context-cmd");
    localDbPath = env.dbPath;
    localConfigDir = env.configDir;

    // Add collection with known name
    const { exitCode, stderr } = await runQmd(
      ["collection", "add", fixturesDir, "--name", collName],
      { dbPath: localDbPath, configDir: localConfigDir }
    );
    if (exitCode !== 0) console.error("collection add failed:", stderr);
    expect(exitCode).toBe(0);
  });

  test("adds context to a path", async () => {
    // Add context to the collection root using virtual path
    const { stdout, exitCode } = await runQmd([
      "context",
      "add",
      `qmd://${collName}/`,
      "Personal notes and meeting logs",
    ], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ Added context");
  });

  test("requires path and text arguments", async () => {
    const { stderr, exitCode } = await runQmd(["context", "add"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(1);
    // Error message goes to stderr
    expect(stderr).toContain("Usage:");
  });
});

describe("CLI Cleanup Command", () => {
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["collection", "add", "."]);
  });

  test("cleans up orphaned entries", async () => {
    const { stdout, exitCode } = await runQmd(["cleanup"]);
    expect(exitCode).toBe(0);
  });
});

describe("CLI Error Handling", () => {
  test("handles unknown command", async () => {
    const { stderr, exitCode } = await runQmd(["unknowncommand"]);
    expect(exitCode).toBe(1);
    // Should indicate unknown command and point users to diagnostics
    expect(stderr).toContain("Unknown command");
    expect(stderr).toContain("qmd doctor");
  });

  test("uses INDEX_PATH environment variable", async () => {
    // Verify the test DB path is being used by creating a separate index
    const customDbPath = join(testDir, "custom.sqlite");
    const { exitCode } = await runQmd(["collection", "add", "."], {
      env: { INDEX_PATH: customDbPath },
    });
    expect(exitCode).toBe(0);

    // The custom database should exist
    expect(existsSync(customDbPath)).toBe(true);
  });
});

describe("CLI Output Formats", () => {
  beforeEach(async () => {
    await runQmd(["collection", "add", "."]);
  });

  test("search with --json flag outputs JSON", async () => {
    const { stdout, exitCode } = await runQmd(["search", "--json", "test"]);
    expect(exitCode).toBe(0);
    // Should be valid JSON
    const parsed = JSON.parse(stdout);
    expect(Array.isArray(parsed)).toBe(true);
  });

  test("search with --files flag outputs file paths", async () => {
    const { stdout, exitCode } = await runQmd(["search", "--files", "meeting"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain(".md");
  });

  test("search output includes snippets by default", async () => {
    const { stdout, exitCode } = await runQmd(["search", "API"]);
    expect(exitCode).toBe(0);
    // If results found, should have snippet content
    if (!stdout.includes("No results")) {
      expect(stdout.toLowerCase()).toContain("api");
    }
  });
});

describe("CLI Search with Collection Filter", () => {
  let localDbPath: string;

  beforeEach(async () => {
    // Use a fresh database for this test suite
    localDbPath = getFreshDbPath();
    // Create multiple collections with explicit names
    await runQmd(["collection", "add", ".", "--name", "notes", "--mask", "notes/*.md"], { dbPath: localDbPath });
    await runQmd(["collection", "add", ".", "--name", "docs", "--mask", "docs/*.md"], { dbPath: localDbPath });
  });

  test("filters search by collection name", async () => {
    const { stdout, stderr, exitCode } = await runQmd([
      "search",
      "-c",
      "notes",
      "meeting",
    ], { dbPath: localDbPath });
    if (exitCode !== 0) {
      console.log("Collection filter search failed:");
      console.log("stdout:", stdout);
      console.log("stderr:", stderr);
    }
    expect(exitCode).toBe(0);
  });
});

describe("CLI Context Management", () => {
  let localDbPath: string;

  beforeEach(async () => {
    // Use a fresh database for this test suite
    localDbPath = getFreshDbPath();
    // Index some files first
    await runQmd(["collection", "add", "."], { dbPath: localDbPath });
  });

  test("add global context with /", async () => {
    const { stdout, exitCode } = await runQmd([
      "context",
      "add",
      "/",
      "Global system context",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ Set global context");
    expect(stdout).toContain("Global system context");
  });

  test("list contexts", async () => {
    // Add a global context first
    await runQmd([
      "context",
      "add",
      "/",
      "Test context",
    ], { dbPath: localDbPath });

    const { stdout, exitCode } = await runQmd([
      "context",
      "list",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Configured Contexts");
    expect(stdout).toContain("Test context");
  });

  test("add context to virtual path", async () => {
    // Collection name should be "fixtures" (basename of the fixtures directory)
    const { stdout, exitCode } = await runQmd([
      "context",
      "add",
      "qmd://fixtures/notes",
      "Context for notes subdirectory",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ Added context for: qmd://fixtures/notes");
  });

  test("remove global context", async () => {
    // Add a global context first
    await runQmd([
      "context",
      "add",
      "/",
      "Global context to remove",
    ], { dbPath: localDbPath });

    const { stdout, exitCode } = await runQmd([
      "context",
      "rm",
      "/",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ Removed");
  });

  test("remove virtual path context", async () => {
    // Add a context first
    await runQmd([
      "context",
      "add",
      "qmd://fixtures/notes",
      "Context to remove",
    ], { dbPath: localDbPath });

    const { stdout, exitCode } = await runQmd([
      "context",
      "rm",
      "qmd://fixtures/notes",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ Removed context for: qmd://fixtures/notes");
  });

  test("fails to remove non-existent context", async () => {
    const { stdout, stderr, exitCode } = await runQmd([
      "context",
      "rm",
      "qmd://nonexistent/path",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(1);
    expect(stderr || stdout).toContain("not found");
  });
});

describe("CLI ls Command", () => {
  let localDbPath: string;

  beforeEach(async () => {
    // Use a fresh database for this test suite
    localDbPath = getFreshDbPath();
    // Index some files first
    await runQmd(["collection", "add", "."], { dbPath: localDbPath });
  });

  test("lists all collections", async () => {
    const { stdout, exitCode } = await runQmd(["ls"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Collections:");
    expect(stdout).toContain("qmd://fixtures/");
  });

  test("lists files in a collection", async () => {
    const { stdout, exitCode } = await runQmd(["ls", "fixtures"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    // handelize preserves original case
    expect(stdout).toContain("qmd://fixtures/README.md");
    expect(stdout).toContain("qmd://fixtures/notes/meeting.md");
  });

  test("lists files with path prefix", async () => {
    const { stdout, exitCode } = await runQmd(["ls", "fixtures/notes"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("qmd://fixtures/notes/meeting.md");
    expect(stdout).toContain("qmd://fixtures/notes/ideas.md");
    // Should not include files outside the prefix (case preserved)
    expect(stdout).not.toContain("qmd://fixtures/README.md");
  });

  test("lists files with virtual path", async () => {
    const { stdout, exitCode } = await runQmd(["ls", "qmd://fixtures/docs"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("qmd://fixtures/docs/api.md");
  });

  test("continues to normalize extra slashes for normal collection virtual paths", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["ls", "qmd:///fixtures/docs"], { dbPath: localDbPath });
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain("qmd://fixtures/docs/api.md");
  });

  test("lists an absolute-path collection from a qmd:/// virtual path", async () => {
    const env = await createIsolatedTestEnv("absolute-qmd-path");
    const absoluteDir = await mkdtemp(join(tmpdir(), "qmd-absolute-collection-"));
    await writeFile(join(absoluteDir, "root.md"), "# Absolute collection\n");
    await writeFile(
      join(env.configDir, "index.yml"),
      `collections:\n  "${absoluteDir}":\n    path: "${absoluteDir}"\n    pattern: "**/*.md"\n`
    );

    const update = await runQmd(["update"], {
      cwd: absoluteDir,
      dbPath: env.dbPath,
      configDir: env.configDir,
    });
    expect(update.exitCode).toBe(0);

    const { stdout, stderr, exitCode } = await runQmd(["ls", `qmd://${absoluteDir}/`], {
      cwd: absoluteDir,
      dbPath: env.dbPath,
      configDir: env.configDir,
    });
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain(`qmd://${absoluteDir}/root.md`);
  });

  test("lists an absolute-path collection from a raw path using the longest prefix match", async () => {
    const env = await createIsolatedTestEnv("absolute-raw-path");
    const parentCollectionName = await mkdtemp(join(tmpdir(), "qmd-absolute-parent-name-"));
    const childCollectionName = join(parentCollectionName, "nested");
    const parentDataDir = await mkdtemp(join(tmpdir(), "qmd-absolute-parent-data-"));
    const childDataDir = await mkdtemp(join(tmpdir(), "qmd-absolute-child-data-"));
    await writeFile(join(parentDataDir, "parent.md"), "# Parent collection\n");
    await writeFile(join(childDataDir, "child.md"), "# Child collection\n");
    await writeFile(
      join(env.configDir, "index.yml"),
      `collections:\n  "${parentCollectionName}":\n    path: "${parentDataDir}"\n    pattern: "**/*.md"\n  "${childCollectionName}":\n    path: "${childDataDir}"\n    pattern: "**/*.md"\n`
    );

    const update = await runQmd(["update"], {
      cwd: parentDataDir,
      dbPath: env.dbPath,
      configDir: env.configDir,
    });
    expect(update.exitCode).toBe(0);

    const { stdout, stderr, exitCode } = await runQmd(["ls", `${childCollectionName}/`], {
      cwd: childDataDir,
      dbPath: env.dbPath,
      configDir: env.configDir,
    });
    expect(stderr).toBe("");
    expect(exitCode).toBe(0);
    expect(stdout).toContain(`qmd://${childCollectionName}/child.md`);
    expect(stdout).not.toContain("No files found");
    expect(stdout).not.toContain(`qmd://${parentCollectionName}/parent.md`);
  });

  test("handles non-existent collection", async () => {
    const { stderr, exitCode } = await runQmd(["ls", "nonexistent"], { dbPath: localDbPath });
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Collection not found");
  });
});

describe("CLI Collection Commands", () => {
  let localDbPath: string;

  beforeEach(async () => {
    // Use a fresh database for this test suite
    localDbPath = getFreshDbPath();
    // Index some files first to create a collection
    await runQmd(["collection", "add", "."], { dbPath: localDbPath });
  });

  test("lists collections", async () => {
    const { stdout, exitCode } = await runQmd(["collection", "list"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Collections");
    expect(stdout).toContain("fixtures");
    expect(stdout).toContain("qmd://fixtures/");
    expect(stdout).toContain("Pattern:");
    expect(stdout).toContain("Files:");
  });

  test("removes a collection", async () => {
    // First verify the collection exists
    const { stdout: listBefore } = await runQmd(["collection", "list"], { dbPath: localDbPath });
    expect(listBefore).toContain("fixtures");

    // Remove it
    const { stdout, exitCode } = await runQmd(["collection", "remove", "fixtures"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ Removed collection 'fixtures'");
    expect(stdout).toContain("Deleted");

    // Verify it's gone
    const { stdout: listAfter } = await runQmd(["collection", "list"], { dbPath: localDbPath });
    expect(listAfter).not.toContain("fixtures");
  });

  test("handles removing non-existent collection", async () => {
    const { stderr, exitCode } = await runQmd(["collection", "remove", "nonexistent"], { dbPath: localDbPath });
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Collection not found");
  });

  test("handles missing remove argument", async () => {
    const { stderr, exitCode } = await runQmd(["collection", "remove"], { dbPath: localDbPath });
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Usage:");
  });

  test("handles unknown subcommand", async () => {
    const { stderr, exitCode } = await runQmd(["collection", "invalid"], { dbPath: localDbPath });
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Unknown subcommand");
  });

  test("renames a collection", async () => {
    // First verify the collection exists
    const { stdout: listBefore } = await runQmd(["collection", "list"], { dbPath: localDbPath });
    expect(listBefore).toContain("qmd://fixtures/");

    // Rename it
    const { stdout, exitCode } = await runQmd(["collection", "rename", "fixtures", "my-fixtures"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("✓ Renamed collection 'fixtures' to 'my-fixtures'");
    expect(stdout).toContain("qmd://fixtures/");
    expect(stdout).toContain("qmd://my-fixtures/");

    // Verify the new name exists and old name is gone
    const { stdout: listAfter } = await runQmd(["collection", "list"], { dbPath: localDbPath });
    expect(listAfter).toContain("qmd://my-fixtures/");
    expect(listAfter).not.toContain("qmd://fixtures/"); // Old collection should not appear
  });

  test("handles renaming non-existent collection", async () => {
    const { stderr, exitCode } = await runQmd(["collection", "rename", "nonexistent", "newname"], { dbPath: localDbPath });
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Collection not found");
  });

  test("handles renaming to existing collection name", async () => {
    // Create a second collection in a temp directory
    const tempDir = await mkdtemp(join(tmpdir(), "qmd-second-"));
    await writeFile(join(tempDir, "test.md"), "# Test");
    const addResult = await runQmd(["collection", "add", tempDir, "--name", "second"], { dbPath: localDbPath });

    if (addResult.exitCode !== 0) {
      console.error("Failed to add second collection:", addResult.stderr);
    }
    expect(addResult.exitCode).toBe(0);

    // Verify both collections exist
    const { stdout: listBoth } = await runQmd(["collection", "list"], { dbPath: localDbPath });
    expect(listBoth).toContain("qmd://fixtures/");
    expect(listBoth).toContain("qmd://second/");

    // Try to rename fixtures to second (which already exists)
    const { stderr, exitCode } = await runQmd(["collection", "rename", "fixtures", "second"], { dbPath: localDbPath });
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Collection name already exists");
  });

  test("handles missing rename arguments", async () => {
    const { stderr: stderr1, exitCode: exitCode1 } = await runQmd(["collection", "rename"], { dbPath: localDbPath });
    expect(exitCode1).toBe(1);
    expect(stderr1).toContain("Usage:");

    const { stderr: stderr2, exitCode: exitCode2 } = await runQmd(["collection", "rename", "fixtures"], { dbPath: localDbPath });
    expect(exitCode2).toBe(1);
    expect(stderr2).toContain("Usage:");
  });
});

// =============================================================================
// Collection Ignore Patterns
// =============================================================================

describe("collection ignore patterns", () => {
  let localDbPath: string;
  let localConfigDir: string;
  let ignoreTestDir: string;

  beforeAll(async () => {
    const env = await createIsolatedTestEnv("ignore-patterns");
    localDbPath = env.dbPath;
    localConfigDir = env.configDir;

    // Create directory structure with subdirectories to ignore
    ignoreTestDir = join(testDir, "ignore-fixtures");
    await mkdir(join(ignoreTestDir, "notes"), { recursive: true });
    await mkdir(join(ignoreTestDir, "sessions"), { recursive: true });
    await mkdir(join(ignoreTestDir, "sessions", "2026-03"), { recursive: true });
    await mkdir(join(ignoreTestDir, "archive"), { recursive: true });

    // Files that should be indexed
    await writeFile(join(ignoreTestDir, "readme.md"), "# Main readme\nThis should be indexed.");
    await writeFile(join(ignoreTestDir, "notes", "note1.md"), "# Note 1\nThis is a personal note.");

    // Files that should be ignored
    await writeFile(join(ignoreTestDir, "sessions", "session1.md"), "# Session 1\nThis session should be ignored.");
    await writeFile(join(ignoreTestDir, "sessions", "2026-03", "session2.md"), "# Session 2\nNested session should also be ignored.");
    await writeFile(join(ignoreTestDir, "archive", "old.md"), "# Old stuff\nThis archive file should be ignored.");
  });

  test("ignore patterns exclude matching files from indexing", async () => {
    // Write YAML config with ignore patterns
    await writeFile(
      join(localConfigDir, "index.yml"),
      `collections:
  ignoretst:
    path: ${ignoreTestDir}
    pattern: "**/*.md"
    ignore:
      - "sessions/**"
      - "archive/**"
`
    );

    const { stdout, exitCode } = await runQmd(["update"], {
      cwd: ignoreTestDir,
      dbPath: localDbPath,
      configDir: localConfigDir,
    });
    expect(exitCode).toBe(0);
    // Should index 2 files (readme.md + notes/note1.md), not 5
    expect(stdout).toContain("2 new");
  });

  test("ignored files are not searchable", async () => {
    const { stdout, exitCode } = await runQmd(["search", "session", "-n", "10"], {
      cwd: ignoreTestDir,
      dbPath: localDbPath,
      configDir: localConfigDir,
    });
    // Should find no results since sessions/ was ignored
    if (exitCode === 0) {
      expect(stdout).not.toContain("session1");
      expect(stdout).not.toContain("session2");
    }
  });

  test("non-ignored files are searchable", async () => {
    const { stdout, exitCode } = await runQmd(["search", "personal note", "-n", "10"], {
      cwd: ignoreTestDir,
      dbPath: localDbPath,
      configDir: localConfigDir,
    });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("note1");
  });

  test("status shows ignore patterns", async () => {
    const { stdout, exitCode } = await runQmd(["collection", "list"], {
      cwd: ignoreTestDir,
      dbPath: localDbPath,
      configDir: localConfigDir,
    });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Ignore:");
    expect(stdout).toContain("sessions/**");
    expect(stdout).toContain("archive/**");
  });

  test("collection without ignore indexes all files", async () => {
    // Create a second collection without ignore
    const env2 = await createIsolatedTestEnv("no-ignore");
    await writeFile(
      join(env2.configDir, "index.yml"),
      `collections:
  allfiles:
    path: ${ignoreTestDir}
    pattern: "**/*.md"
`
    );

    const { stdout, exitCode } = await runQmd(["update"], {
      cwd: ignoreTestDir,
      dbPath: env2.dbPath,
      configDir: env2.configDir,
    });
    expect(exitCode).toBe(0);
    // Should index all 5 files
    expect(stdout).toContain("5 new");
  });
});

// =============================================================================
// Output Format Tests - qmd:// URIs, context, and docid
// =============================================================================

describe("search output formats", () => {
  let localDbPath: string;
  let localConfigDir: string;
  const collName = "fixtures";

  beforeAll(async () => {
    const env = await createIsolatedTestEnv("output-format");
    localDbPath = env.dbPath;
    localConfigDir = env.configDir;

    // Add collection
    const { exitCode, stderr } = await runQmd(
      ["collection", "add", fixturesDir, "--name", collName],
      { dbPath: localDbPath, configDir: localConfigDir }
    );
    if (exitCode !== 0) console.error("collection add failed:", stderr);
    expect(exitCode).toBe(0);

    // Add context
    await runQmd(["context", "add", `qmd://${collName}/`, "Test fixtures for QMD"], { dbPath: localDbPath, configDir: localConfigDir });
  });

  test("search --json includes qmd:// path, docid, and context", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "--json", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    const results = JSON.parse(stdout);
    expect(results.length).toBeGreaterThan(0);

    const result = results[0];
    expect(result.file).toMatch(new RegExp(`^qmd://${collName}/`));
    expect(result.docid).toMatch(/^#[a-f0-9]{6}$/);
    expect(result.context).toBe("Test fixtures for QMD");
    // Ensure no full filesystem paths
    expect(result.file).not.toMatch(/^\/Users\//);
    expect(result.file).not.toMatch(/^\/home\//);
  });

  test("custom-index search links include ?index= and can be passed back to qmd get", async () => {
    const env = await createIsolatedTestEnv("custom-index-links");
    const customColl = "fixtures-alt";
    const customIndex = "release-notes";
    const customCacheDir = join(testDir, `cache-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    await mkdir(customCacheDir, { recursive: true });

    const sharedEnv = {
      INDEX_PATH: "",
      XDG_CACHE_HOME: customCacheDir,
    };

    const addResult = await runQmd(
      ["--index", customIndex, "collection", "add", fixturesDir, "--name", customColl],
      { dbPath: env.dbPath, configDir: env.configDir, env: sharedEnv }
    );
    expect(addResult.exitCode).toBe(0);

    const searchResult = await runQmd(
      ["--index", customIndex, "search", "test", "--json", "-n", "1"],
      { dbPath: env.dbPath, configDir: env.configDir, env: sharedEnv }
    );
    expect(searchResult.exitCode).toBe(0);

    const results = JSON.parse(searchResult.stdout);
    const file = results[0]?.file;
    expect(file).toMatch(new RegExp(`^qmd://${customColl}/.+\\?index=${customIndex}$`));

    const getResult = await runQmd(
      ["get", file, "-l", "2"],
      { dbPath: env.dbPath, configDir: env.configDir, env: sharedEnv }
    );
    expect(getResult.exitCode).toBe(0);
    expect(getResult.stdout.trim().length).toBeGreaterThan(0);
  });

  test("search --files includes qmd:// path, docid, and context", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "--files", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    // Format: #docid,score,qmd://collection/path,"context"
    expect(stdout).toMatch(new RegExp(`^#[a-f0-9]{6},[\\d.]+,qmd://${collName}/`, "m"));
    expect(stdout).toContain("Test fixtures for QMD");
    // Ensure no full filesystem paths
    expect(stdout).not.toMatch(/\/Users\//);
    expect(stdout).not.toMatch(/\/home\//);
  });

  test("search --csv includes qmd:// path, docid, and context", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "--csv", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    // Header should include context
    expect(stdout).toMatch(/^docid,score,file,title,context,line,snippet$/m);
    // Data rows should have qmd:// paths and context
    expect(stdout).toMatch(new RegExp(`#[a-f0-9]{6},[\\d.]+,qmd://${collName}/`));
    expect(stdout).toContain("Test fixtures for QMD");
    // Ensure no full filesystem paths
    expect(stdout).not.toMatch(/\/Users\//);
    expect(stdout).not.toMatch(/\/home\//);
  });

  test("search --md includes docid, context, and qmd:// file line", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "--md", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    expect(stdout).toMatch(/\*\*docid:\*\* `#[a-f0-9]{6}`/);
    expect(stdout).toContain("**context:** Test fixtures for QMD");
    // The file path must be a qmd:// URI so the model can pipe it back into
    // `qmd get` without having to reassemble a collection-relative string.
    expect(stdout).toMatch(new RegExp(`\\*\\*file:\\*\\* \`qmd://${collName}/`));
  });

  test("search --xml includes qmd:// path, docid, and context", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "--xml", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    expect(stdout).toMatch(new RegExp(`<file docid="#[a-f0-9]{6}" name="qmd://${collName}/`));
    expect(stdout).toContain('context="Test fixtures for QMD"');
    // Ensure no full filesystem paths
    expect(stdout).not.toMatch(/\/Users\//);
    expect(stdout).not.toMatch(/\/home\//);
  });

  test("search --full-path --json swaps qmd:// for absolute realpath when cwd is unrelated", async () => {
    // Use "/" as cwd so the fixtures path (under tmpdir) is NOT a subpath of $PWD.
    const { stdout, exitCode } = await runQmd(
      ["search", "test", "--full-path", "--json", "-n", "1"],
      { dbPath: localDbPath, configDir: localConfigDir, cwd: "/" }
    );
    expect(exitCode).toBe(0);
    const results = JSON.parse(stdout);
    expect(results.length).toBeGreaterThan(0);
    const result = results[0];
    expect(result.file).not.toMatch(/^qmd:\/\//);
    // Must be an absolute path ending in .md.
    expect(result.file).toMatch(/^\/.+\.md$/);
    // --full-path: the on-disk path replaces the docid as the identifier.
    expect(result.docid).toBeUndefined();
  });

  test("search --full-path --json uses ./-prefixed $PWD-relative path when in a parent of the file", async () => {
    const { stdout, exitCode } = await runQmd(
      ["search", "test", "--full-path", "--json", "-n", "1"],
      { dbPath: localDbPath, configDir: localConfigDir, cwd: fixturesDir }
    );
    expect(exitCode).toBe(0);
    const results = JSON.parse(stdout);
    expect(results.length).toBeGreaterThan(0);
    const result = results[0];
    expect(result.file).not.toMatch(/^qmd:\/\//);
    // Must start with "./" so it's unambiguously a filesystem path and not
    // mistaken for a bare collection-relative string.
    expect(result.file.startsWith("./")).toBe(true);
    expect(result.file).not.toMatch(/^\.\.\//);
    expect(result.file).toMatch(/\.md$/);
  });

  test("search --full-path default CLI format shows on-disk path and drops the docid", async () => {
    const { stdout, exitCode } = await runQmd(
      ["search", "test", "--full-path", "-n", "1"],
      { dbPath: localDbPath, configDir: localConfigDir, cwd: "/" }
    );
    expect(exitCode).toBe(0);
    // eslint-disable-next-line no-control-regex
    const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "").replace(/\x1b\]8;;[^\x07]*\x07/g, "");
    const plain = stripAnsi(stdout);
    expect(plain).not.toMatch(/qmd:\/\//);
    expect(plain).toMatch(/^\/.+\.md/m);
    // No `#docid` suffix when --full-path is set.
    expect(plain).not.toMatch(/#[a-f0-9]{6}\s*$/m);
  });

  test("search --full-path --md uses on-disk path in heading and drops the docid", async () => {
    const { stdout, exitCode } = await runQmd(
      ["search", "test", "--full-path", "--md", "-n", "1"],
      { dbPath: localDbPath, configDir: localConfigDir, cwd: "/" }
    );
    expect(exitCode).toBe(0);
    expect(stdout).not.toMatch(/qmd:\/\//);
    expect(stdout).not.toMatch(/\*\*docid:\*\*/);
    expect(stdout).toMatch(/\*\*file:\*\* `\/.+\.md`/);
  });

  test("search --format json matches the legacy --json behavior", async () => {
    const a = await runQmd(["search", "test", "--format", "json", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    const b = await runQmd(["search", "test", "--json", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(a.exitCode).toBe(0);
    expect(b.exitCode).toBe(0);
    // Both must yield valid JSON with at least one result.
    const ar = JSON.parse(a.stdout);
    const br = JSON.parse(b.stdout);
    expect(ar.length).toBeGreaterThan(0);
    expect(br.length).toBeGreaterThan(0);
    // Identical first-result file path (the rest may differ in score formatting only).
    expect(ar[0].file).toBe(br[0].file);
  });

  test("search --format md works equivalent to legacy --md", async () => {
    const a = await runQmd(["search", "test", "--format", "md", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(a.exitCode).toBe(0);
    expect(a.stdout).toMatch(/\*\*docid:\*\* `#[a-f0-9]{6}`/);
    expect(a.stdout).toMatch(new RegExp(`\\*\\*file:\\*\\* \`qmd://${collName}/`));
  });

  test("search --format with an unknown kind fails cleanly", async () => {
    const { exitCode, stderr } = await runQmd(["search", "test", "--format", "yaml", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).not.toBe(0);
    expect(stderr).toContain("Unknown --format value");
  });

  test("search default CLI format includes plain qmd:// path, docid, and context in non-TTY mode", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    // runQmd uses piped stdio, so stdout is non-TTY and should not contain OSC 8 links.
    expect(stdout).toMatch(new RegExp(`^qmd://${collName}/.*#[a-f0-9]{6}`, "m"));
    expect(stdout).toContain("Context: Test fixtures for QMD");
    expect(stdout).not.toContain("\x1b]8;;");
    // Ensure no full filesystem paths
    expect(stdout).not.toMatch(/\/Users\//);
    expect(stdout).not.toMatch(/\/home\//);
    // The visible path must NOT be the bare collection-relative form
    // (a leading `${collName}/foo.md` would be "relative to nowhere").
    // Strip ANSI and OSC 8 sequences then assert no result line starts with
    // a bare collection-relative path missing the qmd:// scheme.
    // eslint-disable-next-line no-control-regex
    const stripAnsi = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "").replace(/\x1b\]8;;[^\x07]*\x07/g, "");
    const plain = stripAnsi(stdout);
    expect(plain).not.toMatch(new RegExp(`^${collName}/`, "m"));
  });
});

describe("editor URI templates", () => {
  test("buildEditorUri expands path, line, and col placeholders", () => {
    const uri = buildEditorUri(
      "vscode://file/{path}:{line}:{col}",
      "/tmp/my notes/readme.md",
      42,
      1,
    );

    expect(uri).toBe("vscode://file//tmp/my%20notes/readme.md:42:1");
  });

  test("buildEditorUri supports {column} alias", () => {
    const uri = buildEditorUri(
      "cursor://file/{path}:{line}:{column}",
      "/tmp/docs/api.md",
      7,
      3,
    );

    expect(uri).toBe("cursor://file//tmp/docs/api.md:7:3");
  });

  test("termLink returns plain text when stdout is not a TTY", () => {
    const linked = termLink("docs/api.md:12", "vscode://file//tmp/docs/api.md:12:1", false);

    expect(linked).toBe("docs/api.md:12");
  });

  test("termLink emits OSC 8 hyperlinks when stdout is a TTY", () => {
    const linked = termLink("docs/api.md:12", "vscode://file//tmp/docs/api.md:12:1", true);

    expect(linked).toBe("\x1b]8;;vscode://file//tmp/docs/api.md:12:1\x07docs/api.md:12\x1b]8;;\x07");
  });
});

// =============================================================================
// Get Command Path Normalization Tests
// =============================================================================

describe("get command path normalization", () => {
  let localDbPath: string;
  let localConfigDir: string;
  const collName = "fixtures";

  beforeAll(async () => {
    const env = await createIsolatedTestEnv("get-paths");
    localDbPath = env.dbPath;
    localConfigDir = env.configDir;

    const { exitCode, stderr } = await runQmd(
      ["collection", "add", fixturesDir, "--name", collName],
      { dbPath: localDbPath, configDir: localConfigDir }
    );
    if (exitCode !== 0) console.error("collection add failed:", stderr);
    expect(exitCode).toBe(0);
  });

  test("get with qmd://collection/path format", async () => {
    const { stdout, exitCode } = await runQmd(["get", `qmd://${collName}/test1.md`, "-l", "3"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Test Document 1");
  });

  test("get with collection/path format (no scheme)", async () => {
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md`, "-l", "3"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Test Document 1");
  });

  test("get with //collection/path format", async () => {
    const { stdout, exitCode } = await runQmd(["get", `//${collName}/test1.md`, "-l", "3"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Test Document 1");
  });

  test("get with qmd:////collection/path format (extra slashes)", async () => {
    const { stdout, exitCode } = await runQmd(["get", `qmd:////${collName}/test1.md`, "-l", "3"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Test Document 1");
  });

  test("get with path:line format", async () => {
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md:3`, "-l", "2"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    // Should start from line 3, not line 1
    expect(stdout).not.toMatch(/^# Test Document 1$/m);
  });

  test("get with qmd://path:line format", async () => {
    const { stdout, exitCode } = await runQmd(["get", `qmd://${collName}/test1.md:3`, "-l", "2"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    // Should start from line 3, not line 1
    expect(stdout).not.toMatch(/^# Test Document 1$/m);
  });

  test("get with path:from:count format reads a bounded range", async () => {
    // Lines: 1 "# Test Document 1", 5 "It has multiple lines...",
    //        6 "Line 6 is here.", 7 "Line 7 is here."
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md:5:2`], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("It has multiple lines");
    expect(stdout).toContain("Line 6 is here.");
    // Bounded to 2 lines: must not include the start of the file or line 7
    expect(stdout).not.toMatch(/^# Test Document 1$/m);
    expect(stdout).not.toContain("Line 7 is here.");
  });

  test("get with qmd://path:from:count format reads a bounded range", async () => {
    const { stdout, exitCode } = await runQmd(["get", `qmd://${collName}/test1.md:5:2`], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("It has multiple lines");
    expect(stdout).toContain("Line 6 is here.");
    expect(stdout).not.toMatch(/^# Test Document 1$/m);
    expect(stdout).not.toContain("Line 7 is here.");
  });

  test("explicit -l overrides the :count in path:from:count", async () => {
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md:5:2`, "-l", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("It has multiple lines");
    expect(stdout).not.toContain("Line 6 is here.");
  });

  test("get header includes canonical qmd:// path and a #docid", async () => {
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md`], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    // First line of output identifies the document by path + docid.
    expect(stdout).toMatch(new RegExp(`^qmd://${collName}/test1\\.md\\s+#[a-f0-9]{6}`, "m"));
  });

  test("get shows line numbers by default", async () => {
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md`], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toMatch(/^1: # Test Document 1$/m);
    expect(stdout).toMatch(/^6: Line 6 is here\.$/m);
  });

  test("get --no-line-numbers returns raw content", async () => {
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md`, "--no-line-numbers"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).not.toMatch(/^1: /m);
    expect(stdout).toMatch(/^# Test Document 1$/m);
  });

  test("get line numbers reflect the start line of a range", async () => {
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md:5:2`], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    // Numbering starts at the requested line, not at 1.
    expect(stdout).toMatch(/^5: It has multiple lines/m);
    expect(stdout).not.toMatch(/^1: /m);
  });

  test("get --full-path shows ./-prefixed path when file is under $PWD", async () => {
    // Default runQmd cwd is fixturesDir, and test1.md lives in fixturesDir,
    // so the rendered path must be relative-with-./ prefix.
    const { stdout, exitCode } = await runQmd(["get", `${collName}/test1.md`, "--full-path"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);
    expect(stdout).toMatch(/^\.\/test1\.md$/m);
    expect(stdout).not.toContain("qmd://");
    expect(stdout).not.toMatch(/#[a-f0-9]{6}/);
    // Body still present and line-numbered.
    expect(stdout).toMatch(/^1: # Test Document 1$/m);
  });

  test("get --full-path shows absolute path when file is outside $PWD", async () => {
    const { stdout, exitCode } = await runQmd(
      ["get", `${collName}/test1.md`, "--full-path"],
      { dbPath: localDbPath, configDir: localConfigDir, cwd: "/" }
    );
    expect(exitCode).toBe(0);
    // Absolute realpath (allow macOS /var → /private/var).
    expect(stdout).toMatch(/^\/.+\/test1\.md$/m);
    expect(stdout).not.toMatch(/^\.\//m);
    expect(stdout).not.toContain("qmd://");
    expect(stdout).not.toMatch(/#[a-f0-9]{6}/);
  });

  test("get --full-path falls back to qmd:// + docid when the file is gone", async () => {
    // Index a doc, then delete the underlying file so the fs path no longer exists.
    const env = await createIsolatedTestEnv("full-path-fallback");
    const collectionDir = join(testDir, `gone-fixtures-${Date.now()}`);
    await mkdir(collectionDir, { recursive: true });
    const gonePath = join(collectionDir, "gone.md");
    await writeFile(gonePath, "# Gone\n\nbody line\n");
    const add = await runQmd(["collection", "add", collectionDir, "--name", "gonecoll"], { dbPath: env.dbPath, configDir: env.configDir });
    expect(add.exitCode).toBe(0);
    await rm(gonePath);

    const { stdout, exitCode } = await runQmd(["get", "gonecoll/gone.md", "--full-path"], { dbPath: env.dbPath, configDir: env.configDir });
    expect(exitCode).toBe(0);
    expect(stdout).toMatch(new RegExp(`^qmd://gonecoll/gone\\.md\\s+#[a-f0-9]{6}`, "m"));
  });
});

describe("get command missing and ignored paths", () => {
  let localDbPath: string;
  let localConfigDir: string;
  let getTestDir: string;

  beforeAll(async () => {
    const env = await createIsolatedTestEnv("get-missing-ignored");
    localDbPath = env.dbPath;
    localConfigDir = env.configDir;
    getTestDir = join(testDir, "get-missing-ignored-fixtures");

    await mkdir(join(getTestDir, "public"), { recursive: true });
    await mkdir(join(getTestDir, "private"), { recursive: true });
    await writeFile(join(getTestDir, "public", "prompt-log.md"), "# Public Log\nThis indexed file must not be returned for missing paths.");
    await writeFile(join(getTestDir, "private", "log.md"), "# Private Log\nThis ignored file must not be indexed.");

    await writeFile(
      join(localConfigDir, "index.yml"),
      `collections:
  gettst:
    path: ${getTestDir}
    pattern: "**/*.md"
    ignore:
      - "private/**"
`
    );

    const { exitCode, stderr } = await runQmd(["update"], {
      cwd: getTestDir,
      dbPath: localDbPath,
      configDir: localConfigDir,
    });
    if (exitCode !== 0) console.error("update failed:", stderr);
    expect(exitCode).toBe(0);
  });

  test("get reports ignored paths instead of falling back to same-basename fuzzy match", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["get", "private/log.md"], {
      cwd: getTestDir,
      dbPath: localDbPath,
      configDir: localConfigDir,
    });

    expect(exitCode).toBe(1);
    expect(stdout).not.toContain("Public Log");
    expect(stderr).toContain("excluded by ignore rule");
    expect(stderr).toContain("private/log.md");
    expect(stderr).toContain("private/**");
  });

  test("get does not fall back to same-basename fuzzy match for missing paths", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["get", "missing/log.md"], {
      cwd: getTestDir,
      dbPath: localDbPath,
      configDir: localConfigDir,
    });

    expect(exitCode).toBe(1);
    expect(stdout).not.toContain("Public Log");
    expect(stderr).toContain("Document not found");
  });
});

// =============================================================================
// Status and Collection List - No Full Paths
// =============================================================================

describe("status and collection list hide filesystem paths", () => {
  let localDbPath: string;
  let localConfigDir: string;
  const collName = "fixtures";

  beforeAll(async () => {
    const env = await createIsolatedTestEnv("status-paths");
    localDbPath = env.dbPath;
    localConfigDir = env.configDir;

    const { exitCode, stderr } = await runQmd(
      ["collection", "add", fixturesDir, "--name", collName],
      { dbPath: localDbPath, configDir: localConfigDir }
    );
    if (exitCode !== 0) console.error("collection add failed:", stderr);
    expect(exitCode).toBe(0);
  });

  test("status does not show full filesystem paths", async () => {
    const { stdout, exitCode } = await runQmd(["status"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    // Should show qmd:// URIs
    expect(stdout).toContain(`qmd://${collName}/`);
    // Should NOT show full filesystem paths (except for the index location which is ok)
    const lines = stdout.split('\n').filter(l => !l.includes('Index:'));
    const pathLines = lines.filter(l => l.includes('/Users/') || l.includes('/home/') || l.includes('/tmp/'));
    expect(pathLines.length).toBe(0);
  });

  test("doctor does not show full filesystem paths", async () => {
    const { stdout, exitCode } = await runQmd(["doctor"], {
      dbPath: localDbPath,
      configDir: localConfigDir,
      env: { QMD_DOCTOR_DEVICE_PROBE: "0" },
    });
    expect(exitCode).toBe(0);

    expect(stdout).toContain("QMD Doctor");
    const lines = stdout.split('\n').filter(l => !l.includes('Index:') && !l.includes('INDEX_PATH=') && !l.includes('QMD_CONFIG_DIR='));
    const pathLines = lines.filter(l => l.includes('/Users/') || l.includes('/home/') || l.includes('/tmp/'));
    expect(pathLines.length).toBe(0);
  }, 20000);

  test("collection list does not show full filesystem paths", async () => {
    const { stdout, exitCode } = await runQmd(["collection", "list"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    // Should show qmd:// URIs
    expect(stdout).toContain(`qmd://${collName}/`);
    // Should NOT show Path: lines with filesystem paths
    expect(stdout).not.toMatch(/Path:\s+\//);
  });
});

// =============================================================================
// MCP HTTP Daemon Lifecycle
// =============================================================================

describe("mcp http daemon", () => {
  let daemonTestDir: string;
  let daemonCacheDir: string; // XDG_CACHE_HOME value (the qmd/ subdir is created automatically)
  let daemonDbPath: string;
  let daemonConfigDir: string;

  // Track spawned PIDs for cleanup
  const spawnedPids: number[] = [];

  /** Get path to PID file inside the test cache dir */
  function pidPath(): string {
    return join(daemonCacheDir, "qmd", "mcp.pid");
  }

  /** Run qmd with test-isolated env (cache, db, config) */
  async function runDaemonQmd(
    args: string[],
  ): Promise<{ stdout: string; stderr: string; exitCode: number }> {
    return runQmd(args, {
      dbPath: daemonDbPath,
      configDir: daemonConfigDir,
      env: { XDG_CACHE_HOME: daemonCacheDir },
    });
  }

  /** Spawn a foreground HTTP server (non-blocking) and return the process */
  function spawnHttpServer(
    port: number,
    options: { args?: string[]; env?: Record<string, string> } = {},
  ): import("child_process").ChildProcess {
    const runner = qmdRunnerArgs([...(options.args ?? []), "mcp", "--http", "--port", String(port)]);
    const proc = spawn(runner.command, runner.args, {
      cwd: fixturesDir,
      env: {
        ...process.env,
        INDEX_PATH: daemonDbPath,
        QMD_CONFIG_DIR: daemonConfigDir,
        PWD: fixturesDir,
        ...options.env,
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (proc.pid) spawnedPids.push(proc.pid);
    return proc;
  }

  /** Wait for HTTP server to become ready */
  async function waitForServer(port: number, timeoutMs = 5000): Promise<boolean> {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      try {
        const res = await fetch(`http://localhost:${port}/health`);
        if (res.ok) return true;
      } catch { /* not ready yet */ }
      await sleep(200);
    }
    return false;
  }

  /** Pick a random high port unlikely to conflict */
  function randomPort(): number {
    return 10000 + Math.floor(Math.random() * 50000);
  }

  beforeAll(async () => {
    daemonTestDir = await mkdtemp(join(tmpdir(), "qmd-daemon-test-"));
    daemonCacheDir = join(daemonTestDir, "cache");
    daemonDbPath = join(daemonTestDir, "test.sqlite");
    daemonConfigDir = join(daemonTestDir, "config");

    await mkdir(join(daemonCacheDir, "qmd"), { recursive: true });
    await mkdir(daemonConfigDir, { recursive: true });
    await writeFile(join(daemonConfigDir, "index.yml"), "collections: {}\n");
  });

  afterAll(async () => {
    // Kill any leftover spawned processes
    for (const pid of spawnedPids) {
      try { process.kill(pid, "SIGTERM"); } catch { /* already dead */ }
    }
    // Also clean up via PID file if present
    try {
      const pf = pidPath();
      if (existsSync(pf)) {
        const pid = parseInt(readFileSync(pf, "utf-8").trim());
        try { process.kill(pid, "SIGTERM"); } catch {}
        unlinkSync(pf);
      }
    } catch {}

    await rm(daemonTestDir, { recursive: true, force: true });
  });

  // -------------------------------------------------------------------------
  // Foreground HTTP
  // -------------------------------------------------------------------------

  test("foreground HTTP server starts and responds to health check", async () => {
    const port = randomPort();
    const proc = spawnHttpServer(port);

    try {
      const ready = await waitForServer(port);
      expect(ready).toBe(true);

      const res = await fetch(`http://localhost:${port}/health`);
      expect(res.status).toBe(200);
      const body = await res.json();
      expect(body.status).toBe("ok");
    } finally {
      const closed = new Promise(r => proc.once("close", r));
      proc.kill("SIGTERM");
      await closed;
    }
  });

  test("foreground HTTP server honors --index when selecting the store", async () => {
    const customIndex = "mcp-alt-index";
    const customCacheDir = join(daemonTestDir, `cache-index-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    const customConfigDir = join(daemonTestDir, `config-index-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    await mkdir(customCacheDir, { recursive: true });
    await mkdir(customConfigDir, { recursive: true });

    const addResult = await runQmd(
      ["--index", customIndex, "collection", "add", fixturesDir, "--name", "mcp-fixtures"],
      {
        dbPath: daemonDbPath,
        configDir: customConfigDir,
        env: {
          INDEX_PATH: "",
          XDG_CACHE_HOME: customCacheDir,
        },
      },
    );
    expect(addResult.exitCode).toBe(0);

    const updateResult = await runQmd(
      ["--index", customIndex, "update"],
      {
        dbPath: daemonDbPath,
        configDir: customConfigDir,
        env: {
          INDEX_PATH: "",
          XDG_CACHE_HOME: customCacheDir,
        },
      },
    );
    expect(updateResult.exitCode).toBe(0);

    const port = randomPort();
    const proc = spawnHttpServer(port, {
      args: ["--index", customIndex],
      env: {
        INDEX_PATH: "",
        XDG_CACHE_HOME: customCacheDir,
        QMD_CONFIG_DIR: customConfigDir,
      },
    });

    try {
      const ready = await waitForServer(port);
      expect(ready).toBe(true);

      const res = await fetch(`http://localhost:${port}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ searches: [{ type: "lex", query: "authentication" }], limit: 5, rerank: false }),
      });
      expect(res.status).toBe(200);
      const body = await res.json();
      const files = body.results.map((r: { file: string }) => r.file);
      expect(files.some((file: string) => file.includes("mcp-fixtures/notes/meeting.md"))).toBe(true);
    } finally {
      const closed = new Promise(r => proc.once("close", r));
      proc.kill("SIGTERM");
      await closed;
    }
  }, 10000);

  // -------------------------------------------------------------------------
  // Daemon lifecycle
  // -------------------------------------------------------------------------

  test("--daemon writes PID file and starts server", async () => {
    const port = randomPort();
    const { stdout, exitCode } = await runDaemonQmd([
      "mcp", "--http", "--daemon", "--port", String(port),
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain(`http://localhost:${port}/mcp`);

    // PID file should exist
    expect(existsSync(pidPath())).toBe(true);

    const pid = parseInt(readFileSync(pidPath(), "utf-8").trim());
    spawnedPids.push(pid);

    // Server should be reachable
    const ready = await waitForServer(port);
    expect(ready).toBe(true);

    // Clean up
    process.kill(pid, "SIGTERM");
    await sleep(500);
    try { unlinkSync(pidPath()); } catch {}
  });

  test("stop kills daemon and removes PID file", async () => {
    const port = randomPort();
    // Start daemon
    const { exitCode: startCode } = await runDaemonQmd([
      "mcp", "--http", "--daemon", "--port", String(port),
    ]);
    expect(startCode).toBe(0);

    const pid = parseInt(readFileSync(pidPath(), "utf-8").trim());
    spawnedPids.push(pid);

    await waitForServer(port);

    // Stop it
    const { stdout: stopOut, exitCode: stopCode } = await runDaemonQmd(["mcp", "stop"]);
    expect(stopCode).toBe(0);
    expect(stopOut).toContain("Stopped");

    // PID file should be gone
    expect(existsSync(pidPath())).toBe(false);

    // Process should be dead
    await sleep(500);
    expect(() => process.kill(pid, 0)).toThrow();
  });

  test("stop handles dead PID gracefully (cleans stale file)", async () => {
    // Write a PID file pointing to a dead process
    writeFileSync(pidPath(), "999999999");

    const { stdout, exitCode } = await runDaemonQmd(["mcp", "stop"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("stale");

    // PID file should be cleaned up
    expect(existsSync(pidPath())).toBe(false);
  });

  test("--daemon rejects if already running", async () => {
    const port = randomPort();
    // Start first daemon
    const { exitCode: firstCode } = await runDaemonQmd([
      "mcp", "--http", "--daemon", "--port", String(port),
    ]);
    expect(firstCode).toBe(0);

    const pid = parseInt(readFileSync(pidPath(), "utf-8").trim());
    spawnedPids.push(pid);

    await waitForServer(port);

    // Try to start second daemon — should fail
    const { stderr, exitCode } = await runDaemonQmd([
      "mcp", "--http", "--daemon", "--port", String(port + 1),
    ]);
    expect(exitCode).toBe(1);
    expect(stderr).toContain("Already running");

    // Clean up first daemon
    process.kill(pid, "SIGTERM");
    await sleep(500);
    try { unlinkSync(pidPath()); } catch {}
  });

  test("--daemon cleans stale PID file and starts fresh", async () => {
    // Write a stale PID file
    writeFileSync(pidPath(), "999999999");

    const port = randomPort();
    const { exitCode, stdout } = await runDaemonQmd([
      "mcp", "--http", "--daemon", "--port", String(port),
    ]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain(`http://localhost:${port}/mcp`);

    const pid = parseInt(readFileSync(pidPath(), "utf-8").trim());
    spawnedPids.push(pid);
    expect(pid).not.toBe(999999999);

    // Clean up
    const ready = await waitForServer(port);
    expect(ready).toBe(true);
    process.kill(pid, "SIGTERM");
    await sleep(500);
    try { unlinkSync(pidPath()); } catch {}
  });
});

// =============================================================================
// MCP stdio stdout hygiene
// =============================================================================

describe("mcp stdio launcher", () => {
  test("sets native llama/ggml quiet env before Node starts so stdout stays JSON-RPC only", async () => {
    const tempPackage = await mkdtemp(join(tmpdir(), "qmd-bin-mcp-"));
    try {
      await mkdir(join(tempPackage, "bin"), { recursive: true });
      await mkdir(join(tempPackage, "dist", "cli"), { recursive: true });
      await writeFile(join(tempPackage, "dist", "cli", "qmd.js"), "// fixture\n");
      await mkdir(join(tempPackage, "fake-bin"), { recursive: true });

      const qmdBin = join(tempPackage, "bin", "qmd");
      await copyFile(join(projectRoot, "bin", "qmd"), qmdBin);
      await chmod(qmdBin, 0o755);

      // Force the wrapper down the Node branch, then put our fake `node` first
      // in PATH. The fake node behaves like the native llama/ggml layer: it
      // writes a non-JSON stdout line unless qmd pre-seeded the documented
      // quiet env vars before launching JS.
      await writeFile(join(tempPackage, "package-lock.json"), "{}\n");
      const fakeNode = join(tempPackage, "fake-bin", "node");
      await writeFile(fakeNode, `#!/bin/sh
if [ "$(basename "$1")" = "qmd" ]; then
  exec "${process.execPath}" "$@"
else
  if [ "\${GGML_BACKEND_SILENT:-}" != "1" ]; then
    printf 'llama.cpp native log on stdout\\n'
  fi
  printf '{"jsonrpc":"2.0","id":1,"result":{"ok":true}}\\n'
fi
`);
      await chmod(fakeNode, 0o755);

      const proc = spawn(qmdBin, ["mcp"], {
        cwd: tempPackage,
        env: {
          ...process.env,
          PATH: `${join(tempPackage, "fake-bin")}:${process.env.PATH}`,
          LLAMA_LOG_LEVEL: "",
          GGML_LOG_LEVEL: "",
          GGML_BACKEND_SILENT: "",
        },
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      proc.stdout?.on("data", (chunk: Buffer) => { stdout += chunk.toString(); });
      proc.stderr?.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
      const exitCode = await new Promise<number>((resolve, reject) => {
        proc.once("error", reject);
        proc.on("close", (code) => resolve(code ?? 1));
      });

      expect(exitCode).toBe(0);
      expect(stderr).toBe("");
      const lines = stdout.trim().split("\n").filter(Boolean);
      expect(lines.length).toBeGreaterThan(0);
      for (const line of lines) {
        expect(() => JSON.parse(line)).not.toThrow();
      }
    } finally {
      await rm(tempPackage, { recursive: true, force: true });
    }
  });
});
