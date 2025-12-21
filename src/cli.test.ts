/**
 * CLI Integration Tests
 *
 * Tests all qmd CLI commands using a temporary test database via INDEX_PATH.
 * These tests spawn actual qmd processes to verify end-to-end functionality.
 */

import { describe, test, expect, beforeAll, afterAll, beforeEach } from "bun:test";
import { mkdtemp, rm, writeFile, mkdir } from "fs/promises";
import { tmpdir } from "os";
import { join } from "path";

// Test fixtures directory and database path
let testDir: string;
let testDbPath: string;
let testConfigDir: string;
let fixturesDir: string;
let testCounter = 0; // Unique counter for each test run

// Get the directory where this test file lives (same as qmd.ts)
const qmdDir = import.meta.dir;
const qmdScript = join(qmdDir, "qmd.ts");

// Helper to run qmd command with test database
async function runQmd(
  args: string[],
  options: { cwd?: string; env?: Record<string, string>; dbPath?: string; configDir?: string } = {}
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const workingDir = options.cwd || fixturesDir;
  const dbPath = options.dbPath || testDbPath;
  const configDir = options.configDir || testConfigDir;
  const proc = Bun.spawn(["bun", qmdScript, ...args], {
    cwd: workingDir,
    env: {
      ...process.env,
      INDEX_PATH: dbPath,
      QMD_CONFIG_DIR: configDir, // Use test config directory
      PWD: workingDir, // Must explicitly set PWD since getPwd() checks this
      ...options.env,
    },
    stdout: "pipe",
    stderr: "pipe",
  });

  const stdout = await new Response(proc.stdout).text();
  const stderr = await new Response(proc.stderr).text();
  const exitCode = await proc.exited;

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
  });

  test("shows help with no arguments", async () => {
    const { stdout, exitCode } = await runQmd([]);
    expect(exitCode).toBe(1);
    expect(stdout).toContain("Usage:");
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

  test("shows index status", async () => {
    const { stdout, exitCode } = await runQmd(["status"]);
    expect(exitCode).toBe(0);
    // Should show collection info
    expect(stdout).toContain("Collection");
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

  test("requires query argument", async () => {
    const { stdout, stderr, exitCode } = await runQmd(["search"]);
    expect(exitCode).toBe(1);
    // Error message goes to stderr
    expect(stderr).toContain("Usage:");
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
    // Should indicate unknown command
    expect(stderr).toContain("Unknown command");
  });

  test("uses INDEX_PATH environment variable", async () => {
    // Verify the test DB path is being used by creating a separate index
    const customDbPath = join(testDir, "custom.sqlite");
    const { exitCode } = await runQmd(["collection", "add", "."], {
      env: { INDEX_PATH: customDbPath },
    });
    expect(exitCode).toBe(0);

    // The custom database should exist
    const file = Bun.file(customDbPath);
    expect(await file.exists()).toBe(true);
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
    // handelize converts to lowercase
    expect(stdout).toContain("qmd://fixtures/readme.md");
    expect(stdout).toContain("qmd://fixtures/notes/meeting.md");
  });

  test("lists files with path prefix", async () => {
    const { stdout, exitCode } = await runQmd(["ls", "fixtures/notes"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("qmd://fixtures/notes/meeting.md");
    expect(stdout).toContain("qmd://fixtures/notes/ideas.md");
    // Should not include files outside the prefix (handelize converts to lowercase)
    expect(stdout).not.toContain("qmd://fixtures/readme.md");
  });

  test("lists files with virtual path", async () => {
    const { stdout, exitCode } = await runQmd(["ls", "qmd://fixtures/docs"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("qmd://fixtures/docs/api.md");
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

  test("search --md includes docid and context", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "--md", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    expect(stdout).toMatch(/\*\*docid:\*\* `#[a-f0-9]{6}`/);
    expect(stdout).toContain("**context:** Test fixtures for QMD");
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

  test("search default CLI format includes qmd:// path, docid, and context", async () => {
    const { stdout, exitCode } = await runQmd(["search", "test", "-n", "1"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    // First line should have qmd:// path and docid
    expect(stdout).toMatch(new RegExp(`^qmd://${collName}/.*#[a-f0-9]{6}`, "m"));
    expect(stdout).toContain("Context: Test fixtures for QMD");
    // Ensure no full filesystem paths
    expect(stdout).not.toMatch(/\/Users\//);
    expect(stdout).not.toMatch(/\/home\//);
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

  test("collection list does not show full filesystem paths", async () => {
    const { stdout, exitCode } = await runQmd(["collection", "list"], { dbPath: localDbPath, configDir: localConfigDir });
    expect(exitCode).toBe(0);

    // Should show qmd:// URIs
    expect(stdout).toContain(`qmd://${collName}/`);
    // Should NOT show Path: lines with filesystem paths
    expect(stdout).not.toMatch(/Path:\s+\//);
  });
});
