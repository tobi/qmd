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
let fixturesDir: string;
let testCounter = 0; // Unique counter for each test run

// Get the directory where this test file lives (same as qmd.ts)
const qmdDir = import.meta.dir;
const qmdScript = join(qmdDir, "qmd.ts");

// Helper to run qmd command with test database
async function runQmd(
  args: string[],
  options: { cwd?: string; env?: Record<string, string>; dbPath?: string } = {}
): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  const workingDir = options.cwd || fixturesDir;
  const dbPath = options.dbPath || testDbPath;
  const proc = Bun.spawn(["bun", qmdScript, ...args], {
    cwd: workingDir,
    env: {
      ...process.env,
      INDEX_PATH: dbPath,
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

// Setup test fixtures
beforeAll(async () => {
  // Create temp directory structure
  testDir = await mkdtemp(join(tmpdir(), "qmd-test-"));
  testDbPath = join(testDir, "test.sqlite");
  fixturesDir = join(testDir, "fixtures");

  await mkdir(fixturesDir, { recursive: true });
  await mkdir(join(fixturesDir, "notes"), { recursive: true });
  await mkdir(join(fixturesDir, "docs"), { recursive: true });

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
});

// Cleanup after all tests
afterAll(async () => {
  if (testDir) {
    await rm(testDir, { recursive: true, force: true });
  }
});

describe("CLI Help", () => {
  test("shows help with --help flag", async () => {
    const { stdout, exitCode } = await runQmd(["--help"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Usage:");
    expect(stdout).toContain("qmd add");
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
    const { stdout, exitCode } = await runQmd(["add", "."]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Collection:");
    expect(stdout).toContain("Indexed:");
  });

  test("adds files with custom glob pattern", async () => {
    const { stdout, exitCode } = await runQmd(["add", "notes/*.md"]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Collection:");
    // Should find meeting.md and ideas.md in notes/
    expect(stdout).toContain("notes/*.md");
  });

  test("adds files with --drop flag recreates collection", async () => {
    // First add
    await runQmd(["add", "."]);
    // Then drop and re-add
    const { stdout, exitCode } = await runQmd(["add", "--drop", "."]);
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Dropped collection:");
  });
});

describe("CLI Status Command", () => {
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["add", "."]);
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
    await runQmd(["add", "."]);
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
    await runQmd(["add", "."]);
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
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["add", "."]);
  });

  test("retrieves multiple documents by pattern", async () => {
    const { stdout, exitCode } = await runQmd(["multi-get", "notes/*.md"]);
    expect(exitCode).toBe(0);
    // Should contain content from both notes files
    expect(stdout).toContain("Meeting");
    expect(stdout).toContain("Ideas");
  });

  test("retrieves documents by comma-separated paths", async () => {
    const { stdout, exitCode } = await runQmd([
      "multi-get",
      "README.md,notes/meeting.md",
    ]);
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
    await runQmd(["add", "."], { dbPath: localDbPath });
  });

  test("updates all collections", async () => {
    const { stdout, exitCode } = await runQmd(["update"], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    expect(stdout).toContain("Updating");
  });
});

describe("CLI Add-Context Command", () => {
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["add", "."]);
  });

  test("adds context to a path", async () => {
    const { stdout, exitCode } = await runQmd([
      "add-context",
      "notes",
      "Personal notes and meeting logs",
    ]);
    expect(exitCode).toBe(0);
  });

  test("requires path and text arguments", async () => {
    const { stderr, exitCode } = await runQmd(["add-context"]);
    expect(exitCode).toBe(1);
    // Error message goes to stderr
    expect(stderr).toContain("Usage:");
  });
});

describe("CLI Cleanup Command", () => {
  beforeEach(async () => {
    // Ensure we have indexed files
    await runQmd(["add", "."]);
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
    const { exitCode } = await runQmd(["add", "."], {
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
    await runQmd(["add", "."]);
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
    // Create multiple collections
    await runQmd(["add", "notes/*.md"], { dbPath: localDbPath });
    await runQmd(["add", "docs/*.md"], { dbPath: localDbPath });
  });

  test("filters search by collection name", async () => {
    const { stdout, exitCode } = await runQmd([
      "search",
      "-c",
      "notes",
      "meeting",
    ], { dbPath: localDbPath });
    expect(exitCode).toBe(0);
    // Should find results from notes collection
    expect(stdout.toLowerCase()).toContain("meeting");
  });
});
