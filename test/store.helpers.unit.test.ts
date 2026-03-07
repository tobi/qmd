/**
 * Store helper-level unit tests (pure logic, no model/runtime dependency).
 */

import { describe, test, expect } from "vitest";
import {
  homedir,
  resolve,
  getDefaultDbPath,
  getPwd,
  getRealPath,
  isVirtualPath,
  parseVirtualPath,
  normalizeVirtualPath,
  normalizeDocid,
  isDocid,
  handelize,
  chunkDocumentByTokens,
} from "../src/store";

// =============================================================================
// Path Utilities
// =============================================================================

describe("Path Utilities", () => {
  test("homedir returns HOME environment variable", () => {
    expect(homedir()).toBe(process.env.HOME || "/tmp");
  });

  test("resolve handles absolute paths", () => {
    expect(resolve("/foo/bar")).toBe("/foo/bar");
    expect(resolve("/foo", "/bar")).toBe("/bar");
  });

  test("resolve handles relative paths", () => {
    const pwd = process.env.PWD || process.cwd();
    expect(resolve("foo")).toBe(`${pwd}/foo`);
    expect(resolve("foo", "bar")).toBe(`${pwd}/foo/bar`);
  });

  test("resolve normalizes . and ..", () => {
    expect(resolve("/foo/bar/./baz")).toBe("/foo/bar/baz");
    expect(resolve("/foo/bar/../baz")).toBe("/foo/baz");
    expect(resolve("/foo/bar/../../baz")).toBe("/baz");
  });

  test("getDefaultDbPath throws in test mode without INDEX_PATH", () => {
    const originalIndexPath = process.env.INDEX_PATH;
    delete process.env.INDEX_PATH;

    expect(() => getDefaultDbPath()).toThrow("Database path not set");

    if (originalIndexPath) {
      process.env.INDEX_PATH = originalIndexPath;
    }
  });

  test("getDefaultDbPath uses INDEX_PATH when set", () => {
    const originalIndexPath = process.env.INDEX_PATH;
    process.env.INDEX_PATH = "/tmp/test-index.sqlite";

    expect(getDefaultDbPath()).toBe("/tmp/test-index.sqlite");
    expect(getDefaultDbPath("custom")).toBe("/tmp/test-index.sqlite");

    if (originalIndexPath) {
      process.env.INDEX_PATH = originalIndexPath;
    } else {
      delete process.env.INDEX_PATH;
    }
  });

  test("getPwd returns current working directory", () => {
    const pwd = getPwd();
    expect(pwd).toBeTruthy();
    expect(typeof pwd).toBe("string");
  });

  test("getRealPath resolves symlinks", () => {
    const result = getRealPath("/tmp");
    expect(result).toBeTruthy();
    expect(result === "/tmp" || result === "/private/tmp").toBe(true);
  });
});

// =============================================================================
// Handelize Tests
// =============================================================================

describe("handelize", () => {
  test("converts to lowercase", () => {
    expect(handelize("README.md")).toBe("readme.md");
    expect(handelize("MyFile.MD")).toBe("myfile.md");
  });

  test("preserves folder structure", () => {
    expect(handelize("a/b/c/d.md")).toBe("a/b/c/d.md");
    expect(handelize("docs/api/README.md")).toBe("docs/api/readme.md");
  });

  test("replaces non-word characters with dash", () => {
    expect(handelize("hello world.md")).toBe("hello-world.md");
    expect(handelize("file (1).md")).toBe("file-1.md");
    expect(handelize("foo@bar#baz.md")).toBe("foo-bar-baz.md");
  });

  test("collapses multiple special chars into single dash", () => {
    expect(handelize("hello   world.md")).toBe("hello-world.md");
    expect(handelize("foo---bar.md")).toBe("foo-bar.md");
    expect(handelize("a  -  b.md")).toBe("a-b.md");
  });

  test("removes leading and trailing dashes from segments", () => {
    expect(handelize("-hello-.md")).toBe("hello.md");
    expect(handelize("--test--.md")).toBe("test.md");
    expect(handelize("a/-b-/c.md")).toBe("a/b/c.md");
  });

  test("converts triple underscore to folder separator", () => {
    expect(handelize("foo___bar.md")).toBe("foo/bar.md");
    expect(handelize("notes___2025___january.md")).toBe("notes/2025/january.md");
    expect(handelize("a/b___c/d.md")).toBe("a/b/c/d.md");
  });

  test("handles complex real-world meeting notes", () => {
    const complexName = "Money Movement Licensing Review - 2025／11／19 10:25 EST - Notes by Gemini.md";
    const result = handelize(complexName);
    expect(result).toBe("money-movement-licensing-review-2025-11-19-10-25-est-notes-by-gemini.md");
    expect(result).not.toContain(" ");
    expect(result).not.toContain("／");
    expect(result).not.toContain(":");
  });

  test("handles unicode characters", () => {
    expect(handelize("日本語.md")).toBe("日本語.md");
    expect(handelize("Зоны и проекты.md")).toBe("зоны-и-проекты.md");
    expect(handelize("café-notes.md")).toBe("café-notes.md");
    expect(handelize("naïve.md")).toBe("naïve.md");
    expect(handelize("日本語-notes.md")).toBe("日本語-notes.md");
  });

  test("handles dates and times in filenames", () => {
    expect(handelize("meeting-2025-01-15.md")).toBe("meeting-2025-01-15.md");
    expect(handelize("notes 2025/01/15.md")).toBe("notes-2025/01/15.md");
    expect(handelize("call_10:30_AM.md")).toBe("call-10-30-am.md");
  });

  test("handles special project naming patterns", () => {
    expect(handelize("PROJECT_ABC_v2.0.md")).toBe("project-abc-v2-0.md");
    expect(handelize("[WIP] Feature Request.md")).toBe("wip-feature-request.md");
    expect(handelize("(DRAFT) Proposal v1.md")).toBe("draft-proposal-v1.md");
  });

  test("handles symbol-only route filenames", () => {
    expect(handelize("routes/api/auth/$.ts")).toBe("routes/api/auth/$.ts");
    expect(handelize("app/routes/$id.tsx")).toBe("app/routes/$id.tsx");
  });

  test("filters out empty segments", () => {
    expect(handelize("a//b/c.md")).toBe("a/b/c.md");
    expect(handelize("/a/b/")).toBe("a/b");
    expect(handelize("///test///")).toBe("test");
  });

  test("throws error for invalid inputs", () => {
    expect(() => handelize("" )).toThrow("path cannot be empty");
    expect(() => handelize("   ")).toThrow("path cannot be empty");
    expect(() => handelize(".md")).toThrow("no valid filename content");
    expect(() => handelize("...")).toThrow("no valid filename content");
    expect(() => handelize("___")).toThrow("no valid filename content");
  });

  test("handles minimal valid inputs", () => {
    expect(handelize("a")).toBe("a");
    expect(handelize("1")).toBe("1");
    expect(handelize("a.md")).toBe("a.md");
  });

  test("normalizes virtual paths", () => {
    expect(normalizeVirtualPath("qmd://docs/readme.md")).toBe("qmd://docs/readme.md");
    expect(normalizeVirtualPath("docs/readme.md")).toBe("docs/readme.md");
  });

  test("detects virtual paths", () => {
    expect(isVirtualPath("qmd://docs/readme.md")).toBe(true);
    expect(isVirtualPath("/tmp/file.md")).toBe(false);
  });

  test("parses virtual paths", () => {
    expect(parseVirtualPath("qmd://docs/readme.md")).toEqual({
      collectionName: "docs",
      path: "readme.md",
    });
  });

  test("normalizes docids", () => {
    expect(normalizeDocid("123456")).toBe("123456");
    expect(normalizeDocid("#123456")).toBe("123456");
  });

  test("checks docid validity", () => {
    expect(isDocid("123456")).toBe(true);
    expect(isDocid("#123456")).toBe(true);
    expect(isDocid("bad-id")).toBe(false);
    expect(isDocid("12345")).toBe(false);
  });
});

describe("Token Chunking Fallback", () => {
  test("chunkDocumentByTokens uses char-based fallback when backend cannot tokenize", async () => {
    const originalBackend = process.env.QMD_LLM_BACKEND;
    process.env.QMD_LLM_BACKEND = "api";

    try {
      const content = "This is a document sentence. ".repeat(400);
      const chunks = await chunkDocumentByTokens(content, 120, 18, 40);
      expect(chunks.length).toBeGreaterThan(1);
      for (const chunk of chunks) {
        expect(chunk.tokens).toBeGreaterThan(0);
      }
    } finally {
      if (originalBackend === undefined) {
        delete process.env.QMD_LLM_BACKEND;
      } else {
        process.env.QMD_LLM_BACKEND = originalBackend;
      }
    }
  });
});
