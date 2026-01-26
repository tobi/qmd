/**
 * store-paths.test.ts - Comprehensive unit tests for Windows path support
 *
 * Tests all path-related utility functions for cross-platform compatibility:
 * - isAbsolutePath() - Unix, Windows (C:\, C:/), and Git Bash (/c/) paths
 * - normalizePathSeparators() - backslash to forward slash conversion
 * - getRelativePathFromPrefix() - relative path extraction
 * - resolve() - path resolution with Unix and Windows paths
 *
 * Run with: bun test store-paths.test.ts
 */

import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import {
  isAbsolutePath,
  normalizePathSeparators,
  getRelativePathFromPrefix,
  resolve,
} from "./store.js";

// =============================================================================
// Test Utilities
// =============================================================================

let originalPWD: string | undefined;
let originalProcessCwd: () => string;

beforeEach(() => {
  // Save original environment
  originalPWD = Bun.env.PWD;
  originalProcessCwd = process.cwd;
});

afterEach(() => {
  // Restore original environment
  if (originalPWD !== undefined) {
    Bun.env.PWD = originalPWD;
  } else {
    delete Bun.env.PWD;
  }
  process.cwd = originalProcessCwd;
});

/**
 * Mock the current working directory for testing.
 * Sets both Bun.env.PWD and process.cwd() to simulate different environments.
 */
function mockPWD(path: string): void {
  Bun.env.PWD = path;
  process.cwd = () => path;
}

// =============================================================================
// Path Utilities - Cross-platform Support
// =============================================================================

describe("Path utilities - Cross-platform support", () => {
  
  // ===========================================================================
  // isAbsolutePath
  // ===========================================================================
  
  describe("isAbsolutePath", () => {
    test("Unix absolute paths", () => {
      expect(isAbsolutePath("/path/to/file")).toBe(true);
      expect(isAbsolutePath("/")).toBe(true);
      expect(isAbsolutePath("/home/user/documents")).toBe(true);
      expect(isAbsolutePath("/usr/local/bin")).toBe(true);
    });

    test("Unix relative paths", () => {
      expect(isAbsolutePath("path/to/file")).toBe(false);
      expect(isAbsolutePath("./path/to/file")).toBe(false);
      expect(isAbsolutePath("../path/to/file")).toBe(false);
      expect(isAbsolutePath("./file")).toBe(false);
      expect(isAbsolutePath("../file")).toBe(false);
      expect(isAbsolutePath("file.txt")).toBe(false);
    });

    test("Windows absolute paths (native) - forward slash", () => {
      expect(isAbsolutePath("C:/path/to/file")).toBe(true);
      expect(isAbsolutePath("C:/")).toBe(true);
      expect(isAbsolutePath("D:/Users/Documents")).toBe(true);
      expect(isAbsolutePath("Z:/")).toBe(true);
      expect(isAbsolutePath("c:/lowercase")).toBe(true);
    });

    test("Windows absolute paths (native) - backslash", () => {
      expect(isAbsolutePath("C:\\path\\to\\file")).toBe(true);
      expect(isAbsolutePath("C:\\")).toBe(true);
      expect(isAbsolutePath("D:\\Users\\Documents")).toBe(true);
      expect(isAbsolutePath("Z:\\")).toBe(true);
      expect(isAbsolutePath("c:\\lowercase")).toBe(true);
    });

    test("Windows relative paths", () => {
      expect(isAbsolutePath("path\\to\\file")).toBe(false);
      expect(isAbsolutePath(".\\path\\to\\file")).toBe(false);
      expect(isAbsolutePath("..\\path\\to\\file")).toBe(false);
      expect(isAbsolutePath(".\\file")).toBe(false);
      expect(isAbsolutePath("..\\file")).toBe(false);
      expect(isAbsolutePath("file.txt")).toBe(false);
    });

    test("Git Bash style paths", () => {
      expect(isAbsolutePath("/c/Users/name/file")).toBe(true);
      expect(isAbsolutePath("/C/Users/name/file")).toBe(true);
      expect(isAbsolutePath("/d/Projects")).toBe(true);
      expect(isAbsolutePath("/D/Projects")).toBe(true);
      expect(isAbsolutePath("/z/")).toBe(true);
    });

    test("Edge cases", () => {
      expect(isAbsolutePath("")).toBe(false);
      expect(isAbsolutePath("C:")).toBe(true); // Drive letter only
      expect(isAbsolutePath("C")).toBe(false); // Just a letter
      expect(isAbsolutePath(":")).toBe(false);
      expect(isAbsolutePath("/a")).toBe(true); // Short Unix path
      expect(isAbsolutePath("/1/")).toBe(true); // Number after slash (not Git Bash)
    });
  });

  // ===========================================================================
  // normalizePathSeparators
  // ===========================================================================
  
  describe("normalizePathSeparators", () => {
    test("Windows paths with backslashes", () => {
      expect(normalizePathSeparators("C:\\Users\\name\\file.txt"))
        .toBe("C:/Users/name/file.txt");
      expect(normalizePathSeparators("D:\\Projects\\qmd\\src"))
        .toBe("D:/Projects/qmd/src");
      expect(normalizePathSeparators("\\path\\to\\file"))
        .toBe("/path/to/file");
    });

    test("Mixed separators", () => {
      expect(normalizePathSeparators("C:\\Users/name\\file.txt"))
        .toBe("C:/Users/name/file.txt");
      expect(normalizePathSeparators("path\\to/file/here"))
        .toBe("path/to/file/here");
    });

    test("Unix paths (should remain unchanged)", () => {
      expect(normalizePathSeparators("/path/to/file"))
        .toBe("/path/to/file");
      expect(normalizePathSeparators("/usr/local/bin"))
        .toBe("/usr/local/bin");
      expect(normalizePathSeparators("relative/path"))
        .toBe("relative/path");
    });

    test("Multiple consecutive backslashes", () => {
      expect(normalizePathSeparators("path\\\\to\\\\file"))
        .toBe("path//to//file");
      expect(normalizePathSeparators("C:\\\\Users\\\\name"))
        .toBe("C://Users//name");
    });

    test("Edge cases", () => {
      expect(normalizePathSeparators("")).toBe("");
      expect(normalizePathSeparators("\\")).toBe("/");
      expect(normalizePathSeparators("\\\\")).toBe("//");
      expect(normalizePathSeparators("file.txt")).toBe("file.txt");
    });
  });

  // ===========================================================================
  // getRelativePathFromPrefix
  // ===========================================================================
  
  describe("getRelativePathFromPrefix", () => {
    test("Exact match (path equals prefix)", () => {
      expect(getRelativePathFromPrefix("/home/user", "/home/user")).toBe("");
      expect(getRelativePathFromPrefix("C:/Users/name", "C:/Users/name")).toBe("");
      expect(getRelativePathFromPrefix("/path", "/path")).toBe("");
    });

    test("Path under prefix", () => {
      expect(getRelativePathFromPrefix("/home/user/documents", "/home/user"))
        .toBe("documents");
      expect(getRelativePathFromPrefix("/home/user/documents/file.txt", "/home/user"))
        .toBe("documents/file.txt");
      expect(getRelativePathFromPrefix("C:/Users/name/Documents/file.txt", "C:/Users/name"))
        .toBe("Documents/file.txt");
    });

    test("Path not under prefix", () => {
      expect(getRelativePathFromPrefix("/home/other", "/home/user")).toBeNull();
      expect(getRelativePathFromPrefix("/usr/local", "/home/user")).toBeNull();
      expect(getRelativePathFromPrefix("C:/Users/other", "D:/Users")).toBeNull();
    });

    test("Windows paths with normalized separators", () => {
      // Backslashes should be normalized
      expect(getRelativePathFromPrefix("C:\\Users\\name\\Documents", "C:\\Users\\name"))
        .toBe("Documents");
      expect(getRelativePathFromPrefix("C:\\Users\\name\\Documents\\file.txt", "C:/Users/name"))
        .toBe("Documents/file.txt");
    });

    test("Prefix with trailing slash", () => {
      expect(getRelativePathFromPrefix("/home/user/documents", "/home/user/"))
        .toBe("documents");
      expect(getRelativePathFromPrefix("C:/Users/name/Documents", "C:/Users/name/"))
        .toBe("Documents");
    });

    test("Prefix without trailing slash", () => {
      expect(getRelativePathFromPrefix("/home/user/documents", "/home/user"))
        .toBe("documents");
      expect(getRelativePathFromPrefix("C:/Users/name/Documents", "C:/Users/name"))
        .toBe("Documents");
    });

    test("Edge cases", () => {
      // Empty prefix
      expect(getRelativePathFromPrefix("/path/to/file", "")).toBeNull();
      
      // Path is prefix substring but not in hierarchy
      expect(getRelativePathFromPrefix("/home/username", "/home/user")).toBeNull();
      
      // Root prefix
      expect(getRelativePathFromPrefix("/home/user", "/")).toBe("home/user");
    });
  });

  // ===========================================================================
  // resolve - Unix environment
  // ===========================================================================
  
  describe("resolve - Unix environment", () => {
    beforeEach(() => {
      mockPWD("/home/user");
    });

    test("Unix relative paths", () => {
      expect(resolve("/base", "relative")).toBe("/base/relative");
      expect(resolve("/base", "a/b/c")).toBe("/base/a/b/c");
      expect(resolve("/home", "user/documents")).toBe("/home/user/documents");
    });

    test("Unix absolute paths", () => {
      expect(resolve("/base", "/absolute")).toBe("/absolute");
      expect(resolve("/home/user", "/usr/local")).toBe("/usr/local");
      expect(resolve("/any", "/")).toBe("/");
    });

    test("Path with .. and .", () => {
      expect(resolve("/base", "../other")).toBe("/other");
      expect(resolve("/base/sub", "..")).toBe("/base");
      expect(resolve("/base", "./file")).toBe("/base/file");
      expect(resolve("/base/a/b", "../../c")).toBe("/base/c");
    });

    test("Multiple path segments", () => {
      expect(resolve("/a", "b", "c")).toBe("/a/b/c");
      expect(resolve("/a", "b", "../c")).toBe("/a/c");
      expect(resolve("/a", "b", "/c")).toBe("/c");
    });

    test("Relative path without base (uses PWD)", () => {
      expect(resolve("relative")).toBe("/home/user/relative");
      expect(resolve("a/b/c")).toBe("/home/user/a/b/c");
      expect(resolve("./file")).toBe("/home/user/file");
    });

    test("Absolute path alone", () => {
      expect(resolve("/absolute/path")).toBe("/absolute/path");
      expect(resolve("/")).toBe("/");
    });
  });

  // ===========================================================================
  // resolve - Windows environment
  // ===========================================================================
  
  describe("resolve - Windows environment", () => {
    beforeEach(() => {
      mockPWD("C:/Users/name");
    });

    test("Windows relative paths", () => {
      expect(resolve("C:/base", "relative")).toBe("C:/base/relative");
      expect(resolve("C:/base", "a/b/c")).toBe("C:/base/a/b/c");
      expect(resolve("D:/Projects", "qmd/src")).toBe("D:/Projects/qmd/src");
    });

    test("Windows absolute paths", () => {
      expect(resolve("C:/base", "D:/other")).toBe("D:/other");
      expect(resolve("C:/Users", "C:/Program Files")).toBe("C:/Program Files");
      expect(resolve("D:/any", "E:/other")).toBe("E:/other");
    });

    test("Windows with backslashes", () => {
      expect(resolve("C:\\base", "relative")).toBe("C:/base/relative");
      expect(resolve("C:\\Users\\name", "Documents")).toBe("C:/Users/name/Documents");
      expect(resolve("C:\\base", "a\\b\\c")).toBe("C:/base/a/b/c");
    });

    test("Path with .. and .", () => {
      expect(resolve("C:/base", "../other")).toBe("C:/other");
      expect(resolve("C:/base/sub", "..")).toBe("C:/base");
      expect(resolve("C:/base", "./file")).toBe("C:/base/file");
      expect(resolve("C:/base/a/b", "../../c")).toBe("C:/base/c");
    });

    test("Multiple path segments", () => {
      expect(resolve("C:/a", "b", "c")).toBe("C:/a/b/c");
      expect(resolve("C:/a", "b", "../c")).toBe("C:/a/c");
      expect(resolve("C:/a", "b", "D:/c")).toBe("D:/c");
    });

    test("Relative path without base (uses PWD)", () => {
      expect(resolve("relative")).toBe("C:/Users/name/relative");
      expect(resolve("a/b/c")).toBe("C:/Users/name/a/b/c");
      expect(resolve(".\\file")).toBe("C:/Users/name/file");
    });

    test("Drive letter only", () => {
      expect(resolve("C:")).toBe("C:/");
      expect(resolve("D:")).toBe("D:/");
    });
  });

  // ===========================================================================
  // resolve - Git Bash style paths
  // ===========================================================================
  
  describe("resolve - Git Bash style paths", () => {
    test("Git Bash to Windows conversion", () => {
      expect(resolve("/c/Users/name")).toBe("C:/Users/name");
      expect(resolve("/C/Users/name")).toBe("C:/Users/name");
      expect(resolve("/d/Projects")).toBe("D:/Projects");
      expect(resolve("/D/Projects")).toBe("D:/Projects");
    });

    test("Git Bash with relative paths", () => {
      expect(resolve("/c/base", "relative")).toBe("C:/base/relative");
      expect(resolve("/d/Projects", "qmd/src")).toBe("D:/Projects/qmd/src");
    });

    test("Git Bash with .. and .", () => {
      expect(resolve("/c/base", "../other")).toBe("C:/other");
      expect(resolve("/c/base/sub", "..")).toBe("C:/base");
      expect(resolve("/c/base", "./file")).toBe("C:/base/file");
    });

    test("Multiple Git Bash segments", () => {
      expect(resolve("/c/a", "b", "c")).toBe("C:/a/b/c");
      expect(resolve("/c/a", "b", "/d/c")).toBe("D:/c");
    });
  });

  // ===========================================================================
  // resolve - Edge cases and mixed scenarios
  // ===========================================================================
  
  describe("resolve - Edge cases", () => {
    test("Empty path segments are filtered", () => {
      expect(resolve("/base", "", "file")).toBe("/base/file");
      expect(resolve("C:/base", "", "file")).toBe("C:/base/file");
    });

    test("Multiple consecutive slashes", () => {
      expect(resolve("/base//path///file")).toBe("/base/path/file");
      expect(resolve("C:/base//path///file")).toBe("C:/base/path/file");
    });

    test("Trailing slashes", () => {
      expect(resolve("/base/", "file")).toBe("/base/file");
      expect(resolve("C:/base/", "file")).toBe("C:/base/file");
    });

    test("Complex .. navigation", () => {
      expect(resolve("/a/b/c/d", "../../../e")).toBe("/a/e");
      expect(resolve("C:/a/b/c/d", "../../../e")).toBe("C:/a/e");
    });

    test("Too many .. (should not go above root)", () => {
      expect(resolve("/base", "../../../../other")).toBe("/other");
      expect(resolve("C:/base", "../../../../other")).toBe("C:/other");
    });

    test("Mixed Unix and Windows (normalized)", () => {
      mockPWD("C:/Users/name");
      expect(resolve("/unix/path")).toBe("/unix/path");
      expect(resolve("relative")).toBe("C:/Users/name/relative");
    });

    test("Error on no arguments", () => {
      expect(() => resolve()).toThrow("resolve: at least one path segment is required");
    });
  });
});
