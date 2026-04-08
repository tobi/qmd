import { describe, expect, test, vi } from "vitest";
import {
  createSqliteVecUnavailableError,
  getFreebsdSqliteVecProbePaths,
  getSqliteVecUnavailableHint,
  resolveSqliteVecLoadablePath,
} from "../src/platform/sqlite-vec.js";

describe("resolveSqliteVecLoadablePath", () => {
  test("prefers QMD_SQLITE_VEC_PATH over all other resolution paths", () => {
    const fileExists = vi.fn(() => false);
    const packageResolve = vi.fn((specifier: string) => `/resolved/${specifier}`);

    const result = resolveSqliteVecLoadablePath({
      platform: "freebsd",
      arch: "x64",
      env: {
        QMD_SQLITE_VEC_PATH: " /custom/vec0.so ",
        LOCALBASE: "/opt/local",
      },
      fileExists,
      packageResolve,
    });

    expect(result).toEqual({
      path: "/custom/vec0.so",
      source: "env",
    });
    expect(fileExists).not.toHaveBeenCalled();
    expect(packageResolve).not.toHaveBeenCalled();
  });

  test("uses the first matching FreeBSD system path before npm resolution", () => {
    const fileExists = vi.fn((path: string) => path === "/opt/local/lib/sqlite3/vec0.so");
    const packageResolve = vi.fn((specifier: string) => `/resolved/${specifier}`);

    const result = resolveSqliteVecLoadablePath({
      platform: "freebsd",
      arch: "x64",
      env: {
        LOCALBASE: "/opt/local",
      },
      fileExists,
      packageResolve,
    });

    expect(result).toEqual({
      path: "/opt/local/lib/sqlite3/vec0.so",
      source: "system",
    });
    expect(packageResolve).not.toHaveBeenCalled();
  });

  test("resolves the packaged sqlite-vec binary on supported npm platforms", () => {
    const packageResolve = vi.fn((specifier: string) => `/resolved/${specifier}`);

    const result = resolveSqliteVecLoadablePath({
      platform: "darwin",
      arch: "arm64",
      env: {},
      fileExists: () => false,
      packageResolve,
    });

    expect(packageResolve).toHaveBeenCalledWith("sqlite-vec-darwin-arm64/vec0.dylib");
    expect(result).toEqual({
      path: "/resolved/sqlite-vec-darwin-arm64/vec0.dylib",
      source: "npm",
      packageName: "sqlite-vec-darwin-arm64",
    });
  });

  test("returns null when no path can be resolved", () => {
    const packageResolve = vi.fn(() => {
      throw new Error("module not found");
    });

    const result = resolveSqliteVecLoadablePath({
      platform: "openbsd",
      arch: "x64",
      env: {},
      fileExists: () => false,
      packageResolve,
    });

    expect(result).toEqual({
      path: null,
      source: null,
    });
    expect(packageResolve).not.toHaveBeenCalled();
  });
});

describe("getFreebsdSqliteVecProbePaths", () => {
  test("includes LOCALBASE first and de-duplicates prefixes", () => {
    const paths = getFreebsdSqliteVecProbePaths({
      env: {
        LOCALBASE: "/usr/local",
        PREFIX: "/usr/local",
      },
    });

    expect(paths[0]).toBe("/usr/local/lib/sqlite3/vec0.so");
    expect(new Set(paths).size).toBe(paths.length);
  });
});

describe("sqlite-vec diagnostics", () => {
  test("macOS Bun hint mentions Homebrew and npm fallback", () => {
    const hint = getSqliteVecUnavailableHint({
      platform: "darwin",
      isBun: true,
    });

    expect(hint).toContain("brew install sqlite");
    expect(hint).toContain("npm install -g @tobilu/qmd");
  });

  test("FreeBSD hint mentions QMD_SQLITE_VEC_PATH", () => {
    const hint = getSqliteVecUnavailableHint({
      platform: "freebsd",
      isBun: false,
    });

    expect(hint).toContain("QMD_SQLITE_VEC_PATH");
    expect(hint).toContain("vec0.so");
  });

  test("error formatting avoids duplicate punctuation", () => {
    const err = createSqliteVecUnavailableError("No loadable sqlite-vec extension was found.", {
      platform: "freebsd",
      isBun: false,
    });

    expect(err.message).toContain("sqlite-vec extension is unavailable.");
    expect(err.message).not.toContain("found..");
  });
});
