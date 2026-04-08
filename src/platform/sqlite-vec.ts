import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";

const SQLITE_VEC_ENV_VAR = "QMD_SQLITE_VEC_PATH";
const SQLITE_VEC_ENTRYPOINT = "vec0";
const requireForResolve = createRequire(import.meta.url);
const bunGlobal = globalThis as typeof globalThis & { Bun?: unknown };

export type SqliteVecLoadSource = "env" | "system" | "npm";

export type SqliteVecLoadableResolution = {
  path: string | null;
  source: SqliteVecLoadSource | null;
  packageName?: string;
};

export type SqliteVecResolveOptions = {
  platform?: string;
  arch?: string;
  env?: NodeJS.ProcessEnv;
  fileExists?: (path: string) => boolean;
  packageResolve?: (specifier: string) => string;
};

function uniqueStrings(values: Array<string | undefined>): string[] {
  const deduped = new Set<string>();
  for (const value of values) {
    const normalized = value?.trim();
    if (normalized) deduped.add(normalized);
  }
  return [...deduped];
}

export function isSqliteVecNpmPlatformSupported(
  platform: string = process.platform,
  arch: string = process.arch,
): boolean {
  return (
    (platform === "darwin" && (arch === "arm64" || arch === "x64")) ||
    (platform === "linux" && (arch === "arm64" || arch === "x64")) ||
    (platform === "win32" && arch === "x64")
  );
}

export function getSqliteVecNpmPackageName(
  platform: string = process.platform,
  arch: string = process.arch,
): string | null {
  if (!isSqliteVecNpmPlatformSupported(platform, arch)) {
    return null;
  }
  const os = platform === "win32" ? "windows" : platform;
  return `sqlite-vec-${os}-${arch}`;
}

export function getSqliteVecEntrypointFilename(platform: string = process.platform): string {
  if (platform === "win32") return `${SQLITE_VEC_ENTRYPOINT}.dll`;
  if (platform === "darwin") return `${SQLITE_VEC_ENTRYPOINT}.dylib`;
  return `${SQLITE_VEC_ENTRYPOINT}.so`;
}

export function getFreebsdSqliteVecProbePaths(
  options: Pick<SqliteVecResolveOptions, "env"> = {},
): string[] {
  const env = options.env ?? process.env;
  const prefixes = uniqueStrings([
    env.LOCALBASE,
    env.PREFIX,
    "/usr/local",
  ]);

  const paths: string[] = [];
  for (const prefix of prefixes) {
    paths.push(join(prefix, "lib", "sqlite3", "vec0.so"));
    paths.push(join(prefix, "lib", "sqlite-vec", "vec0.so"));
    paths.push(join(prefix, "lib", "vec0.so"));
    paths.push(join(prefix, "libexec", "sqlite3", "vec0.so"));
    paths.push(join(prefix, "libexec", "sqlite-vec", "vec0.so"));
  }

  return uniqueStrings(paths);
}

export function resolveSqliteVecLoadablePath(
  options: SqliteVecResolveOptions = {},
): SqliteVecLoadableResolution {
  const platform = options.platform ?? process.platform;
  const arch = options.arch ?? process.arch;
  const env = options.env ?? process.env;
  const fileExists = options.fileExists ?? existsSync;
  const packageResolve = options.packageResolve ?? requireForResolve.resolve.bind(requireForResolve);

  const overridePath = env[SQLITE_VEC_ENV_VAR]?.trim();
  if (overridePath) {
    return { path: overridePath, source: "env" };
  }

  if (platform === "freebsd") {
    for (const candidate of getFreebsdSqliteVecProbePaths({ env })) {
      if (fileExists(candidate)) {
        return { path: candidate, source: "system" };
      }
    }
  }

  const packageName = getSqliteVecNpmPackageName(platform, arch);
  if (packageName) {
    try {
      const path = packageResolve(`${packageName}/${getSqliteVecEntrypointFilename(platform)}`);
      return { path, source: "npm", packageName };
    } catch {}
  }

  return { path: null, source: null };
}

export function getSqliteVecUnavailableHint(
  options: { platform?: string; isBun?: boolean } = {},
): string {
  const platform = options.platform ?? process.platform;
  const isBun = options.isBun ?? (typeof bunGlobal.Bun !== "undefined");

  if (isBun && platform === "darwin") {
    return "On macOS with Bun, install Homebrew SQLite: brew install sqlite\n" +
      "Or install qmd with npm instead: npm install -g @tobilu/qmd";
  }

  if (platform === "freebsd") {
    return "On FreeBSD, install SQLite with extension loading support and make vec0.so available. " +
      "Set QMD_SQLITE_VEC_PATH=/path/to/vec0.so if auto-discovery does not find it.";
  }

  if (isSqliteVecNpmPlatformSupported(platform)) {
    return "Ensure the sqlite-vec native module is installed correctly.";
  }

  return "Install a SQLite build with extension loading support and set " +
    "QMD_SQLITE_VEC_PATH=/path/to/vec0.so if needed.";
}

export function createSqliteVecUnavailableError(
  reason: string,
  options: { platform?: string; isBun?: boolean } = {},
): Error {
  const cleanedReason = reason.trim().replace(/[.\s]+$/g, "");
  return new Error(
    `sqlite-vec extension is unavailable. ${cleanedReason}. ${getSqliteVecUnavailableHint(options)}`
  );
}
