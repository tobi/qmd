/**
 * Build/commit identification for `qmd --version`.
 *
 * The commit is *stamped at build time* (scripts/build.mjs writes
 * build-info.json next to the compiled CLI) rather than discovered at runtime.
 * A published tarball carries no git history of its own, so a runtime lookup
 * can only ever find some *other* repository's HEAD: `git -C <dir> rev-parse`
 * walks up the tree, and a global install under a git-managed prefix (e.g.
 * Homebrew's /opt/homebrew) reported that prefix's commit as qmd's.
 *
 * Running from a source checkout has no stamp, so the git lookup remains as a
 * fallback — but only after confirming the enclosing repository is the package
 * we are actually running from.
 */

import { execFileSync } from "node:child_process";
import { readFileSync, realpathSync } from "node:fs";
import { join } from "node:path";

/** Written next to the compiled CLI (dist/cli/) by scripts/build.mjs. */
export const BUILD_INFO_FILENAME = "build-info.json";

export type BuildInfo = {
  commit: string;
  builtAt?: string;
};

/**
 * Read the commit stamped into this build, or "" when there is none.
 *
 * Deliberately looks beside the running script rather than in the package's
 * dist/ directory: a source run (src/cli/) must not pick up the stamp left by
 * an earlier, possibly unrelated, build in dist/.
 */
export function readStampedCommit(scriptDir: string): string {
  try {
    const raw = readFileSync(join(scriptDir, BUILD_INFO_FILENAME), "utf-8");
    const info = JSON.parse(raw) as Partial<BuildInfo>;
    return typeof info.commit === "string" ? info.commit : "";
  } catch {
    // No stamp (source run), unreadable, or malformed — fall through.
    return "";
  }
}

function git(args: string[], cwd: string): string {
  // execFileSync, not execSync: no shell, so a path containing spaces or shell
  // metacharacters is passed through intact instead of silently failing.
  return execFileSync("git", args, {
    cwd,
    encoding: "utf-8",
    stdio: ["pipe", "pipe", "pipe"],
  }).trim();
}

/**
 * Short HEAD of the checkout we are running from, or "" when we are not
 * running from one.
 *
 * The guard is the whole point: an enclosing repository is only qmd's if its
 * top level *is* this package's root. Without that check, any install nested
 * inside an unrelated repository reports that repository's HEAD.
 */
export function gitCommitForCheckout(scriptDir: string, packageRoot: string): string {
  try {
    const top = git(["rev-parse", "--show-toplevel"], scriptDir);
    if (realpathSync(top) !== realpathSync(packageRoot)) return "";

    const commit = git(["rev-parse", "--short", "HEAD"], top);
    if (!commit) return "";

    // Same "-dirty" marker scripts/build.mjs stamps: running edited sources is
    // not the commit it names, and that distinction is the whole point of
    // printing a commit at all.
    const dirty = git(["status", "--porcelain"], top);
    return dirty ? `${commit}-dirty` : commit;
  } catch {
    // Not a git repo, git not installed, or an unborn branch.
    return "";
  }
}

/**
 * The commit to report for this invocation: the build stamp when there is one,
 * otherwise the verified checkout HEAD, otherwise nothing.
 */
export function resolveCommit(scriptDir: string, packageRoot: string): string {
  return readStampedCommit(scriptDir) || gitCommitForCheckout(scriptDir, packageRoot);
}
