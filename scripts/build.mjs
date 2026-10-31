#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { chmodSync, readFileSync, realpathSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(fileURLToPath(new URL("..", import.meta.url)));

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: "inherit",
    // Never shell:true — on Windows, cmd.exe splits unquoted process.execPath
    // at spaces (C:\Program Files\nodejs\node.exe) and the build can exit 0
    // with no dist/. spawnSync can spawn the binary + args array directly. (#681)
    shell: false,
    ...options,
  });
  if (result.error || result.status !== 0) {
    if (result.error) {
      console.error(`build: failed to spawn ${command}: ${result.error.message}`);
    }
    process.exit(result.status ?? 1);
  }
}

run(process.execPath, [join(root, "node_modules", "typescript", "bin", "tsc"), "-p", "tsconfig.build.json"]);

const cliPath = join(root, "dist", "cli", "qmd.js");
const tmpPath = `${cliPath}.tmp`;
const built = readFileSync(cliPath, "utf8");
const withoutExistingShebang = built.startsWith("#!") ? built.slice(built.indexOf("\n") + 1) : built;
writeFileSync(tmpPath, `#!/usr/bin/env node\n${withoutExistingShebang}`);
renameSync(tmpPath, cliPath);
chmodSync(cliPath, 0o755);

// Stamp the commit this build came from, for `qmd --version`.
//
// It has to happen here: a published tarball has no git history of its own, so
// discovering the commit at runtime finds whatever repository the install
// happens to sit inside (a global install under /opt/homebrew reported
// Homebrew's HEAD as qmd's). See src/cli/version.ts.
//
// Uses spawnSync directly rather than run() above, because run() exits the
// build on a non-zero status: no git, no repo, or a source tarball must all
// degrade to an unstamped build, not a failed one.
function git(args) {
  const result = spawnSync("git", args, { cwd: root, encoding: "utf8", stdio: ["pipe", "pipe", "pipe"] });
  if (result.status !== 0 || typeof result.stdout !== "string") return null;
  return result.stdout.trim();
}

function buildCommit() {
  const top = git(["rev-parse", "--show-toplevel"]);
  // Only trust a repository that *is* this package — same rule the runtime
  // fallback applies, so building from inside an unrelated checkout can't
  // stamp that checkout's commit.
  if (top === null || realpathSync(top) !== realpathSync(root)) return "";

  const commit = git(["rev-parse", "--short", "HEAD"]);
  if (!commit) return "";

  // A build from a dirty tree is not the commit it claims to be; say so, so
  // "did my install actually take?" has an answer.
  const status = git(["status", "--porcelain"]);
  return status ? `${commit}-dirty` : commit;
}

writeFileSync(
  join(root, "dist", "cli", "build-info.json"),
  `${JSON.stringify({ commit: buildCommit(), builtAt: new Date().toISOString() }, null, 2)}\n`,
);
