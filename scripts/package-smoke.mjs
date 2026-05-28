#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const pkg = JSON.parse(readFileSync(join(root, "package.json"), "utf8"));

function run(label, command, args, options = {}) {
  console.log(`==> ${label}`);
  const { quiet, ...spawnOptions } = options;
  const result = spawnSync(command, args, {
    cwd: root,
    stdio: quiet ? "pipe" : "inherit",
    shell: process.platform === "win32",
    ...spawnOptions,
  });
  if (result.status !== 0) {
    console.error(`Package smoke failed: ${label}`);
    if (quiet) {
      if (result.stdout) process.stderr.write(result.stdout);
      if (result.stderr) process.stderr.write(result.stderr);
    }
    process.exit(result.status ?? 1);
  }
}

function assertPath(path, label = path) {
  const full = join(root, path);
  if (!existsSync(full)) {
    console.error(`Package smoke failed: missing ${label} (${path})`);
    process.exit(1);
  }
  return full;
}

run("build compiled package", process.execPath, ["scripts/build.mjs"]);
run("AST grammar runtime packages", process.execPath, ["scripts/check-package-grammars.mjs"]);

for (const entry of pkg.files ?? []) {
  assertPath(entry.replace(/\/$/, ""), `package.json files[] entry ${entry}`);
}

for (const [name, binPath] of Object.entries(pkg.bin ?? {})) {
  const full = assertPath(binPath, `bin ${name}`);
  const mode = statSync(full).mode;
  if ((mode & 0o111) === 0) {
    console.error(`Package smoke failed: bin ${name} is not executable (${binPath})`);
    process.exit(1);
  }
}

assertPath("dist/index.js", "compiled main export");
assertPath("dist/index.d.ts", "compiled type export");
assertPath("dist/cli/qmd.js", "compiled CLI");

run("compiled CLI under Node", process.execPath, ["dist/cli/qmd.js", "--help"], { quiet: true });
run("package wrapper", "sh", ["bin/qmd", "--help"], { quiet: true });

if (process.env.QMD_SKIP_BUN_SMOKE === "1") {
  console.log("==> compiled CLI under Bun (skipped by QMD_SKIP_BUN_SMOKE=1)");
} else {
  run("compiled CLI under Bun", "bun", ["dist/cli/qmd.js", "--help"], { quiet: true });
}
