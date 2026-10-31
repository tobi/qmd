import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url);
const pkg = JSON.parse(readFileSync(new URL("package.json", root), "utf8"));

describe("package test task", () => {
  test("runs typecheck, unit tests, and package smoke checks", () => {
    expect(pkg.scripts.test).toContain("scripts/test-all.mjs");

    expect(pkg.scripts["test:types"]).toContain("tsconfig.build.json --noEmit");
    expect(pkg.scripts["test:unit"]).toContain("vitest.mjs");
    expect(pkg.scripts["test:unit"]).toContain("bun test");
    expect(pkg.scripts["test:unit"]).toContain("CI=true");

    expect(pkg.scripts["test:package"]).toContain("scripts/package-smoke.mjs");

    const testAllScript = readFileSync(new URL("scripts/test-all.mjs", root), "utf8");
    expect(testAllScript).toContain("TypeScript build typecheck");
    expect(testAllScript).toContain("Vitest suite under Node");
    expect(testAllScript).toContain("Bun test suite");
    expect(testAllScript).toContain("Package smoke");

    const packageSmokeScript = readFileSync(new URL("scripts/package-smoke.mjs", root), "utf8");
    expect(packageSmokeScript).toContain("scripts/build.mjs");
    expect(packageSmokeScript).toContain("scripts/check-package-grammars.mjs");
    expect(packageSmokeScript).toContain("compiled CLI under Node");
    expect(packageSmokeScript).toContain("compiled CLI under Bun");
    expect(packageSmokeScript).toContain("package wrapper");
  });
});

describe("package grammar distribution", () => {
  test("installs AST grammar wasm packages as required runtime dependencies", () => {
    for (const dep of ["tree-sitter-typescript", "tree-sitter-python", "tree-sitter-go", "tree-sitter-rust"]) {
      expect(pkg.dependencies, `${dep} should be a required dependency`).toHaveProperty(dep);
      expect(pkg.optionalDependencies ?? {}, `${dep} should not be optional`).not.toHaveProperty(dep);
    }
  });

  test("documents a packaging smoke check for grammar wasm availability", () => {
    expect(pkg.scripts, "package.json scripts").toHaveProperty("smoke:package-grammars");
    expect(String(pkg.scripts["smoke:package-grammars"])).toContain("check-package-grammars");

    expect(pkg.files, "published package files").toContain("scripts/build.mjs");
    expect(pkg.files, "published package files").toContain("scripts/check-package-grammars.mjs");
    expect(pkg.files, "published package files").toContain("scripts/package-smoke.mjs");
    expect(pkg.files, "published package files").toContain("scripts/test-all.mjs");
    expect(pkg.files, "published package files").toContain("skills/");
    const qmdSkill = readFileSync(new URL("skills/qmd/SKILL.md", root), "utf8");
    expect(qmdSkill).toContain("# QMD - Query Markdown Documents");
    expect(qmdSkill).toContain("## How search works");
    expect(qmdSkill).toContain("## MCP Tool: `query`");
    expect(qmdSkill).not.toContain("This file is a discovery stub");

    const firstSixtyLines = qmdSkill.split(/\r?\n/).slice(0, 60).join("\n");
    expect(firstSixtyLines).toContain("Search for candidate documents");
    expect(firstSixtyLines).toContain("qmd search");
    expect(firstSixtyLines).toContain('qmd multi-get "#abc123,#def432"');
    expect(firstSixtyLines).toContain("Retrieved:");
    expect(firstSixtyLines).toContain("qmd query");
    // The skill must teach structured, self-authored queries near the top.
    expect(firstSixtyLines).toContain("Default to structured");

    const scriptPath = join(root.pathname, "scripts", "check-package-grammars.mjs");
    const script = readFileSync(scriptPath, "utf8");
    expect(script).toContain("tree-sitter-typescript/tree-sitter-typescript.wasm");
    expect(script).toContain("tree-sitter-typescript/tree-sitter-tsx.wasm");
  });
});

describe("Nix flake package layout", () => {
  test("installPhase copies skills/ next to src/ so findPackageRoot can resolve them (#722)", () => {
    const flake = readFileSync(new URL("flake.nix", root), "utf8");

    // The bun wrapper runs $out/lib/qmd/src/cli/qmd.ts. findPackageRoot() walks
    // up from that file looking for a sibling skills/ directory, so skills must
    // land at the same $out/lib/qmd prefix as src — not only in the source tree.
    expect(flake).toContain("cp -r src $out/lib/qmd/");
    expect(flake).toContain("cp -r skills $out/lib/qmd/");
    expect(flake).toContain("cp package.json $out/lib/qmd/");
  });

  test("makeWrapper seeds the same pre-import env as bin/qmd (#723)", () => {
    const flake = readFileSync(new URL("flake.nix", root), "utf8");
    const launcher = readFileSync(new URL("bin/qmd", root), "utf8");

    // Nix installs skip bin/qmd and exec bun src/cli/qmd.ts. The wrapper must
    // still set these BEFORE the native binding loads, matching the launcher.
    for (const env of [
      "LLAMA_LOG_LEVEL",
      "GGML_LOG_LEVEL",
      "GGML_BACKEND_SILENT",
      "GGML_METAL_NO_RESIDENCY",
      "QMD_METAL_KEEP_RESIDENCY",
    ]) {
      expect(launcher, `bin/qmd should set ${env}`).toContain(env);
      expect(flake, `flake.nix wrapper should set ${env}`).toContain(env);
    }

    expect(flake).toContain('--run');
    expect(flake).toContain('$1" = mcp');
    expect(flake).toContain('$(uname -s)" = Darwin');
    expect(flake).toContain('LLAMA_LOG_LEVEL:-error');
    expect(flake).toContain('GGML_LOG_LEVEL:-error');
    expect(flake).toContain('GGML_BACKEND_SILENT:-1');
    expect(flake).toContain('GGML_METAL_NO_RESIDENCY:-1');
  });
});
