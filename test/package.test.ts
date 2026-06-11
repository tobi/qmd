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
