import { existsSync, mkdtempSync, mkdirSync, writeFileSync, rmSync, realpathSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, describe, expect, test } from "vitest";
import { findLocalConfigPath, getLocalDbPath } from "../src/collections.js";

function cliCommandArgs(command: string): { bin: string; args: string[] } {
  const cliPath = join(process.cwd(), "src/cli/qmd.ts");
  if (process.versions.bun) {
    return { bin: process.execPath, args: [cliPath, command] };
  }
  return {
    bin: process.execPath,
    args: [join(process.cwd(), "node_modules/tsx/dist/cli.mjs"), cliPath, command],
  };
}

const roots: string[] = [];

function tempProject(): string {
  const root = mkdtempSync(join(tmpdir(), "qmd-local-config-"));
  roots.push(root);
  return root;
}

afterEach(() => {
  for (const root of roots.splice(0)) {
    rmSync(root, { recursive: true, force: true });
  }
});

describe("local .qmd project config", () => {
  test("finds .qmd/index.yaml from nested working directories", () => {
    const root = tempProject();
    const configPath = join(root, ".qmd", "index.yaml");
    mkdirSync(join(root, ".qmd"), { recursive: true });
    writeFileSync(configPath, "collections: {}\n");
    const nested = join(root, "wiki", "Shopify");
    mkdirSync(nested, { recursive: true });

    expect(findLocalConfigPath(nested)).toBe(configPath);
  });

  test("prefers index.yaml over index.yml when both exist", () => {
    const root = tempProject();
    mkdirSync(join(root, ".qmd"), { recursive: true });
    const yaml = join(root, ".qmd", "index.yaml");
    const yml = join(root, ".qmd", "index.yml");
    writeFileSync(yaml, "collections: {}\n");
    writeFileSync(yml, "collections: {}\n");

    expect(findLocalConfigPath(root)).toBe(yaml);
  });

  test("uses .qmd/index.sqlite next to the local config", () => {
    const root = tempProject();
    mkdirSync(join(root, ".qmd"), { recursive: true });
    const configPath = join(root, ".qmd", "index.yaml");
    writeFileSync(configPath, "collections: {}\n");

    expect(getLocalDbPath(configPath)).toBe(join(root, ".qmd", "index.sqlite"));
  });

  test("CLI uses local .qmd config and index instead of global cache", () => {
    const root = tempProject();
    mkdirSync(join(root, ".qmd"), { recursive: true });
    mkdirSync(join(root, "docs"), { recursive: true });
    writeFileSync(join(root, "docs", "a.md"), "# A\n\nLocal test document.\n");
    writeFileSync(join(root, ".qmd", "index.yaml"), `collections:\n  docs:\n    path: ${JSON.stringify(join(root, "docs"))}\n    pattern: "**/*.md"\n    context:\n      /: Local test docs\nmodels:\n  embed: local-embed-model\n  rerank: local-rerank-model\n  generate: local-generate-model\n`);

    const home = join(root, "home");
    const { bin, args } = cliCommandArgs("status");
    const output = execFileSync(bin, args, {
      cwd: root,
      encoding: "utf-8",
      env: {
        ...process.env,
        HOME: home,
        XDG_CONFIG_HOME: join(home, ".config"),
        XDG_CACHE_HOME: join(home, ".cache"),
        QMD_EMBED_MODEL: "env-embed-model",
        QMD_RERANK_MODEL: "env-rerank-model",
        QMD_GENERATE_MODEL: "env-generate-model",
      },
    });

    const localIndex = join(root, ".qmd", "index.sqlite");
    expect(output).toContain(`Index: ${realpathSync(localIndex)}`);
    expect(output).toContain("docs (qmd://docs/)");
    expect(output).toContain("Embedding:   local-embed-model");
    expect(output).toContain("Reranking:   local-rerank-model");
    expect(output).toContain("Generation:  local-generate-model");
    expect(output).not.toContain("env-embed-model");
    expect(existsSync(localIndex)).toBe(true);
    expect(existsSync(join(home, ".cache", "qmd", "index.sqlite"))).toBe(false);
  });
});
