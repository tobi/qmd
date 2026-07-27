import { describe, expect, test } from "vitest";
import { readFileSync } from "fs";
import { join } from "path";
import {
  NODE_LLAMA_CPP_UNAVAILABLE_MESSAGE,
  resetNodeLlamaCppModuleForTest,
  setNodeLlamaCppModuleForTest,
} from "../src/llm.js";

describe("LLM module loading", () => {
  test("node-llama-cpp is only dynamically imported by LLM operations", () => {
    const source = readFileSync(join(process.cwd(), "src", "llm.ts"), "utf-8");

    expect(source).not.toMatch(/import\s+(?!type\b)[\s\S]*?from\s+["']node-llama-cpp["']/);
    expect(source).toContain('import("node-llama-cpp")');
  });

  test("importing the CLI for lightweight commands succeeds", async () => {
    const mod = await import("../src/cli/qmd.ts");
    expect(mod).toMatchObject({
      buildEditorUri: expect.any(Function),
      termLink: expect.any(Function),
    });
  });

  test("package declares node-llama-cpp only as an optional dependency", () => {
    const pkg = JSON.parse(readFileSync(join(process.cwd(), "package.json"), "utf-8"));

    expect(pkg.dependencies).not.toHaveProperty("node-llama-cpp");
    expect(pkg.optionalDependencies).toMatchObject({
      "node-llama-cpp": "3.18.1",
    });
  });

  test("CLI LLM availability guard reports the actionable optional dependency error", async () => {
    const { assertNodeLlamaCppAvailableForCli } = await import("../src/cli/qmd.ts");
    setNodeLlamaCppModuleForTest(null);
    try {
      await expect(assertNodeLlamaCppAvailableForCli()).rejects.toThrow(NODE_LLAMA_CPP_UNAVAILABLE_MESSAGE);
    } finally {
      resetNodeLlamaCppModuleForTest();
    }
  });

  test("doctor warns for darwin x64 Node and not for native arm64 or non-darwin", async () => {
    const { rosettaNodeDoctorWarning } = await import("../src/cli/qmd.ts");

    expect(rosettaNodeDoctorWarning("darwin", "x64")).toContain("native arm64 Node");
    expect(rosettaNodeDoctorWarning("darwin", "arm64")).toBeNull();
    expect(rosettaNodeDoctorWarning("linux", "x64")).toBeNull();
  });
});
