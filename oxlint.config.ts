import { defineConfig } from "oxlint";

export default defineConfig({
  // Built-in oxlint categories stay off: this fence is anti-slop, not an unused-var sweep.
  categories: {
    correctness: "off",
    suspicious: "off",
    pedantic: "off",
    perf: "off",
    style: "off",
    restriction: "off",
    nursery: "off",
  },
  plugins: [],
  ignorePatterns: [
    "dist/**",
    "node_modules/**",
    "skills/**",
    "tools/oxlint/anti-slop/**",
    ".agent/**",
    ".agents/**",
    ".claude/**",
    ".claude-plugin/**",
    ".codex/**",
    ".continue/**",
    ".cursor/**",
    ".gemini/**",
    ".github/copilot/**",
    ".opencode/**",
    ".pi/**",
    ".roo/**",
    ".windsurf/**",
  ],
  jsPlugins: [
    { name: "anti-slop", specifier: "./tools/oxlint/anti-slop/index.ts" },
  ],
  rules: {
    "anti-slop/no-module-mocking": "error",
    "anti-slop/no-chained-type-assertions": "error",
    "anti-slop/no-widen-then-assert": "error",
    "anti-slop/no-reflect-get": "error",
    "anti-slop/no-reflect-apply": "error",
    "anti-slop/no-unknown-type-aliases": "error",
    "anti-slop/no-shape-in-symbol-names": "error",
    "anti-slop/no-object-parameters": "error",

    // SQLite row casts and CLI/MCP boundaries are full of these; enabling
    // them now would be hundreds of noisy hits rather than a useful fence.
    "anti-slop/require-safety-comment-for-type-assertion": "off",
    "anti-slop/no-runtime-typeof": "off",
    "anti-slop/no-unknown-parameters": "off",
    "anti-slop/no-unknown-returns": "off",
    "anti-slop/no-conditional-empty-object-spread": "off",
    "anti-slop/no-known-value-widening": "off",
    "anti-slop/no-unsafe-dictionary-type": "off",
  },
});
