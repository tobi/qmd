#!/usr/bin/env node
import { existsSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const grammars = [
  "tree-sitter-typescript/tree-sitter-typescript.wasm",
  "tree-sitter-typescript/tree-sitter-tsx.wasm",
  "tree-sitter-python/tree-sitter-python.wasm",
  "tree-sitter-go/tree-sitter-go.wasm",
  "tree-sitter-rust/tree-sitter-rust.wasm",
];

// Grammars whose npm package ships no .wasm; built with `tree-sitter build
// --wasm` and bundled in assets/grammars/ instead.
const bundledGrammars = [
  "../assets/grammars/tree-sitter-swift.wasm",
];

let ok = true;
for (const grammar of grammars) {
  try {
    const resolved = require.resolve(grammar);
    console.log(`ok ${grammar} -> ${resolved}`);
  } catch (err) {
    ok = false;
    console.error(`missing ${grammar}`);
    console.error(err instanceof Error ? err.message : String(err));
  }
}

for (const grammar of bundledGrammars) {
  const resolved = new URL(grammar, import.meta.url);
  if (existsSync(resolved)) {
    console.log(`ok ${grammar} -> ${resolved.pathname}`);
  } else {
    ok = false;
    console.error(`missing bundled ${grammar}`);
  }
}

if (!ok) {
  console.error("\nAST grammar package smoke check failed. Run `bun install` locally or repair a broken global install with the matching `bun add tree-sitter-...@<version>` command shown by `qmd status`.");
  process.exit(1);
}
