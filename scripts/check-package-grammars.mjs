#!/usr/bin/env node
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

const grammars = [
  "tree-sitter-typescript/tree-sitter-typescript.wasm",
  "tree-sitter-typescript/tree-sitter-tsx.wasm",
  "tree-sitter-python/tree-sitter-python.wasm",
  "tree-sitter-go/tree-sitter-go.wasm",
  "tree-sitter-rust/tree-sitter-rust.wasm",
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

if (!ok) {
  console.error("\nAST grammar package smoke check failed. Run `bun install` locally or repair a broken global install with the matching `bun add tree-sitter-...@<version>` command shown by `qmd status`.");
  process.exit(1);
}
