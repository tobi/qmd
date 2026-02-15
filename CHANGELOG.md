# Changelog

All notable changes to QMD will be documented in this file.

## [1.0.0] - 2026-02-15

### Node.js Compatibility

QMD now runs on both **Node.js (>=22)** and **Bun**. Install with `npm install -g @tobilu/qmd` or `bun install -g @tobilu/qmd` — your choice. The `qmd` wrapper auto-detects Node.js via `tsx` and works out of the box with mise, asdf, nvm, and Homebrew installs.

### Performance

- **Parallel embedding & reranking** — multiple contexts split work across CPU cores (or VRAM on GPU), delivering up to **2.7x faster reranking** and significantly faster embedding on multi-core machines
- **Flash attention** — ~20% less VRAM per reranking context, enabling more parallel contexts on GPU
- **Right-sized contexts** — reranker context dropped from 40960 to 2048 tokens (17x less memory), since chunks are capped at ~900 tokens
- **Adaptive parallelism** — automatically scales context count based on available VRAM (GPU) or CPU math cores
- **CPU thread splitting** — each context runs on its own cores for true parallelism instead of contending on a single context

### GPU Auto-Detection

- Probes for CUDA, Metal, and Vulkan at startup — uses the best available backend
- Falls back gracefully to CPU with a warning if GPU init fails
- `qmd status` now shows device info (GPU type, VRAM usage)

### Test Suite

- Tests split into `src/*.test.ts` (unit), `src/models/*.test.ts` (model), and `src/integration/*.test.ts` (CLI/integration)
- Vitest config for Node.js; bun test still works for Bun
- New `eval-bm25` and `store.helpers.unit` test suites

### Fixes

- Prevent VRAM waste from duplicate context creation during concurrent loads
- Collection-aware FTS filtering for scoped keyword search

---

## [0.9.0] - 2026-02-15

Initial public release.

### Features

- **Hybrid search pipeline** — BM25 full-text + vector similarity + LLM reranking with Reciprocal Rank Fusion
- **Smart chunking** — scored markdown break points keep sections, paragraphs, and code blocks intact (~900 tokens/chunk, 15% overlap)
- **Query expansion** — fine-tuned Qwen3 1.7B model generates search variations for better recall
- **Cross-encoder reranking** — Qwen3-Reranker scores candidates with position-aware blending
- **Vector embeddings** — EmbeddingGemma 300M via node-llama-cpp, all on-device
- **MCP server** — stdio and HTTP transports for Claude Desktop, Claude Code, and any MCP client
- **Collection management** — index multiple directories with glob patterns
- **Context annotations** — add descriptions to collections and paths for richer search
- **Document IDs** — 6-char content hash for stable references across re-indexes
- **Multi-get** — retrieve multiple documents by glob pattern, comma list, or docids
- **Multiple output formats** — JSON, CSV, Markdown, XML, files list
- **Claude Code plugin** — inline status checks and MCP integration

### Fixes

- Handle dense content (code) that tokenizes beyond expected chunk size
- Proper cleanup of Metal GPU resources
- SQLite-vec readiness verification after extension load
- Reactivate deactivated documents on re-index
- BM25 score normalization with Math.abs
- Bun UTF-8 path corruption workaround

[1.0.0]: https://github.com/tobi/qmd/releases/tag/v1.0.0
[0.9.0]: https://github.com/tobi/qmd/releases/tag/v0.9.0

