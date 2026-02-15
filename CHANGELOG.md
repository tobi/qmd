# Changelog

All notable changes to QMD will be documented in this file.

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

[0.9.0]: https://github.com/tobi/qmd/releases/tag/v0.9.0
