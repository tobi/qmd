# Changelog

## [Unreleased]

## [1.0.6] - 2026-02-16

### Changes

- CLI: `qmd status` now shows models with full HuggingFace links instead of
  static names in `--help`. Model info is derived from the actual configured
  URIs so it stays accurate if models change.
- Release tooling: pre-push hook handles non-interactive shells (CI, editors)
  gracefully — warnings auto-proceed instead of hanging on a tty prompt.
  Annotated tags now resolve correctly for CI checks.

## [1.0.5] - 2026-02-16

The npm package now ships compiled JavaScript instead of raw TypeScript,
removing the `tsx` runtime dependency. A new `/release` skill automates the
full release workflow with changelog validation and git hook enforcement.

### Changes

- Build: compile TypeScript to `dist/` via `tsc` so the npm package no longer
  requires `tsx` at runtime. The `qmd` shell wrapper now runs `dist/qmd.js`
  directly.
- Release tooling: new `/release` skill that manages the full release
  lifecycle — validates changelog, installs git hooks, previews release notes,
  and cuts the release. Auto-populates `[Unreleased]` from git history when
  empty.
- Release tooling: `scripts/extract-changelog.sh` extracts cumulative notes
  for the full minor series (e.g. 1.0.0 through 1.0.5) for GitHub releases.
  Includes `[Unreleased]` content in previews.
- Release tooling: `scripts/release.sh` renames `[Unreleased]` to a versioned
  heading and inserts a fresh empty `[Unreleased]` section automatically.
- Release tooling: pre-push git hook blocks `v*` tag pushes unless
  `package.json` version matches the tag, a changelog entry exists, and CI
  passed on GitHub.
- Publish workflow: GitHub Actions now builds TypeScript, creates a GitHub
  release with cumulative notes extracted from the changelog, and publishes
  to npm with provenance.

## [1.0.0] - 2026-02-15

QMD now runs on both Node.js and Bun, with up to 2.7x faster reranking
through parallel GPU contexts. GPU auto-detection replaces the unreliable
`gpu: "auto"` with explicit CUDA/Metal/Vulkan probing.

### Changes

- Runtime: support Node.js (>=22) alongside Bun via a cross-runtime SQLite
  abstraction layer (`src/db.ts`). `bun:sqlite` on Bun, `better-sqlite3` on
  Node. The `qmd` wrapper auto-detects a suitable Node.js install via PATH,
  then falls back to mise, asdf, nvm, and Homebrew locations.
- Performance: parallel embedding & reranking via multiple LlamaContext
  instances — up to 2.7x faster on multi-core machines.
- Performance: flash attention for ~20% less VRAM per reranking context,
  enabling more parallel contexts on GPU.
- Performance: right-sized reranker context (40960 → 2048 tokens, 17x less
  memory) since chunks are capped at ~900 tokens.
- Performance: adaptive parallelism — context count computed from available
  VRAM (GPU) or CPU math cores rather than hardcoded.
- GPU: probe for CUDA, Metal, Vulkan explicitly at startup instead of
  relying on node-llama-cpp's `gpu: "auto"`. `qmd status` shows device info.
- Tests: reorganized into flat `test/` directory with vitest for Node.js and
  bun test for Bun. New `eval-bm25` and `store.helpers.unit` suites.

### Fixes

- Prevent VRAM waste from duplicate context creation during concurrent
  `embedBatch` calls — initialization lock now covers the full path.
- Collection-aware FTS filtering so scoped keyword search actually restricts
  results to the requested collection.

## [0.9.0] - 2026-02-15

First published release on npm as `@tobilu/qmd`. MCP HTTP transport with
daemon mode cuts warm query latency from ~16s to ~10s by keeping models
loaded between requests.

### Changes

- MCP: HTTP transport with daemon lifecycle — `qmd mcp --http --daemon`
  starts a background server, `qmd mcp stop` shuts it down. Models stay warm
  in VRAM between queries. #149 (thanks @igrigorik)
- Search: type-routed query expansion preserves lex/vec/hyde type info and
  routes to the appropriate backend. Eliminates ~4 wasted backend calls per
  query (10.0 → 6.0 calls, 1278ms → 549ms). #149 (thanks @igrigorik)
- Search: unified pipeline — extracted `hybridQuery()` and
  `vectorSearchQuery()` to `store.ts` so CLI and MCP share identical logic.
  Fixes a class of bugs where results differed between the two. #149 (thanks
  @igrigorik)
- MCP: dynamic instructions generated at startup from actual index state —
  LLMs see collection names, doc counts, and content descriptions. #149
  (thanks @igrigorik)
- MCP: tool renames (vsearch → vector_search, query → deep_search) with
  rewritten descriptions for better tool selection. #149 (thanks @igrigorik)
- Integration: Claude Code plugin with inline status checks and MCP
  integration. #99 (thanks @galligan)

### Fixes

- BM25 score normalization — formula was inverted (`1/(1+|x|)` instead of
  `|x|/(1+|x|)`), so strong matches scored *lowest*. Broke `--min-score`
  filtering and made the "strong signal" short-circuit dead code. #76 (thanks
  @dgilperez)
- Normalize Unicode paths to NFC for macOS compatibility. #82 (thanks
  @c-stoeckl)
- Handle dense content (code) that tokenizes beyond expected chunk size.
- Proper cleanup of Metal GPU resources on process exit.
- SQLite-vec readiness verification after extension load.
- Reactivate deactivated documents on re-index instead of creating duplicates.
- Bun UTF-8 path corruption workaround for non-ASCII filenames.
- Disable following symlinks in glob.scan to avoid infinite loops.

## [0.8.0] - 2026-01-28

Fine-tuned query expansion model trained with GRPO replaces the stock Qwen3
0.6B. The training pipeline scores expansions on named entity preservation,
format compliance, and diversity — producing noticeably better lexical
variations and HyDE documents.

### Changes

- LLM: deploy GRPO-trained (Group Relative Policy Optimization) query
  expansion model, hosted on HuggingFace and auto-downloaded on first use.
  Better preservation of proper nouns and technical terms in expansions.
- LLM: `/only:lex` mode for single-type expansions — useful when you know
  which search backend will help.
- LLM: HyDE output moved to first position so vector search can start
  embedding while other expansions generate.
- LLM: session lifecycle management via `withLLMSession()` pattern — ensures
  cleanup even on failure, similar to database transactions.
- Integration: org-mode title extraction support. #50 (thanks @sh54)
- Integration: SQLite extension loading in Nix devshell. #48 (thanks @sh54)
- Integration: AI agent discovery via skills.sh. #64 (thanks @Algiras)

### Fixes

- Use sequential embedding on CPU-only systems — parallel contexts caused a
  race condition where contexts competed for CPU cores, making things slower.
  #54 (thanks @freeman-jiang)
- Fix `collectionName` column in vector search SQL (was still using old
  `collectionId` from before YAML migration). #61 (thanks @jdvmi00)
- Fix Qwen3 sampling params to prevent repetition loops — stock
  temperature/top-p caused occasional infinite repeat patterns.
- Add `--index` option to CLI argument parser (was documented but not wired
  up). #84 (thanks @Tritlo)
- Fix DisposedError during slow batch embedding. #41 (thanks @wuhup)

## [0.7.0] - 2026-01-09

First community contributions. The project gained external contributors,
surfacing bugs that only appear in diverse environments — Homebrew sqlite-vec
paths, case-sensitive model filenames, and sqlite-vec JOIN incompatibilities.

### Changes

- Indexing: native `realpathSync()` replaces `readlink -f` subprocess spawn
  per file. On a 5000-file collection this eliminates 5000 shell spawns,
  ~15% faster. #8 (thanks @burke)
- Indexing: single-pass tokenization — chunking algorithm tokenized each
  document twice (count then split); now tokenizes once and reuses. #9
  (thanks @burke)

### Fixes

- Fix `vsearch` and `query` hanging — sqlite-vec's virtual table doesn't
  support the JOIN pattern used; rewrote to subquery. #23 (thanks @mbrendan)
- Fix MCP server exiting immediately after startup — process had no active
  handles keeping the event loop alive. #29 (thanks @mostlydev)
- Fix collection filter SQL to properly restrict vector search results.
- Support non-ASCII filenames in collection filter.
- Skip empty files during indexing instead of crashing on zero-length content.
- Fix case sensitivity in Qwen3 model filename resolution. #15 (thanks
  @gavrix)
- Fix sqlite-vec loading on macOS with Homebrew (`BREW_PREFIX` detection).
  #42 (thanks @komsit37)
- Fix Nix flake to use correct `src/qmd.ts` path. #7 (thanks @burke)
- Fix docid lookup with quotes support in get command. #36 (thanks
  @JoshuaLelon)
- Fix query expansion model size in documentation. #38 (thanks @odysseus0)

## [0.6.0] - 2025-12-28

Replaced Ollama HTTP API with node-llama-cpp for all LLM operations. Ollama
adds convenience but also a running server dependency. node-llama-cpp loads
GGUF models directly in-process — zero external dependencies. Models
auto-download from HuggingFace on first use.

### Changes

- LLM: structured query expansion via JSON schema grammar constraints.
  Model produces typed expansions — **lexical** (BM25 keywords), **vector**
  (semantic rephrasings), **HyDE** (hypothetical document excerpts) — so each
  routes to the right backend instead of sending everything everywhere.
- LLM: lazy model loading with 2-minute inactivity auto-unload. Keeps memory
  low when idle while avoiding ~3s model load on every query.
- Search: conditional query expansion — when BM25 returns strong results, the
  expensive LLM expansion is skipped entirely.
- Search: multi-chunk reranking — documents with multiple relevant chunks
  scored by aggregating across all chunks rather than best single chunk.
- Search: cosine distance for vector search (was L2).
- Search: embeddinggemma nomic-style prompt formatting.
- Testing: evaluation harness with synthetic test documents and Hit@K metrics
  for BM25, vector, and hybrid RRF.

## [0.5.0] - 2025-12-13

Collections and contexts moved from SQLite tables to YAML at
`~/.config/qmd/index.yml`. SQLite was overkill for config — you can't share
it, and it's opaque. YAML is human-readable and version-controllable. The
migration was extensive (35+ commits) because every part of the system that
touched collections or contexts had to be updated.

### Changes

- Config: YAML-based collections and contexts replace SQLite tables.
  `collections` and `path_contexts` tables dropped from schema. Collections
  support an optional `update:` command (e.g., `git pull`) before re-index.
- CLI: `qmd collection add/list/remove/rename` commands with `--name` and
  `--mask` glob pattern support.
- CLI: `qmd ls` virtual file tree — list collections, files in a collection,
  or files under a path prefix.
- CLI: `qmd context add/list/check/rm` with hierarchical context inheritance.
  A query to `qmd://notes/2024/jan/` inherits context from `notes/`,
  `notes/2024/`, and `notes/2024/jan/`.
- CLI: `qmd context add / "text"` for global context across all collections.
- CLI: `qmd context check` audit command to find paths without context.
- Paths: `qmd://` virtual URI scheme for portable document references.
  `qmd://notes/ideas.md` works regardless of where the collection lives on
  disk. Works in `get`, `multi-get`, `ls`, and context commands.
- CLI: document IDs (docid) — first 6 chars of content hash for stable
  references. Shown as `#abc123` in search results, usable with `get` and
  `multi-get`.
- CLI: `--line-numbers` flag for get command output.

## [0.4.0] - 2025-12-10

MCP server for AI agent integration. Without it, agents had to shell out to
`qmd search` and parse CLI output. The monolithic `qmd.ts` (1840 lines) was
split into focused modules with the project's first test suite (215 tests).

### Changes

- MCP: stdio server with tools for search, vector search, hybrid query,
  document retrieval, and status. Runs over stdio transport for Claude
  Desktop and MCP clients.
- MCP: spec-compliant with June 2025 MCP specification — removed non-spec
  `mimeType`, added `isError: true` to errors, `structuredContent` for
  machine-readable results, proper URI encoding.
- MCP: simplified tool naming (`qmd_search` → `search`) since MCP already
  namespaces by server.
- Architecture: extract `store.ts` (1221 LOC), `llm.ts` (539 LOC),
  `formatter.ts` (359 LOC), `mcp.ts` (503 LOC) from monolithic `qmd.ts`.
- Testing: 215 tests (store: 96, llm: 60, mcp: 59) with mocked Ollama for
  fast, deterministic runs. Before this: zero tests.

## [0.3.0] - 2025-12-08

Document chunking for vector search. A 5000-word document about many topics
gets a single embedding that averages everything together, matching poorly for
specific queries. Chunking produces one embedding per ~900-token section with
focused semantic signal.

### Changes

- Search: markdown-aware chunking — prefers heading boundaries, then paragraph
  breaks, then sentence boundaries. 15% overlap between chunks ensures
  cross-boundary queries still match.
- Search: multi-chunk scoring bonus (+0.02 per additional chunk, capped at
  +0.1 for 5+ chunks). Documents relevant in multiple sections rank higher.
- CLI: display paths show collection-relative paths and extracted titles
  (from H1 headings or YAML frontmatter) instead of raw filesystem paths.
- CLI: `--all` flag returns all matches (use with `--min-score` to filter).
- CLI: byte-based progress bar with ETA for `embed` command.
- CLI: human-readable time formatting ("15m 4s" instead of "904.2s").
- CLI: documents >64KB truncated with warning during embedding.

## [0.2.0] - 2025-12-08

### Changes

- CLI: `--json`, `--csv`, `--files`, `--md`, `--xml` output format flags.
  `--json` for programmatic access, `--files` for piping, `--md`/`--xml` for
  LLM consumption, `--csv` for spreadsheets.
- CLI: `qmd status` shows index health — document count, size, embedding
  coverage, time since last update.
- Search: weighted RRF — original query gets 2x weight relative to expanded
  queries since the user's actual words are a more reliable signal.

## [0.1.0] - 2025-12-07

Initial implementation. Built in a single day for searching personal markdown
notes, journals, and meeting transcripts.

### Changes

- Search: SQLite FTS5 with BM25 ranking. Chose SQLite over Elasticsearch
  because QMD is a personal tool — single binary, no server dependencies.
- Search: sqlite-vec for vector similarity. Same rationale: in-process, no
  external vector database.
- Search: Reciprocal Rank Fusion to combine BM25 and vector results. RRF is
  parameter-free and handles missing signals gracefully.
- LLM: Ollama for embeddings, reranking, and query expansion. Later replaced
  with node-llama-cpp in 0.6.0.
- CLI: `qmd add`, `qmd embed`, `qmd search`, `qmd vsearch`, `qmd query`,
  `qmd get`. ~1800 lines of TypeScript in a single `qmd.ts` file.

[Unreleased]: https://github.com/tobi/qmd/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tobi/qmd/releases/tag/v1.0.0
[0.9.0]: https://github.com/tobi/qmd/compare/v0.8.0...v0.9.0

