# Changelog

## [Unreleased]

### Fixed

- **build**: Remove `shell: true` from `spawnSync` in `scripts/build.mjs`. On
  Windows, passing `shell: true` caused `cmd.exe` to misparse `process.execPath`
  when the Node.js installation path contains spaces (e.g.
  `C:\Program Files\nodejs\node.exe`). Since `run()` always receives a real
  binary path + args array, no shell is needed — `spawnSync` can spawn
  executables directly. (#681)

## [2.5.3] - 2026-05-28

### Features

- `qmd get` now accepts a `:from:count` suffix on a path or docid (e.g.
  `qmd get "#abc123:120:40"` reads 40 lines starting at line 120). Explicit
  `--from`/`-l` flags still override the suffix. The MCP `get` tool accepts the
  same suffix.
- `qmd get` and `qmd multi-get` are now **line-numbered by default** and print
  the document's `#docid` and `qmd://` path in the output header. Disable line
  numbers with `--no-line-numbers`. The MCP `get`/`multi_get` tools default
  `lineNumbers` to `true` to match.
- `qmd multi-get` now includes the `#docid` in every output format
  (`--md`, `--json`, `--csv`, `--xml`, `--files`, and the default CLI view),
  consistent with `qmd search`.
- `qmd get` and `qmd multi-get` accept `--full-path`, which replaces the
  `qmd://` path + `#docid` with the document's on-disk filesystem path (handy for
  piping into `Read`/`Edit`/an editor). Falls back to the canonical `qmd://` +
  docid header when the file no longer exists on disk.
- `qmd search` / `qmd query` now show a clearer hit identifier: the default CLI
  view (and the new `**file:**` line in `--md` output) always prints the full
  `qmd://collection/path` URI so you can pipe it straight back into `qmd get`.
- `qmd search` / `qmd query` accept `--full-path` with the same semantics as
  `qmd get`: the result label becomes the file's on-disk path — `./`-prefixed
  relative path when the file lives in a subfolder of `$PWD`, absolute realpath
  otherwise — and the per-result `#docid` is dropped because the path is the
  identifier. The leading `./` is intentional so the output is unambiguously a
  filesystem path. Applies to all output formats.
- `qmd get` and `qmd multi-get` now also use the `./`-prefixed convention when
  `--full-path` renders a path under `$PWD`, matching `search`/`query`.
- New `--format <kind>` flag selects the output format (`cli` | `json` | `csv` |
  `md` | `xml` | `files`) for `search`, `query`, and `multi-get`. The legacy
  boolean aliases (`--json`/`--csv`/`--md`/`--xml`/`--files`) still work but are
  no longer in `--help`; prefer `--format`.

### Fixes

- Launcher: source-mode runner selection now prefers Node + tsx over Bun when
  both `package-lock.json` and `bun.lock` are present in the package root,
  mirroring the dist-mode "npm priority" rule. Fixes pnpm-global installs that
  copy the entire working tree (including `.git` and `bun.lock`) into the
  install dir and previously routed through Bun, causing ABI mismatches with
  the Node-built `better-sqlite3` / `sqlite-vec` native modules.
- Darwin Metal: llama-using commands (`query`, `vsearch`, `embed`) no longer
  dump a multi-kB GGML/Metal backtrace at process exit even when output
  succeeded. The libggml-metal static `ggml_metal_device` destructor asserts
  `[rsets->data count] == 0` during `__cxa_finalize_ranges`, but the
  buffer-free path never calls the symmetric `ggml_metal_device_rsets_rm`
  to remove released rsets from the device collection (upstream
  ggml-org/llama.cpp#22593, one-line fix open as PR #22595). The assertion
  only fires when `process.exit()` skips Node's `beforeExit` hook, which is
  what node-llama-cpp uses to auto-dispose Metal contexts. Primary fix:
  `finishSuccessfulCliCommand` now sets `process.exitCode = 0` and returns
  instead of calling `process.exit(0)`, so `beforeExit` fires and the native
  binding cleans up before libc's static destructor runs. Defense-in-depth:
  the launcher (`bin/qmd`) and the npm test driver (`scripts/test-all.mjs`
  + the `test:bun` / `test:unit` package.json scripts) also set
  `GGML_METAL_NO_RESIDENCY=1` on darwin before spawning node/bun, covering
  error paths and tests that still terminate via `process.exit()`. The env
  var must be set before node/bun start — libggml-metal reads it via libc
  `getenv` at module-load time, and Bun does not propagate `process.env`
  mutations to libc `setenv` — so it lives in the launcher rather than in
  test-preload. Residency sets give no measurable speedup for QMD's
  short-lived CLI workflow (benchmarked on M3 Pro). Opt back in with
  `QMD_METAL_KEEP_RESIDENCY=1` for long-lived qmd processes (e.g. the MCP
  daemon may benefit on hot reload) or to triage the upstream fix.
  `qmd doctor` reports the mitigation state. Minimal reproduction:
  `scripts/repro-metal-rsets-crash.mjs`.

### Docs

- qmd skill: emphasize reading line ranges with `get`'s built-in
  `:from:count` suffix / `--from`/`-l` flags instead of piping through
  `sed`/`head`/`tail`; cite the docid and line numbers now present in retrieval
  output; and author structured `intent:`/`lex:`/`vec:`/`hyde:` queries yourself
  rather than relying on built-in query expansion.

## [2.5.2] - 2026-05-22

### Fixes

- Launcher: Rewrite `bin/qmd` as a Node-based shebang polyglot to fix global npm installation execution failures on Windows (#668 / #452), while supporting seamless fallback to Bun in Node-less environments.


## [2.5.1] - 2026-05-20

### Changes

- Release: publish from GitHub Actions via npm Trusted Publishing/OIDC instead of a long-lived `NPM_TOKEN` secret.

## [2.5.0] - 2026-05-19

### Changes

- Dependencies: update core SQLite/config/chunking packages (`better-sqlite3`, `yaml`, `web-tree-sitter`, `tree-sitter-go`, and `tree-sitter-python`) while keeping incompatible `zod`, `tsx`, and `vitest` majors pinned.
- Agent skills: add `qmd skills list|get|path` to serve version-matched runtime skill instructions from the installed CLI, and make `qmd skill install` write a stable discovery stub so installed agent skills do not go stale after QMD upgrades.
- CLI: add `qmd doctor` for index/runtime diagnostics, including SQLite/sqlite-vec versions, embedding fingerprint freshness, mixed-fingerprint detection, safe legacy fingerprint adoption, and content-hash sampling.

### Fixes

- Launcher: prefer runnable TypeScript source in git checkouts even when ignored `dist/` artifacts exist, while packaged installs continue to run `dist/`.
- GPU: keep node-llama-cpp's documented `gpu: "auto"` initialization as the primary path, then perform no-build packaged CUDA/Vulkan/Metal probes only if auto falls back to CPU.
- CLI: move GPU/CPU runtime diagnostics out of `qmd status`; use `qmd doctor` for device probing and related environment guidance.
- CLI: point unexpected command/setup failures toward `qmd doctor` so diagnostics are the default next step when QMD behaves incorrectly.
- Doctor: explicitly warn when `content_vectors` contains multiple non-empty embedding fingerprint names, with the per-fingerprint document/chunk breakdown.
- Embed: make the TTY progress line label byte-based input progress explicitly, show embedded chunks as a count, and shorten the displayed model name.
- Embed: retain per-chunk failure details, retry failed chunks after later successful embeds and again when no other chunks remain, clear recovered errors, and cap retries to avoid endless loops.
- Tests: expand the container smoke harness to cover npm-global, npx-style, and Bun-global install scenarios, always checking auto and `QMD_FORCE_CPU=1` doctor modes, with opt-in tiny `qmd embed` and GPU probe runs for supported container runtimes.
- Embedding: fingerprint vector metadata using the active embedding model and formatting/chunking parameters so stale vectors are treated as pending after search semantics change. Legacy `content_vectors` columns are migrated lazily on first vector-health/write use to preserve fast QMD startup.

- Skill: expand the packaged QMD skill with retrieval-first workflows, structured query examples, wiki/source collection guidance, and safe fallbacks when model-backed search is unavailable.
- Tests: make `bun run test` execute the local unit suite under both Node/Vitest and Bun (`test:node` + `test:bun`) so runtime-specific regressions are caught before CI.
- Model config: centralize embedding/rerank/generation model resolution so `qmd embed`, `status`, `query`, `vsearch`, `pull`, SDK vector search, and `bench` use the same active `.qmd/index.yaml` model hints and environment fallbacks.
- GPU/status: `qmd status` now uses the same embedding model identity as `qmd embed` when computing pending embeddings, so URI-backed embeddings are not incorrectly reported as pending under the legacy `embeddinggemma` alias.
- GPU status: `qmd status` now always shows GPU mode/configuration without unsafe native probing, and CPU-fallback warnings point to `QMD_STATUS_DEVICE_PROBE=1 qmd status` for an actual backend probe. The no-GPU warning is emitted once per process instead of once per LLM instance during benchmarks.
- GPU: add `QMD_FORCE_CPU=1` / `--no-gpu` to bypass CUDA/Vulkan/Metal probing entirely, and route native llama.cpp stdout noise to stderr so JSON output stays parseable during search/query commands.
- Snippet line numbers: `qmd_query` (MCP), HTTP `/query`, and `qmd query`
  (CLI JSON output and snippet headers) now return absolute source-file
  line numbers instead of chunk-local ones, so the `line` field can be
  passed back to `qmd_get` as `fromLine` without a separate lookup.
  Snippet selection remains scoped to the best matching chunk
  (preserves #149).
- CLI: `qmd query --full` now emits the full document body in all output
  formats (json, csv, md, xml), restoring the documented behavior of the
  flag. Previously it returned only the best matching chunk (~3.6KB max
  per result). Output payload for `--full` queries is now proportional
  to total document size.
- macOS Metal: `qmd query --json` now flushes successful JSON output and uses a safe immediate-exit path on Darwin to avoid ggml Metal finalizer aborts; other commands still dispose LLM contexts/models before the llama runtime. #368
- Embedding: require complete chunk coverage before treating a document as
  embedded, remove partial vectors when chunk/session failures leave a
  document incomplete, and keep `qmd status` pending counts honest after
  interrupted long embed runs. #637 #378
- Embedding: `qmd embed -c <collection>` now scopes pending-doc selection
  to the requested collection instead of embedding global pending work.
  Scoped `--force` clears only collection-owned vectors, preserves shared
  hashes referenced by sibling collections, and drops `vectors_vec` only
  when the scoped clear empties all vectors.
- Hybrid search: weight RRF lists by query type so original FTS and original vector evidence get the intended 2x boost, instead of accidentally boosting the first lexical expansion. #591
- MCP: seed llama.cpp/GGML quiet env vars before launching `qmd mcp` so native logs cannot pollute stdio JSON-RPC framing. #593
- CLI: remove CommonJS `require()` calls from ESM index path normalization so `qmd --index <path>` no longer crashes with `ERR_AMBIGUOUS_MODULE_SYNTAX` on Node 22+. #634
- Windows CUDA: serialize llama.cpp embedding/reranking contexts by default to avoid intermittent `ggml-cuda.cu:98` crashes in `qmd query`; set `QMD_EMBED_PARALLELISM` to opt back into parallel contexts if your driver is stable. #519
- MCP: make `qmd mcp --index <name>` use the selected index for both foreground and daemon HTTP servers instead of falling back to the default store. #343
- Embedding: respect `QMD_EMBED_MODEL` consistently for vector indexing and vector-backed search, with default-model fallback when unset.
- Config: use one home-directory resolver for YAML config and the default SQLite cache path, avoiding Windows CLI/MCP split-brain when `HOME` is unset.
- GPU: respect explicit `QMD_LLAMA_GPU=metal|vulkan|cuda` backend overrides instead of always using auto GPU selection. #529
- Fix: preserve original filename case in `handelize()`. The previous
  `.toLowerCase()` call made indexed paths unreachable on case-sensitive
  filesystems (Linux). `qmd update` automatically migrates legacy
  lowercase paths without re-embedding.
- CLI: make `qmd status` skip native `node-llama-cpp` device probing by
  default so status stays safe on machines with broken or unsupported GPU
  drivers. Set `QMD_STATUS_DEVICE_PROBE=1` to opt in.
- CLI: lazy-load `node-llama-cpp` so lightweight commands such as
  `qmd status` do not import native ML dependencies or trigger llama.cpp
  builds on ARM/no-GPU machines. #491
- Store: keep content rows referenced by inactive documents during orphan
  cleanup so `qmd update` preserves soft-deleted tombstones for removed
  files. #585
- Packaging: install AST grammar WASM packages as required dependencies so
  Bun global installs include TypeScript/TSX/JavaScript grammars, and add a
  `smoke:package-grammars` verification command. #595
- Launcher: add wrapper smoke coverage for scoped package, npm/npx,
  Homebrew/Linuxbrew, Bun global symlink layouts, and `$BUN_INSTALL`
  false-positive runtime selection regressions. #351 #353 #354 #356 #358 #359

## [2.1.0] - 2026-04-05

Code files now chunk at function and class boundaries via tree-sitter,
clickable editor links land you at the right line from search results,
and per-collection model configuration means you can point different
collections at different embedding models. 25+ community PRs fix
embedding stability, BM25 accuracy, and cross-platform launcher issues.

### Changes

- AST-aware chunking for code files via `web-tree-sitter`. Supported
  languages: TypeScript/JavaScript, Python, Go, and Rust. Code files
  are chunked at function, class, and import boundaries instead of
  arbitrary text positions. Markdown and unknown file types are unchanged.
  `--chunk-strategy <auto|regex>` flag on `qmd embed` and `qmd query`
  (default `regex`). SDK: `chunkStrategy` option on `embed()` and
  `search()`. `qmd status` shows grammar availability.
- `qmd bench <fixture.json>` command for search quality benchmarks.
  Measures precision@k, recall, MRR, and F1 across BM25, vector, hybrid,
  and full pipeline backends. Ships with an example fixture against
  the eval-docs test collection. #470 (thanks @jmilinovich)
- `models:` section in `index.yml` lets you configure `embed`, `rerank`,
  and `generate` model URIs per collection. Resolution order is
  config > env var (`QMD_EMBED_MODEL`, `QMD_RERANK_MODEL`,
  `QMD_GENERATE_MODEL`) > built-in default. #502
  (thanks @JohnRichardEnders)
- CLI search output now emits clickable OSC 8 terminal hyperlinks when
  stdout is a TTY. Links resolve `qmd://` paths to absolute filesystem
  paths and open in editors via URI templates (default:
  `vscode://file/{path}:{line}:{col}`). Configure with `QMD_EDITOR_URI`
  or `editor_uri` in the YAML config. #508 (thanks @danmackinlay)
- `--no-rerank` flag skips the reranking step in `qmd query` — useful
  when you want fast results or don't have a GPU. Also exposed as
  `rerank: false` on the MCP `query` tool. #370 (thanks @mvanhorn),
  #478 (thanks @zestyboy)
- ONNX conversion script for deploying embedding models via
  Transformers.js. #399 (thanks @shreyaskarnik)
- GitHub Actions workflow to build the Nix flake on Linux and macOS.

### Fixes

- Embedding: prevent `qmd embed` from running indefinitely when the
  embedding loop stalls. #458 (thanks @ccc-fff)
- Embedding: truncate oversized text before embedding to prevent GGML
  crash, and bound memory usage during batch embedding. #393
  (thanks @lskun), #395 (thanks @ProgramCaiCai)
- Embedding: set explicit embed context size (default 2048, configurable
  via `QMD_EMBED_CONTEXT_SIZE`) instead of using the model's full
  window. #500
- Embedding: error on dimension mismatch instead of silently rebuilding
  the vec0 table. #501
- Embedding: handle vec0 `OR REPLACE` limitation in `insertEmbedding`.
  #456 (thanks @antonio-mello-ai)
- Embedding: fix model selection when multiple models are configured.
  #494
- BM25: correct field weights to include all 3 FTS columns — title,
  body, and path were not weighted correctly. #462 (thanks @goldsr09)
- BM25: handle hyphenated tokens in FTS5 lex queries so terms like
  "real-time" match correctly. #463 (thanks @goldsr09)
- BM25: preserve underscores in search terms instead of stripping them.
  #404
- BM25: use CTE in `searchFTS` to prevent query planner regression with
  collection filter.
- Reranker: increase default context size 2048→4096 and make
  configurable via `QMD_RERANK_CONTEXT_SIZE`. Fix template overhead
  underestimate 200→512. #453 (thanks @builderjarvis)
- GPU: catch initialization failures and fall back to CPU instead of
  crashing.
- MCP: read version from `package.json` instead of hardcoding. #431
- MCP: include collection name in status output. #416
- Multi-get: support brace expansion patterns in glob matching. #424
- Launcher: prioritize `package-lock.json` to prevent Bun false
  positive. #385 (thanks @rymalia)
- Launcher: remove `$BUN_INSTALL` check that caused false Bun detection.
  #362 (thanks @syedair)
- Launcher: skip Git Bash path detection on WSL. #371
  (thanks @oysteinkrog)
- Model cache: respect `XDG_CACHE_HOME` for model cache directory. #457
  (thanks @antonio-mello-ai)
- SQLite: add macOS Homebrew SQLite support for Bun and restore
  actionable errors. #377 (thanks @serhii12)
- Pin zod to exact 4.2.1 to fix `tsc` build failure. #382
  (thanks @rymalia)
- Preserve dots and original case in `handelize()` — filenames like
  `MEMORY.md` no longer become `memory-md`. #475 (thanks @alexei-led)
- Include `line` in `--json` search output so editor integrations can
  jump directly to `file:line`. #506 (thanks @danmackinlay)
- Nix: fix paths in flake and make Bun dependency a fixed-output
  derivation so sandboxed Linux builds work offline. #479
  (thanks @surma-dump)
- Sync stale `bun.lock` (`better-sqlite3` 11.x → 12.x). CI and release
  script now use `--frozen-lockfile` to prevent recurrence. #386
  (thanks @Mic92)
- Approve native build scripts in pnpm so `better-sqlite3` and
  tree-sitter modules compile correctly. Update vitest ^3.0.0 → ^3.2.4.

## [2.0.1] - 2026-03-10

### Changes

- `qmd skill install` copies the packaged QMD skill into
  `~/.claude/commands/` for one-command setup. #355 (thanks @nibzard)

### Fixes

- Fix Qwen3-Embedding GGUF filename case — HuggingFace filenames are
  case-sensitive, the lowercase variant returned 404. #349 (thanks @byheaven)
- Resolve symlinked global launcher path so `qmd` works correctly when
  installed via `npm i -g`. #352 (thanks @nibzard)

## [2.0.0] - 2026-03-10

QMD 2.0 declares a stable library API. The SDK is now the primary interface —
the MCP server is a clean consumer of it, and the source is organized into
`src/cli/` and `src/mcp/`. Also: Node 25 support and a runtime-aware bin wrapper
for bun installs.

### Changes

- Stable SDK API with `QMDStore` interface — search, retrieval, collection/context
  management, indexing, lifecycle
- Unified `search()`: pass `query` for auto-expansion or `queries` for
  pre-expanded lex/vec/hyde — replaces the old query/search/structuredSearch split
- New `getDocumentBody()`, `getDefaultCollectionNames()`, `Maintenance` class
- MCP server rewritten as a clean SDK consumer — zero internal store access
- CLI and MCP organized into `src/cli/` and `src/mcp/` subdirectories
- Runtime-aware `bin/qmd` wrapper detects bun vs node to avoid ABI mismatches.
  Closes #319
- `better-sqlite3` bumped to ^12.4.5 for Node 25 support. Closes #257
- Utility exports: `extractSnippet`, `addLineNumbers`, `DEFAULT_MULTI_GET_MAX_BYTES`

### Fixes

- Remove unused `import { resolve }` in store.ts that shadowed local export

## [1.1.6] - 2026-03-09

QMD can now be used as a library. `import { createStore } from '@tobilu/qmd'`
gives you the full search and indexing API — hybrid query, BM25, structured
search, collection/context management — without shelling out to the CLI.

### Changes

- **SDK / library mode**: `createStore({ dbPath, config })` returns a
  `QMDStore` with `query()`, `search()`, `structuredSearch()`, `get()`,
  `multiGet()`, and collection/context management methods. Supports inline
  config (no files needed) or a YAML config path.
- **Package exports**: `package.json` now declares `main`, `types`, and
  `exports` so bundlers and TypeScript resolve `@tobilu/qmd` correctly.

## [1.1.5] - 2026-03-07

Ambiguous queries like "performance" now produce dramatically better results
when the caller knows what they mean. The new `intent` parameter steers all
five pipeline stages — expansion, strong-signal bypass, chunk selection,
reranking, and snippet extraction — without searching on its own. Design and
original implementation by Ilya Grigorik (@vyalamar) in #180.

### Changes

- **Intent parameter**: optional `intent` string disambiguates queries across
  the entire search pipeline. Available via CLI (`--intent` flag or `intent:`
  line in query documents), MCP (`intent` field on the query tool), and
  programmatic API. Adapted from PR #180 (thanks @vyalamar).
- **Query expansion**: when intent is provided, the expansion LLM prompt
  includes `Query intent: {intent}`, matching the finetune training data
  format for better-aligned expansions.
- **Reranking**: intent is prepended to the rerank query so Qwen3-Reranker
  scores with domain context.
- **Chunk selection**: intent terms scored at 0.5× weight alongside query
  terms (1.0×) when selecting the best chunk per document for reranking.
- **Snippet extraction**: intent terms scored at 0.3× weight to nudge
  snippets toward intent-relevant lines without overriding query anchoring.
- **Strong-signal bypass disabled with intent**: when intent is provided, the
  BM25 strong-signal shortcut is skipped — the obvious keyword match may not
  be what the caller wants.
- **MCP instructions**: callers are now guided to provide `intent` on every
  search call for disambiguation.
- **Query document syntax**: `intent:` recognized as a line type. At most one
  per document, cannot appear alone. Grammar updated in `docs/SYNTAX.md`.

## [1.1.2] - 2026-03-07

13 community PRs merged. GPU initialization replaced with node-llama-cpp's
built-in `autoAttempt` — deleting ~220 lines of manual fallback code and
fixing GPU issues reported across 10+ PRs in one shot. Reranking is faster
through chunk deduplication and a parallelism cap that prevents VRAM
exhaustion.

### Changes

- **GPU init**: use node-llama-cpp's `build: "autoAttempt"` instead of manual
  GPU backend detection. Automatically tries Metal/CUDA/Vulkan and falls back
  gracefully. #310 (thanks @giladgd — the node-llama-cpp author)
- **Query `--explain`**: `qmd query --explain` exposes retrieval score traces
  — backend scores, per-list RRF contributions, top-rank bonus, reranker
  score, and final blended score. Works in JSON and CLI output. #242
  (thanks @vyalamar)
- **Collection ignore patterns**: `ignore: ["Sessions/**", "*.tmp"]` in
  collection config to exclude files from indexing. #304 (thanks @sebkouba)
- **Multilingual embeddings**: `QMD_EMBED_MODEL` env var lets you swap in
  models like Qwen3-Embedding for non-English collections. #273 (thanks
  @daocoding)
- **Configurable expansion context**: `QMD_EXPAND_CONTEXT_SIZE` env var
  (default 2048) — previously used the model's full 40960-token window,
  wasting VRAM. #313 (thanks @0xble)
- **`candidateLimit` exposed**: `-C` / `--candidate-limit` flag and MCP
  parameter to tune how many candidates reach the reranker. #255 (thanks
  @pandysp)
- **MCP multi-session**: HTTP transport now supports multiple concurrent
  client sessions, each with its own server instance. #286 (thanks @joelev)

### Fixes

- **Reranking performance**: cap parallel rerank contexts at 4 to prevent
  VRAM exhaustion on high-core machines. Deduplicate identical chunk texts
  before reranking — same content from different files now shares a single
  reranker call. Cache scores by content hash instead of file path.
- Deactivate stale docs when all files are removed from a collection and
  `qmd update` is run. #312 (thanks @0xble)
- Handle emoji-only filenames (`🐘.md` → `1f418.md`) instead of crashing.
  #308 (thanks @debugerman)
- Skip unreadable files during indexing (e.g. iCloud-evicted files returning
  EAGAIN) instead of crashing. #253 (thanks @jimmynail)
- Suppress progress bar escape sequences when stderr is not a TTY. #230
  (thanks @dgilperez)
- Emit format-appropriate empty output (`[]` for JSON, CSV header for CSV,
  etc.) instead of plain text "No results." #228 (thanks @amsminn)
- Correct Windows sqlite-vec package name (`sqlite-vec-windows-x64`) and add
  `sqlite-vec-linux-arm64`. #225 (thanks @ilepn)
- Fix claude plugin setup CLI commands in README. #311 (thanks @gi11es)

## [1.1.1] - 2026-03-06

### Fixes

- Reranker: truncate documents exceeding the 2048-token context window
  instead of silently producing garbage scores. Long chunks (e.g. from
  PDF ingestion) now get a fair ranking.
- Nix: add python3 and cctools to build dependencies. #214 (thanks
  @pcasaretto)

## [1.1.0] - 2026-02-20

QMD now speaks in **query documents** — structured multi-line queries where every line is typed (`lex:`, `vec:`, `hyde:`), combining keyword precision with semantic recall. A single plain query still works exactly as before (it's treated as an implicit `expand:` and auto-expanded by the LLM). Lex now supports quoted phrases and negation (`"C++ performance" -sports -athlete`), making intent-aware disambiguation practical. The formal query grammar is documented in `docs/SYNTAX.md`.

The npm package now uses the standard `#!/usr/bin/env node` bin convention, replacing the custom bash wrapper. This fixes native module ABI mismatches when installed via bun and works on any platform with node >= 22 on PATH.

### Changes

- **Query document format**: multi-line queries with typed sub-queries (`lex:`, `vec:`, `hyde:`). Plain queries remain the default (`expand:` implicit, but not written inside the document). First sub-query gets 2× fusion weight — put your strongest signal first. Formal grammar in `docs/SYNTAX.md`.
- **Lex syntax**: full BM25 operator support. `"exact phrase"` for verbatim matching; `-term` and `-"phrase"` for exclusions. Essential for disambiguation when a term is overloaded across domains (e.g. `performance -sports -athlete`).
- **`expand:` shortcut**: send a single plain query (or start the document with `expand:` on its only line) to auto-expand via the local LLM. Query documents themselves are limited to `lex`, `vec`, and `hyde` lines.
- **MCP `query` tool** (renamed from `structured_search`): rewrote the tool description to fully teach AI agents the query document format, lex syntax, and combination strategy. Includes worked examples with intent-aware lex.
- **HTTP `/query` endpoint** (renamed from `/search`; `/search` kept as silent alias).
- **`collections` array filter**: filter by multiple collections in a single query (`collections: ["notes", "brain"]`). Removed the single `collection` string param — array only.
- **Collection `include`/`exclude`**: `includeByDefault: false` hides a collection from all queries unless explicitly named via `collections`. CLI: `qmd collection exclude <name>` / `qmd collection include <name>`.
- **Collection `update-cmd`**: attach a shell command that runs before every `qmd update` (e.g. `git stash && git pull --rebase --ff-only && git stash pop`). CLI: `qmd collection update-cmd <name> '<cmd>'`.
- **`qmd status` tips**: shows actionable tips when collections lack context descriptions or update commands.
- **`qmd collection` subcommands**: `show`, `update-cmd`, `include`, `exclude`. Bare `qmd collection` now prints help.
- **Packaging**: replaced custom bash wrapper with standard `#!/usr/bin/env node` shebang on `dist/qmd.js`. Fixes native module ABI mismatches when installed via bun, and works on any platform where node >= 22 is on PATH.
- **Removed MCP tools** `search`, `vector_search`, `deep_search` — all superseded by `query`.
- **Removed** `qmd context check` command.
- **CLI timing**: each LLM step (expand, embed, rerank) prints elapsed time inline (`Expanding query... (4.2s)`).

### Fixes

- `qmd collection list` shows `[excluded]` tag for collections with `includeByDefault: false`.
- Default searches now respect `includeByDefault` — excluded collections are skipped unless explicitly named.
- Fix main module detection when installed globally via npm/bun (symlink resolution).

## [1.0.7] - 2026-02-18

### Changes

- LLM: add LiquidAI LFM2-1.2B as an alternative base model for query
  expansion fine-tuning. LFM2's hybrid architecture (convolutions + attention)
  is 2x faster at decode/prefill vs standard transformers — good fit for
  on-device inference.
- CLI: support multiple `-c` flags to search across several collections at
  once (e.g. `qmd search -c notes -c journals "query"`). #191 (thanks
  @openclaw)

### Fixes

- Return empty JSON array `[]` instead of no output when `--json` search
  finds no results.
- Resolve relative paths passed to `--index` so they don't produce malformed
  config entries.
- Respect `XDG_CONFIG_HOME` for collection config path instead of always
  using `~/.config`. #190 (thanks @openclaw)
- CLI: empty-collection hint now shows the correct `collection add` command.
  #200 (thanks @vincentkoc)

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
