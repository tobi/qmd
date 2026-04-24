# QMD — Query Markup Documents

An on-device search engine for your markdown knowledge base. Index notes, meeting transcripts, research papers, and documentation. Search with keywords, natural language, or structured metadata filters — all running locally, nothing leaves your machine.

---

## Quick Start

### Install

**Via Claude Code marketplace (recommended):**

```sh
claude plugin marketplace add idanariav/pkm-query-tools
claude plugin install qmd@pkm-query-tools
```

**Via npm:**

```sh
npm install -g @tobilu/qmd
```

Requires Node.js ≥ 22.

### Get running in 3 steps

```sh
# 1. Register your notes as a collection
qmd collection add ~/notes --name notes

# 2. Add context to help the search understand your content
qmd context add qmd://notes "Personal notes, ideas, and research"

# 3. Generate semantic embeddings
qmd embed
```

### Popular commands

```sh
# Hybrid search — query expansion + reranking (best quality)
qmd hsearch "quarterly planning process"

# Keyword search — fast, no LLM needed
qmd tsearch "project timeline"

# Semantic search — find by meaning
qmd vsearch "how to handle failure gracefully"

# Filter by metadata — tags, dates, frontmatter fields
qmd fsearch 'tag=productivity AND modified > 30d'

# Retrieve a specific document
qmd get "notes/meeting-2025-01-15.md"

# Get a document by its short ID (shown in search results)
qmd get "#abc123"
```

---

## Use Cases

QMD is built for knowledge management workflows — Zettelkasten, Obsidian vaults, research archives, meeting notes. It answers questions you can't easily ask a file browser or basic grep.

**Find an idea you vaguely remember writing about**

> *"I wrote something about the tradeoff between speed and correctness in distributed systems — where is it?"*

```sh
qmd hsearch "speed vs correctness distributed systems tradeoff"
```

**Catch up on recent work**

> *"What did I write about authentication in the last two weeks?"*

```sh
qmd fsearch 'content ~= "authentication" AND modified > 14d'
```

**Surface notes on a concept across different collections**

> *"Find everything I have on cognitive bias, across my book notes and research papers."*

```sh
qmd hsearch "cognitive bias" -c books -c research_papers
```

**Find incomplete or draft material to revisit**

> *"Which notes are tagged as drafts and haven't been touched in over a month?"*

```sh
qmd fsearch 'tag=draft AND modified < 30d'
```

**Pull relevant context for an AI prompt**

> *"Give me all notes related to API design patterns as structured JSON for an LLM."*

```sh
qmd hsearch "API design patterns" --json -n 15
```

**Batch-retrieve a set of documents for review**

> *"Get all my journal entries from May 2025."*

```sh
qmd multi-get "journals/2025-05*.md"
```

---

## Commands

### Collection management

Collections are indexed folders. You can have multiple collections for different areas of your life or work.

```sh
qmd collection add <path> --name <name>   # Register a folder
qmd collection add <path> --name <name> --mask "**/*.md"  # Custom glob
qmd collection list                        # Show all collections with stats
qmd collection remove <name>              # Remove a collection
qmd collection rename <old> <new>         # Rename a collection
qmd ls <name>                             # List files in a collection
qmd ls <name>/subfolder                   # List files under a path
```

### Context

Context is short descriptive text attached to a collection or subfolder. It is returned alongside search results and helps an LLM understand what it's looking at.

```sh
qmd context add qmd://notes "Personal notes and ideas"
qmd context add qmd://notes/work "Work-related notes"
qmd context add qmd://docs/api "API reference documentation"
qmd context add / "My personal knowledge base"   # global — applies to all collections
qmd context list                                  # Show all contexts
qmd context rm qmd://notes/old                   # Remove a context
```

### Indexing and embeddings

```sh
qmd update              # Re-index all collections (scan filesystem for changes)
qmd update --pull       # Git pull first, then re-index
qmd embed               # Generate vector embeddings for semantic search
qmd embed -f            # Force re-embed all documents
qmd embed --chunk-strategy auto  # AST-aware chunking for code files
qmd status              # Show index health, collection stats, pending embeddings
```

### Search commands

#### `hsearch` — Hybrid search *(recommended)*

Full pipeline: query expansion → BM25 + vector retrieval → RRF fusion → LLM reranking. Best quality, requires LLM models.

```sh
qmd hsearch "how does authentication work"
qmd hsearch "error handling best practices" -c docs -n 10
qmd hsearch "API rate limiting" --min-score 0.4 --all --files
qmd hsearch --json --explain "quarterly planning"   # include score traces
qmd hsearch $'lex: auth token\nvec: how users log in'   # structured sub-queries
```

| Option | Description |
|--------|-------------|
| `-n <num>` | Number of results (default: 5, or 20 for `--json`/`--files`) |
| `-c <name>` | Restrict to collection (repeatable) |
| `--all` | Return all matches above threshold |
| `--min-score <num>` | Minimum relevance score (0.0–1.0) |
| `--full` | Include full document body in output |
| `--explain` | Show per-result RRF + rerank score breakdown |
| `--no-rerank` | Skip LLM reranking, use RRF scores only (faster) |
| `--intent <text>` | Domain hint to improve query expansion |
| `--json` / `--csv` / `--md` / `--xml` / `--files` | Output format |

#### `tsearch` — Text search

BM25 full-text keyword search. Fast, no LLM required.

```sh
qmd tsearch "project timeline"
qmd tsearch "authentication" -c docs --json -n 10
qmd tsearch "error handling" --md --full
```

Supports quoted phrases (`"exact match"`), prefix matching, and `-negation`.

| Option | Description |
|--------|-------------|
| `-n <num>` | Number of results |
| `-c <name>` | Restrict to collection |
| `--all` | Return all matches |
| `--min-score <num>` | Minimum score threshold |
| `--json` / `--csv` / `--md` / `--xml` / `--files` | Output format |

#### `vsearch` — Vector (semantic) search

Embedding-only similarity search. Finds documents by meaning, not exact words. No reranking.

```sh
qmd vsearch "how to handle failure gracefully"
qmd vsearch "startup fundraising strategy" --min-score 0.3
```

| Option | Description |
|--------|-------------|
| `-n <num>` | Number of results |
| `-c <name>` | Restrict to collection |
| `--min-score <num>` | Default: 0.3 |

#### `fsearch` — Filter search

DSL-based structured filter. Matches documents by frontmatter fields, tags, headings, dates, or word count — without any LLM.

```sh
qmd fsearch 'tag=productivity AND modified > 30d'
qmd fsearch 'section ~= "Summary" AND missing:status'
qmd fsearch 'collection=research_papers AND word_count > 1000'
qmd fsearch 'NOT (tag=draft) AND created > 2025-01-01'
qmd fsearch 'status = "complete" AND modified > 2025-01-01' --json
```

**DSL syntax:**

| Operator | Example | Meaning |
|----------|---------|---------|
| `=` | `tag=productivity` | Exact match |
| `~=` | `title ~= "budget"` | Contains (case-insensitive) |
| `~/regex/` | `title ~/^Q[0-9]/` | Regex match |
| `>` / `<` | `modified > 30d` | Date/number comparison |
| `AND` / `OR` / `NOT` | `tag=a AND NOT tag=b` | Boolean logic |
| `missing:field` | `missing:status` | Field absent from frontmatter |
| `empty:field` | `empty:summary` | Field present but blank |
| `no:headings` | `no:headings` | Document has no headings |

**Special fields:** `tag`, `section`, `level`, `content`, `title`, `collection`, `modified`, `created`, `word_count`, or any frontmatter key.

**Date values:** ISO dates (`2025-01-01`) or relative durations (`7d`, `2w`, `3m`, `1y`).

| Option | Description |
|--------|-------------|
| `-c <name>` | Restrict to collection |
| `-n <num>` | Max results (default: 50) |
| `--all` | Return all matches |
| `--json` / `--csv` / `--md` / `--files` | Output format |

### Document retrieval

#### `get` — Retrieve a single document

```sh
qmd get "notes/meeting-2025-01-15.md"     # by path
qmd get "#abc123"                          # by docid (from search results)
qmd get "notes/meeting.md:50"             # start at line 50
qmd get "notes/meeting.md" -l 100         # max 100 lines
qmd get "notes/meeting.md" --from 50 -l 100
qmd get "notes/meeting.md" --line-numbers
```

#### `multi-get` — Retrieve multiple documents

```sh
qmd multi-get "journals/2025-05*.md"           # glob pattern
qmd multi-get "doc1.md, doc2.md, #abc123"      # comma-separated list
qmd multi-get "docs/*.md" --max-bytes 20480    # skip files over 20KB
qmd multi-get "docs/*.md" -l 50               # max 50 lines per file
qmd multi-get "docs/*.md" --json              # JSON output for agent processing
```

### MCP server

QMD exposes an MCP (Model Context Protocol) server for tight integration with AI agents and IDEs.

**Tools:**
- `hsearch` — Search with typed sub-queries (`lex`/`vec`/`hyde`), combined via RRF + reranking
- `fsearch` — Filter by frontmatter, tags, dates, or sections using the DSL
- `get` — Retrieve a document by path or docid (with fuzzy match suggestions)
- `multi_get` — Batch retrieve by glob pattern or comma-separated list
- `toc` — Return the heading tree for a document
- `status` — Index health and collection info

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["mcp"]
    }
  }
}
```

**HTTP transport** — for a long-lived shared server that avoids reloading models on every request:

```sh
qmd mcp --http              # start on localhost:8181
qmd mcp --http --port 8080  # custom port
qmd mcp --http --daemon     # background daemon
qmd mcp stop                # stop the daemon
```

---

## Methodology

### Indexing

When you run `qmd collection add`, QMD scans the folder for markdown files and stores their content, title, and metadata in a local SQLite database. Each document gets a short content hash (docid) that stays stable across re-indexes unless the content changes.

When you run `qmd embed`, documents are split into overlapping ~900-token chunks and passed through a local embedding model (embeddinggemma-300M). The resulting vectors are stored in a sqlite-vec index alongside the text index.

**Smart chunking** splits at natural markdown boundaries rather than hard token counts — headings, code fences, and paragraph breaks all get scored, and the algorithm finds the best break point near the 900-token target. For code files, `--chunk-strategy auto` uses tree-sitter to split at function and class boundaries instead.

### Hybrid search (`hsearch`)

`hsearch` runs a full pipeline:

1. **Query expansion** — a fine-tuned local LLM (qmd-query-expansion-1.7B) rewrites your query into typed sub-queries: BM25 keywords (`lex`), a semantic question (`vec`), and a hypothetical answer passage (`hyde`). The original query is kept and weighted 2× so exact matches are preserved.
2. **Parallel retrieval** — each sub-query runs against both the FTS5 BM25 index and the vector index simultaneously.
3. **RRF fusion** — all result lists are merged using Reciprocal Rank Fusion (k=60). Documents that appear at the top of multiple lists get a bonus (+0.05 for #1, +0.02 for #2–3).
4. **LLM reranking** — the top 30 candidates are scored by a cross-encoder reranker (Qwen3-Reranker-0.6B) that rates each document's relevance as a yes/no with logprob confidence.
5. **Position-aware blending** — RRF and reranker scores are blended based on position: top results trust retrieval more (75% RRF), lower results trust the reranker more (60% reranker), preventing the reranker from overriding high-confidence exact matches.

### Keyword search (`tsearch`)

Pure BM25 full-text search via SQLite FTS5. No LLM, no embeddings required. Supports quoted phrases, prefix matching, and negation. Fast and deterministic.

### Semantic search (`vsearch`)

Embedding-only similarity search. Your query is embedded with the same model used at index time, and the nearest vectors are returned by cosine distance. No BM25, no reranking.

### Filter search (`fsearch`)

A structured DSL that queries document metadata directly from SQLite — no LLM, no embeddings. Useful for date-range queries, frontmatter field lookups, tag filtering, and other structural questions where semantic similarity is irrelevant.

### Models

All three models run locally via node-llama-cpp and are downloaded automatically on first use from HuggingFace:

| Model | Purpose | Size |
|-------|---------|------|
| `embeddinggemma-300M-Q8_0` | Vector embeddings | ~300 MB |
| `qwen3-reranker-0.6b-q8_0` | Cross-encoder reranking | ~640 MB |
| `qmd-query-expansion-1.7B-q4_k_m` | Query expansion (fine-tuned) | ~1.1 GB |

Models are cached in `~/.cache/qmd/models/`. You can override the embedding model with `QMD_EMBED_MODEL` for multilingual corpora (e.g. Qwen3-Embedding-0.6B covers 119 languages including CJK).

---

## Privacy and Security

**Everything runs locally. No data ever leaves your device.**

- The SQLite index (`~/.cache/qmd/index.sqlite`) lives entirely on your filesystem.
- All three models run on-device via node-llama-cpp. No API calls, no cloud inference.
- No telemetry, no analytics, no network requests of any kind during search or indexing.
- Your notes, queries, and search results are never transmitted anywhere.

This makes QMD safe to use with sensitive personal notes, confidential work documents, or proprietary research — without needing to trust a third party with your data.

---

## Other Plugins

QMD is part of a family of local-first knowledge management tools for Claude Code:

### [qnode](https://github.com/idanariav/qnode) — Knowledge graph queries

qnode builds a typed, directional graph from your wikilinks and answers structural questions that text search can't: shortest path between two concepts, which notes share a parent topic, which notes are the most central hubs in your knowledge base.

```sh
qnode siblings "epistemology.md"
qnode path "rationalism.md" "empiricism.md"
qnode metrics show --top 20 --sort pagerank
```

### [qimg](https://github.com/idanariav/qimg) — Image search

qimg is the visual equivalent of qmd — on-device hybrid search for your image library using a local SigLIP model. Find images by meaning, description, visual concept, or image-to-image similarity.

```sh
qimg hsearch "person looking contemplative"
qimg vsearch --image ./reference.jpg
```

### [qvoid](https://github.com/idanariav/qvoid) — Unresolved link management

qvoid indexes every wikilink in your vault that points to a note that doesn't exist yet, clusters near-duplicates, and helps you decide which ghost notes to create, merge, or discard.

```sh
qvoid query --destination idea
qvoid find-similar --cluster
```

All four plugins are available together from the Claude Code marketplace:

```sh
claude plugin marketplace add idanariav/pkm-query-tools
```
