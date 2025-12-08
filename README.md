# QMD - Quick Markdown Search

A CLI tool for searching markdown knowledge bases using hybrid retrieval: combining BM25 full-text search, vector semantic search, and LLM re-ranking.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QMD Hybrid Search Pipeline                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   User Query    │
                              └────────┬────────┘
                                       │
                        ┌──────────────┴──────────────┐
                        ▼                             ▼
               ┌────────────────┐            ┌────────────────┐
               │ Query Expansion│            │  Original Query│
               │  (qwen3:0.6b)  │            │   (×2 weight)  │
               └───────┬────────┘            └───────┬────────┘
                       │                             │
                       │ 2 alternative queries       │
                       └──────────────┬──────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
     │ Original Query  │     │ Expanded Query 1│     │ Expanded Query 2│
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                       │                       │
      ┌───────┴───────┐       ┌───────┴───────┐       ┌───────┴───────┐
      ▼               ▼       ▼               ▼       ▼               ▼
  ┌───────┐       ┌───────┐ ┌───────┐     ┌───────┐ ┌───────┐     ┌───────┐
  │ BM25  │       │Vector │ │ BM25  │     │Vector │ │ BM25  │     │Vector │
  │(FTS5) │       │Search │ │(FTS5) │     │Search │ │(FTS5) │     │Search │
  └───┬───┘       └───┬───┘ └───┬───┘     └───┬───┘ └───┬───┘     └───┬───┘
      │               │         │             │         │             │
      └───────┬───────┘         └──────┬──────┘         └──────┬──────┘
              │                        │                       │
              └────────────────────────┼───────────────────────┘
                                       │
                                       ▼
                          ┌───────────────────────┐
                          │   RRF Fusion + Bonus  │
                          │  Original query: ×2   │
                          │  Top-rank bonus: +0.05│
                          │     Top 30 Kept       │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │    LLM Re-ranking     │
                          │  (qwen3-reranker)     │
                          │  Yes/No + logprobs    │
                          └───────────┬───────────┘
                                      │
                                      ▼
                          ┌───────────────────────┐
                          │  Position-Aware Blend │
                          │  Top 1-3:  75% RRF    │
                          │  Top 4-10: 60% RRF    │
                          │  Top 11+:  40% RRF    │
                          └───────────────────────┘
```

## Score Normalization & Fusion

### Search Backends

| Backend | Raw Score | Conversion | Range |
|---------|-----------|------------|-------|
| **FTS (BM25)** | SQLite FTS5 BM25 | `Math.abs(score)` | 0 to ~25+ |
| **Vector** | Cosine distance | `1 / (1 + distance)` | 0.0 to 1.0 |
| **Reranker** | LLM 0-10 rating | `score / 10` | 0.0 to 1.0 |

### Fusion Strategy

The `query` command uses **Reciprocal Rank Fusion (RRF)** with position-aware blending:

1. **Query Expansion**: Original query (×2 for weighting) + 1 LLM variation
2. **Parallel Retrieval**: Each query searches both FTS and vector indexes
3. **RRF Fusion**: Combine all result lists using `score = Σ(1/(k+rank+1))` where k=60
4. **Top-Rank Bonus**: Documents ranking #1 in any list get +0.05, #2-3 get +0.02
5. **Top-K Selection**: Take top 30 candidates for reranking
6. **Re-ranking**: LLM scores each document (yes/no with logprobs confidence)
7. **Position-Aware Blending**:
   - RRF rank 1-3: 75% retrieval, 25% reranker (preserves exact matches)
   - RRF rank 4-10: 60% retrieval, 40% reranker
   - RRF rank 11+: 40% retrieval, 60% reranker (trust reranker more)

**Why this approach**: Pure RRF can dilute exact matches when expanded queries don't match. The top-rank bonus preserves documents that score #1 for the original query. Position-aware blending prevents the reranker from destroying high-confidence retrieval results.

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 0.8 - 1.0 | Highly relevant |
| 0.5 - 0.8 | Moderately relevant |
| 0.2 - 0.5 | Somewhat relevant |
| 0.0 - 0.2 | Low relevance |

## Requirements

### System Requirements

- **Bun** >= 1.0.0
- **macOS**: Homebrew SQLite (for extension support)
  ```sh
  brew install sqlite
  ```
- **Ollama** running locally (default: `http://localhost:11434`)

### Ollama Models

QMD uses three models (auto-pulled if missing):

| Model | Purpose | Size |
|-------|---------|------|
| `embeddinggemma` | Vector embeddings | ~1.6GB |
| `ExpedientFalcon/qwen3-reranker:0.6b-q8_0` | Re-ranking (trained) | ~640MB |
| `qwen3:0.6b` | Query expansion | ~400MB |

```sh
# Pre-pull models (optional)
ollama pull embeddinggemma
ollama pull ExpedientFalcon/qwen3-reranker:0.6b-q8_0
ollama pull qwen3:0.6b
```

## Installation

```sh
bun install
```

## Usage

### Index Markdown Files

```sh
# Index all .md files in current directory
qmd add .

# Index with custom glob pattern
qmd add "docs/**/*.md"

# Drop and re-add a collection
qmd add --drop .
```

### Generate Vector Embeddings

```sh
# Embed all indexed documents (chunked into ~6KB pieces)
qmd embed

# Force re-embed everything
qmd embed -f
```

### Add Context

```sh
# Add context description for files in a path
qmd add-context . "Project documentation and guides"
qmd add-context ./meetings "Internal meeting transcripts"
```

### Search Commands

```
┌──────────────────────────────────────────────────────────────────┐
│                        Search Modes                              │
├──────────┬───────────────────────────────────────────────────────┤
│ search   │ BM25 full-text search only                           │
│ vsearch  │ Vector semantic search only                          │
│ query    │ Hybrid: FTS + Vector + Query Expansion + Re-ranking  │
└──────────┴───────────────────────────────────────────────────────┘
```

```sh
# Full-text search (fast, keyword-based)
qmd search "authentication flow"

# Vector search (semantic similarity)
qmd vsearch "how to login"

# Hybrid search with re-ranking (best quality)
qmd query "user authentication"
```

### Options

```sh
-n <num>           # Number of results (default: 5, or 20 for --files/--json)
--min-score <num>  # Minimum score threshold (default: 0)
--full             # Show full document content
--files            # Output: score,filepath,context
--json             # JSON output with snippets
--csv              # CSV output with snippets
--md               # Markdown output
--xml              # XML output
--index <name>     # Use named index
```

### Output Format

Default output is colorized CLI format (respects `NO_COLOR` env):

```
 93%  ~/docs/guide.md:42
  │ This section covers the **craftsmanship** of building
  │ quality software with attention to detail.
  │ See also: engineering principles

 67%  ~/notes/meeting.md:15
  │ Discussion about code quality and craftsmanship
  │ in the development process.
```

- **Score**: Color-coded (green >70%, yellow >40%, dim otherwise)
- **Path**: Relative to $HOME (`~/...`)
- **Line**: Line number where match was found
- **Snippet**: Context around match with query terms highlighted

### Examples

```sh
# Get 10 results with minimum score 0.3
qmd query -n 10 --min-score 0.3 "API design patterns"

# Output as markdown for LLM context
qmd search --md --full "error handling"

# JSON output for scripting
qmd query --json "quarterly reports"

# Use separate index for different knowledge base
qmd --index work search "quarterly reports"
```

### Manage Collections

```sh
# Show index status and collections with contexts
qmd status

# Re-index all collections
qmd update-all

# Get document body by filepath
qmd get ~/notes/meeting.md

# Clean up cache and orphaned data
qmd cleanup
```

## Data Storage

Index stored in: `~/.cache/qmd/index.sqlite`

### Schema

```sql
collections     -- Indexed directories and glob patterns
path_contexts   -- Context descriptions by path prefix
documents       -- Markdown content with metadata
documents_fts   -- FTS5 full-text index
content_vectors -- Embedding chunks (hash, seq, pos)
vectors_vec     -- sqlite-vec vector index (hash_seq key)
ollama_cache    -- Cached API responses
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API endpoint |
| `XDG_CACHE_HOME` | `~/.cache` | Cache directory location |

## How It Works

### Indexing Flow

```
Markdown Files ──► Parse Title ──► Hash Content ──► Store in SQLite
                      │                                    │
                      └──────────► FTS5 Index ◄────────────┘
```

### Embedding Flow

Documents are chunked into ~6KB pieces to fit the embedding model's token window:

```
Document ──► Chunk (~6KB each) ──► Format each chunk ──► Ollama API ──► Store Vectors
                │                    "title | text"        /api/embed
                │
                └─► Chunks stored with:
                    - hash: document hash
                    - seq: chunk sequence (0, 1, 2...)
                    - pos: character position in original
```

### Query Flow (Hybrid)

```
Query ──► LLM Expansion ──► [Original, Variant 1, Variant 2]
                │
      ┌─────────┴─────────┐
      ▼                   ▼
   For each query:     FTS (BM25)
      │                   │
      ▼                   ▼
   Vector Search      Ranked List
      │
      ▼
   Ranked List
      │
      └─────────┬─────────┘
                ▼
         RRF Fusion (k=60)
         Original query ×2 weight
         Top-rank bonus: +0.05/#1, +0.02/#2-3
                │
                ▼
         Top 30 candidates
                │
                ▼
         LLM Re-ranking
         (yes/no + logprob confidence)
                │
                ▼
         Position-Aware Blend
         Rank 1-3:  75% RRF / 25% reranker
         Rank 4-10: 60% RRF / 40% reranker
         Rank 11+:  40% RRF / 60% reranker
                │
                ▼
         Final Results
```

## Model Configuration

Models are configured as constants in `qmd.ts`:

```typescript
const DEFAULT_EMBED_MODEL = "embeddinggemma";
const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
const DEFAULT_QUERY_MODEL = "qwen3:0.6b";
```

### EmbeddingGemma Prompt Format

```
// For queries
"task: search result | query: {query}"

// For documents
"title: {title} | text: {content}"
```

### Qwen3-Reranker

A dedicated reranker model trained on relevance classification:

```
System: Judge whether the Document meets the requirements based on the Query
        and the Instruct provided. Note that the answer can only be "yes" or "no".

User: <Instruct>: Given a search query, determine if the document is relevant...
      <Query>: {query}
      <Document>: {doc}
```

- Uses `logprobs: true` to extract token probabilities
- Outputs yes/no with confidence score (0.0 - 1.0)
- `num_predict: 1` - Only need the yes/no token

### Qwen3 (Query Expansion)

- `num_predict: 150` - For generating query variations

## License

MIT
