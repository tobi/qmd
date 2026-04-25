# LLM Usage in QMD

QMD uses model inference to improve retrieval quality. It does **not** use an LLM to write final answers. The main output of QMD is ranked search results and retrieved markdown content; any answer synthesis is handled by the calling agent or application.

By default, QMD runs local GGUF models loaded in-process through `node-llama-cpp` and cached under `~/.cache/qmd/models/` (or `$XDG_CACHE_HOME/qmd/models/`). The same embedding, query-expansion, and reranking paths can also run against a remotely deployed OpenAI-compatible LLM service.

## Summary

| Function | Used by | Local default model | Remote behavior | Purpose |
|---|---|---|---|---|
| Document embeddings | `qmd embed`, SDK `store.embed()` | `embeddinggemma-300M-Q8_0` | `POST /embeddings` | Converts indexed document chunks into vectors for semantic search. |
| Query embeddings | `qmd query`, `qmd vsearch`, SDK vector/structured search | `embeddinggemma-300M-Q8_0` | `POST /embeddings` | Converts natural-language, `vec`, and `hyde` queries into vectors for sqlite-vec lookup. |
| Query expansion | Plain `qmd query`, CLI `qmd vsearch`, SDK `store.search({ query })`, SDK `store.expandQuery()` | `qmd-query-expansion-1.7B-q4_k_m` | `POST /chat/completions` | Expands a raw query into typed `lex`, `vec`, and `hyde` sub-queries. |
| HyDE generation | Query expansion | `qmd-query-expansion-1.7B-q4_k_m` | `POST /chat/completions` | Generates a hypothetical answer/document passage for vector retrieval. |
| Reranking | `qmd query`, MCP `query`, SDK `store.search()` by default | `qwen3-reranker-0.6b-q8_0` | rerank endpoint or chat scoring fallback | Scores candidate chunks for relevance after BM25/vector retrieval. |

## Model configuration

Defaults are defined in `src/llm.ts`:

```ts
const DEFAULT_EMBED_MODEL = "hf:ggml-org/embeddinggemma-300M-GGUF/embeddinggemma-300M-Q8_0.gguf";
const DEFAULT_RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const DEFAULT_GENERATE_MODEL = "hf:tobil/qmd-query-expansion-1.7B-gguf/qmd-query-expansion-1.7B-q4_k_m.gguf";
```

The embedding model can be overridden with `QMD_EMBED_MODEL`. Rerank, generation, and backend selection can be overridden through environment variables or the `models:` section in the QMD collections config.

### Remote LLM backend

Set `QMD_LLM_BACKEND=remote` (or define `QMD_REMOTE_BASE_URL`) to use a remotely deployed OpenAI-compatible service instead of loading local GGUF models:

```sh
export QMD_LLM_BACKEND=remote
export QMD_REMOTE_BASE_URL="https://llm.example.com/v1"
export QMD_REMOTE_API_KEY="..."            # optional for trusted/internal endpoints
export QMD_REMOTE_EMBED_MODEL="text-embedding-3-small"
export QMD_REMOTE_GENERATE_MODEL="gpt-4o-mini"
export QMD_REMOTE_RERANK_MODEL="gpt-4o-mini"
```

Remote endpoints used by QMD:

- `POST {baseUrl}/embeddings` with `{ model, input }` for document/query embeddings.
- `POST {baseUrl}/chat/completions` for query expansion and, if needed, chat-based reranking.
- Optional `QMD_REMOTE_RERANK_URL` for native reranker services. The endpoint may be absolute or relative to `QMD_REMOTE_BASE_URL`; QMD sends `{ model, query, documents, top_n, return_documents }` and accepts common `results[{ index, relevance_score|score }]` or `scores[]` responses.

Remote backend configuration can also live in the QMD YAML config. `provider: remote` is explicit; a `models.baseUrl` also selects the remote backend:

```yaml
models:
  provider: remote
  baseUrl: https://llm.example.com/v1
  apiKeyEnv: QMD_REMOTE_API_KEY
  embed: text-embedding-3-small
  generate: gpt-4o-mini
  rerank: gpt-4o-mini
  # Optional native reranker endpoint:
  # rerankUrl: /rerank
```

When the remote backend is selected, `OPENAI_API_KEY` and `OPENAI_BASE_URL` are also recognized. If `OPENAI_API_KEY` is set and no base URL is provided, QMD defaults to `https://api.openai.com/v1`.

Remote embedding model names default to raw text formatting. Local GGUF/HuggingFace model names keep QMD’s historical embeddinggemma format. Override with `QMD_EMBED_FORMAT=raw|nomic|qwen3` if your remote embedding model expects a different prompt format.

## Where LLMs are used

### 1. Embedding documents

Command/API:

```sh
qmd embed
```

```ts
await store.embed()
```

What happens:

1. QMD finds active documents that do not have vectors yet.
2. Each document is chunked.
3. Each chunk is formatted for the active embedding model.
4. The embedding model converts each chunk to a vector, either locally or through the remote `/embeddings` endpoint.
5. Vectors are stored in `content_vectors` and `vectors_vec`.

This enables semantic search. `qmd update` indexes file contents and metadata, but `qmd embed` is the step that calls the embedding model.

### 2. Embedding queries for vector search

Vector search needs a vector for the user query or each semantic sub-query.

Used by:

- `qmd query` when vector embeddings exist
- `qmd vsearch`
- Structured queries containing `vec:` or `hyde:` lines
- SDK `store.searchVector()`
- SDK `store.search({ queries: [...] })` when queries include `vec` or `hyde`

`lex:` queries are routed to BM25 and do not require embeddings. `vec:` and `hyde:` queries are embedded and searched against `vectors_vec`.

### 3. Query expansion

Plain queries are expanded by the configured generation backend: the local generation model by default, or the remote `/chat/completions` endpoint when the remote backend is selected.

Examples:

```sh
qmd query "authentication flow"
qmd vsearch "how do users log in"
```

```ts
await store.search({ query: "authentication flow" })
await store.expandQuery("authentication flow")
```

The expansion model emits typed sub-queries such as:

```text
lex: authentication login flow
vec: how does user authentication work
hyde: The authentication flow validates user credentials, creates a session, and redirects the user after login.
```

Routing:

- `lex` -> BM25 full-text search
- `vec` -> vector semantic search
- `hyde` -> vector semantic search using a hypothetical document passage

A structured query skips this internal expansion step because the caller supplies the typed sub-queries directly:

```sh
qmd query $'lex: "connection pool" timeout\nvec: why database connections time out under load'
```

```ts
await store.search({
  queries: [
    { type: "lex", query: "\"connection pool\" timeout" },
    { type: "vec", query: "why database connections time out under load" },
  ],
})
```

### 4. HyDE

HyDE means “Hypothetical Document Embeddings.” A `hyde:` query is written as a short passage that looks like the answer or relevant document. QMD embeds that passage and searches for documents with similar meaning.

HyDE can be generated automatically during query expansion or provided manually:

```sh
qmd query $'hyde: The API uses a token bucket rate limiter. Requests over the burst capacity return HTTP 429 until tokens refill.'
```

HyDE is useful when the exact vocabulary in the corpus is unknown or when the user query is conceptual.

### 5. Reranking candidates

`qmd query` gathers candidates from BM25 and vector search, combines them with Reciprocal Rank Fusion (RRF), selects the best chunk per candidate, and sends those chunks to the reranker model.

The reranker returns relevance scores. QMD blends those scores with the retrieval/RRF rank so strong exact matches are not discarded too aggressively.

With a remote backend, QMD first uses `QMD_REMOTE_RERANK_URL`/`models.rerankUrl` if configured. Without a native reranker endpoint, QMD falls back to chat-based scoring through `/chat/completions`, asking the remote model to return JSON scores for small batches of candidate chunks.

Reranking is enabled by default for full query search and MCP `query`. Disable it when speed matters more than quality:

```sh
qmd query "authentication flow" --no-rerank
```

```ts
await store.search({ query: "authentication flow", rerank: false })
```

## Where LLMs are not used

These operations do not require an LLM:

- `qmd search` / SDK `store.searchLex()` — BM25 keyword search only.
- `qmd get` and `qmd multi-get` — direct document retrieval.
- `qmd collection ...` — collection metadata management.
- `qmd update` — filesystem scanning and SQLite indexing.
- `qmd ls`, `qmd status`, context management, and cleanup operations.

Some full-query paths may also avoid specific LLM steps:

- Structured queries skip internal query expansion.
- `--no-rerank` / `rerank: false` skips reranking.
- `lex:`-only structured queries do not need query embeddings, unless reranking is enabled afterward.
- If no vector index exists, vector search is skipped.
- A strong BM25 signal can skip query expansion, though later embedding/reranking may still run depending on query options and index state.

## MCP behavior

The MCP `query` tool expects typed sub-queries (`lex`, `vec`, `hyde`). In that path, the calling LLM or agent usually creates the query document, so QMD skips its own query-expansion model and executes the provided searches.

By default, MCP `query` still uses QMD’s configured LLM backend for:

- embeddings for `vec`/`hyde` searches
- reranking, unless `rerank: false` is passed

If `QMD_LLM_BACKEND=remote` or `models.provider: remote` is configured for the MCP server process, MCP searches use the remote embedding/reranking backend too.

## Caching and lifecycle

QMD caches LLM-derived results in the `llm_cache` table:

- query expansion results
- rerank scores

Embeddings are stored separately in `content_vectors` and `vectors_vec`.

Local models are lazy-loaded on first use. Contexts are released after inactivity, while models are kept warm by default to avoid repeated load overhead.

Remote backends do not keep local model state; QMD only maintains the SQLite index/cache and sends HTTP requests on demand.
