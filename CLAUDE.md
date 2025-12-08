# QMD - Quick Markdown Search

Use Bun instead of Node.js (`bun` not `node`, `bun install` not `npm install`).

## Commands

```sh
qmd add .              # Index markdown files in current directory
qmd status             # Show index status and collections
qmd update-all         # Re-index all collections
qmd embed              # Generate vector embeddings (requires Ollama)
qmd search <query>     # BM25 full-text search
qmd vsearch <query>    # Vector similarity search
qmd query <query>      # Hybrid search with reranking (best quality)
```

## Development

```sh
bun qmd.ts <command>   # Run from source
bun link               # Install globally as 'qmd'
```

## Architecture

- SQLite FTS5 for full-text search (BM25)
- sqlite-vec for vector similarity search
- Ollama for embeddings (embeddinggemma) and reranking (qwen3-reranker)
- Reciprocal Rank Fusion (RRF) for combining results
