# QMD MCP Server Setup

## Quick Start

1. **Install QMD**
   ```bash
   npm install -g @tobilu/qmd
   ```

2. **Configure your client** (see below)

3. **Index your content**
   ```bash
   qmd collection add ~/path/to/markdown --name myknowledge
   qmd embed  # Generate embeddings for semantic search
   ```

## Client Configuration

### Claude Code

Add to `~/.claude/settings.json`:

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

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

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

### OpenClaw

Add to `~/.openclaw/openclaw.json`:

```json
{
  "mcp": {
    "servers": {
      "qmd": {
        "command": "qmd",
        "args": ["mcp"]
      }
    }
  }
}
```

### HTTP Mode (for remote/multi-client)

```bash
# Start HTTP server (default port 8181)
qmd mcp --http

# Or as a background daemon
qmd mcp --http --daemon

# Stop daemon
qmd mcp stop
```

## MCP Tools

### structured_search ⭐ Recommended

Execute pre-expanded search queries. **Use this** — you're a capable LLM that generates better query expansions than the local model.

```json
{
  "searches": [
    { "type": "lex", "query": "keyword phrases here" },
    { "type": "vec", "query": "natural language question" },
    { "type": "hyde", "query": "A hypothetical answer passage..." }
  ],
  "limit": 10,
  "collection": "optional-filter",
  "minScore": 0.0
}
```

**Search types:**
- `lex` — BM25 keyword search. Short phrases, 2-5 terms.
- `vec` — Vector similarity. Write a natural language *question*.
- `hyde` — Vector similarity. Write a hypothetical *answer* (50-100 words).

Both `vec` and `hyde` use vector search — the difference is what you write.

### search

Fast BM25 keyword search (~30ms).

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query |
| `collection` | string? | Filter by collection |
| `limit` | number? | Max results (default: 5) |
| `minScore` | number? | Min relevance 0-1 |

### vector_search

Semantic similarity search (~2s).

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Natural language query |
| `collection` | string? | Filter by collection |
| `limit` | number? | Max results (default: 5) |
| `minScore` | number? | Min relevance 0-1 |

### deep_search

Hybrid search with automatic query expansion (~10s). Uses a small local model to expand your query. **Prefer `structured_search`** — you generate better expansions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | string | Search query |
| `collection` | string? | Filter by collection |
| `limit` | number? | Max results (default: 5) |
| `minScore` | number? | Min relevance 0-1 |

### get

Retrieve a document by path or docid.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | string | File path or `#docid` |
| `full` | boolean? | Return full content |
| `lineNumbers` | boolean? | Add line numbers |

### multi_get

Retrieve multiple documents by glob or list.

| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | string | Glob pattern or comma-separated paths/docids |
| `maxBytes` | number? | Skip files larger than this (default: 10KB) |

### status

Get index health and collection info. No parameters.

## Troubleshooting

**MCP server not starting**
- Check qmd is in PATH: `which qmd`
- Run manually to see errors: `qmd mcp`
- Verify bun installed: `bun --version`

**No results / empty index**
- Check collections: `qmd collection list`
- Verify status: `qmd status`
- Generate embeddings: `qmd embed`

**Slow first search**
- Normal — models load on first use (~3GB)
- Subsequent searches are fast

**structured_search not found**
- Update QMD: `npm install -g @tobilu/qmd`
- Requires v1.0.8+
