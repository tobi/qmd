# QMD MCP Server Setup

## Install

```bash
npm install -g @tobilu/qmd
qmd collection add ~/path/to/markdown --name myknowledge
qmd embed
```

## Configure MCP Client

**Claude Code** (`~/.claude/settings.json`):
```json
{
  "mcpServers": {
    "qmd": { "command": "qmd", "args": ["mcp"] }
  }
}
```

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "qmd": { "command": "qmd", "args": ["mcp"] }
  }
}
```

**OpenClaw** (`~/.openclaw/openclaw.json`):
```json
{
  "mcp": {
    "servers": {
      "qmd": { "command": "qmd", "args": ["mcp"] }
    }
  }
}
```

## HTTP Mode

```bash
qmd mcp --http              # Port 8181
qmd mcp --http --daemon     # Background
qmd mcp stop                # Stop daemon
```

Embedding, reranking, and query-generation resources each unload after five
idle minutes by default. Configure them independently with
`QMD_EMBED_IDLE_TIMEOUT_MINUTES`, `QMD_RERANK_IDLE_TIMEOUT_MINUTES`, and
`QMD_GENERATE_IDLE_TIMEOUT_MINUTES`. `0` keeps that resource group warm; the
maximum is `34560` minutes. For example:

```bash
QMD_EMBED_IDLE_TIMEOUT_MINUTES=0 QMD_RERANK_IDLE_TIMEOUT_MINUTES=10 QMD_GENERATE_IDLE_TIMEOUT_MINUTES=30 qmd mcp --http
```

The `query` tool uses embedding resources for `vec`/`hyde` searches and uses
reranking unless the caller disables it. Its searches are already expanded, so
the MCP tool does not invoke query generation itself; the CLI `qmd query` and a
simple SDK `store.search()` can use all three groups. Each group's activity and
timer are independent. Retrieved Markdown content is index data, not a live GPU
context. Stopping the server explicitly releases all contexts and models,
including groups configured with `0`. Use `qmd doctor` to inspect the effective
values.

## Tools

### query

Search with pre-expanded queries.

```json
{
  "searches": [
    { "type": "lex", "query": "keyword phrases" },
    { "type": "vec", "query": "natural language question" },
    { "type": "hyde", "query": "hypothetical answer passage..." }
  ],
  "limit": 10,
  "collection": "optional",
  "minScore": 0.0
}
```

| Type | Method | Input |
|------|--------|-------|
| `lex` | BM25 | Keywords (2-5 terms) |
| `vec` | Vector | Question |
| `hyde` | Vector | Answer passage (50-100 words) |

### get

Retrieve document by path or `#docid`.

| Param | Type | Description |
|-------|------|-------------|
| `path` | string | File path or `#docid` |
| `full` | bool? | Return full content |
| `lineNumbers` | bool? | Add line numbers |

### multi_get

Retrieve multiple documents.

| Param | Type | Description |
|-------|------|-------------|
| `pattern` | string | Glob or comma-separated list |
| `maxBytes` | number? | Skip large files (default 64KB) |

### status

Index health and collections. No params.

## Troubleshooting

- **Not starting**: `which qmd`, `qmd mcp` manually
- **No results**: `qmd collection list`, `qmd embed`
- **Slow first search**: Normal, models loading (~3GB)
