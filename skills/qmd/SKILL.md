---
name: qmd
description: Search markdown knowledge bases, notes, and documentation using QMD. Use when users ask to search notes, find documents, or look up information.
license: MIT
compatibility: Requires qmd CLI or MCP server. Install via `npm install -g @tobilu/qmd`.
metadata:
  author: tobi
  version: "1.3.0"
allowed-tools: Bash(qmd:*), mcp__qmd__*
---

# QMD - Quick Markdown Search

Local search engine for markdown content. Indexes notes, docs, and knowledge bases.

## Status

!`qmd status 2>/dev/null || echo "Not installed: npm install -g @tobilu/qmd"`

## MCP Search — `structured_search`

Pass 1-4 sub-queries with type `lex`, `vec`, or `hyde`:

```json
{
  "searches": [
    { "type": "lex", "query": "CAP theorem consistency" },
    { "type": "vec", "query": "tradeoff between consistency and availability" }
  ]
}
```

| Type | Method | What to Write |
|------|--------|---------------|
| `lex` | BM25 keywords | Short phrases — exact terms, names, code |
| `vec` | Vector search | Natural language question |
| `hyde` | Vector search | Hypothetical answer (50-100 words) |

**Tips:**
- Quick lookup → single `lex` query
- Don't know exact terms → use `vec`
- Best results → combine `lex` + `vec` (+ `hyde` for complex topics)
- First query gets 2x weight

## MCP Tools

| Tool | Use |
|------|-----|
| `structured_search` | Search with lex/vec/hyde queries |
| `get` | Retrieve doc by path or `#docid` |
| `multi_get` | Retrieve multiple docs by glob/list |
| `status` | Index health and collections |

## CLI

```bash
qmd search "keywords"           # BM25 keyword search
qmd vsearch "question"          # Vector similarity
qmd query "question"            # Auto-expand + rerank
qmd query $'lex: X\nvec: Y'     # Structured (same as MCP)
qmd get "#abc123"               # Retrieve by docid
```

## Setup

```bash
npm install -g @tobilu/qmd
qmd collection add ~/notes --name notes
qmd embed                       # Generate embeddings
```

MCP config for Claude Code (`~/.claude/settings.json`):
```json
{ "mcpServers": { "qmd": { "command": "qmd", "args": ["mcp"] } } }
```
