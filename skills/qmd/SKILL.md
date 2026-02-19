---
name: qmd
description: Search markdown knowledge bases, notes, and documentation using QMD. Use when users ask to search notes, find documents, or look up information.
license: MIT
compatibility: Requires qmd CLI or MCP server. Install via `npm install -g @tobilu/qmd`.
metadata:
  author: tobi
  version: "2.0.0"
allowed-tools: Bash(qmd:*), mcp__qmd__*
---

# QMD - Quick Markdown Search

Local search engine for markdown content.

## Status

!`qmd status 2>/dev/null || echo "Not installed: npm install -g @tobilu/qmd"`

## MCP: `query`

```json
{
  "searches": [
    { "type": "lex", "query": "CAP theorem consistency" },
    { "type": "vec", "query": "tradeoff between consistency and availability" }
  ],
  "collections": ["docs"],
  "limit": 10
}
```

### Query Types

| Type | Method | Input |
|------|--------|-------|
| `lex` | BM25 | Keywords — exact terms, names, code |
| `vec` | Vector | Question — natural language |
| `hyde` | Vector | Answer — hypothetical result (50-100 words) |
| `expand` | LLM | Auto-expand via local model (max 1 per query) |

### Writing Good Queries

**lex (keyword)**
- 2-5 terms, no filler words
- Exact phrase: `"connection pool"` (quoted)
- Exclude terms: `performance -sports` (minus prefix)
- Code identifiers work: `handleError async`

**vec (semantic)**
- Full natural language question
- Be specific: `"how does the rate limiter handle burst traffic"`
- Include context: `"in the payment service, how are refunds processed"`

**hyde (hypothetical document)**
- Write 50-100 words of what the *answer* looks like
- Use the vocabulary you expect in the result

**expand (auto-expand)**
- Let the local LLM generate lex/vec/hyde variations
- Good when you don't know exact terms
- Max one expand: per query

### Combining Types

| Goal | Approach |
|------|----------|
| Know exact terms | `lex` only |
| Don't know vocabulary | `vec` or `expand` |
| Best recall | `lex` + `vec` |
| Complex topic | `lex` + `vec` + `hyde` |

First query gets 2x weight in fusion — put your best guess first.

### Lex Query Syntax

| Syntax | Meaning | Example |
|--------|---------|---------|
| `term` | Prefix match | `perf` matches "performance" |
| `"phrase"` | Exact phrase | `"rate limiter"` |
| `-term` | Exclude | `performance -sports` |

Note: `-term` only works in lex queries, not vec/hyde.

### Collection Filtering

```json
{ "collections": ["docs"] }              // Single
{ "collections": ["docs", "notes"] }     // Multiple (OR)
```

Omit to search all collections.

## Other MCP Tools

| Tool | Use |
|------|-----|
| `get` | Retrieve doc by path or `#docid` |
| `multi_get` | Retrieve multiple by glob/list |
| `status` | Collections and health |

## CLI

```bash
qmd query "question"              # Auto-expand + rerank
qmd query $'lex: X\nvec: Y'       # Structured
qmd query $'expand: question'     # Explicit expand
qmd search "keywords"             # BM25 only (no LLM)
qmd get "#abc123"                 # By docid
```

## HTTP API

```bash
curl -X POST http://localhost:8181/query \
  -H "Content-Type: application/json" \
  -d '{"searches": [{"type": "lex", "query": "test"}]}'
```

## Setup

```bash
npm install -g @tobilu/qmd
qmd collection add ~/notes --name notes
qmd embed
```
