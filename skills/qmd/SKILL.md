---
name: qmd
description: Search personal markdown knowledge bases, notes, meeting transcripts, and documentation using QMD - a local hybrid search engine. Combines BM25 keyword search, vector semantic search, and LLM re-ranking. Use when users ask to search notes, find documents, look up information in their knowledge base, retrieve meeting notes, or search documentation. Triggers on "search markdown files", "search my notes", "find in docs", "look up", "what did I write about", "meeting notes about".
license: MIT
compatibility: Requires qmd CLI or MCP server. Install via `npm install -g @tobilu/qmd`.
metadata:
  author: tobi
  version: "1.2.0"
allowed-tools: Bash(qmd:*), mcp__qmd__*
---

# QMD - Quick Markdown Search

QMD is a local, on-device search engine for markdown content. It indexes your notes, meeting transcripts, documentation, and knowledge bases for fast retrieval.

## QMD Status

!`qmd status 2>/dev/null || echo "Not installed. See installation instructions below."`

## Installation

### Install QMD

```bash
npm install -g @tobilu/qmd
```

### Configure MCP Server

**Claude Code** — add to `~/.claude/settings.json`:
```json
{
  "mcpServers": {
    "qmd": { "command": "qmd", "args": ["mcp"] }
  }
}
```

**Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "qmd": { "command": "qmd", "args": ["mcp"] }
  }
}
```

**OpenClaw** — add to `~/.openclaw/openclaw.json` under `mcp.servers`:
```json
{
  "mcp": {
    "servers": {
      "qmd": { "command": "qmd", "args": ["mcp"] }
    }
  }
}
```

### Index Your Content

```bash
# Add a collection (indexes all markdown files)
qmd collection add ~/Documents/notes --name notes

# Generate embeddings for semantic search
qmd embed

# Check status
qmd status
```

## Search Strategy — Use `structured_search`

**You are a capable LLM.** Use `structured_search` instead of `deep_search` — you generate better query expansions than the local model.

### How structured_search Works

You provide 2-4 sub-searches, each with a type:

| Type | Search Method | What to Write |
|------|---------------|---------------|
| `lex` | BM25 keyword | Short keyword phrases — exact terms, names, identifiers |
| `vec` | Vector similarity | Natural language question — what you're asking |
| `hyde` | Vector similarity | Hypothetical answer — what the result looks like (50-100 words) |

Both `vec` and `hyde` use vector similarity search. The difference is input format:
- **vec**: Write a *question* ("what is X?")
- **hyde**: Write a *hypothetical answer* ("X is a concept that...")

### Example: Finding CAP Theorem Docs

```json
{
  "searches": [
    { "type": "lex", "query": "CAP theorem consistency availability partition" },
    { "type": "vec", "query": "distributed systems tradeoff between data consistency and availability" },
    { "type": "hyde", "query": "The CAP theorem proves that a distributed system cannot simultaneously provide consistency, availability, and partition tolerance. You must choose two." }
  ],
  "limit": 10
}
```

### Guidelines for Query Expansion

1. **lex queries**: 2-5 keyword terms. Include synonyms and related terms.
2. **vec queries**: Full natural language questions. Be specific.
3. **hyde queries**: 50-100 words. Write what the answer *looks like*, not the question.
4. **Order matters**: First search gets 2x weight in fusion.

### When to Use Each Search Type

| Situation | Approach |
|-----------|----------|
| Know exact terms (names, code, acronyms) | Start with `lex` |
| Conceptual search, don't know vocabulary | Lead with `vec` |
| Complex topic, want best recall | Use all three types |
| Quick lookup | Single `lex` query is fine |

## MCP Tools Reference

| Tool | Speed | Use Case |
|------|-------|----------|
| `structured_search` | ~5s | **Recommended** — you provide query expansions |
| `search` | ~30ms | Fast keyword lookup (BM25) |
| `vector_search` | ~2s | Semantic similarity |
| `deep_search` | ~10s | Auto-expands query (uses small local model) |
| `get` | instant | Retrieve doc by path or `#docid` |
| `multi_get` | instant | Retrieve multiple docs |
| `status` | instant | Index health |

## CLI Fallback

If MCP isn't configured, use the CLI:

```bash
# Keyword search
qmd search "your query" -n 10

# Semantic search  
qmd vsearch "your query"

# Hybrid with re-ranking (auto-expands)
qmd query "your query"

# Retrieve document
qmd get "#abc123" --full
```

## Score Interpretation

| Score | Meaning |
|-------|---------|
| 0.8+ | Highly relevant — show to user |
| 0.5-0.8 | Moderately relevant — include if few results |
| 0.2-0.5 | Weak match — only if user wants more |
| <0.2 | Skip |

## Workflow Example

1. **Check collections**: `qmd status` or `status` tool
2. **Search with structured_search**: Generate lex + vec + hyde queries
3. **Review results**: Check scores and snippets
4. **Retrieve full docs**: Use `get` with `#docid` from results
5. **Iterate**: Refine queries based on what you find
