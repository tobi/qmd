---
name: qmd
description: Search personal markdown knowledge bases, notes, meeting transcripts, and documentation using QMD - a local hybrid search engine. Combines BM25 keyword search, vector semantic search, and LLM re-ranking. Use when users ask to search notes, find documents, look up information in their knowledge base, retrieve meeting notes, or search documentation. Triggers on "search my notes", "find in docs", "look up", "what did I write about", "meeting notes about".
license: MIT
compatibility: Requires qmd installed via `bun install -g https://github.com/tobi/qmd`. Works with Claude Code CLI and MCP-compatible agents.
metadata:
  author: tobi
  version: "1.0"
allowed-tools: Bash(qmd:*)
---

# QMD - Quick Markdown Search

QMD is a local, on-device search engine for markdown content. It indexes your notes, meeting transcripts, documentation, and knowledge bases for fast retrieval.

## When to Use This Skill

- User asks to search their notes, documents, or knowledge base
- User needs to find information in their markdown files
- User wants to retrieve specific documents or search across collections
- User asks "what did I write about X" or "find my notes on Y"
- User needs semantic search (conceptual similarity) not just keyword matching
- User mentions meeting notes, transcripts, or documentation lookup

## Search Commands

Choose the right search mode for the task:

| Command | Use When | Speed |
|---------|----------|-------|
| `qmd search` | Exact keyword matches needed | Fast |
| `qmd vsearch` | Keywords aren't working, need conceptual matches | Medium |
| `qmd query` | Best results needed, speed not critical | Slower |

```sh
# Fast keyword search (BM25)
qmd search "your query"

# Semantic vector search (finds conceptually similar content)
qmd vsearch "your query"

# Hybrid search with re-ranking (best quality)
qmd query "your query"
```

## Common Options

```sh
-n <num>                 # Number of results (default: 5)
-c, --collection <name>  # Restrict to specific collection
--all                    # Return all matches
--min-score <num>        # Minimum score threshold (0.0-1.0)
--full                   # Show full document content
--json                   # JSON output for processing
--files                  # List files with scores
--line-numbers           # Add line numbers to output
```

## Document Retrieval

```sh
# Get document by path
qmd get "collection/path/to/doc.md"

# Get document by docid (shown in search results as #abc123)
qmd get "#abc123"

# Get with line numbers for code review
qmd get "docs/api.md" --line-numbers

# Get multiple documents by glob pattern
qmd multi-get "docs/*.md"

# Get multiple documents by list
qmd multi-get "doc1.md, doc2.md, #abc123"
```

## Index Management

```sh
# Check index status and available collections
qmd status

# List all collections
qmd collection list

# List files in a collection
qmd ls <collection-name>

# Update index (re-scan files for changes)
qmd update
```

## Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 0.8 - 1.0 | Highly relevant | Show to user |
| 0.5 - 0.8 | Moderately relevant | Include if few results |
| 0.2 - 0.5 | Somewhat relevant | Only if user wants more |
| 0.0 - 0.2 | Low relevance | Usually skip |

## Recommended Workflow

1. **Check what's available**: `qmd status`
2. **Start with keyword search**: `qmd search "topic" -n 10`
3. **Try semantic if needed**: `qmd vsearch "describe the concept"`
4. **Use hybrid for best results**: `qmd query "question" --min-score 0.4`
5. **Retrieve full documents**: `qmd get "#docid" --full`

## Example: Finding Meeting Notes

```sh
# Search for meetings about a topic
qmd search "quarterly review" -c meetings -n 5

# Get semantic matches
qmd vsearch "performance discussion" -c meetings

# Retrieve the full meeting notes
qmd get "#abc123" --full
```

## Example: Research Across All Notes

```sh
# Hybrid search for best results
qmd query "authentication implementation" --min-score 0.3 --json

# Get all relevant files for deeper analysis
qmd query "auth flow" --all --files --min-score 0.4
```

## MCP Server Integration

QMD also works as an MCP server, providing these tools directly:

| MCP Tool | Equivalent CLI | Purpose |
|----------|---------------|---------|
| `qmd_search` | `qmd search` | Fast BM25 keyword search |
| `qmd_vsearch` | `qmd vsearch` | Semantic vector search |
| `qmd_query` | `qmd query` | Hybrid search with reranking |
| `qmd_get` | `qmd get` | Retrieve document by path or docid |
| `qmd_multi_get` | `qmd multi-get` | Retrieve multiple documents |
| `qmd_status` | `qmd status` | Index health and collection info |

To enable MCP tools, add to `~/.claude/settings.json`:

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

When MCP is configured, prefer using the `qmd_*` tools directly instead of Bash commands for better integration.
