# QMD MCP Server Setup

Manual MCP configuration for use without the qmd plugin.

> **Note**: If using the qmd plugin, MCP configuration is included automatically. This is only needed for manual setup.

## Claude Code

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

## Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

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

## Available MCP Tools

Once configured, these tools become available:

### qmd_search
Fast BM25 keyword search.

**Parameters:**
- `query` (required): Search query string
- `collection` (optional): Restrict to specific collection
- `limit` (optional): Number of results (default: 10)
- `minScore` (optional): Minimum relevance score

### qmd_vsearch
Semantic vector search for conceptual similarity.

**Parameters:**
- `query` (required): Search query string
- `collection` (optional): Restrict to specific collection
- `limit` (optional): Number of results (default: 10)
- `minScore` (optional): Minimum relevance score

### qmd_query
Hybrid search combining BM25, vector search, and LLM re-ranking.

**Parameters:**
- `query` (required): Search query string
- `collection` (optional): Restrict to specific collection
- `limit` (optional): Number of results (default: 10)
- `minScore` (optional): Minimum relevance score

### qmd_get
Retrieve a document by path or docid.

**Parameters:**
- `file` (required): Document path or docid (e.g., `#abc123`)
- `fromLine` (optional): Start from this line number (1-indexed)
- `maxLines` (optional): Maximum number of lines to return
- `lineNumbers` (optional): Include line numbers

### qmd_multi_get
Retrieve multiple documents.

**Parameters:**
- `pattern` (required): Glob pattern or comma-separated list
- `maxBytes` (optional): Skip files larger than this (default: 10KB)

### qmd_status
Get index health and collection information.

**Parameters:** None

### qmd_collection_add
Create a new collection and index files from a directory.

**Parameters:**
- `path` (required): Absolute path to a directory to index
- `name` (optional): Collection name (default: directory basename)
- `pattern` (optional): Glob pattern (default: `**/*.md`)

### qmd_collection_list
List all configured collections with document counts and metadata.

**Parameters:** None

### qmd_context_add
Add context text for a global path (`/`) or virtual path (`qmd://collection/path`).

**Parameters:**
- `path` (required): `/` for global context, or `qmd://collection/path`
- `context` (required): Context description

### qmd_context_list
List all configured context descriptions.

**Parameters:** None

## Troubleshooting

### MCP server not starting
- Ensure qmd is in your PATH: `which qmd`
- Try running `qmd mcp` manually to see errors
- Check that Bun is installed: `bun --version`

### No results returned
- Verify collections exist: `qmd collection list`
- Check index status: `qmd status`
- Ensure embeddings are generated: `qmd embed`

### Slow searches
- For faster results, use `qmd_search` instead of `qmd_query`
- The first search may be slow while models load (~3GB)
- Subsequent searches are much faster

## Choosing Between CLI and MCP

| Scenario | Recommendation |
|----------|---------------|
| MCP configured | Use `qmd_*` tools directly |
| No MCP | Use Bash with `qmd` commands |
| Complex pipelines | Bash may be more flexible |
| Simple lookups | MCP tools are cleaner |
