# QMD MCP Server Setup

Detailed instructions for configuring QMD as an MCP (Model Context Protocol) server.

## Prerequisites

1. Install qmd globally:
   ```sh
   bun install -g https://github.com/tobi/qmd
   ```

2. Verify installation:
   ```sh
   qmd --help
   ```

3. Set up at least one collection:
   ```sh
   qmd collection add ~/Documents/notes --name notes
   qmd embed  # Generate vector embeddings
   ```

## Claude Code Configuration

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

## Claude Desktop Configuration

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
- `limit` (optional): Number of results (default: 5)
- `minScore` (optional): Minimum relevance score

### qmd_vsearch
Semantic vector search for conceptual similarity.

**Parameters:**
- `query` (required): Search query string
- `collection` (optional): Restrict to specific collection
- `limit` (optional): Number of results (default: 5)
- `minScore` (optional): Minimum relevance score

### qmd_query
Hybrid search combining BM25, vector search, and LLM re-ranking.

**Parameters:**
- `query` (required): Search query string
- `collection` (optional): Restrict to specific collection
- `limit` (optional): Number of results (default: 5)
- `minScore` (optional): Minimum relevance score

### qmd_get
Retrieve a document by path or docid.

**Parameters:**
- `path` (required): Document path or docid (e.g., `#abc123`)
- `full` (optional): Return full content (default: true)
- `lineNumbers` (optional): Include line numbers

### qmd_multi_get
Retrieve multiple documents.

**Parameters:**
- `pattern` (required): Glob pattern or comma-separated list
- `maxBytes` (optional): Skip files larger than this (default: 10KB)

### qmd_status
Get index health and collection information.

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
