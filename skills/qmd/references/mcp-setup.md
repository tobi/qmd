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

## Protocol

QMD speaks MCP **2026-07-28** (SDK 2.x) and dual-speaks 2025-era stdio clients.

- **stdio** (`qmd mcp`): hosts still launch this as a subprocess. Opening
  `initialize` (2025) or a `_meta`-enveloped request / `server/discover`
  (2026) pins the connection's era.
- **HTTP** (`qmd mcp --http`): sessionless Streamable HTTP. No
  `Mcp-Session-Id`, no handshake. Each POST is independent. 2026 clients
  MUST send `MCP-Protocol-Version`, `Mcp-Method`, and (for `tools/call`)
  `Mcp-Name`. Version/caps travel in `_meta`. Call `server/discover` to
  learn supported versions and capabilities. `tools/list` is cacheable
  (`ttlMs` / `cacheScope`) and returns tools in a stable order.

## HTTP Mode

```bash
qmd mcp --http              # Port 8181
qmd mcp --http --daemon     # Background
qmd mcp stop                # Stop daemon
```

`POST /mcp` is the MCP endpoint (JSON). `GET /health` is a liveness check.
There is no session GET stream and no idle-session TTL.

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

## Compare a local runbook with upstream docs

Use QMD to find what your team configured. If you also want to verify the
upstream default, ask your MCP client to look it up with a separate web tool.

Try it with a sample runbook in a scratch directory:

```bash
example_dir=$(mktemp -d)
cd "$example_dir"
mkdir runbooks
qmd init
cat > runbooks/sqlite.md <<'EOF'
# SQLite runbook

Our service sets SQLite WAL autocheckpoint to 2000 pages.
EOF
qmd collection add ./runbooks --name runbooks
```

From the same directory, add the local QMD server to Claude Code. The second
command adds Parallel as an optional web connector using anonymous access.

```bash
claude mcp add --scope local --transport stdio qmd-example-local -- qmd mcp
claude mcp add --scope local --transport http qmd-example-web \
  https://mcp.parallel.ai/mcp --header "User-Agent: qmd-mcp-example"
```

Start Claude Code in that directory and ask:

> Use `qmd-example-local` to run a `query` with a single `lex` search for
> "SQLite WAL autocheckpoint" in `runbooks` and set `rerank` to false. Read
> the match with `get` using the returned docid. Then use `qmd-example-web`
> to search for "SQLite WAL default autocheckpoint threshold" and fetch
> the matching page on sqlite.org. Send only the public lookup to the web
> tool. Cite the runbook path and line for our setting and the sqlite.org URL
> for the default.

You should see 2,000 pages from `qmd://runbooks/sqlite.md` (line 3) and
1,000 pages from [SQLite's WAL docs](https://sqlite.org/wal.html#automatic_checkpoint).
If the web lookup is unavailable or you decline it, report the runbook
setting and leave the upstream default unverified. Keep local excerpts,
paths, and internal settings out of web requests.

Remove the optional connector with
`claude mcp remove --scope local qmd-example-web`. When you're done with the
example, remove the local entry with
`claude mcp remove --scope local qmd-example-local`.

## Troubleshooting

- **Not starting**: `which qmd`, `qmd mcp` manually
- **No results**: `qmd collection list`, `qmd embed`
- **Slow first search**: Normal, models loading (~3GB)
