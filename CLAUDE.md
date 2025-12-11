# QMD - Quick Markdown Search

Use Bun instead of Node.js (`bun` not `node`, `bun install` not `npm install`).

## Commands

```sh
qmd init               # Initialize .qmd/ directory for project-local index
qmd doctor             # Check system health and diagnose issues
qmd add .              # Index markdown files in current directory
qmd update             # Re-index all collections
qmd update <id>        # Re-index specific collection by ID
qmd status             # Show index status and collections
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
- Ollama for embeddings (nomic-embed-text) and reranking (qwen3-reranker)
- Reciprocal Rank Fusion (RRF) for combining results
- Project-local indexes via `.qmd/` directory (like `.git/`)

## Index Location Priority

QMD searches for indexes in this order:
1. **`.qmd/` directory** - Walks up from current directory (project-local)
2. **`QMD_CACHE_DIR`** - Environment variable for custom location
3. **`~/.cache/qmd/`** - Global default (respects `XDG_CACHE_HOME`)

Example workflows:
```sh
# Project-local index (recommended)
qmd init                    # Creates .qmd/ directory
qmd add .                   # Uses .qmd/default.sqlite

# Custom location via environment variable
export QMD_CACHE_DIR=/custom/path
qmd add .                   # Uses /custom/path/default.sqlite

# Global index (no .qmd/ in tree)
qmd add ~/Documents/notes   # Uses ~/.cache/qmd/default.sqlite
```

## Important: Do NOT run automatically

- Never run `qmd add`, `qmd init`, or `qmd embed` automatically
- Never modify the SQLite database directly
- Write out example commands for the user to run manually
- Index location: `.qmd/` (project-local) or `~/.cache/qmd/` (global default)

## Build & Distribution

### ⚠️ Compilation Does NOT Work
- **Never use `bun build --compile`** - produces a binary that runs but outputs nothing
- Not specifically a sqlite-vec issue - oclif's dynamic imports aren't compatible with Bun's compiler
- See `BUILD.md` for detailed testing results

### ✅ Working Distribution Methods
1. **Install Bun on target machine** (recommended)
   ```sh
   curl -fsSL https://bun.sh/install | bash
   git clone <repo> && cd qmd && bun install
   ```
2. **Docker container** - packages everything including Bun runtime
3. **Package manager** - publish with Bun as peer dependency

### Build Scripts (For Testing Only)
- `bun run build` - Creates compiled binary (doesn't work)
- `bun run build:bundle` - Attempts bundling (doesn't work)
- `./builds/` directory is git-ignored

The shell wrapper approach (`./qmd` → `bin/run`) is the correct solution.