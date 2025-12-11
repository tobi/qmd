# QMD Commands Reference

Complete reference for all QMD commands.

## Table of Contents

- [Project Setup](#project-setup)
- [Indexing](#indexing)
- [Searching](#searching)
- [Information](#information)
- [Maintenance](#maintenance)

---

## Project Setup

### `qmd init`

Initialize `.qmd/` directory for project-local index.

**Usage:**
```bash
qmd init [--with-index] [--force] [--config]
```

**Flags:**
- `--with-index` - Index markdown files after initialization
- `--force` - Overwrite existing `.qmd/` directory
- `--config` - Create `config.json` with default settings

**Examples:**
```bash
# Basic initialization
qmd init

# Initialize and index immediately
qmd init --with-index

# Initialize with config file
qmd init --config

# Force reinitialize
qmd init --force
```

**What It Creates:**
```
.qmd/
├── .gitignore          # Ignores *.sqlite, keeps config
└── config.json         # (if --config used)
```

---

### `qmd doctor`

Check system health and diagnose issues.

**Usage:**
```bash
qmd doctor [--fix] [--verbose] [--json] [--index <name>]
```

**Flags:**
- `--fix` - Attempt to auto-fix common issues
- `--verbose` - Show detailed diagnostic information
- `--json` - Output results as JSON (CI/CD friendly)
- `--index <name>` - Index name to check (default: "default")

**Examples:**
```bash
# Basic health check
qmd doctor

# Detailed diagnostics
qmd doctor --verbose

# Auto-fix issues
qmd doctor --fix

# JSON output for scripts
qmd doctor --json
```

**Checks:**
- ✓ Project Configuration (.qmd/ directory, index exists)
- ✓ Dependencies (Bun runtime, sqlite-vec extension)
- ✓ Services (Ollama server, available models)
- ✓ Index Health (embeddings, WAL mode, FTS5)

---

## Indexing

### `qmd add`

Index markdown files in current directory.

**Usage:**
```bash
qmd add [pattern] [--index <name>]
```

**Arguments:**
- `pattern` - Glob pattern (default: "." which expands to "**/*.md")

**Flags:**
- `--index <name>` - Index name (default: "default")

**Examples:**
```bash
# Index all markdown files
qmd add .

# Index specific directory (ALWAYS QUOTE GLOBS!)
qmd add "docs/**/*.md"

# Custom pattern (QUOTE IT!)
qmd add "src/**/*.md"

# Named index
qmd add . --index work
```

**⚠️ Important: Quote Glob Patterns**

Always quote glob patterns to prevent shell expansion:

```bash
# ✓ Correct
qmd add "**/*.md"
qmd add "docs/**/*.md"

# ✗ Wrong (shell expands before qmd sees it)
qmd add **/*.md        # Error: Unexpected argument
qmd add docs/**/*.md   # Error: Unexpected argument
```

**What Happens Without Quotes:**
When you run `qmd add **/*.md`, your shell expands it to `qmd add file1.md file2.md file3.md`, causing an error.

**Behavior:**
- Creates or updates collection for (pwd, pattern)
- Detects new, updated, removed files
- Shows statistics: indexed, updated, unchanged, removed
- Warns if pattern looks like a file instead of glob

---

### `qmd update`

Re-index one or all collections.

**Usage:**
```bash
qmd update [collection-id] [--all] [--index <name>]
```

**Arguments:**
- `collection-id` - Collection ID to update (optional)

**Flags:**
- `--all` - Update all collections (same as omitting ID)
- `--index <name>` - Index name (default: "default")

**Examples:**
```bash
# Update all collections
qmd update

# Update all (explicit)
qmd update --all

# Update specific collection
qmd update 1

# Update in named index
qmd update --index work
```

**Output:**
```
Updating 2 collection(s)...

Collection 1: /home/user/project1
  Pattern: **/*.md
Indexed: 0 new, 2 updated, 5 unchanged, 1 removed

Collection 2: /home/user/project2
  Pattern: **/*.md
Indexed: 1 new, 0 updated, 3 unchanged, 0 removed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary:
  Collections updated: 2/2
  Documents indexed: 1 new
  Documents updated: 2
  Documents removed: 1
  Documents unchanged: 8
```

---

### `qmd embed`

Generate vector embeddings for indexed documents.

**Usage:**
```bash
qmd embed [--index <name>]
```

**Flags:**
- `--index <name>` - Index name (default: "default")

**Requirements:**
- Ollama server running
- Embedding model installed (e.g., `nomic-embed-text`)

**Examples:**
```bash
# Generate embeddings
qmd embed

# For named index
qmd embed --index work
```

**Note:** Required for `qmd vsearch` and `qmd query` commands.

---

## Searching

### `qmd search`

Full-text search using BM25 (fast, keyword-based).

**Usage:**
```bash
qmd search <query> [--limit <n>] [--index <name>]
```

**Arguments:**
- `query` - Search query (required)

**Flags:**
- `--limit <n>` - Maximum results (default: 10)
- `--index <name>` - Index name (default: "default")

**Examples:**
```bash
# Basic search
qmd search "docker containers"

# More results
qmd search "kubernetes" --limit 20

# Search in named index
qmd search "API" --index work
```

**Best For:**
- Exact keyword matches
- Known terminology
- Fast lookups

---

### `qmd vsearch`

Vector similarity search (semantic understanding).

**Usage:**
```bash
qmd vsearch <query> [--limit <n>] [--index <name>]
```

**Arguments:**
- `query` - Search query (required)

**Flags:**
- `--limit <n>` - Maximum results (default: 10)
- `--index <name>` - Index name (default: "default")

**Requirements:**
- Embeddings generated (`qmd embed`)
- Ollama server running

**Examples:**
```bash
# Semantic search
qmd vsearch "how to deploy applications"

# Find similar concepts
qmd vsearch "error handling patterns"
```

**Best For:**
- Conceptual searches
- Semantic understanding
- Finding similar content

---

### `qmd query`

Hybrid search with RRF fusion and reranking (best quality).

**Usage:**
```bash
qmd query <query> [--limit <n>] [--index <name>]
```

**Arguments:**
- `query` - Search query (required)

**Flags:**
- `--limit <n>` - Maximum results (default: 10)
- `--index <name>` - Index name (default: "default")

**Requirements:**
- Embeddings generated (`qmd embed`)
- Ollama server running
- Reranking model installed (e.g., `qwen3-reranker`)

**Examples:**
```bash
# Best quality search
qmd query "kubernetes deployment strategies"

# Complex queries
qmd query "error handling in microservices"
```

**Process:**
1. Full-text search (BM25)
2. Vector search (semantic)
3. Reciprocal Rank Fusion (RRF)
4. LLM reranking (quality boost)

**Best For:**
- Complex queries
- Best result quality
- Mixed keyword + semantic needs

---

## Information

### `qmd status`

Show index status and collections.

**Usage:**
```bash
qmd status [--index <name>]
```

**Flags:**
- `--index <name>` - Index name (default: "default")

**Examples:**
```bash
# Show status
qmd status

# Named index
qmd status --index work
```

**Output:**
```
📊 Index: default
📁 Location: /project/.qmd/default.sqlite

Collections (2):
  /home/user/project1
    Pattern: **/*.md
    Documents: 47
    Created: 12/9/2025, 6:00:00 PM

  /home/user/project2
    Pattern: docs/**/*.md
    Documents: 23
    Created: 12/9/2025, 7:00:00 PM

Total: 70 documents in 2 collections
```

---

### `qmd get`

Retrieve document content by file path.

**Usage:**
```bash
qmd get <path> [--index <name>]
```

**Arguments:**
- `path` - File path to retrieve (required)

**Flags:**
- `--index <name>` - Index name (default: "default")

**Examples:**
```bash
# Get document
qmd get docs/readme.md

# Get from named index
qmd get architecture.md --index work
```

---

## Maintenance

### Index Management

```bash
# View all collections
qmd status

# Update all collections
qmd update

# Update specific collection
qmd update <id>

# Re-generate embeddings
qmd embed
```

### Named Indexes

```bash
# Create named index
qmd add . --index work

# Search in named index
qmd search "query" --index work

# Status of named index
qmd status --index work
```

### Environment Variables

```bash
# Custom cache directory
export QMD_CACHE_DIR=/custom/path
qmd add .  # Uses /custom/path/default.sqlite

# Custom Ollama URL
export OLLAMA_URL=http://localhost:11434

# Custom embedding model
export QMD_EMBED_MODEL=nomic-embed-text

# Custom reranking model
export QMD_RERANK_MODEL=qwen3-reranker:0.6b-q8_0
```

---

## Command Cheat Sheet

```bash
# Setup
qmd init --with-index      # Initialize + index
qmd doctor                 # Health check

# Indexing
qmd add .                  # Index current dir
qmd update                 # Re-index all
qmd embed                  # Generate embeddings

# Searching
qmd search "query"         # Full-text (fast)
qmd vsearch "query"        # Vector (semantic)
qmd query "query"          # Hybrid (best)

# Info
qmd status                 # Show collections
qmd get path/to/file.md    # Get document
```
