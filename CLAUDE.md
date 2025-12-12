# QMD - Quick Markdown Search

**Note**: This project uses [bd (beads)](https://github.com/steveyegge/beads) for issue tracking. Use `bd` commands instead of markdown TODOs. See AGENTS.md for workflow details.

Use Bun instead of Node.js (`bun` not `node`, `bun install` not `npm install`).

## Commands

```sh
qmd collection add . --name <n>   # Create/index collection
qmd collection list               # List all collections with details
qmd collection remove <name>      # Remove a collection by name
qmd collection rename <old> <new> # Rename a collection
qmd ls [collection[/path]]        # List collections or files in a collection
qmd context add [path] "text"     # Add context for path (defaults to current dir)
qmd context list                  # List all contexts
qmd context rm <path>             # Remove context
qmd get <file>                    # Get document content (fuzzy matches if not found)
qmd multi-get <pattern>           # Get multiple docs by glob or comma-separated list
qmd status                        # Show index status and collections
qmd update                        # Re-index all collections
qmd embed                         # Generate vector embeddings (requires Ollama)
qmd search <query>                # BM25 full-text search
qmd vsearch <query>               # Vector similarity search
qmd query <query>                 # Hybrid search with reranking (best quality)
```

## Collection Management

```sh
# List all collections
qmd collection list

# Create a collection with explicit name
qmd collection add ~/Documents/notes --name mynotes --mask '**/*.md'

# Remove a collection
qmd collection remove mynotes

# Rename a collection
qmd collection rename mynotes my-notes

# List all files in a collection
qmd ls mynotes

# List files with a path prefix
qmd ls journals/2025
qmd ls qmd://journals/2025
```

## Context Management

```sh
# Add context to current directory (auto-detects collection)
qmd context add "Description of these files"

# Add context to a specific path
qmd context add /subfolder "Description for subfolder"

# Add global context to all collections (system message)
qmd context add / "Always include this context"

# Add context using virtual paths
qmd context add qmd://journals/2024 "Journal entries from 2024"

# List all contexts
qmd context list

# Remove context
qmd context rm qmd://journals/2024
qmd context rm /  # Remove global context
```

## Options

```sh
# Search & retrieval
-c, --collection <name>  # Restrict search to a collection (matches pwd suffix)
-n <num>                 # Number of results
--all                    # Return all matches
--min-score <num>        # Minimum score threshold
--full                   # Show full document content

# Multi-get specific
-l <num>                 # Maximum lines per file
--max-bytes <num>        # Skip files larger than this (default 10KB)

# Output formats (search and multi-get)
--json, --csv, --md, --xml, --files
```

## Development

```sh
bun qmd.ts <command>   # Run from source
bun link               # Install globally as 'qmd'
```

## Architecture

- SQLite FTS5 for full-text search (BM25)
- sqlite-vec for vector similarity search
- Ollama for embeddings (embeddinggemma) and reranking (qwen3-reranker)
- Reciprocal Rank Fusion (RRF) for combining results

## Important: Do NOT run automatically

- Never run `qmd collection add`, `qmd embed`, or `qmd update` automatically
- Never modify the SQLite database directly
- Write out example commands for the user to run manually
- Index is stored at `~/.cache/qmd/index.sqlite`

## Do NOT compile

- Never run `bun build --compile` - it overwrites the shell wrapper and breaks sqlite-vec
- The `qmd` file is a shell script that runs `bun qmd.ts` - do not replace it