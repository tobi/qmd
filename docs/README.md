# QMD Documentation

Complete documentation for QMD (Quick Markdown Search) features and commands.

## Table of Contents

- [Getting Started](getting-started.md) - Quick start guide
- [Commands](commands.md) - Complete command reference
- [Project Setup](project-setup.md) - Setting up project-local indexes
- [Index Management](index-management.md) - Managing collections and indexes
- [CI/CD Integration](ci-cd.md) - GitHub Actions workflow
- [Architecture](architecture.md) - Index location priority and design

## Quick Reference

### Essential Commands

```bash
# Initialize project
qmd init                    # Create .qmd/ directory
qmd init --with-index       # Init + index files
qmd doctor                  # Check system health

# Indexing
qmd add .                   # Index current directory
qmd update                  # Re-index all collections
qmd update <id>             # Re-index specific collection
qmd embed                   # Generate embeddings

# Searching
qmd search "query"          # Full-text search (BM25)
qmd vsearch "query"         # Vector similarity search
qmd query "query"           # Hybrid search (best quality)

# Information
qmd status                  # Show collections and stats
qmd get <path>              # Get document by path
```

## Core Concepts

### Project-Local Indexes

QMD uses a `.qmd/` directory (like `.git/`) for project-local indexes:

```
myproject/
├── .qmd/
│   ├── default.sqlite      # Index database
│   ├── .gitignore          # Ignores *.sqlite files
│   └── config.json         # Optional config
├── docs/
│   └── readme.md
└── src/
    └── index.ts
```

### Index Location Priority

QMD searches for indexes in this order:

1. **`.qmd/` directory** - Project-local (walks up tree)
2. **`QMD_CACHE_DIR`** - Environment variable override
3. **`~/.cache/qmd/`** - Global default

### Collections

A collection is a set of indexed files from one directory with one glob pattern:

```bash
qmd add .                   # Creates collection: (pwd, **/*.md)
qmd add "docs/**/*.md"      # Creates collection: (pwd, docs/**/*.md)
```

## Features

### ✅ Project Initialization (`qmd init`)
- Zero-config setup for project-local indexes
- Automatic `.gitignore` generation
- Optional configuration file
- Immediate indexing with `--with-index`

### ✅ Health Diagnostics (`qmd doctor`)
- Check project configuration
- Validate dependencies (Bun, sqlite-vec)
- Test services (Ollama)
- Examine index health
- Auto-fix capability

### ✅ Smart Index Location
- Auto-detects `.qmd/` directory
- Works from subdirectories
- Environment variable support
- Global fallback

### ✅ Collection Updates (`qmd update`)
- Re-index all collections
- Update specific collection by ID
- No need to cd into directories
- Detailed statistics

### ✅ CI/CD Integration
- GitHub Actions workflow
- Multi-platform testing
- Code coverage with Codecov
- Type checking and build verification

## Examples

### Single Project Workflow

```bash
# Setup
cd myproject
qmd init --with-index

# Work in subdirectories
cd docs
qmd search "architecture"   # Finds .qmd/ in parent

# Update after changes
git pull
qmd update                  # Refresh index
```

### Multi-Project Workflow

```bash
# Index multiple projects
cd ~/work/project1 && qmd add .
cd ~/work/project2 && qmd add .
cd ~/work/project3 && qmd add .

# View all
qmd status

# Update all at once
qmd update
```

### Environment Variable Override

```bash
# Custom cache location
export QMD_CACHE_DIR=/mnt/ssd/qmd-indexes
qmd add .                   # Uses custom location

# Or with direnv (.envrc)
echo 'export QMD_CACHE_DIR=.qmd' >> .envrc
direnv allow
```

## Next Steps

- Read [Getting Started](getting-started.md) for detailed setup
- See [Commands](commands.md) for complete command reference
- Check [Project Setup](project-setup.md) for best practices
- Review [CI/CD Integration](ci-cd.md) for automated workflows

## Support

- GitHub Issues: https://github.com/ddebowczyk/qmd/issues
- Architecture: See [ARCHITECTURE.md](../ARCHITECTURE.md)
- Claude Guide: See [CLAUDE.md](../CLAUDE.md)
