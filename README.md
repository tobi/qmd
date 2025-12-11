# QMD Documentation

Complete documentation for QMD (Quick Markdown Search) features and commands.

## Table of Contents

- [Installation](#installation) - How to install QMD
- [Configuration](#configuration) - Unified config system (CLI > Env > File > Defaults)
- [Getting Started](docs/user/getting-started.md) - Quick start guide
- [Commands](docs/user/commands.md) - Complete command reference
- [Project Setup](docs/user/project-setup.md) - Setting up project-local indexes
- [Index Management](docs/user/index-management.md) - Managing collections and indexes
- [Architecture](docs/user/architecture.md) - Index location priority and design

## Installation

### Prerequisites

QMD requires [Bun](https://bun.sh) runtime:

```bash
# Install Bun (if not already installed)
curl -fsSL https://bun.sh/install | bash
```

### Install QMD

#### Option 1: Install Globally from Source

```bash
# Clone the repository
git clone https://github.com/ddebowczyk/qmd.git
cd qmd

# Install dependencies
bun install

# Link globally (creates 'qmd' command)
bun link

# Verify installation
qmd doctor
```

#### Option 2: Run from Source

```bash
# Clone and install dependencies
git clone https://github.com/ddebowczyk/qmd.git
cd qmd
bun install

# Run directly
bun qmd.ts init
bun qmd.ts add .
bun qmd.ts search "query"
```

### Optional: Ollama for Embeddings

For vector search (`qmd embed`, `qmd vsearch`, `qmd query`), install [Ollama](https://ollama.ai):

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull nomic-embed-text    # For embeddings
ollama pull qwen3-reranker      # For reranking (hybrid search)
```

### Verify Installation

```bash
# Check system health
qmd doctor

# Initialize a project
qmd init

# Run your first search
qmd add .
qmd search "markdown"
```

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

## Configuration

QMD uses a unified configuration system with clear precedence:

**Priority:** CLI flags > Environment variables > Config file > Defaults

### Config File (`.qmd/config.json`)

Create a config file for project-specific settings:

```bash
# Create with defaults
qmd init --config

# Or create manually
cat > .qmd/config.json <<'EOF'
{
  "embedModel": "nomic-embed-text",
  "rerankModel": "qwen3-reranker:0.6b-q8_0",
  "defaultGlob": "**/*.md",
  "excludeDirs": ["node_modules", ".git", "dist", "build", ".cache"],
  "ollamaUrl": "http://localhost:11434"
}
EOF
```

**Commit this file** to share settings with your team.

### Environment Variables

Quick overrides for machine-specific settings:

```bash
# Model configuration
export QMD_EMBED_MODEL=custom-model
export QMD_RERANK_MODEL=custom-reranker
export OLLAMA_URL=http://remote:11434

# Infrastructure
export QMD_CACHE_DIR=/custom/cache  # Cache location override

# Standard
export NO_COLOR=1  # Disable terminal colors
```

### CLI Flags

Override any setting at runtime:

```bash
qmd embed --embed-model custom-model
qmd vsearch "query" --embed-model nomic-embed-text
qmd query "query" --rerank-model qwen3-reranker
```

### Example: Team Configuration

```bash
# 1. Create project config (commit to git)
qmd init --config

# 2. Team members clone repo (config is shared)
git clone your-repo
cd your-repo

# 3. Override locally if needed
export QMD_EMBED_MODEL=faster-model  # Personal preference
export OLLAMA_URL=http://localhost:11434  # Local Ollama

# 4. Or override per-command
qmd embed --embed-model production-model
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

- Read [Getting Started](docs/user/getting-started.md) for detailed setup
- See [Commands](docs/user/commands.md) for complete command reference
- Check [Project Setup](docs/user/project-setup.md) for best practices

## Support

- GitHub Issues: https://github.com/ddebowczyk/qmd/issues
- Architecture: See [ARCHITECTURE.md](docs/dev/ARCHITECTURE.md)
- Claude Guide: See [CLAUDE.md](CLAUDE.md)
