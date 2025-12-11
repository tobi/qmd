# Getting Started with QMD

Quick guide to get started with QMD for markdown search.

## Prerequisites

- **Bun** >= 1.0.0 (runtime)
- **Ollama** (optional, for embeddings and reranking)

## Installation

```bash
# Clone repository
git clone https://github.com/ddebowczyk/qmd.git
cd qmd

# Install dependencies
bun install

# Link globally
bun link

# Verify installation
qmd --version
```

## First-Time Setup

### Option 1: Project-Local Index (Recommended)

```bash
# Navigate to your markdown project
cd ~/Documents/my-notes

# Initialize QMD
qmd init --with-index

# Output:
# ✓ Created .qmd/ directory
# ✓ Created .qmd/.gitignore
# Indexing markdown files...
# ✓ Indexed 47 new documents
```

### Option 2: Global Index

```bash
# Index from any directory
cd ~
qmd add ~/Documents/my-notes

# Index is stored in ~/.cache/qmd/
```

## Basic Usage

### Search Your Documents

```bash
# Full-text search (fast)
qmd search "docker containers"

# Vector search (semantic)
qmd vsearch "how to deploy apps"

# Hybrid search (best quality)
qmd query "kubernetes deployment"
```

### Check Status

```bash
qmd status

# Output:
# 📊 Index: default
# 📁 Location: /path/to/project/.qmd/default.sqlite
#
# Collections (1):
#   /path/to/project
#     Pattern: **/*.md
#     Documents: 47
#     Created: 12/9/2025, 6:00:00 PM
#
# Total: 47 documents in 1 collections
```

### Update Index

```bash
# After editing files
qmd update

# Or just the current project
cd project && qmd add .
```

## Setting Up Embeddings (Optional)

Embeddings enable vector search and hybrid search with reranking.

### Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start server
ollama serve
```

### Pull Required Models

```bash
# Embedding model (required for vsearch/query)
ollama pull nomic-embed-text

# Reranking model (optional, for query)
ollama pull qwen3-reranker:0.6b-q8_0
```

### Generate Embeddings

```bash
# After indexing documents
qmd embed

# This may take a while for large collections
```

## Health Check

Verify everything is set up correctly:

```bash
qmd doctor

# Output:
# 🔍 QMD Health Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#
# ✓ Project Configuration
#   ✓ .qmd/ directory found
#   ✓ Index database exists (2.4 MB)
#   ✓ 47 documents indexed
#
# ✓ Dependencies
#   ✓ Bun runtime: v1.3.0
#   ✓ sqlite-vec extension: loaded
#
# ✓ Services
#   ✓ Ollama server: running at http://localhost:11434
#   ✓ 26 Ollama models available
#
# ✓ Index Health
#   ✓ All documents have embeddings
#   ✓ WAL mode: enabled
#   ✓ FTS5 index: created
#
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✓ All checks passed! QMD is ready to use.
```

## Common Workflows

### Daily Use

```bash
# Morning: Update index
qmd update

# Search as needed
qmd search "meeting notes"
qmd query "project architecture"

# Evening: Check what changed
qmd status
```

### Team Collaboration

```bash
# .gitignore (recommended)
.qmd/*.sqlite
.qmd/*.sqlite-shm
.qmd/*.sqlite-wal

# Commit config (optional)
git add .qmd/config.json
git commit -m "Add QMD config"

# Teammates clone and setup
git clone repo
cd repo
qmd init --with-index  # Uses team config
```

### Multiple Projects

```bash
# Each project gets its own index
cd ~/work/project-a && qmd init
cd ~/work/project-b && qmd init
cd ~/work/project-c && qmd init

# Update all at once
qmd update

# Or update specific project
qmd status  # Note the collection ID
qmd update 2  # Update collection #2
```

## Troubleshooting

### Index Not Found

```bash
# Check location
qmd status

# Initialize if needed
qmd init
```

### Ollama Not Running

```bash
# Start Ollama
ollama serve

# Or set custom URL
export OLLAMA_URL=http://custom-host:11434
```

### Embeddings Missing

```bash
qmd doctor

# If warned about missing embeddings:
qmd embed
```

### Out of Date Index

```bash
# Quick fix
qmd update

# Or start fresh
rm -rf .qmd/
qmd init --with-index
```

## Next Steps

- Learn all commands: [Commands Reference](commands.md)
- Set up project properly: [Project Setup](project-setup.md)
- Understand indexes: [Index Management](index-management.md)
- Add to CI/CD: [CI/CD Integration](../dev/ci-cd.md)
