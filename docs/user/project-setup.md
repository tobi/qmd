# Project Setup Guide

Best practices for setting up QMD in your projects.

## Overview

QMD supports project-local indexes using a `.qmd/` directory (similar to `.git/`). This guide covers setup strategies and team collaboration.

## Quick Setup

### Single Command

```bash
cd myproject
qmd init --with-index
```

This creates:
- `.qmd/` directory
- `.qmd/.gitignore` (ignores `*.sqlite` files)
- Indexes all markdown files
- Ready to search immediately

## Directory Structure

### Recommended Layout

```
myproject/
├── .qmd/
│   ├── default.sqlite      # Index database (gitignored)
│   ├── default.sqlite-shm  # SQLite shared memory (gitignored)
│   ├── default.sqlite-wal  # Write-ahead log (gitignored)
│   ├── .gitignore          # Auto-generated
│   └── config.json         # Optional project config
├── docs/
│   └── architecture.md
├── src/
│   └── index.ts
└── README.md
```

### What to Commit

**Commit:**
- `.qmd/.gitignore` - Ensures teammates ignore database files
- `.qmd/config.json` - Shared project configuration (optional)

**Don't Commit:**
- `.qmd/*.sqlite` - Database files (auto-generated)
- `.qmd/*.sqlite-shm` - Temp files
- `.qmd/*.sqlite-wal` - Temp files

## Configuration

### Project Config (.qmd/config.json)

Create with `qmd init --config`:

```json
{
  "embedModel": "nomic-embed-text",
  "rerankModel": "qwen3-reranker:0.6b-q8_0",
  "defaultGlob": "**/*.md",
  "excludeDirs": ["node_modules", ".git", "dist", "build", ".cache"],
  "ollamaUrl": "http://localhost:11434"
}
```

**Benefits:**
- Team shares same settings
- Consistent embedding models
- Custom glob patterns per project

### Environment Variables

For per-developer customization:

```bash
# .envrc (with direnv)
export OLLAMA_URL=http://custom-host:11434
export QMD_EMBED_MODEL=custom-model
```

## Team Collaboration

### Initial Setup (Project Owner)

```bash
# 1. Initialize project
cd myproject
qmd init --config

# 2. Configure .gitignore
cat > .gitignore <<EOF
.qmd/*.sqlite
.qmd/*.sqlite-shm
.qmd/*.sqlite-wal
EOF

# 3. Commit configuration
git add .qmd/config.json .qmd/.gitignore .gitignore
git commit -m "Add QMD configuration"
git push
```

### Teammate Setup

```bash
# 1. Clone repository
git clone repo myproject
cd myproject

# 2. Initialize (uses team config)
qmd init --with-index

# 3. Verify
qmd doctor
qmd status
```

**Zero friction** - teammates just run `qmd init --with-index`.

## Multiple Projects

### Independent Indexes

Each project gets its own isolated index:

```bash
# Project A
cd ~/work/project-a
qmd init
qmd add .

# Project B
cd ~/work/project-b
qmd init
qmd add .

# Search in specific project
cd ~/work/project-a
qmd search "architecture"  # Only searches project-a
```

### Shared Global Index

Alternative: Use global index for multiple projects:

```bash
# Don't use qmd init, just add from home dir
cd ~
qmd add ~/work/project-a
qmd add ~/work/project-b
qmd add ~/work/project-c

# Search all projects
qmd search "docker"  # Searches all 3 projects
```

## Custom Index Locations

### Per-Project Custom Location

```bash
# Use environment variable
echo 'export QMD_CACHE_DIR=.qmd' >> .envrc
direnv allow

qmd add .  # Creates .qmd/default.sqlite
```

### Network Drive

```bash
export QMD_CACHE_DIR=/mnt/network/qmd-indexes
qmd add .
```

## Subdirectory Usage

QMD finds `.qmd/` by walking up the directory tree:

```bash
# Initialize at project root
cd ~/myproject
qmd init

# Use from any subdirectory
cd ~/myproject/docs/api
qmd search "endpoint"       # Finds ~/myproject/.qmd/

cd ~/myproject/src/components
qmd status                  # Finds ~/myproject/.qmd/
```

**Works like Git** - no need to be in root directory.

## Best Practices

### Do's

✅ **Initialize per project** - Use `qmd init` for project-local indexes
✅ **Commit config** - Share `.qmd/config.json` with team
✅ **Gitignore databases** - Never commit `*.sqlite` files
✅ **Use qmd doctor** - Verify setup with health checks
✅ **Update regularly** - Run `qmd update` after major changes

### Don'ts

❌ **Don't commit .sqlite files** - They're large and user-specific
❌ **Don't share indexes** - Each developer generates their own
❌ **Don't nest .qmd/** - One per project root
❌ **Don't mix global + local** - Choose one strategy

## Migration Strategies

### From Global to Project-Local

```bash
# 1. Check current collections
qmd status

# 2. Note the paths
# Collections (2):
#   /home/user/project1
#   /home/user/project2

# 3. Initialize each project
cd /home/user/project1
qmd init --with-index

cd /home/user/project2
qmd init --with-index

# 4. Old global index still works
# New project-local indexes take priority
```

### From Project-Local to Global

```bash
# Remove .qmd/ directories
rm -rf .qmd/

# Add to global index
cd ~
qmd add ~/projects/project1
qmd add ~/projects/project2
```

## Troubleshooting

### .qmd/ Not Found

```bash
# Check current directory
pwd

# Verify .qmd/ exists
ls -la .qmd/

# Initialize if missing
qmd init
```

### Wrong Index Being Used

```bash
# Check which index is active
qmd status

# Shows: 📁 Location: /path/to/index.sqlite

# If wrong location:
# 1. Check for .qmd/ in parent directories
# 2. Check QMD_CACHE_DIR environment variable
# 3. See priority: .qmd/ > QMD_CACHE_DIR > ~/.cache/qmd/
```

### Team Member Can't Find Index

```bash
# Verify .qmd/ is in .gitignore
cat .gitignore | grep .qmd

# Should see:
# .qmd/*.sqlite
# .qmd/*.sqlite-shm
# .qmd/*.sqlite-wal

# Teammate should:
qmd init --with-index
```

## Examples

### Monorepo Setup

```bash
# Option 1: One index for entire monorepo
cd monorepo-root
qmd init
qmd add .  # Indexes all packages

# Option 2: Per-package indexes
cd monorepo-root/packages/api
qmd init && qmd add .

cd monorepo-root/packages/web
qmd init && qmd add .
```

### Documentation Project

```bash
cd docs-site
qmd init --config

# Custom glob for specific files
qmd add "content/**/*.md"

# Exclude drafts
# Edit .qmd/config.json:
# "excludeDirs": ["drafts", "archive"]
```

### Multi-Language Projects

```bash
# Index markdown AND other formats
qmd add "**/*.{md,mdx,txt}"

# Or separate collections
qmd add "**/*.md"
qmd add "**/*.mdx"
```
