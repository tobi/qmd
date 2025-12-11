# Index Management Guide

Complete guide to managing QMD indexes and collections.

## Overview

QMD organizes indexed files into **collections**. Each collection represents one directory with one glob pattern.

## Understanding Collections

### What is a Collection?

A collection stores:
- **pwd**: Working directory path
- **glob_pattern**: File pattern (e.g., `**/*.md`)
- **documents**: Indexed files with metadata

### Example

```bash
# Create collection 1
cd ~/project1
qmd add .
# Collection: (/home/user/project1, **/*.md)

# Create collection 2
cd ~/project2
qmd add "docs/**/*.md"
# Collection: (/home/user/project2, docs/**/*.md)
```

## Viewing Collections

### List All Collections

```bash
qmd status

# Output:
# 📊 Index: default
# 📁 Location: ~/.cache/qmd/default.sqlite
#
# Collections (2):
#   /home/user/project1
#     Pattern: **/*.md
#     Documents: 47
#     Created: 12/9/2025, 6:00:00 PM
#
#   /home/user/project2
#     Pattern: docs/**/*.md
#     Documents: 23
#     Created: 12/9/2025, 7:00:00 PM
#
# Total: 70 documents in 2 collections
```

### Collection IDs

Collections are numbered sequentially (1, 2, 3...). Use IDs for targeted updates.

## Updating Collections

### Update All Collections

Re-index all collections without cd'ing:

```bash
qmd update

# Output:
# Updating 2 collection(s)...
#
# Collection 1: /home/user/project1
#   Pattern: **/*.md
# Indexed: 0 new, 2 updated, 45 unchanged, 0 removed
#
# Collection 2: /home/user/project2
#   Pattern: docs/**/*.md
# Indexed: 1 new, 0 updated, 23 unchanged, 0 removed
#
# Summary:
#   Collections updated: 2/2
#   Documents indexed: 1 new
#   Documents updated: 2
#   ...
```

### Update Specific Collection

```bash
# Get collection ID from status
qmd status

# Update by ID
qmd update 1

# Output:
# Updating 1 collection(s)...
#
# Collection 1: /home/user/project1
#   Pattern: **/*.md
# Indexed: 0 new, 2 updated, 45 unchanged, 0 removed
```

### Update Current Directory

```bash
cd ~/project1
qmd add .

# This updates the collection for current directory
# If collection exists: updates it
# If collection doesn't exist: creates it
```

## Index Operations

### Full Re-index

```bash
# Method 1: Update all collections
qmd update

# Method 2: Remove and recreate
rm -rf .qmd/
qmd init --with-index
```

### Incremental Updates

```bash
# After editing files
qmd add .

# QMD detects:
# - New files (indexed)
# - Modified files (updated)
# - Deleted files (removed)
# - Unchanged files (skipped)
```

### Embeddings

Generate vector embeddings for semantic search:

```bash
# After indexing or updating
qmd embed

# Progress shown:
# Generating embeddings...
# Processed: 47/47 documents
```

## Index Location

### Priority System

QMD finds indexes in this order:

1. **`.qmd/` directory** - Project-local
   ```bash
   ~/myproject/.qmd/default.sqlite
   ```

2. **`QMD_CACHE_DIR`** - Environment variable
   ```bash
   export QMD_CACHE_DIR=/custom/path
   # Uses: /custom/path/default.sqlite
   ```

3. **`~/.cache/qmd/`** - Global default
   ```bash
   ~/.cache/qmd/default.sqlite
   ```

### Check Active Index

```bash
qmd status

# Shows:
# 📁 Location: /path/to/index.sqlite
```

## Named Indexes

Use multiple independent indexes:

```bash
# Work index
qmd add . --index work
qmd search "meeting" --index work

# Personal index
qmd add ~/notes --index personal
qmd search "recipe" --index personal

# Each has separate collections
qmd status --index work
qmd status --index personal
```

## Collection Lifecycle

### Creating Collections

```bash
# Automatic creation
cd ~/project
qmd add .
# Collection created: (~/project, **/*.md)

# With custom pattern
qmd add "docs/**/*.md"
# Collection created: (~/project, docs/**/*.md)
```

### Updating Collections

```bash
# Collections are updated automatically
cd ~/project
qmd add .  # Updates existing collection

# Or from anywhere
qmd update 1  # Update collection ID 1
```

### Removing Collections

Currently no command to remove collections. Workarounds:

```bash
# Method 1: Delete database
rm .qmd/default.sqlite
qmd init --with-index  # Start fresh

# Method 2: Start new index
qmd add . --index new-index
```

## Statistics & Monitoring

### Document Counts

```bash
qmd status

# Shows per collection:
#   Documents: 47
#
# Shows total:
# Total: 70 documents in 2 collections
```

### Index Size

```bash
# Check database file size
ls -lh .qmd/default.sqlite

# Or
du -h .qmd/default.sqlite
```

### Embedding Status

```bash
qmd doctor

# Shows:
# ⚠ 23 documents need embeddings
#   Fix: Run 'qmd embed' to generate embeddings
```

## Performance Optimization

### WAL Mode

QMD uses Write-Ahead Logging for better performance:

```bash
qmd doctor

# Shows:
# ✓ WAL mode: enabled
```

### Vacuum Database

Reclaim space after many updates:

```bash
# Direct SQLite command
sqlite3 .qmd/default.sqlite "VACUUM;"
```

### Batch Operations

```bash
# Update all at once (faster than one-by-one)
qmd update

# Instead of:
# cd project1 && qmd add .
# cd project2 && qmd add .
# cd project3 && qmd add .
```

## Common Patterns

### Multi-Project Management

```bash
# Setup: Index all projects
cd ~/work/project1 && qmd add .
cd ~/work/project2 && qmd add .
cd ~/work/project3 && qmd add .

# Daily: Update all
qmd update

# Specific: Update one project
qmd update 2
```

### Scheduled Updates

```bash
# Cron job: Update all projects nightly
0 2 * * * qmd update

# Or per project
0 2 * * * cd ~/project && qmd add .
```

### CI/CD Integration

```bash
# In GitHub Actions
- name: Update search index
  run: |
    qmd add .
    qmd embed
```

## Troubleshooting

### Index Not Updating

```bash
# Verify files changed
git status

# Force re-index
rm .qmd/default.sqlite
qmd init --with-index
```

### Collection Not Found

```bash
# Check collection exists
qmd status

# Note the ID
# Collections (2):
#   [ID 1] /path/to/project1
#   [ID 2] /path/to/project2

# Update by ID
qmd update 1
```

### Slow Updates

```bash
# Check database size
ls -lh .qmd/default.sqlite

# Large database? Consider:
# 1. Remove old collections (delete & recreate)
# 2. Use named indexes for separation
# 3. Vacuum database
```

### Embeddings Out of Sync

```bash
qmd doctor

# If shows warnings:
# ⚠ 23 documents need embeddings

# Fix:
qmd embed
```

## Advanced Topics

### Database Schema

QMD uses SQLite with:
- **FTS5** - Full-text search index
- **sqlite-vec** - Vector similarity search
- **WAL mode** - Write-ahead logging

### Manual Inspection

```bash
# Open database
sqlite3 .qmd/default.sqlite

# List collections
SELECT * FROM collections;

# List documents
SELECT * FROM documents WHERE active = 1;

# Check FTS index
SELECT * FROM documents_fts WHERE documents_fts MATCH 'docker';
```

### Backup & Restore

```bash
# Backup
cp .qmd/default.sqlite .qmd/default.sqlite.backup

# Restore
cp .qmd/default.sqlite.backup .qmd/default.sqlite

# Or just re-index
qmd init --with-index
```
