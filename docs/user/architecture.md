# Architecture & Design Decisions

Technical overview of QMD's architecture and design choices.

## Overview

QMD is built with a layered architecture using oclif for CLI, Bun for runtime, SQLite for storage, and Ollama for AI features.

## Core Architecture

### Technology Stack

- **Runtime**: Bun (JavaScript/TypeScript)
- **CLI Framework**: oclif (command structure)
- **Database**: SQLite with FTS5 and sqlite-vec
- **Search**: BM25 (full-text), vector similarity, hybrid with RRF
- **AI**: Ollama (embeddings, reranking)

### Layer Structure

```
Commands (CLI)
    ↓
Services (Business Logic)
    ↓
Repositories (Data Access)
    ↓
Database (Storage)
```

**See:** [`ARCHITECTURE.md`](../dev/ARCHITECTURE.md) for detailed layer information.

## Index Location Priority

### Design Goal

Enable flexible index placement while defaulting to project-local for best UX.

### Priority Cascade

```
1. .qmd/ directory     (project-local, walks up tree)
    ↓ if not found
2. QMD_CACHE_DIR       (environment variable)
    ↓ if not set
3. ~/.cache/qmd/       (global default, XDG compliant)
```

### Implementation

**File:** `src/utils/paths.ts`

```typescript
export function getDbPath(indexName: string = "index"): string {
  let qmdCacheDir: string;

  // Priority 1: Check for .qmd/ directory
  const projectQmdDir = findQmdDir();
  if (projectQmdDir) {
    qmdCacheDir = projectQmdDir;
  }
  // Priority 2: Check QMD_CACHE_DIR env var
  else if (process.env.QMD_CACHE_DIR) {
    qmdCacheDir = resolve(process.env.QMD_CACHE_DIR);
  }
  // Priority 3: Use XDG_CACHE_HOME or ~/.cache/qmd
  else {
    const cacheDir = process.env.XDG_CACHE_HOME || resolve(homedir(), ".cache");
    qmdCacheDir = resolve(cacheDir, "qmd");
  }

  return resolve(qmdCacheDir, `${indexName}.sqlite`);
}

export function findQmdDir(startDir?: string): string | null {
  let dir = startDir || getPwd();
  const root = resolve('/');

  // Walk up directory tree
  while (dir !== root) {
    const qmdDir = resolve(dir, '.qmd');
    if (existsSync(qmdDir)) {
      return qmdDir;
    }
    dir = resolve(dir, '..');
  }

  return null;
}
```

### Rationale

**Why .qmd/ first?**
- Zero-config project setup (like `.git/`)
- Team collaboration (shared config, ignored databases)
- Project isolation (each project has own index)
- Works from subdirectories (walks up tree)

**Why environment variable second?**
- Power user control (custom locations)
- Per-project override (via `.envrc` + direnv)
- CI/CD flexibility (temporary locations)

**Why global default last?**
- Backward compatibility (existing behavior)
- XDG compliance (respects `XDG_CACHE_HOME`)
- Simple fallback (just works without config)

## Database Design

### Schema

**Collections Table:**
```sql
CREATE TABLE collections (
  id INTEGER PRIMARY KEY,
  pwd TEXT NOT NULL,
  glob_pattern TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(pwd, glob_pattern)
);
```

**Documents Table:**
```sql
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  collection_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  title TEXT NOT NULL,
  hash TEXT NOT NULL,
  filepath TEXT NOT NULL UNIQUE,
  display_path TEXT,
  body TEXT NOT NULL,
  created_at TEXT NOT NULL,
  modified_at TEXT NOT NULL,
  active INTEGER DEFAULT 1,
  FOREIGN KEY (collection_id) REFERENCES collections(id)
);
```

**FTS5 Index:**
```sql
CREATE VIRTUAL TABLE documents_fts USING fts5(
  name,
  title,
  body,
  content='documents',
  content_rowid='id'
);
```

**Vector Table (sqlite-vec):**
```sql
CREATE VIRTUAL TABLE vectors_vec USING vec0(
  hash_seq TEXT PRIMARY KEY,
  embedding float[128]
);
```

### Indexing Strategy

**Content Hashing:**
- SHA-256 hash of document body
- Detects changes efficiently
- Deduplicates identical content

**Active Flag:**
- Documents marked `active=0` when deleted
- Preserves history
- Enables soft deletes

**Display Paths:**
- Computed minimal unique paths
- User-friendly (e.g., `docs/api.md` not `/full/path/to/docs/api.md`)
- Collision-resistant

## Search Architecture

### Full-Text Search (BM25)

**Algorithm:** SQLite FTS5 with BM25 ranking

```sql
SELECT * FROM documents_fts
WHERE documents_fts MATCH ?
ORDER BY rank
LIMIT ?
```

**Use Cases:**
- Keyword search
- Exact phrase matching
- Boolean queries

### Vector Search

**Algorithm:** Cosine similarity via sqlite-vec

```sql
SELECT * FROM vectors_vec
WHERE embedding MATCH ?
ORDER BY distance
LIMIT ?
```

**Use Cases:**
- Semantic search
- Concept matching
- Find similar documents

### Hybrid Search (RRF)

**Algorithm:** Reciprocal Rank Fusion + LLM Reranking

**Process:**
1. Run full-text search → results A
2. Run vector search → results B
3. Merge with RRF: `score = 1/(rank_A + k) + 1/(rank_B + k)`
4. Rerank top results with LLM
5. Return final ranked list

**Use Cases:**
- Best quality results
- Complex queries
- Mixed keyword + semantic needs

## Command Architecture

### Command Pattern

All commands follow oclif structure:

```typescript
export default class CommandName extends Command {
  static description = 'Description';
  static args = { /* ... */ };
  static flags = { /* ... */ };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(CommandName);
    // Implementation
  }
}
```

### Shared Services

Commands call shared services for business logic:

```typescript
// indexing.ts
export async function indexFiles(
  db: Database,
  globPattern: string,
  pwd: string
): Promise<Stats> {
  // Indexing logic
}
```

**Benefits:**
- Reusable across commands
- Testable independently
- MCP server can use same logic

## Design Decisions

### Why Bun?

- Fast startup (CLI responsiveness)
- Built-in SQLite support
- Native TypeScript
- Smaller binary than Node.js

### Why SQLite?

- Embedded (no server required)
- Fast for local search
- FTS5 built-in
- sqlite-vec extension for vectors

### Why oclif?

- Professional CLI framework
- Auto-generated help
- Type-safe args/flags
- Plugin architecture

### Why .qmd/ Directory?

**Inspired by:**
- `.git/` (version control)
- `.beads/` (issue tracking)
- `node_modules/` (dependencies)

**Benefits:**
- Familiar pattern
- Discoverable (visible with `ls -la`)
- Team-friendly (shared config)
- Zero-config (auto-detected)

### Why Not MCP Server Migration?

Original plan included MCP server for Claude Desktop integration. **Decision:** Excluded from v2.

**Rationale:**
- Questionable value (CLI sufficient)
- Complexity vs benefit
- Focus on core features
- Can add later if needed

## Performance Considerations

### WAL Mode

SQLite Write-Ahead Logging enabled for:
- Better concurrency
- Faster writes
- Atomic commits

### Prepared Statements

All SQL uses prepared statements for:
- SQL injection prevention
- Query plan caching
- Performance

### Caching

Ollama API results cached for:
- Embeddings reuse
- Faster queries
- Reduced API calls

## Security

### SQL Injection Prevention

**All queries use prepared statements:**

```typescript
// ✓ Safe
db.prepare('SELECT * FROM docs WHERE id = ?').get(id);

// ✗ Unsafe (never do this)
db.query(`SELECT * FROM docs WHERE id = ${id}`);
```

**See:** [`SQL_SAFETY.md`](../SQL_SAFETY.md) for complete guidelines.

### File System Safety

- No arbitrary file writes
- Index database in known locations
- User-provided paths validated

## Extensibility

### Named Indexes

```bash
qmd add . --index work
qmd add . --index personal
```

**Use Cases:**
- Separate work/personal docs
- Different projects
- Testing/production

### Custom Models

```bash
export QMD_EMBED_MODEL=custom-model
export QMD_RERANK_MODEL=custom-reranker
```

**Flexibility:**
- Any Ollama-compatible model
- Model-specific config
- Easy switching

## Future Considerations

### Planned Features

- `qmd doctor --fix` - Auto-fix implementation
- Collection deletion command
- Index compression
- Multi-index search

### Architectural Flexibility

Design allows for:
- REST API layer
- Web UI
- MCP server revival
- Plugin system

## References

- [ARCHITECTURE.md](../dev/ARCHITECTURE.md) - Detailed layer docs
- [SQL_SAFETY.md](../SQL_SAFETY.md) - SQL injection prevention
- [CLAUDE.md](../../CLAUDE.md) - Development guidelines
