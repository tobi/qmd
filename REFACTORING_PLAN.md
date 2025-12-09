# QMD TypeScript Refactoring Plan

## Overview

Refactor qmd.ts (2545 lines) from a monolithic file into a modular TypeScript architecture following best practices for maintainability, testability, and scalability.

## Current State

- **Single file**: qmd.ts (~2500 lines)
- **Contains**: Database logic, search functions, embedding operations, CLI parsing, MCP server, utilities
- **Issues**: Hard to maintain, test, and extend

## Goals

1. Separate concerns into logical modules
2. Follow TypeScript best practices
3. Maintain functionality during refactoring
4. Improve testability and maintainability
5. Keep it simple - don't over-engineer

---

## Proposed Directory Structure

```
qmd/
├── src/
│   ├── index.ts                 # Main entry point (CLI router)
│   ├── config/
│   │   ├── constants.ts         # Constants, env vars, defaults
│   │   └── terminal.ts          # Colors, progress, cursor utilities
│   ├── database/
│   │   ├── db.ts                # Database initialization & schema
│   │   ├── migrations.ts        # Schema migrations
│   │   └── cache.ts             # Ollama cache operations
│   ├── models/
│   │   └── types.ts             # TypeScript types & interfaces
│   ├── services/
│   │   ├── ollama.ts            # Ollama API integration
│   │   ├── embedding.ts         # Embedding operations
│   │   └── reranking.ts         # Reranking operations
│   ├── indexing/
│   │   ├── collections.ts       # Collection CRUD operations
│   │   ├── documents.ts         # Document indexing & updates
│   │   └── chunking.ts          # Document chunking logic
│   ├── search/
│   │   ├── fts.ts               # BM25/FTS5 search
│   │   ├── vector.ts            # Vector similarity search
│   │   ├── hybrid.ts            # Hybrid search with RRF
│   │   └── query-expansion.ts   # Query expansion logic
│   ├── output/
│   │   ├── formatters.ts        # All output format implementations
│   │   └── snippets.ts          # Snippet extraction & highlighting
│   ├── commands/
│   │   ├── add.ts               # qmd add
│   │   ├── embed.ts             # qmd embed
│   │   ├── search.ts            # qmd search/vsearch/query
│   │   ├── get.ts               # qmd get
│   │   ├── status.ts            # qmd status
│   │   └── cleanup.ts           # qmd cleanup
│   ├── mcp/
│   │   ├── server.ts            # MCP server setup
│   │   └── tools.ts             # MCP tool definitions
│   ├── cli/
│   │   ├── parser.ts            # CLI argument parsing
│   │   └── help.ts              # Help text
│   └── utils/
│       ├── paths.ts             # Path resolution utilities
│       ├── hash.ts              # Hashing functions
│       └── formatters.ts        # Time, bytes, score formatting
├── qmd.ts                       # Entry point (imports src/index.ts)
└── package.json
```

---

## 10-Phase Implementation Plan

### Phase 1: Setup & Foundation (Low Risk) - 2-3 hours

**Goal**: Create structure and extract zero-dependency modules

**Steps**:
1. Create directory structure: `mkdir -p src/{config,database,models,services,indexing,search,output,commands,mcp,cli,utils}`
2. Extract `src/models/types.ts` - all type/interface declarations
3. Extract `src/utils/` - paths.ts, hash.ts, formatters.ts
4. Extract `src/config/` - constants.ts, terminal.ts
5. Update qmd.ts imports

**Test**: `bun qmd.ts --version`, `bun qmd.ts status`

### Phase 2: Database Layer (Medium Risk) - 1-2 hours

**Goal**: Isolate database operations

**Steps**:
1. Extract `src/database/db.ts` - getDb(), getDbPath(), schema
2. Extract `src/database/cache.ts` - cache operations
3. Extract `src/database/migrations.ts` - migration logic
4. Update imports

**Test**: `qmd status`, `qmd search "test"`

### Phase 3: Services (Medium Risk) - 2-3 hours

**Goal**: Isolate external service integrations

**Steps**:
1. Extract `src/services/ollama.ts` - ensureModelAvailable()
2. Extract `src/services/embedding.ts` - getEmbedding(), formatters
3. Extract `src/services/reranking.ts` - rerank(), rerankSingle()
4. Update imports

**Test**: `qmd embed`, `qmd vsearch "test"`, `qmd query "test"`

### Phase 4: Indexing (Medium-High Risk) - 3-4 hours

**Goal**: Separate document indexing logic

**Steps**:
1. Extract `src/indexing/chunking.ts` - chunkDocument()
2. Extract `src/indexing/collections.ts` - collection management
3. Extract `src/indexing/documents.ts` - indexFiles(), extractTitle(), etc.
4. Update imports

**Test**: `qmd add .`, `qmd add-context`, `qmd update-all`

### Phase 5: Search (Medium-High Risk) - 2-3 hours

**Goal**: Isolate search algorithms

**Steps**:
1. Extract `src/search/fts.ts` - BM25 search
2. Extract `src/search/vector.ts` - vector search
3. Extract `src/search/hybrid.ts` - RRF fusion
4. Extract `src/search/query-expansion.ts` - query expansion
5. Update imports

**Test**: All search commands, verify result ordering matches

### Phase 6: Output (Low-Medium Risk) - 1-2 hours

**Goal**: Separate output formatting

**Steps**:
1. Extract `src/output/snippets.ts` - snippet extraction
2. Extract `src/output/formatters.ts` - all format implementations
3. Update imports

**Test**: Try all formats: --csv, --json, --md, --xml, --files

### Phase 7: Commands (Low Risk) - 2-3 hours

**Goal**: Create command modules

**Steps**:
1. Extract `src/commands/status.ts`, `get.ts`, `cleanup.ts`
2. Extract `src/commands/search.ts` - all search commands
3. Extract `src/commands/add.ts`, `embed.ts`
4. Update imports

**Test**: All CLI commands

### Phase 8: MCP Server (Low Risk) - 1 hour

**Goal**: Isolate MCP server

**Steps**:
1. Extract `src/mcp/server.ts` - startMcpServer()
2. Extract `src/mcp/tools.ts` - tool definitions
3. Update imports

**Test**: `qmd mcp` should start server

### Phase 9: CLI & Main Entry (Low Risk) - 1-2 hours

**Goal**: Create main orchestrator

**Steps**:
1. Extract `src/cli/parser.ts` - parseCLI()
2. Extract `src/cli/help.ts` - showHelp()
3. Create `src/index.ts` - main entry with command router
4. Update qmd.ts to minimal wrapper
5. Update imports

**Test**: All commands should work

### Phase 10: Final Cleanup - 1-2 hours

**Goal**: Polish and document

**Steps**:
1. Simplify qmd.ts to thin wrapper
2. Add barrel exports (index.ts in each dir)
3. Add module-level documentation
4. Create ARCHITECTURE.md explaining design
5. Update README with new structure

**Test**: Full integration test suite

---

## Risk Mitigation Strategy

### Testing After Each Phase
```bash
# Smoke tests
bun qmd.ts --version
bun qmd.ts status
bun qmd.ts search "test query"
bun qmd.ts vsearch "test query"
bun qmd.ts query "test query"
bun qmd.ts add .
bun qmd.ts embed
```

### Git Strategy
- Create branch for each phase
- Commit after each successful step
- Easy rollback if needed

### Backward Compatibility
- All CLI commands work identically
- No syntax or output changes
- Database schema unchanged
- qmd shell wrapper unchanged

---

## Module Responsibilities

### Core Layers

**Config Layer** (`src/config/`)
- Constants, environment variables
- Terminal utilities (colors, progress)
- No dependencies

**Database Layer** (`src/database/`)
- Schema management
- Database initialization
- Cache operations
- Depends on: config

**Models Layer** (`src/models/`)
- TypeScript types and interfaces
- No dependencies (pure declarations)

**Utils Layer** (`src/utils/`)
- Path utilities
- Formatting functions
- Hash functions
- Minimal dependencies

### Service Layer

**Services** (`src/services/`)
- External service integrations (Ollama)
- Embedding generation
- Reranking operations
- Depends on: config, database, models

### Business Logic Layer

**Indexing** (`src/indexing/`)
- Document chunking
- Collection management
- File indexing
- Depends on: database, services, utils

**Search** (`src/search/`)
- FTS5 search
- Vector search
- Hybrid search with RRF
- Query expansion
- Depends on: database, services, models

**Output** (`src/output/`)
- Result formatting (CLI, CSV, JSON, etc.)
- Snippet extraction
- Highlighting
- Depends on: models, utils

### Application Layer

**Commands** (`src/commands/`)
- CLI command implementations
- Orchestrates business logic
- Depends on: indexing, search, output

**MCP** (`src/mcp/`)
- MCP server setup
- Tool definitions
- Depends on: commands

**CLI** (`src/cli/`)
- Argument parsing
- Help text
- Depends on: config

**Main Entry** (`src/index.ts`)
- Command routing
- Top-level orchestration
- Depends on: all commands

---

## Benefits

### Maintainability
- Clear module boundaries
- Easy to locate code
- Better organization

### Testability
- Unit test individual modules
- Mock dependencies easily
- Isolated algorithm testing

### Reusability
- Services can be reused
- Output formatters extensible
- Search algorithms composable

### Scalability
- Easy to add commands
- New output formats simple
- New search strategies pluggable

### Developer Experience
- Clear imports show dependencies
- Easier onboarding
- Better IDE support

---

## Post-Refactoring Opportunities

Once refactored:

1. **Unit tests**: Test algorithms independently
2. **Alternative backends**: Swap database easily
3. **Performance profiling**: Optimize per module
4. **Plugin system**: Custom formatters/strategies
5. **API layer**: Expose as HTTP API
6. **Better documentation**: Generate from types

---

## Time Estimate

- **Development**: 17-26 hours (focused work)
- **Testing**: +8-9 hours (between phases)
- **Total**: ~25-35 hours

---

## Success Criteria

✅ All CLI commands work identically
✅ No breaking changes to user interface
✅ Database schema unchanged
✅ All tests pass
✅ Code is more maintainable
✅ Modules have clear responsibilities
✅ Documentation is complete

---

## Notes

- Keep qmd.ts as entry point (shell wrapper requirement)
- Extract incrementally (one module at a time)
- Test after each extraction
- Commit frequently for easy rollback
- Focus on pragmatic refactoring, not perfection
