# QMD Architecture

## Directory Structure

```
src/
├── commands/           # oclif CLI commands (thin controllers)
│   ├── add.ts
│   ├── embed.ts
│   ├── get.ts
│   ├── query.ts
│   ├── search.ts
│   ├── status.ts
│   └── vsearch.ts
├── services/           # Business logic layer
│   ├── embedding.ts    # Vector embedding & chunking
│   ├── indexing.ts     # Document indexing
│   ├── ollama.ts       # Ollama API client
│   ├── reranking.ts    # LLM-based reranking
│   └── search.ts       # Search algorithms (FTS, vector, hybrid)
├── database/           # Data access layer
│   ├── db.ts           # Connection & schema
│   ├── index.ts
│   └── repositories/
│       ├── collections.ts
│       ├── documents.ts
│       ├── index.ts
│       ├── path-contexts.ts
│       └── vectors.ts
├── models/             # TypeScript types
│   └── types.ts
├── utils/              # Pure utility functions
│   ├── formatters.ts
│   ├── hash.ts
│   └── paths.ts
└── config/             # Configuration
    ├── constants.ts
    └── terminal.ts
```

## Architecture Layers

### Layer 1: Commands (CLI Interface)
**Location**: `src/commands/`
**Responsibility**: Parse arguments, call services, format output
**Dependencies**: Services, Repositories

### Layer 2: Services (Business Logic)
**Location**: `src/services/`
**Responsibility**: Implement algorithms, orchestrate repositories
**Dependencies**: Repositories, Utils

### Layer 3: Repositories (Data Access)
**Location**: `src/database/repositories/`
**Responsibility**: SQL queries with prepared statements
**Dependencies**: Database, Models

### Layer 4: Database (Infrastructure)
**Location**: `src/database/`
**Responsibility**: Schema, migrations, connections
**Dependencies**: None

## Design Principles

1. **Separation of Concerns** - Each layer has a single responsibility
2. **Dependency Rule** - Dependencies point inward (Commands → Services → Repositories → Database)
3. **SQL Injection Safety** - All queries use prepared statements (see `SQL_SAFETY.md`)
4. **Testability** - Pure functions, dependency injection ready
5. **Reusability** - Services work in any context (CLI, API, etc.)

## Command Flow Example

```
User: qmd search "query"
    ↓
StatusCommand.run()
    ↓
fullTextSearch(db, query, limit)  [service]
    ↓
DocumentRepository.searchFTS(query)  [repository]
    ↓
db.prepare("SELECT ... WHERE MATCH ?")  [database]
    ↓
Results → Service → Command → User
```

## Design Changes from Original Plan

### Original Plan (REFACTORING_PLAN.md)
- Separate directories: `cli/`, `indexing/`, `search/`, `output/`, `mcp/`
- Manual CLI parsing in `cli/` directory
- Separate search module

### Final Design (V2 with oclif)
- **oclif commands** - Professional CLI framework with auto-generated help
- **Consolidated services** - `indexing.ts` and `search.ts` in `services/`
- **No separate CLI layer** - oclif handles this
- **No separate output layer** - Commands handle their own output formatting
- **MCP server not migrated** - Questionable value, excluded

### Why the Changes?
1. **oclif** provides better CLI structure than manual parsing
2. **Services** naturally group related business logic
3. **Commands** are thin enough to handle their own output
4. **Simpler** is better - fewer directories, clearer responsibilities

## File Count & Lines of Code

| Layer | Files | Lines | Avg per file |
|-------|-------|-------|--------------|
| Commands | 7 | ~660 | ~94 |
| Services | 5 | ~870 | ~174 |
| Repositories | 4 | ~540 | ~135 |
| Database | 2 | ~260 | ~130 |
| Utils | 3 | ~220 | ~73 |
| Config | 2 | ~50 | ~25 |
| Models | 1 | ~100 | ~100 |
| **Total** | **24** | **~2700** | **~113** |

Compare to original: 1 file × 2538 lines = unmaintainable
