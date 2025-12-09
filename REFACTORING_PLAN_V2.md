# QMD Refactoring Plan V2 - with oclif CLI Framework

## Major Update: Using oclif for CLI

After initial assessment, we're adopting **oclif** (Open CLI Framework) for proper separation of concerns:

**Why oclif:**
- ✅ Command classes separate controller from business logic
- ✅ Built-in argument/flag parsing
- ✅ Auto-generated help
- ✅ TypeScript-first design
- ✅ Plugin system for extensibility
- ✅ Industry standard (Heroku, Salesforce CLI use it)

## Revised Architecture

```
qmd/
├── src/
│   ├── commands/           # oclif Command classes (thin controllers)
│   │   ├── add.ts         # class AddCommand extends Command
│   │   ├── embed.ts       # class EmbedCommand extends Command
│   │   ├── search.ts      # class SearchCommand extends Command
│   │   ├── vsearch.ts     # class VSearchCommand extends Command
│   │   ├── query.ts       # class QueryCommand extends Command
│   │   ├── get.ts         # class GetCommand extends Command
│   │   ├── status.ts      # class StatusCommand extends Command
│   │   └── mcp.ts         # class McpCommand extends Command
│   │
│   ├── services/           # Business logic (testable, reusable)
│   │   ├── ollama.ts      # Ollama API client
│   │   ├── embedding.ts   # Embedding service
│   │   ├── reranking.ts   # Reranking service
│   │   ├── indexing.ts    # Document indexing service
│   │   ├── search.ts      # Search service (FTS, vector, hybrid)
│   │   └── mcp-server.ts  # MCP server service
│   │
│   ├── database/
│   │   ├── db.ts          # Database connection
│   │   ├── repositories/  # Data access layer
│   │   │   ├── documents.ts    # Document repository
│   │   │   ├── collections.ts  # Collection repository
│   │   │   └── vectors.ts      # Vector repository
│   │   └── queries.ts     # Raw SQL queries with prepared statements
│   │
│   ├── models/
│   │   └── types.ts       # TypeScript interfaces
│   │
│   ├── utils/
│   │   ├── paths.ts
│   │   ├── hash.ts
│   │   └── formatters.ts
│   │
│   └── config/
│       ├── constants.ts
│       └── terminal.ts
│
├── bin/
│   └── run            # oclif entry point
├── qmd.ts             # Thin wrapper to bin/run
└── package.json
```

## oclif Command Structure

### Example: SearchCommand

```typescript
// src/commands/search.ts
import { Command, Flags, Args } from '@oclif/core';
import { SearchService } from '../services/search.js';
import { OutputService } from '../services/output.js';

export default class SearchCommand extends Command {
  static description = 'Full-text search (BM25)';

  static flags = {
    n: Flags.integer({
      description: 'Number of results',
      default: 5,
    }),
    'min-score': Flags.string({
      description: 'Minimum score threshold',
    }),
    full: Flags.boolean({
      description: 'Show full document',
      default: false,
    }),
    json: Flags.boolean({
      description: 'JSON output',
      default: false,
    }),
    // ... other flags
  };

  static args = {
    query: Args.string({
      description: 'Search query',
      required: true,
    }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(SearchCommand);

    // Thin controller - delegates to services
    const searchService = new SearchService();
    const outputService = new OutputService();

    const results = await searchService.fullTextSearch(
      args.query,
      {
        limit: flags.n,
        minScore: parseFloat(flags['min-score'] || '0'),
      }
    );

    outputService.render(results, {
      format: flags.json ? 'json' : 'cli',
      full: flags.full,
    });
  }
}
```

### Example: Service (Business Logic)

```typescript
// src/services/search.ts
import { Database } from 'bun:sqlite';
import { SearchResult, SearchOptions } from '../models/types.js';
import { DocumentRepository } from '../database/repositories/documents.js';

export class SearchService {
  private db: Database;
  private docRepo: DocumentRepository;

  constructor(db?: Database) {
    this.db = db || getDb();
    this.docRepo = new DocumentRepository(this.db);
  }

  /**
   * Perform full-text search using BM25
   * @param query - Search query
   * @param options - Search options (limit, minScore, etc.)
   * @returns Array of search results
   */
  async fullTextSearch(
    query: string,
    options: SearchOptions
  ): Promise<SearchResult[]> {
    // Business logic here - fully testable
    const sanitized = this.sanitizeQuery(query);
    const results = this.docRepo.searchFTS(sanitized, options.limit);

    return results
      .filter(r => r.score >= options.minScore)
      .map(r => this.enrichResult(r));
  }

  /**
   * Perform vector similarity search
   */
  async vectorSearch(
    query: string,
    model: string,
    options: SearchOptions
  ): Promise<SearchResult[]> {
    const embeddingService = new EmbeddingService();
    const queryVector = await embeddingService.embed(query, model, true);

    return this.docRepo.searchVector(queryVector, options.limit);
  }

  /**
   * Hybrid search with RRF fusion
   */
  async hybridSearch(
    query: string,
    embedModel: string,
    rerankModel: string,
    options: SearchOptions
  ): Promise<SearchResult[]> {
    // Combine FTS and vector search
    const [ftsResults, vecResults] = await Promise.all([
      this.fullTextSearch(query, options),
      this.vectorSearch(query, embedModel, options),
    ]);

    // Apply RRF fusion
    const fused = this.fuseResults([ftsResults, vecResults]);

    // Rerank top candidates
    const rerankService = new RerankService();
    return rerankService.rerank(query, fused.slice(0, 30), rerankModel);
  }

  private sanitizeQuery(query: string): string {
    // FTS5 query sanitization logic
  }

  private enrichResult(result: any): SearchResult {
    // Add display paths, snippets, etc.
  }

  private fuseResults(lists: SearchResult[][]): SearchResult[] {
    // RRF fusion algorithm
  }
}
```

## Benefits of This Approach

### 1. Separation of Concerns
- **Commands**: Thin controllers (parse args, call services, format output)
- **Services**: Business logic (testable, reusable, no CLI knowledge)
- **Repositories**: Data access (SQL, prepared statements)

### 2. Testability
```typescript
// Easy to test services in isolation
describe('SearchService', () => {
  test('fullTextSearch returns results', async () => {
    const service = new SearchService(testDb);
    const results = await service.fullTextSearch('query', { limit: 5 });
    expect(results).toHaveLength(5);
  });
});

// Mock-free testing
```

### 3. Reusability
```typescript
// Services can be used by:
// - CLI commands
// - MCP server
// - HTTP API (future)
// - Tests

const searchService = new SearchService();
const results = await searchService.fullTextSearch('query', options);
```

### 4. Auto-generated Help
```bash
$ qmd search --help
Full-text search (BM25)

USAGE
  $ qmd search QUERY

ARGUMENTS
  QUERY  Search query

FLAGS
  -n, --n=<value>          [default: 5] Number of results
  --min-score=<value>      Minimum score threshold
  --full                   Show full document
  --json                   JSON output
```

## Revised Implementation Phases

### Phase 0: Setup oclif (NEW) - 1-2 hours

1. Install oclif: `npm install @oclif/core`
2. Create `bin/run` entry point
3. Set up basic oclif structure
4. Test: `qmd --help` should work

### Phase 1: Extract Types, Utils, Config - 1-2 hours

(Same as before - zero dependencies)

### Phase 2: Extract Database Layer - 2-3 hours

- Create `src/database/db.ts`
- Create `src/database/repositories/` with Document, Collection, Vector repos
- Create `src/database/queries.ts` (prepared statements only)
- **Security focus**: SQL injection tests

### Phase 3: Extract Services - 3-4 hours

- `src/services/ollama.ts` - API client
- `src/services/embedding.ts` - Embedding service
- `src/services/reranking.ts` - Reranking service
- `src/services/indexing.ts` - Document indexing
- `src/services/search.ts` - Search algorithms (FTS, vector, hybrid)
- `src/services/output.ts` - Output formatting

**Key**: Services are CLI-agnostic, pure business logic

### Phase 4: Create oclif Commands - 2-3 hours

- `src/commands/add.ts` - AddCommand
- `src/commands/embed.ts` - EmbedCommand
- `src/commands/search.ts` - SearchCommand
- `src/commands/vsearch.ts` - VSearchCommand
- `src/commands/query.ts` - QueryCommand
- `src/commands/get.ts` - GetCommand
- `src/commands/status.ts` - StatusCommand
- `src/commands/mcp.ts` - McpCommand

**Key**: Commands are thin - just parse, delegate, output

### Phase 5: Update Entry Points - 1 hour

- Update `qmd.ts` to call oclif
- Update `bin/run` to use oclif
- Remove old CLI parsing code

### Phase 6: Write Tests - 3-4 hours

- Unit tests for services (business logic)
- Integration tests for repositories (database)
- Command tests (mocked services)
- E2E tests (full workflows)

## Migration Strategy

### 1. Parallel Implementation

Keep old code working while building new structure:

```typescript
// qmd.ts
if (process.env.USE_OCLIF) {
  // New oclif path
  await runOclif();
} else {
  // Old monolithic path (current)
  // ... existing code
}
```

### 2. Service-by-Service Migration

Extract services first, commands later:

1. Extract SearchService
2. Test SearchService
3. Create SearchCommand using SearchService
4. Switch to new command
5. Remove old code

### 3. Feature Flag per Command

```bash
# Use new search command
USE_OCLIF_SEARCH=1 qmd search "query"

# Use old search command
qmd search "query"
```

## Updated Time Estimates

- **Phase 0** (oclif setup): 1-2 hours
- **Phase 1** (types, utils, config): 1-2 hours
- **Phase 2** (database + repos): 2-3 hours
- **Phase 3** (services): 3-4 hours
- **Phase 4** (oclif commands): 2-3 hours
- **Phase 5** (entry points): 1 hour
- **Phase 6** (tests): 3-4 hours

**Total**: 13-19 hours (focused work)

## Testing Strategy with oclif

### Test Services (No Mocks Needed)

```typescript
describe('SearchService', () => {
  let db: Database;
  let service: SearchService;

  beforeEach(() => {
    db = new Database(':memory:');
    // ... create schema
    service = new SearchService(db);
  });

  test('searches documents', async () => {
    const results = await service.fullTextSearch('query', { limit: 5 });
    expect(results).toBeDefined();
  });
});
```

### Test Commands (Mock Services)

```typescript
import { SearchCommand } from '../src/commands/search';

describe('SearchCommand', () => {
  test('calls search service', async () => {
    const mockService = {
      fullTextSearch: jest.fn(() => Promise.resolve([])),
    };

    const cmd = new SearchCommand(['query'], {});
    await cmd.run();

    expect(mockService.fullTextSearch).toHaveBeenCalledWith('query', expect.any(Object));
  });
});
```

## Next Steps

1. Install oclif: `npm install @oclif/core`
2. Create basic oclif structure
3. Extract first service (SearchService)
4. Create first command (SearchCommand)
5. Test both independently
6. Migrate remaining commands

## Decision: Start with oclif or Continue Current Plan?

**Option A**: Add oclif now (cleaner, but adds setup time)
**Option B**: Continue modularization, add oclif later (faster, but may need refactoring)

**Recommendation**: **Option A** - Add oclif now for proper architecture from the start.
