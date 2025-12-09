# QMD Testing Strategy

## Framework: Bun Test

Using Bun's built-in test runner for maximum speed and zero dependencies.

## Test Structure

```
qmd/
├── src/
│   ├── models/
│   │   ├── types.ts
│   │   └── types.test.ts       # Type guards, validators
│   ├── utils/
│   │   ├── paths.ts
│   │   ├── paths.test.ts       # Pure functions, easy to test
│   │   ├── hash.ts
│   │   ├── hash.test.ts
│   │   └── formatters.test.ts
│   ├── database/
│   │   ├── db.ts
│   │   ├── db.test.ts          # Schema, initialization
│   │   ├── queries.ts
│   │   └── queries.test.ts     # SQL injection tests
│   ├── search/
│   │   ├── fts.test.ts         # BM25 search tests
│   │   ├── vector.test.ts      # Vector search tests
│   │   └── hybrid.test.ts      # RRF fusion tests
│   └── ...
└── tests/
    ├── integration/
    │   ├── indexing.test.ts    # Full indexing flow
    │   ├── search.test.ts      # End-to-end search
    │   └── mcp.test.ts         # MCP server tests
    └── fixtures/
        ├── sample.md           # Test documents
        └── test-db.sqlite      # Test database
```

## Test Types

### 1. Unit Tests (Fast, Isolated)

Test individual functions in isolation:

```typescript
// src/utils/formatters.test.ts
import { describe, test, expect } from "bun:test";
import { formatBytes, formatScore } from "./formatters";

describe("formatBytes", () => {
  test("formats bytes correctly", () => {
    expect(formatBytes(0)).toBe("0.0 B");
    expect(formatBytes(1024)).toBe("1.0 KB");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(1048576)).toBe("1.0 MB");
  });
});

describe("formatScore", () => {
  test("formats scores as percentages", () => {
    expect(formatScore(1.0)).toBe("100%");
    expect(formatScore(0.856)).toBe("86%");
    expect(formatScore(0.1)).toBe("10%");
  });
});
```

### 2. Integration Tests (Database, Services)

Test modules working together:

```typescript
// src/database/db.test.ts
import { describe, test, expect, beforeEach, afterEach } from "bun:test";
import { Database } from "bun:sqlite";
import { getDb, ensureVecTable } from "./db";

describe("Database", () => {
  let testDb: Database;

  beforeEach(() => {
    // Use in-memory database for tests
    testDb = new Database(":memory:");
  });

  afterEach(() => {
    testDb.close();
  });

  test("creates schema correctly", () => {
    const tables = testDb.prepare(
      "SELECT name FROM sqlite_master WHERE type='table'"
    ).all();

    expect(tables).toContainEqual({ name: "documents" });
    expect(tables).toContainEqual({ name: "collections" });
  });
});
```

### 3. SQL Injection Tests (Security)

Critical security tests:

```typescript
// src/database/queries.test.ts
import { describe, test, expect } from "bun:test";
import { searchDocuments } from "./queries";

describe("SQL Injection Prevention", () => {
  test("handles malicious input safely", () => {
    const maliciousInputs = [
      "'; DROP TABLE documents; --",
      "' OR 1=1 --",
      "admin'--",
      "' UNION SELECT * FROM users --",
    ];

    for (const input of maliciousInputs) {
      // Should not throw, should return empty or safe results
      expect(() => {
        searchDocuments(testDb, { hash: input });
      }).not.toThrow();
    }
  });

  test("uses prepared statements", () => {
    // Verify queries use ? placeholders
    const result = searchDocuments(testDb, {
      hash: "test'; DROP TABLE documents; --"
    });

    // Should return no results, not execute DROP
    expect(result).toEqual([]);

    // Verify table still exists
    const tables = testDb.prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
    ).all();
    expect(tables).toHaveLength(1);
  });
});
```

### 4. Search Algorithm Tests (Correctness)

Test search quality:

```typescript
// src/search/hybrid.test.ts
import { describe, test, expect } from "bun:test";
import { reciprocalRankFusion } from "./hybrid";

describe("Reciprocal Rank Fusion", () => {
  test("combines multiple result lists correctly", () => {
    const list1 = [
      { file: "a.md", score: 0.9 },
      { file: "b.md", score: 0.8 },
    ];
    const list2 = [
      { file: "b.md", score: 0.95 },
      { file: "c.md", score: 0.7 },
    ];

    const fused = reciprocalRankFusion([list1, list2]);

    // b.md should rank highest (appears in both lists)
    expect(fused[0].file).toBe("b.md");
  });

  test("handles weighted lists", () => {
    const original = [{ file: "a.md", score: 0.9 }];
    const expanded = [{ file: "b.md", score: 0.85 }];

    // Original query weighted 2x
    const fused = reciprocalRankFusion([original, expanded], [2.0, 1.0]);

    expect(fused[0].file).toBe("a.md");
  });
});
```

### 5. End-to-End Tests (Full Flow)

Test complete user workflows:

```typescript
// tests/integration/search.test.ts
import { describe, test, expect, beforeAll } from "bun:test";
import { $ } from "bun";

describe("Search Integration", () => {
  beforeAll(async () => {
    // Set up test database
    await $`bun qmd.ts --index test add tests/fixtures/*.md`;
  });

  test("search command returns results", async () => {
    const result = await $`bun qmd.ts --index test search "test query" --json`.text();
    const json = JSON.parse(result);

    expect(json).toHaveProperty("results");
    expect(Array.isArray(json.results)).toBe(true);
  });

  test("vsearch requires embeddings", async () => {
    try {
      await $`bun qmd.ts --index test vsearch "test query"`.text();
    } catch (error) {
      expect(error.message).toContain("need embedding");
    }
  });
});
```

## Running Tests

### Basic Commands

```bash
# Run all tests
bun test

# Run specific file
bun test src/utils/formatters.test.ts

# Watch mode (auto-rerun on changes)
bun test --watch

# Coverage report
bun test --coverage

# Bail on first failure
bun test --bail

# Run tests matching pattern
bun test --test-name-pattern "formatBytes"
```

### Test Scripts

Add to `package.json`:

```json
{
  "scripts": {
    "test": "bun test",
    "test:watch": "bun test --watch",
    "test:coverage": "bun test --coverage",
    "test:unit": "bun test src/**/*.test.ts",
    "test:integration": "bun test tests/integration/**/*.test.ts"
  }
}
```

## Testing Best Practices

### 1. Arrange-Act-Assert Pattern

```typescript
test("example test", () => {
  // Arrange - Set up test data
  const input = "test";

  // Act - Execute function
  const result = myFunction(input);

  // Assert - Verify result
  expect(result).toBe("expected");
});
```

### 2. Descriptive Test Names

```typescript
// ❌ Bad
test("works", () => { ... });

// ✅ Good
test("returns empty array when no results match", () => { ... });
```

### 3. Test Edge Cases

```typescript
describe("chunkDocument", () => {
  test("handles empty document", () => { ... });
  test("handles single character", () => { ... });
  test("handles document smaller than chunk size", () => { ... });
  test("handles document at exact chunk boundary", () => { ... });
  test("handles very large document", () => { ... });
  test("handles unicode characters correctly", () => { ... });
});
```

### 4. Use Test Fixtures

```typescript
// tests/fixtures/documents.ts
export const sampleDocs = {
  simple: "# Title\n\nContent here.",
  withCode: "# Code Example\n\n```js\ncode\n```",
  large: "x".repeat(10000),
};
```

### 5. Mock External Dependencies

```typescript
import { mock } from "bun:test";

test("getEmbedding calls Ollama API", async () => {
  const fetchMock = mock((url, options) => {
    return Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ embeddings: [[0.1, 0.2, 0.3]] })
    });
  });

  global.fetch = fetchMock;

  await getEmbedding("test text", "nomic-embed-text");

  expect(fetchMock).toHaveBeenCalledWith(
    expect.stringContaining("/api/embed"),
    expect.any(Object)
  );
});
```

## Test Coverage Goals

| Module | Target Coverage | Priority |
|--------|----------------|----------|
| **utils/** | 90%+ | High - Pure functions |
| **search/** | 85%+ | High - Core algorithms |
| **database/** | 80%+ | High - Data integrity |
| **services/** | 70%+ | Medium - External APIs |
| **output/** | 75%+ | Medium - Formatting |
| **commands/** | 60%+ | Low - CLI glue code |

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: oven-sh/setup-bun@v1
        with:
          bun-version: latest
      - run: bun install
      - run: bun test --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Mocking Strategies

### Mock Database

```typescript
import { Database } from "bun:sqlite";

function createTestDb(): Database {
  const db = new Database(":memory:");
  // Run schema creation
  return db;
}
```

### Mock Ollama API

```typescript
function mockOllamaEmbed(embeddings: number[][]) {
  return mock(() =>
    Promise.resolve({
      embeddings,
    })
  );
}
```

### Mock File System

```typescript
import { mock } from "bun:test";

const mockGlob = mock(() => ["file1.md", "file2.md"]);
```

## Performance Testing

```typescript
import { describe, test, expect } from "bun:test";

describe("Performance", () => {
  test("searches 1000 documents in <100ms", async () => {
    const start = performance.now();
    await searchFTS(testDb, "query", 10);
    const duration = performance.now() - start;

    expect(duration).toBeLessThan(100);
  });
});
```

## Testing Database Migrations

```typescript
describe("Schema Migrations", () => {
  test("adds display_path column", () => {
    const db = createTestDb();

    // Check column exists
    const columns = db.prepare(
      "PRAGMA table_info(documents)"
    ).all();

    expect(columns).toContainEqual(
      expect.objectContaining({ name: "display_path" })
    );
  });
});
```

## Continuous Testing

Enable watch mode during development:

```bash
# Terminal 1: Development
vim src/utils/formatters.ts

# Terminal 2: Tests auto-run
bun test --watch src/utils/formatters.test.ts
```

## Test Pyramid

```
     /\
    /  \   E2E Tests (Few, Slow)
   /────\
  /      \ Integration Tests (Some, Medium)
 /────────\
/          \ Unit Tests (Many, Fast)
────────────
```

**Distribution**:
- 70% Unit Tests (fast, isolated functions)
- 20% Integration Tests (modules working together)
- 10% E2E Tests (full user workflows)

## Next Steps

1. ✅ Choose framework: **Bun Test**
2. ⏳ Write first unit test for `formatters.ts`
3. ⏳ Add test script to package.json
4. ⏳ Test each module as we extract it
5. ⏳ Set up CI/CD with test automation
6. ⏳ Achieve 80%+ coverage
