/**
 * Tests for Zod schemas
 * Validates runtime type checking and error messages
 */

import { describe, test, expect } from 'bun:test';
import {
  CollectionSchema,
  DocumentSchema,
  ContentVectorSchema,
  PathContextSchema,
  OllamaCacheSchema,
  SearchResultSchema,
  RankedResultSchema,
  OutputOptionsSchema,
  RerankResponseSchema,
} from './schemas.ts';

describe('CollectionSchema', () => {
  test('validates valid collection', () => {
    const valid = {
      id: 1,
      pwd: '/home/user/docs',
      glob_pattern: '**/*.md',
      created_at: '2024-01-01T00:00:00Z',
      context: 'Documentation folder',
    };

    const result = CollectionSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('validates collection without optional context', () => {
    const valid = {
      id: 1,
      pwd: '/home/user/docs',
      glob_pattern: '**/*.md',
      created_at: '2024-01-01T00:00:00Z',
    };

    const result = CollectionSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('rejects invalid id', () => {
    const invalid = {
      id: -1,
      pwd: '/home/user/docs',
      glob_pattern: '**/*.md',
      created_at: '2024-01-01T00:00:00Z',
    };

    const result = CollectionSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });

  test('rejects empty pwd', () => {
    const invalid = {
      id: 1,
      pwd: '',
      glob_pattern: '**/*.md',
      created_at: '2024-01-01T00:00:00Z',
    };

    const result = CollectionSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });
});

describe('DocumentSchema', () => {
  test('validates valid document', () => {
    const valid = {
      id: 1,
      collection_id: 1,
      name: 'README.md',
      filepath: '/home/user/docs/README.md',
      hash: 'a'.repeat(64),
      title: 'README',
      body: 'Document content',
      active: 1,
      created_at: '2024-01-01T00:00:00Z',
      modified_at: '2024-01-01T00:00:00Z',
      display_path: 'docs/README.md',
    };

    const result = DocumentSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('validates document with minimal fields', () => {
    const valid = {
      id: 1,
      collection_id: 1,
      filepath: '/home/user/docs/README.md',
      hash: 'a'.repeat(64),
      title: 'README',
      body: 'Document content',
      active: 1,
      modified_at: '2024-01-01T00:00:00Z',
    };

    const result = DocumentSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('rejects invalid hash format', () => {
    const invalid = {
      id: 1,
      collection_id: 1,
      filepath: '/home/user/docs/README.md',
      hash: 'not-a-valid-hash',
      title: 'README',
      body: 'Document content',
      active: 1,
      modified_at: '2024-01-01T00:00:00Z',
    };

    const result = DocumentSchema.safeParse(invalid);
    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues[0].message).toContain('SHA-256');
    }
  });

  test('rejects invalid active value', () => {
    const invalid = {
      id: 1,
      collection_id: 1,
      filepath: '/home/user/docs/README.md',
      hash: 'a'.repeat(64),
      title: 'README',
      body: 'Document content',
      active: 2, // Must be 0 or 1
      modified_at: '2024-01-01T00:00:00Z',
    };

    const result = DocumentSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });
});

describe('ContentVectorSchema', () => {
  test('validates valid content vector', () => {
    const valid = {
      hash: 'a'.repeat(64),
      seq: 0,
      pos: 0,
      embedding: new Float32Array([0.1, 0.2, 0.3]),
    };

    const result = ContentVectorSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('rejects negative seq', () => {
    const invalid = {
      hash: 'a'.repeat(64),
      seq: -1,
      pos: 0,
      embedding: new Float32Array([0.1, 0.2, 0.3]),
    };

    const result = ContentVectorSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });

  test('rejects non-Float32Array embedding', () => {
    const invalid = {
      hash: 'a'.repeat(64),
      seq: 0,
      pos: 0,
      embedding: [0.1, 0.2, 0.3], // Should be Float32Array
    };

    const result = ContentVectorSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });
});

describe('PathContextSchema', () => {
  test('validates valid path context', () => {
    const valid = {
      id: 1,
      path_prefix: '/docs',
      context: 'Documentation directory',
      created_at: '2024-01-01T00:00:00Z',
    };

    const result = PathContextSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('validates without optional fields', () => {
    const valid = {
      path_prefix: '/docs',
      context: 'Documentation directory',
    };

    const result = PathContextSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('rejects empty path_prefix', () => {
    const invalid = {
      path_prefix: '',
      context: 'Documentation directory',
    };

    const result = PathContextSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });
});

describe('OllamaCacheSchema', () => {
  test('validates valid cache entry', () => {
    const valid = {
      hash: 'cache-key-123',
      result: 'cached result',
      created_at: '2024-01-01T00:00:00Z',
    };

    const result = OllamaCacheSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('rejects empty hash', () => {
    const invalid = {
      hash: '',
      result: 'cached result',
      created_at: '2024-01-01T00:00:00Z',
    };

    const result = OllamaCacheSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });
});

describe('SearchResultSchema', () => {
  test('validates valid search result', () => {
    const valid = {
      file: '/docs/README.md',
      displayPath: 'docs/README.md',
      title: 'README',
      body: 'Content',
      score: 0.85,
      source: 'fts' as const,
      chunkPos: 100,
    };

    const result = SearchResultSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('validates without optional chunkPos', () => {
    const valid = {
      file: '/docs/README.md',
      displayPath: 'docs/README.md',
      title: 'README',
      body: 'Content',
      score: 0.85,
      source: 'vec' as const,
    };

    const result = SearchResultSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('rejects invalid source', () => {
    const invalid = {
      file: '/docs/README.md',
      displayPath: 'docs/README.md',
      title: 'README',
      body: 'Content',
      score: 0.85,
      source: 'invalid',
    };

    const result = SearchResultSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });

  test('rejects score out of range', () => {
    const invalid = {
      file: '/docs/README.md',
      displayPath: 'docs/README.md',
      title: 'README',
      body: 'Content',
      score: 1.5,
      source: 'fts' as const,
    };

    const result = SearchResultSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });
});

describe('OutputOptionsSchema', () => {
  test('validates valid options', () => {
    const valid = {
      format: 'cli' as const,
      full: false,
      limit: 20,
      minScore: 0.5,
      all: false,
    };

    const result = OutputOptionsSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('validates without optional all', () => {
    const valid = {
      format: 'json' as const,
      full: true,
      limit: 10,
      minScore: 0.0,
    };

    const result = OutputOptionsSchema.safeParse(valid);
    expect(result.success).toBe(true);
  });

  test('rejects invalid format', () => {
    const invalid = {
      format: 'invalid',
      full: false,
      limit: 20,
      minScore: 0.5,
    };

    const result = OutputOptionsSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });

  test('rejects negative limit', () => {
    const invalid = {
      format: 'cli' as const,
      full: false,
      limit: -1,
      minScore: 0.5,
    };

    const result = OutputOptionsSchema.safeParse(invalid);
    expect(result.success).toBe(false);
  });
});

describe('Type inference', () => {
  test('inferred types match manual types', () => {
    // This is a compile-time test
    // If types don't match, TypeScript will error
    const doc: z.infer<typeof DocumentSchema> = {
      id: 1,
      collection_id: 1,
      filepath: '/docs/README.md',
      hash: 'a'.repeat(64),
      title: 'README',
      body: 'Content',
      active: 1,
      modified_at: '2024-01-01T00:00:00Z',
    };

    expect(doc).toBeDefined();
  });
});

describe('Error messages', () => {
  test('provides clear error messages', () => {
    const invalid = {
      id: 'not-a-number',
      pwd: '',
      glob_pattern: '',
      created_at: 'invalid-date',
    };

    const result = CollectionSchema.safeParse(invalid);
    expect(result.success).toBe(false);

    if (!result.success) {
      // Should have multiple errors
      expect(result.error.issues.length).toBeGreaterThan(0);

      // Each error should have a path and message
      for (const issue of result.error.issues) {
        expect(issue.path).toBeDefined();
        expect(issue.message).toBeTruthy();
      }
    }
  });
});
