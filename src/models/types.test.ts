/**
 * Tests for type definitions
 * Target coverage: 70%+
 */

import { describe, test, expect } from 'bun:test';
import type {
  LogProb,
  RerankResponse,
  SearchResult,
  RankedResult,
  OutputFormat,
  OutputOptions,
  Collection,
  Document,
  ContentVector,
  PathContext,
  OllamaCache,
} from './types.ts';

describe('Type Definitions', () => {
  test('LogProb type structure', () => {
    const logprob: LogProb = {
      token: 'yes',
      logprob: -0.5,
    };

    expect(logprob.token).toBe('yes');
    expect(logprob.logprob).toBe(-0.5);
    expect(typeof logprob.token).toBe('string');
    expect(typeof logprob.logprob).toBe('number');
  });

  test('RerankResponse type structure', () => {
    const response: RerankResponse = {
      model: 'test-model',
      created_at: '2024-01-01T00:00:00Z',
      response: 'yes',
      done: true,
      logprobs: [
        { token: 'yes', logprob: -0.1 },
      ],
    };

    expect(response.model).toBe('test-model');
    expect(response.done).toBe(true);
    expect(response.logprobs).toBeDefined();
    expect(Array.isArray(response.logprobs)).toBe(true);
  });

  test('RerankResponse with optional fields', () => {
    const minimal: RerankResponse = {
      model: 'test',
      created_at: '2024-01-01T00:00:00Z',
      response: 'no',
      done: false,
    };

    expect(minimal.done_reason).toBeUndefined();
    expect(minimal.total_duration).toBeUndefined();
    expect(minimal.logprobs).toBeUndefined();
  });

  test('SearchResult type structure with fts source', () => {
    const result: SearchResult = {
      file: '/path/to/file.md',
      displayPath: 'file',
      title: 'Test Document',
      body: 'Test content',
      score: 0.95,
      source: 'fts',
    };

    expect(result.source).toBe('fts');
    expect(result.score).toBeGreaterThan(0);
    expect(result.chunkPos).toBeUndefined();
  });

  test('SearchResult type structure with vec source', () => {
    const result: SearchResult = {
      file: '/path/to/file.md',
      displayPath: 'file',
      title: 'Test Document',
      body: 'Test content',
      score: 0.85,
      source: 'vec',
      chunkPos: 0,
    };

    expect(result.source).toBe('vec');
    expect(result.chunkPos).toBe(0);
  });

  test('RankedResult type structure', () => {
    const result: RankedResult = {
      file: '/path/to/file.md',
      displayPath: 'file',
      title: 'Test Document',
      body: 'Test content',
      score: 0.90,
    };

    expect(result.file).toBeTruthy();
    expect(result.score).toBeGreaterThan(0);
  });

  test('OutputFormat literal types', () => {
    const formats: OutputFormat[] = ['cli', 'csv', 'md', 'xml', 'files', 'json'];

    for (const format of formats) {
      const options: OutputOptions = {
        format,
        full: false,
        limit: 10,
        minScore: 0.5,
      };

      expect(options.format).toBe(format);
    }
  });

  test('OutputOptions type structure', () => {
    const options: OutputOptions = {
      format: 'json',
      full: true,
      limit: 20,
      minScore: 0.7,
      all: false,
    };

    expect(options.format).toBe('json');
    expect(options.full).toBe(true);
    expect(options.limit).toBe(20);
    expect(options.minScore).toBe(0.7);
    expect(options.all).toBe(false);
  });

  test('Collection interface structure', () => {
    const collection: Collection = {
      id: 1,
      pwd: '/home/user/projects',
      glob_pattern: '**/*.md',
      created_at: '2024-01-01T00:00:00Z',
    };

    expect(collection.id).toBe(1);
    expect(collection.pwd).toBeTruthy();
    expect(collection.glob_pattern).toContain('*.md');
  });

  test('Document interface structure', () => {
    const doc: Document = {
      id: 1,
      collection_id: 1,
      filepath: '/path/to/file.md',
      hash: 'abc123',
      title: 'Test',
      body: 'Content',
      active: 1,
      modified_at: '2024-01-01T00:00:00Z',
    };

    expect(doc.active).toBe(1);
    expect(doc.display_path).toBeUndefined();
  });

  test('Document with display_path', () => {
    const doc: Document = {
      id: 1,
      collection_id: 1,
      filepath: '/path/to/file.md',
      hash: 'abc123',
      title: 'Test',
      body: 'Content',
      active: 1,
      modified_at: '2024-01-01T00:00:00Z',
      display_path: 'docs/file',
    };

    expect(doc.display_path).toBe('docs/file');
  });

  test('ContentVector interface structure', () => {
    const vector: ContentVector = {
      hash: 'abc123',
      seq: 0,
      pos: 0,
      embedding: new Float32Array([0.1, 0.2, 0.3]),
    };

    expect(vector.hash).toBe('abc123');
    expect(vector.seq).toBe(0);
    expect(vector.embedding).toBeInstanceOf(Float32Array);
    expect(vector.embedding.length).toBe(3);
  });

  test('PathContext interface structure', () => {
    const context: PathContext = {
      path_prefix: '/home/user/docs',
      context: 'This is documentation',
    };

    expect(context.path_prefix).toBeTruthy();
    expect(context.context).toBeTruthy();
  });

  test('OllamaCache interface structure', () => {
    const cache: OllamaCache = {
      hash: 'key123',
      result: 'cached result',
      created_at: '2024-01-01T00:00:00Z',
    };

    expect(cache.hash).toBe('key123');
    expect(cache.result).toBe('cached result');
  });
});

describe('Type Compatibility', () => {
  test('SearchResult can be converted to RankedResult', () => {
    const searchResult: SearchResult = {
      file: '/path/to/file.md',
      displayPath: 'file',
      title: 'Test',
      body: 'Content',
      score: 0.9,
      source: 'fts',
    };

    const rankedResult: RankedResult = {
      file: searchResult.file,
      displayPath: searchResult.displayPath,
      title: searchResult.title,
      body: searchResult.body,
      score: searchResult.score,
    };

    expect(rankedResult.file).toBe(searchResult.file);
    expect(rankedResult.score).toBe(searchResult.score);
  });

  test('RerankResponse logprobs are LogProb array', () => {
    const logprobs: LogProb[] = [
      { token: 'yes', logprob: -0.1 },
      { token: 'no', logprob: -2.5 },
    ];

    const response: RerankResponse = {
      model: 'test',
      created_at: '2024-01-01T00:00:00Z',
      response: 'yes',
      done: true,
      logprobs,
    };

    expect(response.logprobs).toBe(logprobs);
    expect(response.logprobs?.length).toBe(2);
  });
});

describe('Type Guards and Validation', () => {
  test('SearchResult source is strictly typed', () => {
    const ftsResult: SearchResult = {
      file: 'test.md',
      displayPath: 'test',
      title: 'Test',
      body: 'Content',
      score: 0.9,
      source: 'fts',
    };

    const vecResult: SearchResult = {
      file: 'test.md',
      displayPath: 'test',
      title: 'Test',
      body: 'Content',
      score: 0.9,
      source: 'vec',
    };

    expect(ftsResult.source).toBe('fts');
    expect(vecResult.source).toBe('vec');
  });

  test('OutputFormat is restricted to valid values', () => {
    const validFormats: OutputFormat[] = ['cli', 'csv', 'md', 'xml', 'files', 'json'];

    for (const format of validFormats) {
      const options: OutputOptions = {
        format,
        full: false,
        limit: 10,
        minScore: 0.5,
      };
      expect(validFormats).toContain(options.format);
    }
  });

  test('Document active is numeric (0 or 1)', () => {
    const activeDoc: Document = {
      id: 1,
      collection_id: 1,
      filepath: 'test.md',
      hash: 'abc',
      title: 'Test',
      body: 'Content',
      active: 1,
      modified_at: '2024-01-01T00:00:00Z',
    };

    const inactiveDoc: Document = {
      id: 2,
      collection_id: 1,
      filepath: 'test2.md',
      hash: 'def',
      title: 'Test 2',
      body: 'Content 2',
      active: 0,
      modified_at: '2024-01-01T00:00:00Z',
    };

    expect(activeDoc.active).toBe(1);
    expect(inactiveDoc.active).toBe(0);
  });
});

describe('Complex Type Scenarios', () => {
  test('Array of SearchResults', () => {
    const results: SearchResult[] = [
      {
        file: 'file1.md',
        displayPath: 'file1',
        title: 'Title 1',
        body: 'Body 1',
        score: 0.9,
        source: 'fts',
      },
      {
        file: 'file2.md',
        displayPath: 'file2',
        title: 'Title 2',
        body: 'Body 2',
        score: 0.8,
        source: 'vec',
        chunkPos: 5,
      },
    ];

    expect(results).toHaveLength(2);
    expect(results[0].source).toBe('fts');
    expect(results[1].source).toBe('vec');
    expect(results[1].chunkPos).toBe(5);
  });

  test('Nested ContentVector with varying dimensions', () => {
    const vectors: ContentVector[] = [
      {
        hash: 'hash1',
        seq: 0,
        pos: 0,
        embedding: new Float32Array(128),
      },
      {
        hash: 'hash1',
        seq: 1,
        pos: 1024,
        embedding: new Float32Array(128),
      },
    ];

    expect(vectors).toHaveLength(2);
    expect(vectors[0].seq).toBe(0);
    expect(vectors[1].seq).toBe(1);
    expect(vectors[0].embedding.length).toBe(vectors[1].embedding.length);
  });

  test('OutputOptions with all optional fields', () => {
    const minimalOptions: OutputOptions = {
      format: 'cli',
      full: false,
      limit: 10,
      minScore: 0.0,
    };

    const fullOptions: OutputOptions = {
      format: 'json',
      full: true,
      limit: 50,
      minScore: 0.7,
      all: true,
    };

    expect(minimalOptions.all).toBeUndefined();
    expect(fullOptions.all).toBe(true);
  });
});
