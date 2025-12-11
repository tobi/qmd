/**
 * Tests for Embedding Service
 * Target coverage: 85%+
 */

import { describe, test, expect, beforeEach, afterEach, mock } from 'bun:test';
import { Database } from 'bun:sqlite';
import { embedText, embedDocument, chunkDocument } from './embedding.ts';
import { createTestDbWithVectors, cleanupDb } from '../../tests/fixtures/helpers/test-db.ts';
import { mockOllamaEmbed } from '../../tests/fixtures/helpers/mock-ollama.ts';
import { VectorRepository } from '../database/repositories/vectors.ts';

describe('chunkDocument', () => {
  test('returns single chunk for small text', () => {
    const text = 'Short text';
    const chunks = chunkDocument(text, 1000, 200);

    expect(chunks).toHaveLength(1);
    expect(chunks[0].text).toBe(text);
    expect(chunks[0].pos).toBe(0);
  });

  test('chunks large text with default settings', () => {
    const text = 'a'.repeat(2500);
    const chunks = chunkDocument(text);

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[0].pos).toBe(0);
    expect(chunks[1].pos).toBe(800); // 1000 - 200 overlap
  });

  test('applies overlap correctly', () => {
    const text = 'a'.repeat(1500);
    const chunks = chunkDocument(text, 1000, 200);

    expect(chunks).toHaveLength(2);
    expect(chunks[0].pos).toBe(0);
    expect(chunks[1].pos).toBe(800);
  });

  test('handles custom chunk size and overlap', () => {
    const text = 'a'.repeat(500);
    const chunks = chunkDocument(text, 200, 50);

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks[1].pos).toBe(150); // 200 - 50
  });

  test('chunks contain correct text slices', () => {
    const text = '0123456789';
    const chunks = chunkDocument(text, 5, 2);

    expect(chunks[0].text).toBe('01234');
    expect(chunks[1].text).toBe('34567');
    expect(chunks[2].text).toBe('6789');
  });

  test('handles edge case of exact chunk size', () => {
    const text = 'a'.repeat(1000);
    const chunks = chunkDocument(text, 1000, 200);

    expect(chunks).toHaveLength(1);
  });
});

describe('embedText', () => {
  beforeEach(() => {
    global.fetch = fetch;
  });

  test('returns Float32Array embedding', async () => {
    global.fetch = mockOllamaEmbed([[0.1, 0.2, 0.3]]);

    const result = await embedText('test text', 'test-model');

    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(3);
    expect(result[0]).toBeCloseTo(0.1, 1);
  });

  test('passes query flag to getEmbedding', async () => {
    global.fetch = mockOllamaEmbed([[0.5, 0.6]]);

    const result = await embedText('query', 'test-model', true);

    expect(result).toBeInstanceOf(Float32Array);
  });

  test('passes title to getEmbedding', async () => {
    global.fetch = mockOllamaEmbed([[0.7, 0.8]]);

    const result = await embedText('doc', 'test-model', false, 'Title');

    expect(result).toBeInstanceOf(Float32Array);
  });
});

describe('embedDocument', () => {
  let db: Database;
  let repo: VectorRepository;

  beforeEach(() => {
    db = createTestDbWithVectors(128);
    repo = new VectorRepository(db);
    global.fetch = fetch;
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('embeds and stores single chunk', async () => {
    global.fetch = mockOllamaEmbed([Array(128).fill(0.1)]);

    const chunks = [{ text: 'test content', pos: 0, title: 'Test' }];
    await embedDocument(db, 'doc-hash', chunks, 'test-model');

    const vectors = repo.findByHash('doc-hash');
    expect(vectors).toHaveLength(1);
    expect(vectors[0].seq).toBe(0);
  });

  test('embeds and stores multiple chunks', async () => {
    global.fetch = mockOllamaEmbed([
      Array(128).fill(0.1),
      Array(128).fill(0.2),
      Array(128).fill(0.3),
    ]);

    const chunks = [
      { text: 'chunk 1', pos: 0 },
      { text: 'chunk 2', pos: 100 },
      { text: 'chunk 3', pos: 200 },
    ];
    await embedDocument(db, 'multi-hash', chunks, 'test-model');

    const vectors = repo.findByHash('multi-hash');
    expect(vectors).toHaveLength(3);
    expect(vectors[0].pos).toBe(0);
    expect(vectors[1].pos).toBe(100);
    expect(vectors[2].pos).toBe(200);
  });

  test('deletes existing vectors before inserting', async () => {
    global.fetch = mockOllamaEmbed([
      Array(128).fill(0.1),
      Array(128).fill(0.2),
    ]);

    // First embedding
    await embedDocument(db, 'replace-hash', [{ text: 'original', pos: 0 }], 'test-model');
    expect(repo.findByHash('replace-hash')).toHaveLength(1);

    // Re-embed with different chunks
    await embedDocument(db, 'replace-hash', [{ text: 'new', pos: 0 }], 'test-model');
    const vectors = repo.findByHash('replace-hash');

    expect(vectors).toHaveLength(1);
  });

  test('handles documents with titles', async () => {
    global.fetch = mockOllamaEmbed([Array(128).fill(0.1)]);

    const chunks = [{ text: 'content', pos: 0, title: 'Document Title' }];
    await embedDocument(db, 'titled-hash', chunks, 'test-model');

    const vectors = repo.findByHash('titled-hash');
    expect(vectors).toHaveLength(1);
  });

  test('creates vectors_vec table with correct dimensions', async () => {
    global.fetch = mockOllamaEmbed([Array(256).fill(0.1)]);

    const chunks = [{ text: 'test', pos: 0 }];
    await embedDocument(db, 'dim-test', chunks, 'test-model');

    // Verify table has correct dimensions
    const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE name='vectors_vec'`).get() as { sql: string };
    expect(tableInfo.sql).toContain('float[256]');
  });
});
