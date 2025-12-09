/**
 * Tests for Vector Repository
 * Target coverage: 80%+
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { VectorRepository } from './vectors.ts';
import { createTestDbWithVectors, cleanupDb } from '../../../tests/fixtures/helpers/test-db.ts';

describe('VectorRepository', () => {
  let db: Database;
  let repo: VectorRepository;

  beforeEach(() => {
    db = createTestDbWithVectors(128);
    repo = new VectorRepository(db);
  });

  afterEach(() => {
    cleanupDb(db);
  });

  describe('findByHash', () => {
    test('returns all vectors for a hash', () => {
      const vectors = repo.findByHash('hash123');

      expect(vectors).toHaveLength(1);
      expect(vectors[0].hash).toBe('hash123');
      expect(vectors[0].seq).toBe(0);
    });

    test('returns empty array when no vectors found', () => {
      const vectors = repo.findByHash('nonexistent');
      expect(vectors).toHaveLength(0);
    });

    test('returns multiple vectors ordered by seq', () => {
      // Insert additional vectors
      const embedding = new Float32Array(128).fill(0.2);
      repo.insert('multi-hash', 0, 0, embedding, 'test-model');
      repo.insert('multi-hash', 1, 512, embedding, 'test-model');
      repo.insert('multi-hash', 2, 1024, embedding, 'test-model');

      const vectors = repo.findByHash('multi-hash');

      expect(vectors).toHaveLength(3);
      expect(vectors[0].seq).toBe(0);
      expect(vectors[1].seq).toBe(1);
      expect(vectors[2].seq).toBe(2);
    });
  });

  describe('findByHashAndSeq', () => {
    test('returns specific vector chunk', () => {
      const vector = repo.findByHashAndSeq('hash123', 0);

      expect(vector).not.toBeNull();
      expect(vector?.hash).toBe('hash123');
      expect(vector?.seq).toBe(0);
    });

    test('returns null when vector not found', () => {
      const vector = repo.findByHashAndSeq('nonexistent', 0);
      expect(vector).toBeNull();
    });

    test('returns null when seq does not match', () => {
      const vector = repo.findByHashAndSeq('hash123', 999);
      expect(vector).toBeNull();
    });
  });

  describe('hasEmbedding', () => {
    test('returns true when document has embeddings', () => {
      const hasEmbedding = repo.hasEmbedding('hash123');
      expect(hasEmbedding).toBe(true);
    });

    test('returns false when document has no embeddings', () => {
      const hasEmbedding = repo.hasEmbedding('nonexistent');
      expect(hasEmbedding).toBe(false);
    });
  });

  describe('insert', () => {
    test('inserts vector embedding', () => {
      const embedding = new Float32Array(128).fill(0.5);

      repo.insert('new-hash', 0, 0, embedding, 'test-model');

      const vector = repo.findByHashAndSeq('new-hash', 0);
      expect(vector).not.toBeNull();
      expect(vector?.hash).toBe('new-hash');
      expect(vector?.seq).toBe(0);
      expect(vector?.pos).toBe(0);
    });

    test('inserts multiple chunks for same hash', () => {
      const embedding = new Float32Array(128).fill(0.5);

      repo.insert('chunked-hash', 0, 0, embedding, 'test-model');
      repo.insert('chunked-hash', 1, 512, embedding, 'test-model');

      const vectors = repo.findByHash('chunked-hash');
      expect(vectors).toHaveLength(2);
    });

    test('stores position correctly', () => {
      const embedding = new Float32Array(128).fill(0.5);

      repo.insert('pos-test', 0, 1024, embedding, 'test-model');

      const vector = repo.findByHashAndSeq('pos-test', 0);
      expect(vector?.pos).toBe(1024);
    });
  });

  describe('deleteByHash', () => {
    test('deletes all vectors for a hash', () => {
      const embedding = new Float32Array(128).fill(0.5);
      repo.insert('delete-me', 0, 0, embedding, 'test-model');
      repo.insert('delete-me', 1, 512, embedding, 'test-model');

      expect(repo.hasEmbedding('delete-me')).toBe(true);

      repo.deleteByHash('delete-me');

      expect(repo.hasEmbedding('delete-me')).toBe(false);
      expect(repo.findByHash('delete-me')).toHaveLength(0);
    });

    test('deleting non-existent hash does not error', () => {
      expect(() => {
        repo.deleteByHash('nonexistent');
      }).not.toThrow();
    });
  });

  describe('countDocumentsWithEmbeddings', () => {
    test('returns correct count of unique hashes', () => {
      const count = repo.countDocumentsWithEmbeddings();
      expect(count).toBe(1); // hash123 from createTestDbWithVectors
    });

    test('counts distinct hashes correctly', () => {
      const embedding = new Float32Array(128).fill(0.5);
      repo.insert('hash2', 0, 0, embedding, 'test-model');
      repo.insert('hash2', 1, 512, embedding, 'test-model');
      repo.insert('hash3', 0, 0, embedding, 'test-model');

      const count = repo.countDocumentsWithEmbeddings();
      expect(count).toBe(3); // hash123, hash2, hash3
    });
  });

  describe('countVectorChunks', () => {
    test('returns total number of chunks', () => {
      const count = repo.countVectorChunks();
      expect(count).toBe(1); // One chunk from createTestDbWithVectors
    });

    test('counts all chunks correctly', () => {
      const embedding = new Float32Array(128).fill(0.5);
      repo.insert('hash2', 0, 0, embedding, 'test-model');
      repo.insert('hash2', 1, 512, embedding, 'test-model');

      const count = repo.countVectorChunks();
      expect(count).toBe(3); // hash123:0, hash2:0, hash2:1
    });
  });
});

describe('SQL Injection Prevention (Basic)', () => {
  let db: Database;
  let repo: VectorRepository;

  beforeEach(() => {
    db = createTestDbWithVectors(128);
    repo = new VectorRepository(db);
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('handles malicious input safely', () => {
    const maliciousInput = "'; DROP TABLE content_vectors; --";

    // Should not throw or execute injection
    expect(() => {
      repo.findByHash(maliciousInput);
      repo.findByHashAndSeq(maliciousInput, 0);
      repo.hasEmbedding(maliciousInput);
      repo.deleteByHash(maliciousInput);
    }).not.toThrow();

    // Verify tables still exist
    const tables = db.prepare(`
      SELECT name FROM sqlite_master
      WHERE type='table' AND name IN ('content_vectors', 'vectors_vec')
    `).all();
    expect(tables).toHaveLength(2);

    // Verify original data intact
    const count = repo.countDocumentsWithEmbeddings();
    expect(count).toBeGreaterThan(0);
  });
});
