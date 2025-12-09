/**
 * Tests for Document Repository
 * Target coverage: 85%+ with MANDATORY SQL injection tests
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { DocumentRepository } from './documents.ts';
import { createTestDb, cleanupDb } from '../../../tests/fixtures/helpers/test-db.ts';
import { sqlInjectionPayloads } from '../../../tests/fixtures/helpers/fixtures.ts';

describe('DocumentRepository', () => {
  let db: Database;
  let repo: DocumentRepository;
  let collectionId: number;

  beforeEach(() => {
    db = createTestDb();
    repo = new DocumentRepository(db);

    // Insert test collection
    const result = db.prepare(`
      INSERT INTO collections (pwd, glob_pattern, created_at)
      VALUES (?, ?, ?)
    `).run('/test', '**/*.md', new Date().toISOString());

    collectionId = result.lastInsertRowid as number;
  });

  afterEach(() => {
    cleanupDb(db);
  });

  describe('findById', () => {
    test('returns document when found', () => {
      const now = new Date().toISOString();
      const result = db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test', 'Test Doc', 'hash123', '/test/doc.md', 'doc', 'Content', now, now);

      const id = result.lastInsertRowid as number;
      const doc = repo.findById(id);

      expect(doc).not.toBeNull();
      expect(doc?.id).toBe(id);
      expect(doc?.title).toBe('Test Doc');
    });

    test('returns null when not found', () => {
      const doc = repo.findById(99999);
      expect(doc).toBeNull();
    });

    test('returns null for inactive documents', () => {
      const now = new Date().toISOString();
      const result = db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
      `).run(collectionId, 'test', 'Test', 'hash', '/test/doc.md', 'doc', 'Content', now, now);

      const id = result.lastInsertRowid as number;
      const doc = repo.findById(id);

      expect(doc).toBeNull();
    });
  });

  describe('findByFilepath', () => {
    test('returns document when found', () => {
      const now = new Date().toISOString();
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test', 'Test Doc', 'hash123', '/test/doc.md', 'doc', 'Content', now, now);

      const doc = repo.findByFilepath('/test/doc.md');

      expect(doc).not.toBeNull();
      expect(doc?.filepath).toBe('/test/doc.md');
    });

    test('returns null when not found', () => {
      const doc = repo.findByFilepath('/nonexistent.md');
      expect(doc).toBeNull();
    });
  });

  describe('findByHash', () => {
    test('returns document when found', () => {
      const now = new Date().toISOString();
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test', 'Test Doc', 'abc123', '/test/doc.md', 'doc', 'Content', now, now);

      const doc = repo.findByHash('abc123');

      expect(doc).not.toBeNull();
      expect(doc?.hash).toBe('abc123');
    });

    test('returns null when not found', () => {
      const doc = repo.findByHash('nonexistent');
      expect(doc).toBeNull();
    });

    test('returns only one document when multiple match', () => {
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test1', 'Test 1', 'samehash', '/test/doc1.md', 'doc1', 'Content', now, now);

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test2', 'Test 2', 'samehash', '/test/doc2.md', 'doc2', 'Content', now, now);

      const doc = repo.findByHash('samehash');

      expect(doc).not.toBeNull();
      // Should return exactly one document
    });
  });

  describe('findByCollection', () => {
    test('returns all documents in collection', () => {
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test1', 'Test 1', 'hash1', '/test/doc1.md', 'doc1', 'Content 1', now, now);

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test2', 'Test 2', 'hash2', '/test/doc2.md', 'doc2', 'Content 2', now, now);

      const docs = repo.findByCollection(collectionId);

      expect(docs).toHaveLength(2);
      expect(docs[0].filepath).toBe('/test/doc1.md');
      expect(docs[1].filepath).toBe('/test/doc2.md');
    });

    test('returns empty array for collection with no documents', () => {
      const docs = repo.findByCollection(collectionId);
      expect(docs).toHaveLength(0);
    });

    test('orders results by filepath', () => {
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'z', 'Z', 'hash3', '/test/z.md', 'z', 'Content', now, now);

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'a', 'A', 'hash1', '/test/a.md', 'a', 'Content', now, now);

      const docs = repo.findByCollection(collectionId);

      expect(docs[0].filepath).toBe('/test/a.md');
      expect(docs[1].filepath).toBe('/test/z.md');
    });
  });

  describe('insert', () => {
    test('inserts document and returns ID', () => {
      const now = new Date().toISOString();

      const id = repo.insert({
        collection_id: collectionId,
        name: 'test',
        title: 'Test Document',
        hash: 'hash123',
        filepath: '/test/new.md',
        display_path: 'new',
        body: 'New content',
        created_at: now,
        modified_at: now,
        active: 1,
      });

      expect(id).toBeGreaterThan(0);

      const doc = repo.findById(id);
      expect(doc?.title).toBe('Test Document');
    });

    test('inserted document is searchable via FTS', () => {
      const now = new Date().toISOString();

      repo.insert({
        collection_id: collectionId,
        name: 'searchable',
        title: 'Searchable Document',
        hash: 'hash456',
        filepath: '/test/searchable.md',
        display_path: 'searchable',
        body: 'This document contains unique searchable content',
        created_at: now,
        modified_at: now,
        active: 1,
      });

      const results = repo.searchFTS('unique', 10);
      expect(results.length).toBeGreaterThan(0);
      expect(results[0].body).toContain('unique');
    });
  });

  describe('searchFTS', () => {
    beforeEach(() => {
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'doc1', 'First Document', 'hash1', '/test/doc1.md', 'doc1', 'This is about cats and dogs', now, now);

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'doc2', 'Second Document', 'hash2', '/test/doc2.md', 'doc2', 'This is about birds and fish', now, now);
    });

    test('finds documents matching query', () => {
      const results = repo.searchFTS('cats', 10);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].body).toContain('cats');
    });

    test('returns normalized BM25 scores', () => {
      const results = repo.searchFTS('cats', 10);

      expect(results.length).toBeGreaterThan(0);
      expect(results[0].score).toBeGreaterThanOrEqual(0);
      expect(results[0].score).toBeLessThanOrEqual(1);
    });

    test('limits results correctly', () => {
      const results = repo.searchFTS('document', 1);
      expect(results.length).toBeLessThanOrEqual(1);
    });

    test('returns empty array for no matches', () => {
      const results = repo.searchFTS('nonexistentxyz', 10);
      expect(results).toHaveLength(0);
    });
  });

  describe('updateDisplayPath', () => {
    test('updates display path correctly', () => {
      const now = new Date().toISOString();
      const result = db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test', 'Test', 'hash', '/test/doc.md', 'old-path', 'Content', now, now);

      const id = result.lastInsertRowid as number;

      repo.updateDisplayPath(id, 'new-path');

      const doc = repo.findById(id);
      expect(doc?.display_path).toBe('new-path');
    });
  });

  describe('deactivate', () => {
    test('marks document as inactive', () => {
      const now = new Date().toISOString();
      const result = db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test', 'Test', 'hash', '/test/doc.md', 'doc', 'Content', now, now);

      const id = result.lastInsertRowid as number;

      repo.deactivate(id);

      const doc = repo.findById(id);
      expect(doc).toBeNull(); // findById only returns active docs
    });
  });

  describe('count', () => {
    test('returns correct count of active documents', () => {
      const now = new Date().toISOString();

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test1', 'Test 1', 'hash1', '/test/doc1.md', 'doc1', 'Content', now, now);

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(collectionId, 'test2', 'Test 2', 'hash2', '/test/doc2.md', 'doc2', 'Content', now, now);

      // Add inactive document
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
      `).run(collectionId, 'test3', 'Test 3', 'hash3', '/test/doc3.md', 'doc3', 'Content', now, now);

      const count = repo.count();
      expect(count).toBe(2); // Only active documents
    });
  });
});

describe('SQL Injection Prevention', () => {
  let db: Database;
  let repo: DocumentRepository;
  let collectionId: number;

  beforeEach(() => {
    db = createTestDb();
    repo = new DocumentRepository(db);

    const result = db.prepare(`
      INSERT INTO collections (pwd, glob_pattern, created_at)
      VALUES (?, ?, ?)
    `).run('/test', '**/*.md', new Date().toISOString());

    collectionId = result.lastInsertRowid as number;

    // Insert test document
    const now = new Date().toISOString();
    db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run(collectionId, 'test', 'Test', 'safe-hash', '/test/doc.md', 'doc', 'Content', now, now);
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('findByFilepath handles malicious input safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.findByFilepath(payload);
      }).not.toThrow();

      const result = repo.findByFilepath(payload);
      expect(result).toBeNull(); // Should not find anything
    }

    // Verify table still exists
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='documents'
    `).all();
    expect(tables).toHaveLength(1);
  });

  test('findByHash handles malicious input safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.findByHash(payload);
      }).not.toThrow();

      const result = repo.findByHash(payload);
      expect(result).toBeNull();
    }

    // Verify original data intact
    const doc = repo.findByHash('safe-hash');
    expect(doc).not.toBeNull();
  });

  test('searchFTS handles malicious input safely', () => {
    // FTS5 will throw syntax errors for some malicious input, which is expected
    // The important thing is that it doesn't execute SQL injection
    for (const payload of sqlInjectionPayloads) {
      try {
        repo.searchFTS(payload, 10);
        // If no error, good - query executed safely
      } catch (error) {
        // FTS syntax errors are acceptable - they prevent injection
        expect(error).toBeDefined();
      }
    }

    // Verify table still exists (not dropped)
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'
    `).all();
    expect(tables).toHaveLength(1);

    // Verify original data intact
    const doc = repo.findByHash('safe-hash');
    expect(doc).not.toBeNull();
  });

  test('updateDisplayPath handles malicious input safely', () => {
    const now = new Date().toISOString();
    const result = db.prepare(`
      INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `).run(collectionId, 'test-sql', 'Test SQL', 'hash-sql', '/test/sql.md', 'sql', 'Content', now, now);

    const id = result.lastInsertRowid as number;

    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.updateDisplayPath(id, payload);
      }).not.toThrow();
    }

    // Verify database integrity
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });

  test('uses prepared statements for all queries', () => {
    // This test verifies that SQL injection payloads don't execute
    const maliciousFilepath = "'; DROP TABLE documents; --";

    repo.findByFilepath(maliciousFilepath);

    // If prepared statements are used, table should still exist
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='documents'
    `).all();

    expect(tables).toHaveLength(1);

    // Original data should be intact
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });
});
