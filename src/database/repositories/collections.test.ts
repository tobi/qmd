/**
 * Tests for Collection Repository
 * Target coverage: 85%+ with MANDATORY SQL injection tests
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import { CollectionRepository } from './collections.ts';
import { createTestDb, cleanupDb } from '../../../tests/fixtures/helpers/test-db.ts';
import { sqlInjectionPayloads } from '../../../tests/fixtures/helpers/fixtures.ts';

describe('CollectionRepository', () => {
  let db: Database;
  let repo: CollectionRepository;

  beforeEach(() => {
    db = createTestDb();
    repo = new CollectionRepository(db);
  });

  afterEach(() => {
    cleanupDb(db);
  });

  describe('findById', () => {
    test('returns collection when found', () => {
      const id = repo.insert('/test/path', '**/*.md');
      const collection = repo.findById(id);

      expect(collection).not.toBeNull();
      expect(collection?.id).toBe(id);
      expect(collection?.pwd).toBe('/test/path');
      expect(collection?.glob_pattern).toBe('**/*.md');
    });

    test('returns null when not found', () => {
      const collection = repo.findById(99999);
      expect(collection).toBeNull();
    });
  });

  describe('findByPwdAndPattern', () => {
    test('returns collection when found', () => {
      repo.insert('/test/path', '**/*.md');

      const collection = repo.findByPwdAndPattern('/test/path', '**/*.md');

      expect(collection).not.toBeNull();
      expect(collection?.pwd).toBe('/test/path');
      expect(collection?.glob_pattern).toBe('**/*.md');
    });

    test('returns null when pwd does not match', () => {
      repo.insert('/test/path', '**/*.md');

      const collection = repo.findByPwdAndPattern('/other/path', '**/*.md');
      expect(collection).toBeNull();
    });

    test('returns null when pattern does not match', () => {
      repo.insert('/test/path', '**/*.md');

      const collection = repo.findByPwdAndPattern('/test/path', '*.txt');
      expect(collection).toBeNull();
    });

    test('returns null when both do not match', () => {
      repo.insert('/test/path', '**/*.md');

      const collection = repo.findByPwdAndPattern('/other', '*.txt');
      expect(collection).toBeNull();
    });
  });

  describe('findAll', () => {
    test('returns all collections', () => {
      repo.insert('/path1', '*.md');
      repo.insert('/path2', '**/*.md');
      repo.insert('/path3', 'docs/*.md');

      const collections = repo.findAll();

      expect(collections).toHaveLength(3);
    });

    test('returns empty array when no collections', () => {
      const collections = repo.findAll();
      expect(collections).toHaveLength(0);
    });

    test('orders by created_at descending', () => {
      repo.insert('/path1', '*.md');
      repo.insert('/path2', '*.md');
      repo.insert('/path3', '*.md');

      const collections = repo.findAll();

      expect(collections).toHaveLength(3);

      // Verify they have created_at timestamps
      for (const collection of collections) {
        expect(collection.created_at).toBeDefined();
      }

      // Most recent should have newer or equal timestamp
      const first = new Date(collections[0].created_at).getTime();
      const last = new Date(collections[2].created_at).getTime();
      expect(first).toBeGreaterThanOrEqual(last);
    });
  });

  describe('findAllWithCounts', () => {
    test('returns collections with document counts', () => {
      const id = repo.insert('/test', '**/*.md');

      // Insert documents
      const now = new Date().toISOString();
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(id, 'doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content', now, now);

      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(id, 'doc2', 'Doc 2', 'hash2', '/test/doc2.md', 'doc2', 'Content', now, now);

      const collections = repo.findAllWithCounts();

      expect(collections).toHaveLength(1);
      expect(collections[0].active_count).toBe(2);
    });

    test('counts only active documents', () => {
      const id = repo.insert('/test', '**/*.md');

      const now = new Date().toISOString();

      // Active document
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(id, 'doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content', now, now);

      // Inactive document
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
      `).run(id, 'doc2', 'Doc 2', 'hash2', '/test/doc2.md', 'doc2', 'Content', now, now);

      const collections = repo.findAllWithCounts();

      expect(collections[0].active_count).toBe(1);
    });

    test('includes last_doc_update timestamp', () => {
      const id = repo.insert('/test', '**/*.md');

      const now = new Date().toISOString();
      db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(id, 'doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content', now, now);

      const collections = repo.findAllWithCounts();

      expect(collections[0].last_doc_update).toBe(now);
    });

    test('returns null last_doc_update for empty collection', () => {
      repo.insert('/test', '**/*.md');

      const collections = repo.findAllWithCounts();

      expect(collections[0].active_count).toBe(0);
      expect(collections[0].last_doc_update).toBeNull();
    });
  });

  describe('insert', () => {
    test('inserts collection and returns ID', () => {
      const id = repo.insert('/test/path', '**/*.md');

      expect(id).toBeGreaterThan(0);

      const collection = repo.findById(id);
      expect(collection?.pwd).toBe('/test/path');
      expect(collection?.glob_pattern).toBe('**/*.md');
    });

    test('sets created_at timestamp', () => {
      const id = repo.insert('/test/path', '**/*.md');
      const collection = repo.findById(id);

      expect(collection?.created_at).toBeDefined();
      expect(typeof collection?.created_at).toBe('string');
    });

    test('allows multiple collections with different paths', () => {
      const id1 = repo.insert('/path1', '**/*.md');
      const id2 = repo.insert('/path2', '**/*.md');

      expect(id1).not.toBe(id2);
      expect(repo.count()).toBe(2);
    });
  });

  describe('delete', () => {
    test('deletes collection', () => {
      const id = repo.insert('/test', '**/*.md');

      repo.delete(id);

      const collection = repo.findById(id);
      expect(collection).toBeNull();
    });

    test('deactivates associated documents', () => {
      const id = repo.insert('/test', '**/*.md');

      const now = new Date().toISOString();
      const docResult = db.prepare(`
        INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
      `).run(id, 'doc1', 'Doc 1', 'hash1', '/test/doc1.md', 'doc1', 'Content', now, now);

      const docId = docResult.lastInsertRowid as number;

      repo.delete(id);

      // Check document is deactivated
      const doc = db.prepare(`SELECT active FROM documents WHERE id = ?`).get(docId) as { active: number };
      expect(doc.active).toBe(0);
    });

    test('deleting non-existent collection does not error', () => {
      expect(() => {
        repo.delete(99999);
      }).not.toThrow();
    });
  });

  describe('count', () => {
    test('returns correct count', () => {
      expect(repo.count()).toBe(0);

      repo.insert('/path1', '*.md');
      expect(repo.count()).toBe(1);

      repo.insert('/path2', '*.md');
      expect(repo.count()).toBe(2);
    });

    test('count decreases after delete', () => {
      const id = repo.insert('/test', '*.md');
      expect(repo.count()).toBe(1);

      repo.delete(id);
      expect(repo.count()).toBe(0);
    });
  });
});

describe('SQL Injection Prevention', () => {
  let db: Database;
  let repo: CollectionRepository;

  beforeEach(() => {
    db = createTestDb();
    repo = new CollectionRepository(db);

    // Insert test collection
    repo.insert('/test/safe', '**/*.md');
  });

  afterEach(() => {
    cleanupDb(db);
  });

  test('findByPwdAndPattern handles malicious pwd safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.findByPwdAndPattern(payload, '**/*.md');
      }).not.toThrow();

      const result = repo.findByPwdAndPattern(payload, '**/*.md');
      expect(result).toBeNull(); // Should not find anything
    }

    // Verify table still exists
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='collections'
    `).all();
    expect(tables).toHaveLength(1);
  });

  test('findByPwdAndPattern handles malicious glob_pattern safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.findByPwdAndPattern('/test/safe', payload);
      }).not.toThrow();

      const result = repo.findByPwdAndPattern('/test/safe', payload);
      expect(result).toBeNull();
    }

    // Verify original data intact
    const collection = repo.findByPwdAndPattern('/test/safe', '**/*.md');
    expect(collection).not.toBeNull();
  });

  test('insert handles malicious pwd safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.insert(payload, '**/*.md');
      }).not.toThrow();
    }

    // Verify database integrity
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });

  test('insert handles malicious glob_pattern safely', () => {
    for (const payload of sqlInjectionPayloads) {
      expect(() => {
        repo.insert('/test/path', payload);
      }).not.toThrow();
    }

    // Verify database integrity
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });

  test('uses prepared statements for all queries', () => {
    // Test that SQL injection payloads don't execute
    const maliciousPwd = "'; DROP TABLE collections; --";

    repo.findByPwdAndPattern(maliciousPwd, '**/*.md');

    // If prepared statements are used, table should still exist
    const tables = db.prepare(`
      SELECT name FROM sqlite_master WHERE type='table' AND name='collections'
    `).all();

    expect(tables).toHaveLength(1);

    // Original data should be intact
    const count = repo.count();
    expect(count).toBeGreaterThan(0);
  });

  test('delete with malicious-looking ID does not cause issues', () => {
    // Even though ID is numeric, test edge cases
    const id = repo.insert('/test', '*.md');

    expect(() => {
      repo.delete(-1);
      repo.delete(0);
      repo.delete(999999);
    }).not.toThrow();

    // Original collection should still exist
    const collection = repo.findById(id);
    expect(collection).not.toBeNull();
  });
});
