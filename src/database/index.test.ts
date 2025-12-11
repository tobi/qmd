/**
 * Tests for database module exports
 */

import { describe, test, expect } from 'bun:test';
import {
  getDb,
  ensureVecTable,
  getHashesNeedingEmbedding,
  checkIndexHealth,
  DocumentRepository,
  CollectionRepository,
  VectorRepository,
  PathContextRepository,
} from './index.ts';

describe('Database Module Exports', () => {
  test('getDb is exported', () => {
    expect(getDb).toBeDefined();
    expect(typeof getDb).toBe('function');
  });

  test('ensureVecTable is exported', () => {
    expect(ensureVecTable).toBeDefined();
    expect(typeof ensureVecTable).toBe('function');
  });

  test('getHashesNeedingEmbedding is exported', () => {
    expect(getHashesNeedingEmbedding).toBeDefined();
    expect(typeof getHashesNeedingEmbedding).toBe('function');
  });

  test('checkIndexHealth is exported', () => {
    expect(checkIndexHealth).toBeDefined();
    expect(typeof checkIndexHealth).toBe('function');
  });

  test('DocumentRepository is re-exported', () => {
    expect(DocumentRepository).toBeDefined();
    expect(typeof DocumentRepository).toBe('function');
  });

  test('CollectionRepository is re-exported', () => {
    expect(CollectionRepository).toBeDefined();
    expect(typeof CollectionRepository).toBe('function');
  });

  test('VectorRepository is re-exported', () => {
    expect(VectorRepository).toBeDefined();
    expect(typeof VectorRepository).toBe('function');
  });

  test('PathContextRepository is re-exported', () => {
    expect(PathContextRepository).toBeDefined();
    expect(typeof PathContextRepository).toBe('function');
  });
});
