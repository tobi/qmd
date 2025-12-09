/**
 * Tests for repository exports
 */

import { describe, test, expect } from 'bun:test';
import {
  DocumentRepository,
  CollectionRepository,
  VectorRepository,
  PathContextRepository,
} from './index.ts';

describe('Repository Exports', () => {
  test('DocumentRepository is exported', () => {
    expect(DocumentRepository).toBeDefined();
    expect(typeof DocumentRepository).toBe('function');
  });

  test('CollectionRepository is exported', () => {
    expect(CollectionRepository).toBeDefined();
    expect(typeof CollectionRepository).toBe('function');
  });

  test('VectorRepository is exported', () => {
    expect(VectorRepository).toBeDefined();
    expect(typeof VectorRepository).toBe('function');
  });

  test('PathContextRepository is exported', () => {
    expect(PathContextRepository).toBeDefined();
    expect(typeof PathContextRepository).toBe('function');
  });
});
