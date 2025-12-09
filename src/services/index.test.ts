/**
 * Tests for services module exports
 */

import { describe, test, expect } from 'bun:test';
import {
  ensureModelAvailable,
  getEmbedding,
  generateCompletion,
  embedText,
  embedDocument,
  chunkDocument,
  rerank,
  fullTextSearch,
  vectorSearch,
  reciprocalRankFusion,
  hybridSearch,
  extractSnippet,
} from './index.ts';

describe('Services Module Exports', () => {
  test('ollama exports are defined', () => {
    expect(ensureModelAvailable).toBeDefined();
    expect(typeof ensureModelAvailable).toBe('function');
    expect(getEmbedding).toBeDefined();
    expect(typeof getEmbedding).toBe('function');
    expect(generateCompletion).toBeDefined();
    expect(typeof generateCompletion).toBe('function');
  });

  test('embedding exports are defined', () => {
    expect(embedText).toBeDefined();
    expect(typeof embedText).toBe('function');
    expect(embedDocument).toBeDefined();
    expect(typeof embedDocument).toBe('function');
    expect(chunkDocument).toBeDefined();
    expect(typeof chunkDocument).toBe('function');
  });

  test('reranking exports are defined', () => {
    expect(rerank).toBeDefined();
    expect(typeof rerank).toBe('function');
  });

  test('search exports are defined', () => {
    expect(fullTextSearch).toBeDefined();
    expect(typeof fullTextSearch).toBe('function');
    expect(vectorSearch).toBeDefined();
    expect(typeof vectorSearch).toBe('function');
    expect(reciprocalRankFusion).toBeDefined();
    expect(typeof reciprocalRankFusion).toBe('function');
    expect(hybridSearch).toBeDefined();
    expect(typeof hybridSearch).toBe('function');
    expect(extractSnippet).toBeDefined();
    expect(typeof extractSnippet).toBe('function');
  });
});
