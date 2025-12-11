/**
 * Zod schemas for runtime type validation
 *
 * These schemas provide runtime validation and serve as the single source
 * of truth for type definitions. Types can be inferred from schemas using
 * z.infer<typeof Schema>.
 */

import { z } from 'zod';

/**
 * Collection schema - Groups documents by directory + glob pattern
 */
export const CollectionSchema = z.object({
  id: z.number().int().positive(),
  pwd: z.string().min(1),
  glob_pattern: z.string().min(1),
  created_at: z.string().datetime(),
  context: z.string().optional(),
});

/**
 * Document schema - Indexed markdown files
 */
export const DocumentSchema = z.object({
  id: z.number().int().positive(),
  collection_id: z.number().int().positive(),
  name: z.string().optional(),
  filepath: z.string().min(1),
  hash: z.string().regex(/^[a-f0-9]{64}$/, 'Must be valid SHA-256 hash'),
  title: z.string(),
  body: z.string(),
  active: z.number().int().min(0).max(1),
  created_at: z.string().datetime().optional(),
  modified_at: z.string().datetime(),
  display_path: z.string().optional(),
});

/**
 * ContentVector schema - Embedding vectors for document chunks
 */
export const ContentVectorSchema = z.object({
  hash: z.string().regex(/^[a-f0-9]{64}$/, 'Must be valid SHA-256 hash'),
  seq: z.number().int().min(0),
  pos: z.number().int().min(0),
  embedding: z.instanceof(Float32Array),
});

/**
 * PathContext schema - Contextual metadata for file paths
 */
export const PathContextSchema = z.object({
  id: z.number().int().positive().optional(),
  path_prefix: z.string().min(1),
  context: z.string(),
  created_at: z.string().datetime().optional(),
});

/**
 * OllamaCache schema - Cache for Ollama API responses
 */
export const OllamaCacheSchema = z.object({
  hash: z.string().min(1),
  result: z.string(),
  created_at: z.string().datetime(),
});

/**
 * SearchResult schema - Results from FTS or vector search
 */
export const SearchResultSchema = z.object({
  file: z.string(),
  displayPath: z.string(),
  title: z.string(),
  body: z.string(),
  score: z.number().min(0).max(1),
  source: z.enum(['fts', 'vec']),
  chunkPos: z.number().int().min(0).optional(),
});

/**
 * RankedResult schema - Reranked hybrid search results
 */
export const RankedResultSchema = z.object({
  file: z.string(),
  displayPath: z.string(),
  title: z.string(),
  body: z.string(),
  score: z.number(),
});

/**
 * OutputOptions schema - Display configuration
 */
export const OutputOptionsSchema = z.object({
  format: z.enum(['cli', 'csv', 'md', 'xml', 'files', 'json']),
  full: z.boolean(),
  limit: z.number().int().positive(),
  minScore: z.number().min(0).max(1),
  all: z.boolean().optional(),
});

/**
 * RerankResponse schema - Ollama reranking API response
 */
export const RerankResponseSchema = z.object({
  model: z.string(),
  created_at: z.string(),
  response: z.string(),
  done: z.boolean(),
  done_reason: z.string().optional(),
  total_duration: z.number().optional(),
  load_duration: z.number().optional(),
  prompt_eval_count: z.number().optional(),
  prompt_eval_duration: z.number().optional(),
  eval_count: z.number().optional(),
  eval_duration: z.number().optional(),
  logprobs: z.array(z.object({
    token: z.string(),
    logprob: z.number(),
  })).optional(),
});

// Export inferred types (alternative to manual type definitions)
// These can gradually replace the types in types.ts
export type Collection = z.infer<typeof CollectionSchema>;
export type Document = z.infer<typeof DocumentSchema>;
export type ContentVector = z.infer<typeof ContentVectorSchema>;
export type PathContext = z.infer<typeof PathContextSchema>;
export type OllamaCache = z.infer<typeof OllamaCacheSchema>;
export type SearchResult = z.infer<typeof SearchResultSchema>;
export type RankedResult = z.infer<typeof RankedResultSchema>;
export type OutputOptions = z.infer<typeof OutputOptionsSchema>;
export type RerankResponse = z.infer<typeof RerankResponseSchema>;
