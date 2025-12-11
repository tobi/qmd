/**
 * Search service - Combines FTS, vector, and hybrid search
 */

import { Database } from 'bun:sqlite';
import type { SearchResult, RankedResult } from '../models/types.ts';
import { DocumentRepository, PathContextRepository } from '../database/repositories/index.ts';
import { embedText } from './embedding.ts';
import { rerank } from './reranking.ts';

/**
 * Build FTS5 query from user input
 * @param input - User query
 * @returns FTS5 query string or null if invalid
 */
function buildFTS5Query(input: string): string | null {
  const trimmed = input.trim();
  if (!trimmed) return null;

  // If already has FTS5 operators, use as-is
  if (/[*"{}]/.test(trimmed) || /\b(AND|OR|NOT)\b/.test(trimmed)) {
    return trimmed;
  }

  // Simple quote escape
  const escaped = trimmed.replace(/"/g, '""');
  return `"${escaped}"`;
}

/**
 * Extract snippet from text around query terms
 * @param text - Full text
 * @param query - Search query
 * @param maxLength - Maximum snippet length
 * @returns Snippet with context
 */
export function extractSnippet(
  text: string,
  query: string,
  maxLength: number = 300
): { snippet: string; position: number } {
  const lowerText = text.toLowerCase();
  const lowerQuery = query.toLowerCase();

  // Find first occurrence of query
  const position = lowerText.indexOf(lowerQuery);

  if (position === -1) {
    // Query not found, return beginning
    return {
      snippet: text.slice(0, maxLength),
      position: 0,
    };
  }

  // Extract context around match
  const start = Math.max(0, position - 100);
  const end = Math.min(text.length, position + maxLength);

  let snippet = text.slice(start, end);
  if (start > 0) snippet = '...' + snippet;
  if (end < text.length) snippet = snippet + '...';

  return { snippet, position };
}

/**
 * Full-text search using BM25
 * @param db - Database instance
 * @param query - Search query
 * @param limit - Maximum results
 * @returns Search results with scores
 */
export async function fullTextSearch(
  db: Database,
  query: string,
  limit: number = 20
): Promise<SearchResult[]> {
  const docRepo = new DocumentRepository(db);
  const pathCtxRepo = new PathContextRepository(db);

  const ftsQuery = buildFTS5Query(query);
  if (!ftsQuery) return [];

  const results = docRepo.searchFTS(ftsQuery, limit);

  // Add context to results
  return results.map(r => ({
    ...r,
    context: pathCtxRepo.findForPath(r.file)?.context || null,
  }));
}

/**
 * Vector similarity search
 * @param db - Database instance
 * @param query - Search query
 * @param model - Embedding model
 * @param limit - Maximum results
 * @returns Search results with scores
 */
export async function vectorSearch(
  db: Database,
  query: string,
  model: string,
  limit: number = 20
): Promise<SearchResult[]> {
  const docRepo = new DocumentRepository(db);
  const pathCtxRepo = new PathContextRepository(db);

  // Embed query
  const queryEmbedding = await embedText(query, model, true);

  // Search vectors
  const results = docRepo.searchVector(queryEmbedding, limit);

  // Add context to results
  return results.map(r => ({
    ...r,
    context: pathCtxRepo.findForPath(r.file)?.context || null,
  }));
}

/**
 * Reciprocal Rank Fusion - Combine multiple ranked lists
 * @param lists - Arrays of ranked results
 * @param weights - Weight for each list (default: equal)
 * @param k - RRF constant (default: 60)
 * @returns Fused ranked results
 */
export function reciprocalRankFusion(
  lists: SearchResult[][],
  weights?: number[],
  k: number = 60
): SearchResult[] {
  const fileScores = new Map<string, { score: number; result: SearchResult }>();

  for (let listIdx = 0; listIdx < lists.length; listIdx++) {
    const list = lists[listIdx];
    const weight = weights ? weights[listIdx] : 1.0;

    for (let rank = 0; rank < list.length; rank++) {
      const result = list[rank];
      const rrfScore = weight / (k + rank + 1);

      const existing = fileScores.get(result.file);
      if (existing) {
        existing.score += rrfScore;
      } else {
        fileScores.set(result.file, {
          score: rrfScore,
          result,
        });
      }
    }
  }

  // Sort by RRF score descending
  const fused = Array.from(fileScores.values())
    .sort((a, b) => b.score - a.score)
    .map(item => ({
      ...item.result,
      score: item.score,
    }));

  return fused;
}

/**
 * Hybrid search with RRF fusion and reranking
 * @param db - Database instance
 * @param query - Search query
 * @param embedModel - Embedding model
 * @param rerankModel - Reranking model
 * @param limit - Final result limit
 * @returns Reranked search results
 */
export async function hybridSearch(
  db: Database,
  query: string,
  embedModel: string,
  rerankModel: string,
  limit: number = 10
): Promise<RankedResult[]> {
  // Get FTS and vector results
  const [ftsResults, vecResults] = await Promise.all([
    fullTextSearch(db, query, 50),
    vectorSearch(db, query, embedModel, 50),
  ]);

  // Apply RRF fusion (weight original query 2x, expansion 1x)
  const weights = [2.0, 1.0];
  const fused = reciprocalRankFusion([ftsResults, vecResults], weights);

  // Take top 30 candidates for reranking
  const candidates = fused.slice(0, 30);

  // Rerank
  const reranked = await rerank(
    query,
    candidates.map(c => ({ file: c.file, text: c.body })),
    rerankModel,
    db
  );

  // Blend RRF and rerank scores
  const candidateMap = new Map(
    candidates.map(c => [c.file, { displayPath: c.displayPath, title: c.title, body: c.body, context: c.context }])
  );
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

  const finalResults = reranked.map(r => {
    const rrfRank = rrfRankMap.get(r.file) || candidates.length;
    let rrfWeight: number;
    if (rrfRank <= 3) rrfWeight = 0.75;
    else if (rrfRank <= 10) rrfWeight = 0.60;
    else rrfWeight = 0.40;

    const rrfScore = 1 / rrfRank;
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * r.score;
    const candidate = candidateMap.get(r.file);

    return {
      file: candidate?.displayPath || "",
      title: candidate?.title || "",
      score: Math.round(blendedScore * 100) / 100,
      context: candidate?.context || null,
      snippet: extractSnippet(candidate?.body || "", query, 300).snippet,
    };
  });

  return finalResults.slice(0, limit);
}
