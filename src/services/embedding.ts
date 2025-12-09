/**
 * Embedding service for vector generation
 */

import { Database } from 'bun:sqlite';
import { getEmbedding } from './ollama.ts';
import { VectorRepository } from '../database/repositories/index.ts';
import { ensureVecTable } from '../database/db.ts';

/**
 * Embed a document chunk
 * @param text - Text to embed
 * @param model - Embedding model
 * @param isQuery - True if query (vs document)
 * @param title - Document title (for documents)
 * @returns Embedding vector as Float32Array
 */
export async function embedText(
  text: string,
  model: string,
  isQuery: boolean = false,
  title?: string
): Promise<Float32Array> {
  const embedding = await getEmbedding(text, model, isQuery, title);
  return new Float32Array(embedding);
}

/**
 * Embed document and store vectors
 * @param db - Database instance
 * @param hash - Document hash
 * @param chunks - Text chunks to embed
 * @param model - Embedding model
 */
export async function embedDocument(
  db: Database,
  hash: string,
  chunks: Array<{ text: string; pos: number; title?: string }>,
  model: string
): Promise<void> {
  const vectorRepo = new VectorRepository(db);

  // Delete existing vectors for this hash
  vectorRepo.deleteByHash(hash);

  // Get embedding dimensions from first chunk
  const firstEmbedding = await embedText(chunks[0].text, model, false, chunks[0].title);
  ensureVecTable(db, firstEmbedding.length);

  // Embed all chunks
  for (let seq = 0; seq < chunks.length; seq++) {
    const chunk = chunks[seq];
    const embedding = seq === 0 ? firstEmbedding : await embedText(chunk.text, model, false, chunk.title);
    vectorRepo.insert(hash, seq, chunk.pos, embedding, model);
  }
}

/**
 * Chunk document into smaller pieces for embedding
 * @param text - Full document text
 * @param chunkSize - Size of each chunk in characters
 * @param overlap - Overlap between chunks in characters
 * @returns Array of chunks with positions
 */
export function chunkDocument(
  text: string,
  chunkSize: number = 1000,
  overlap: number = 200
): Array<{ text: string; pos: number }> {
  const chunks: Array<{ text: string; pos: number }> = [];

  if (text.length <= chunkSize) {
    // Document fits in one chunk
    return [{ text, pos: 0 }];
  }

  let pos = 0;
  while (pos < text.length) {
    const end = Math.min(pos + chunkSize, text.length);
    const chunkText = text.slice(pos, end);
    chunks.push({ text: chunkText, pos });

    // Move forward, accounting for overlap
    pos += chunkSize - overlap;

    // Avoid infinite loop on last chunk
    if (pos + overlap >= text.length) break;
  }

  return chunks;
}
