/**
 * Vector repository - Data access layer for embeddings and vector search
 * All queries use prepared statements to prevent SQL injection
 */

import { Database } from 'bun:sqlite';
import type { ContentVector } from '../../models/types.ts';

export class VectorRepository {
  constructor(private db: Database) {}

  /**
   * Find all vectors for a document hash
   * @param hash - Document content hash
   * @returns Array of content vectors
   */
  findByHash(hash: string): ContentVector[] {
    const stmt = this.db.prepare(`
      SELECT hash, seq, pos, embedding
      FROM content_vectors cv
      JOIN vectors_vec v ON v.hash_seq = cv.hash || '_' || cv.seq
      WHERE hash = ?
      ORDER BY seq
    `);
    return stmt.all(hash) as ContentVector[];
  }

  /**
   * Find a specific chunk vector
   * @param hash - Document content hash
   * @param seq - Chunk sequence number
   * @returns Content vector or null if not found
   */
  findByHashAndSeq(hash: string, seq: number): ContentVector | null {
    const stmt = this.db.prepare(`
      SELECT hash, seq, pos, embedding
      FROM content_vectors cv
      JOIN vectors_vec v ON v.hash_seq = cv.hash || '_' || cv.seq
      WHERE hash = ? AND seq = ?
    `);
    return stmt.get(hash, seq) as ContentVector | null;
  }

  /**
   * Check if a document has embeddings
   * @param hash - Document content hash
   * @returns True if document has at least one embedding
   */
  hasEmbedding(hash: string): boolean {
    const stmt = this.db.prepare(`
      SELECT 1 FROM content_vectors
      WHERE hash = ?
      LIMIT 1
    `);
    return !!stmt.get(hash);
  }

  /**
   * Insert vector embedding for a document chunk
   * @param hash - Document content hash
   * @param seq - Chunk sequence number
   * @param pos - Character position in document
   * @param embedding - Embedding vector
   * @param model - Model name used for embedding
   */
  insert(hash: string, seq: number, pos: number, embedding: Float32Array, model: string): void {
    // Insert into content_vectors table
    const cvStmt = this.db.prepare(`
      INSERT INTO content_vectors (hash, seq, pos, model, embedded_at)
      VALUES (?, ?, ?, ?, ?)
    `);
    cvStmt.run(hash, seq, pos, model, new Date().toISOString());

    // Insert into vectors_vec table
    const hashSeq = `${hash}_${seq}`;
    const vecStmt = this.db.prepare(`
      INSERT INTO vectors_vec (hash_seq, embedding)
      VALUES (?, ?)
    `);
    vecStmt.run(hashSeq, embedding);
  }

  /**
   * Delete all vectors for a document hash
   * @param hash - Document content hash
   */
  deleteByHash(hash: string): void {
    // Get all seq numbers for this hash
    const seqs = this.db.prepare(`
      SELECT seq FROM content_vectors WHERE hash = ?
    `).all(hash) as { seq: number }[];

    // Delete from vectors_vec
    for (const { seq } of seqs) {
      const hashSeq = `${hash}_${seq}`;
      this.db.prepare(`DELETE FROM vectors_vec WHERE hash_seq = ?`).run(hashSeq);
    }

    // Delete from content_vectors
    this.db.prepare(`DELETE FROM content_vectors WHERE hash = ?`).run(hash);
  }

  /**
   * Get count of documents with embeddings
   * @returns Number of unique document hashes with embeddings
   */
  countDocumentsWithEmbeddings(): number {
    const stmt = this.db.prepare(`
      SELECT COUNT(DISTINCT hash) as count
      FROM content_vectors
    `);
    return (stmt.get() as { count: number }).count;
  }

  /**
   * Get count of total vector chunks
   * @returns Total number of embedded chunks
   */
  countVectorChunks(): number {
    const stmt = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM content_vectors
    `);
    return (stmt.get() as { count: number }).count;
  }
}
