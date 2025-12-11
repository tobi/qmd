/**
 * Document repository - Data access layer for documents table
 * All queries use prepared statements to prevent SQL injection
 */

import { Database } from 'bun:sqlite';
import type { Document, SearchResult } from '../../models/types.ts';

export class DocumentRepository {
  constructor(private db: Database) {}

  /**
   * Find document by ID
   * @param id - Document ID
   * @returns Document or null if not found
   */
  findById(id: number): Document | null {
    const stmt = this.db.prepare(`
      SELECT id, collection_id, filepath, hash, title, body, active, modified_at, display_path
      FROM documents
      WHERE id = ? AND active = 1
    `);
    return stmt.get(id) as Document | null;
  }

  /**
   * Find document by filepath
   * @param filepath - File path
   * @returns Document or null if not found
   */
  findByFilepath(filepath: string): Document | null {
    const stmt = this.db.prepare(`
      SELECT id, collection_id, filepath, hash, title, body, active, modified_at, display_path
      FROM documents
      WHERE filepath = ? AND active = 1
    `);
    return stmt.get(filepath) as Document | null;
  }

  /**
   * Find document by hash
   * @param hash - Content hash
   * @returns Document or null if not found
   */
  findByHash(hash: string): Document | null {
    const stmt = this.db.prepare(`
      SELECT id, collection_id, filepath, hash, title, body, active, modified_at, display_path
      FROM documents
      WHERE hash = ? AND active = 1
      LIMIT 1
    `);
    return stmt.get(hash) as Document | null;
  }

  /**
   * Find all documents in a collection
   * @param collectionId - Collection ID
   * @returns Array of documents
   */
  findByCollection(collectionId: number): Document[] {
    const stmt = this.db.prepare(`
      SELECT id, collection_id, filepath, hash, title, body, active, modified_at, display_path
      FROM documents
      WHERE collection_id = ? AND active = 1
      ORDER BY filepath
    `);
    return stmt.all(collectionId) as Document[];
  }

  /**
   * Search documents using FTS5 (BM25)
   * @param query - FTS5 query string
   * @param limit - Maximum number of results
   * @returns Array of search results with scores
   */
  searchFTS(query: string, limit: number = 20): SearchResult[] {
    // BM25 weights: title=10, body=1
    const stmt = this.db.prepare(`
      SELECT d.filepath, d.display_path, d.title, d.body, bm25(documents_fts, 10.0, 1.0) as score
      FROM documents_fts f
      JOIN documents d ON d.id = f.rowid
      WHERE documents_fts MATCH ? AND d.active = 1
      ORDER BY score
      LIMIT ?
    `);

    const results = stmt.all(query, limit) as { filepath: string; display_path: string; title: string; body: string; score: number }[];

    return results.map(r => ({
      file: r.filepath,
      displayPath: r.display_path,
      title: r.title,
      body: r.body,
      score: this.normalizeBM25(r.score),
      source: 'fts' as const,
    }));
  }

  /**
   * Search documents using vector similarity
   * @param embedding - Query embedding vector
   * @param limit - Maximum number of results
   * @returns Array of search results with scores
   */
  searchVector(embedding: Float32Array, limit: number = 20): SearchResult[] {
    const stmt = this.db.prepare(`
      SELECT
        cv.hash,
        cv.seq,
        cv.pos,
        d.filepath,
        d.display_path,
        d.title,
        d.body,
        vec_distance_cosine(v.embedding, ?) as distance
      FROM vectors_vec v
      JOIN content_vectors cv ON v.hash_seq = cv.hash || '_' || cv.seq
      JOIN documents d ON d.hash = cv.hash
      WHERE d.active = 1
      ORDER BY distance
      LIMIT ?
    `);

    const results = stmt.all(embedding, limit) as {
      hash: string;
      seq: number;
      pos: number;
      filepath: string;
      display_path: string;
      title: string;
      body: string;
      distance: number;
    }[];

    return results.map(r => ({
      file: r.filepath,
      displayPath: r.display_path,
      title: r.title,
      body: r.body,
      score: 1 - r.distance, // Convert distance to similarity
      source: 'vec' as const,
      chunkPos: r.pos,
    }));
  }

  /**
   * Insert a new document
   * @param doc - Document data (without id)
   * @returns Inserted document ID
   */
  insert(doc: Omit<Document, 'id'>): number {
    const stmt = this.db.prepare(`
      INSERT INTO documents (
        collection_id, name, title, hash, filepath, display_path,
        body, created_at, modified_at, active
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    `);

    const result = stmt.run(
      doc.collection_id,
      doc.filepath.split('/').pop() || '',
      doc.title,
      doc.hash,
      doc.filepath,
      doc.display_path || '',
      doc.body,
      doc.created_at || new Date().toISOString(),
      doc.modified_at || new Date().toISOString()
    );

    return Number(result.lastInsertRowid);
  }

  /**
   * Update document display path
   * @param id - Document ID
   * @param displayPath - New display path
   */
  updateDisplayPath(id: number, displayPath: string): void {
    const stmt = this.db.prepare(`
      UPDATE documents
      SET display_path = ?
      WHERE id = ?
    `);
    stmt.run(displayPath, id);
  }

  /**
   * Mark document as inactive (soft delete)
   * @param id - Document ID
   */
  deactivate(id: number): void {
    const stmt = this.db.prepare(`
      UPDATE documents
      SET active = 0
      WHERE id = ?
    `);
    stmt.run(id);
  }

  /**
   * Get all documents with empty display paths
   * @returns Array of documents needing display paths
   */
  findWithoutDisplayPath(): Array<{ id: number; filepath: string; pwd: string }> {
    const stmt = this.db.prepare(`
      SELECT d.id, d.filepath, c.pwd
      FROM documents d
      JOIN collections c ON c.id = d.collection_id
      WHERE d.active = 1 AND (d.display_path IS NULL OR d.display_path = '')
      ORDER BY c.id, d.filepath
    `);
    return stmt.all() as Array<{ id: number; filepath: string; pwd: string }>;
  }

  /**
   * Get all existing display paths (for uniqueness checking)
   * @returns Set of display paths
   */
  getExistingDisplayPaths(): Set<string> {
    const stmt = this.db.prepare(`
      SELECT display_path
      FROM documents
      WHERE active = 1 AND display_path != ''
    `);
    const results = stmt.all() as { display_path: string }[];
    return new Set(results.map(r => r.display_path));
  }

  /**
   * Get total count of active documents
   * @returns Document count
   */
  count(): number {
    const stmt = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM documents
      WHERE active = 1
    `);
    return (stmt.get() as { count: number }).count;
  }

  /**
   * Normalize BM25 score (negative) to 0-1 range
   * @param rawScore - Raw BM25 score
   * @returns Normalized score
   */
  private normalizeBM25(rawScore: number): number {
    const normalized = 1 / (1 + Math.abs(rawScore) / 10);
    return Math.round(normalized * 1000) / 1000;
  }
}
