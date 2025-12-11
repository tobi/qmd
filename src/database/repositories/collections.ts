/**
 * Collection repository - Data access layer for collections table
 * All queries use prepared statements to prevent SQL injection
 */

import { Database } from 'bun:sqlite';
import type { Collection } from '../../models/types.ts';

export class CollectionRepository {
  constructor(private db: Database) {}

  /**
   * Find collection by ID
   * @param id - Collection ID
   * @returns Collection or null if not found
   */
  findById(id: number): Collection | null {
    const stmt = this.db.prepare(`
      SELECT id, pwd, glob_pattern, created_at
      FROM collections
      WHERE id = ?
    `);
    return stmt.get(id) as Collection | null;
  }

  /**
   * Find collection by pwd and glob pattern
   * @param pwd - Working directory
   * @param globPattern - Glob pattern
   * @returns Collection or null if not found
   */
  findByPwdAndPattern(pwd: string, globPattern: string): Collection | null {
    const stmt = this.db.prepare(`
      SELECT id, pwd, glob_pattern, created_at
      FROM collections
      WHERE pwd = ? AND glob_pattern = ?
    `);
    return stmt.get(pwd, globPattern) as Collection | null;
  }

  /**
   * Get all collections
   * @returns Array of collections
   */
  findAll(): Collection[] {
    const stmt = this.db.prepare(`
      SELECT id, pwd, glob_pattern, created_at
      FROM collections
      ORDER BY created_at DESC
    `);
    return stmt.all() as Collection[];
  }

  /**
   * Get all collections with document counts
   * @returns Array of collections with active document counts
   */
  findAllWithCounts(): Array<Collection & { active_count: number; last_doc_update: string | null }> {
    const stmt = this.db.prepare(`
      SELECT c.id, c.pwd, c.glob_pattern, c.created_at,
             COUNT(d.id) as active_count,
             MAX(d.modified_at) as last_doc_update
      FROM collections c
      LEFT JOIN documents d ON d.collection_id = c.id AND d.active = 1
      GROUP BY c.id
      ORDER BY last_doc_update DESC
    `);
    return stmt.all() as Array<Collection & { active_count: number; last_doc_update: string | null }>;
  }

  /**
   * Insert a new collection
   * @param pwd - Working directory
   * @param globPattern - Glob pattern
   * @returns Inserted collection ID
   */
  insert(pwd: string, globPattern: string): number {
    const stmt = this.db.prepare(`
      INSERT INTO collections (pwd, glob_pattern, created_at)
      VALUES (?, ?, ?)
    `);

    const result = stmt.run(pwd, globPattern, new Date().toISOString());
    return Number(result.lastInsertRowid);
  }

  /**
   * Delete a collection (and cascade to documents)
   * @param id - Collection ID
   */
  delete(id: number): void {
    // First deactivate all documents in this collection
    const deactivateStmt = this.db.prepare(`
      UPDATE documents
      SET active = 0
      WHERE collection_id = ?
    `);
    deactivateStmt.run(id);

    // Then delete the collection
    const deleteStmt = this.db.prepare(`
      DELETE FROM collections
      WHERE id = ?
    `);
    deleteStmt.run(id);
  }

  /**
   * Get total count of collections
   * @returns Collection count
   */
  count(): number {
    const stmt = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM collections
    `);
    return (stmt.get() as { count: number }).count;
  }
}
