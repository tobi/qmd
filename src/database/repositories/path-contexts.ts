/**
 * PathContext repository - Data access layer for path_contexts table
 * All queries use prepared statements to prevent SQL injection
 */

import { Database } from 'bun:sqlite';
import type { PathContext } from '../../models/types.ts';

export class PathContextRepository {
  constructor(private db: Database) {}

  /**
   * Find context for a file path (longest matching prefix)
   * @param filepath - File path to find context for
   * @returns Path context or null if not found
   */
  findForPath(filepath: string): PathContext | null {
    const stmt = this.db.prepare(`
      SELECT id, path_prefix, context, created_at
      FROM path_contexts
      WHERE ? LIKE path_prefix || '%'
      ORDER BY LENGTH(path_prefix) DESC
      LIMIT 1
    `);
    return stmt.get(filepath) as PathContext | null;
  }

  /**
   * Get all path contexts
   * @returns Array of path contexts
   */
  findAll(): PathContext[] {
    const stmt = this.db.prepare(`
      SELECT id, path_prefix, context, created_at
      FROM path_contexts
      ORDER BY path_prefix
    `);
    return stmt.all() as PathContext[];
  }

  /**
   * Insert or update a path context
   * @param pathPrefix - Path prefix (directory or file pattern)
   * @param contextText - Context description
   */
  upsert(pathPrefix: string, contextText: string): void {
    const stmt = this.db.prepare(`
      INSERT INTO path_contexts (path_prefix, context, created_at)
      VALUES (?, ?, ?)
      ON CONFLICT(path_prefix) DO UPDATE SET context = excluded.context
    `);
    stmt.run(pathPrefix, contextText, new Date().toISOString());
  }

  /**
   * Delete a path context
   * @param pathPrefix - Path prefix to delete
   */
  delete(pathPrefix: string): void {
    const stmt = this.db.prepare(`
      DELETE FROM path_contexts
      WHERE path_prefix = ?
    `);
    stmt.run(pathPrefix);
  }

  /**
   * Get total count of path contexts
   * @returns Path context count
   */
  count(): number {
    const stmt = this.db.prepare(`
      SELECT COUNT(*) as count
      FROM path_contexts
    `);
    return (stmt.get() as { count: number }).count;
  }
}
