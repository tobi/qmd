/**
 * Context management operations for store.ts
 * These will be integrated into store.ts
 */

import { Database } from "bun:sqlite";

// =============================================================================
// Context Management Operations
// =============================================================================

/**
 * Insert or update a context for a specific collection and path prefix.
 */
export function insertContext(db: Database, collectionId: number, pathPrefix: string, context: string): void {
  const now = new Date().toISOString();
  db.prepare(`
    INSERT INTO path_contexts (collection_id, path_prefix, context, created_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(collection_id, path_prefix) DO UPDATE SET context = excluded.context
  `).run(collectionId, pathPrefix, context, now);
}

/**
 * Delete a context for a specific collection and path prefix.
 * Returns the number of contexts deleted.
 */
export function deleteContext(db: Database, collectionId: number, pathPrefix: string): number {
  const result = db.prepare(`
    DELETE FROM path_contexts
    WHERE collection_id = ? AND path_prefix = ?
  `).run(collectionId, pathPrefix);
  return result.changes;
}

/**
 * Delete all global contexts (contexts with empty path_prefix).
 * Returns the number of contexts deleted.
 */
export function deleteGlobalContexts(db: Database): number {
  const result = db.prepare(`DELETE FROM path_contexts WHERE path_prefix = ''`).run();
  return result.changes;
}

/**
 * List all contexts, grouped by collection.
 * Returns contexts ordered by collection name, then by path prefix length (longest first).
 */
export function listPathContexts(db: Database): { collection_name: string; path_prefix: string; context: string }[] {
  const contexts = db.prepare(`
    SELECT c.name as collection_name, pc.path_prefix, pc.context
    FROM path_contexts pc
    JOIN collections c ON c.id = pc.collection_id
    ORDER BY c.name, LENGTH(pc.path_prefix) DESC, pc.path_prefix
  `).all() as { collection_name: string; path_prefix: string; context: string }[];
  return contexts;
}

/**
 * Get all collections (id and name).
 */
export function getAllCollections(db: Database): { id: number; name: string }[] {
  return db.prepare(`SELECT id, name FROM collections`).all() as { id: number; name: string }[];
}
