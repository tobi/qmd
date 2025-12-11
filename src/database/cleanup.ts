/**
 * Database cleanup functions for removing soft-deleted documents
 */

import { Database } from 'bun:sqlite';

export interface CleanupOptions {
  olderThanDays?: number;  // Default: 30
  dryRun?: boolean;
  all?: boolean;
  vacuum?: boolean;
}

export interface CleanupResult {
  documents_deleted: number;
  vectors_deleted: number;
  cache_entries_deleted: number;
  space_reclaimed_mb: number;
}

/**
 * Get database file size in bytes
 */
function getDatabaseSize(db: Database): number {
  const result = db.prepare(`
    SELECT page_count * page_size as size
    FROM pragma_page_count(), pragma_page_size()
  `).get() as { size: number };
  return result.size;
}

/**
 * Preview what would be deleted without actually deleting
 */
function previewCleanup(db: Database, cutoff: Date, options: CleanupOptions): CleanupResult {
  const result: CleanupResult = {
    documents_deleted: 0,
    vectors_deleted: 0,
    cache_entries_deleted: 0,
    space_reclaimed_mb: 0,
  };

  // Count documents that would be deleted
  const docStmt = options.all
    ? db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 0`)
    : db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 0 AND modified_at < ?`);

  const docCount = options.all
    ? (docStmt.get() as { count: number })
    : (docStmt.get(cutoff.toISOString()) as { count: number });
  result.documents_deleted = docCount.count;

  // Count orphaned vectors if --vacuum
  if (options.vacuum) {
    const vecCount = db.prepare(`
      SELECT COUNT(*) as count FROM content_vectors
      WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
    `).get() as { count: number };
    result.vectors_deleted = vecCount.count;

    // Count old cache entries
    const cacheCount = db.prepare(`
      SELECT COUNT(*) as count FROM ollama_cache
      WHERE created_at < datetime('now', '-7 days')
    `).get() as { count: number };
    result.cache_entries_deleted = cacheCount.count;
  }

  return result;
}

/**
 * Permanently delete soft-deleted documents and optionally cleanup orphaned data
 */
export function cleanup(db: Database, options: CleanupOptions = {}): CleanupResult {
  const cutoff = new Date();
  const daysToSubtract = options.olderThanDays ?? 30;
  cutoff.setDate(cutoff.getDate() - daysToSubtract);

  const result: CleanupResult = {
    documents_deleted: 0,
    vectors_deleted: 0,
    cache_entries_deleted: 0,
    space_reclaimed_mb: 0,
  };

  // Dry run - just preview
  if (options.dryRun) {
    return previewCleanup(db, cutoff, options);
  }

  // Get DB size before cleanup
  const sizeBefore = getDatabaseSize(db);

  // Perform cleanup in transaction
  db.transaction(() => {
    // Delete old inactive documents
    const docStmt = options.all
      ? db.prepare(`DELETE FROM documents WHERE active = 0`)
      : db.prepare(`DELETE FROM documents WHERE active = 0 AND modified_at < ?`);

    const docResult = options.all
      ? docStmt.run()
      : docStmt.run(cutoff.toISOString());
    result.documents_deleted = docResult.changes || 0;

    // Delete orphaned vectors and cache (optional)
    if (options.vacuum) {
      // Delete vectors with no corresponding active document
      const vecStmt = db.prepare(`
        DELETE FROM content_vectors
        WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
      `);
      result.vectors_deleted = vecStmt.run().changes || 0;

      // Also delete from vec table if it exists
      const vecTableExists = db.prepare(`
        SELECT COUNT(*) as count FROM sqlite_master
        WHERE type='table' AND name='vectors_vec'
      `).get() as { count: number };

      if (vecTableExists.count > 0) {
        // Get list of active hashes
        const activeHashes = db.prepare(`
          SELECT DISTINCT hash FROM documents WHERE active = 1
        `).all() as Array<{ hash: string }>;

        const activeHashSet = new Set(activeHashes.map(h => h.hash));

        // Delete vec table entries for orphaned vectors
        // (This is a simplified version - in production you'd want to batch this)
        const allVecEntries = db.prepare(`SELECT hash_seq FROM vectors_vec`).all() as Array<{ hash_seq: string }>;
        for (const { hash_seq } of allVecEntries) {
          const hash = hash_seq.split('_')[0];
          if (!activeHashSet.has(hash)) {
            db.prepare(`DELETE FROM vectors_vec WHERE hash_seq = ?`).run(hash_seq);
          }
        }
      }

      // Cleanup old cache entries (older than 7 days)
      const cacheStmt = db.prepare(`
        DELETE FROM ollama_cache
        WHERE created_at < datetime('now', '-7 days')
      `);
      result.cache_entries_deleted = cacheStmt.run().changes || 0;
    }
  })();

  // Reclaim space if requested
  if (options.vacuum) {
    db.exec('VACUUM');
  }

  // Calculate space reclaimed
  const sizeAfter = getDatabaseSize(db);
  result.space_reclaimed_mb = (sizeBefore - sizeAfter) / (1024 * 1024);

  return result;
}
