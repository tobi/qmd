/**
 * Performance optimization utilities for database operations
 */

import { Database } from 'bun:sqlite';

/**
 * Optimize database query planner by running ANALYZE
 * Should be called after large indexing/update operations
 */
export function analyzeDatabase(db: Database): void {
  db.exec('ANALYZE');
}

/**
 * Get database statistics
 */
export function getDatabaseStats(db: Database): {
  page_count: number;
  page_size: number;
  size_mb: number;
} {
  const result = db.prepare(`
    SELECT page_count, page_size,
           (page_count * page_size) / (1024.0 * 1024.0) as size_mb
    FROM pragma_page_count(), pragma_page_size()
  `).get() as { page_count: number; page_size: number; size_mb: number };

  return result;
}

/**
 * Check if database should be analyzed (heuristic: >1000 docs or large changes)
 */
export function shouldAnalyze(db: Database, documentsChanged: number = 0): boolean {
  const totalDocs = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number };

  // Analyze if:
  // - Changed more than 100 documents
  // - Database has more than 1000 documents and hasn't been analyzed recently
  if (documentsChanged > 100) {
    return true;
  }

  if (totalDocs.count > 1000) {
    // Check when last analyzed (sqlite_stat1 table is created by ANALYZE)
    const hasStats = db.prepare(`
      SELECT COUNT(*) as count FROM sqlite_master
      WHERE type='table' AND name='sqlite_stat1'
    `).get() as { count: number };

    // If no stats table, definitely analyze
    return hasStats.count === 0;
  }

  return false;
}

/**
 * Batch insert documents (transaction wrapper)
 */
export function batchInsertDocuments<T>(
  db: Database,
  items: T[],
  insertFn: (item: T) => void
): number {
  let inserted = 0;

  db.transaction(() => {
    for (const item of items) {
      insertFn(item);
      inserted++;
    }
  })();

  return inserted;
}

/**
 * Get query performance hints
 */
export function getPerformanceHints(db: Database): string[] {
  const hints: string[] = [];

  // Check for missing ANALYZE
  const hasStats = db.prepare(`
    SELECT COUNT(*) as count FROM sqlite_master
    WHERE type='table' AND name='sqlite_stat1'
  `).get() as { count: number };

  if (hasStats.count === 0) {
    hints.push("Run ANALYZE to optimize query planner (happens automatically after large operations)");
  }

  // Check database size
  const stats = getDatabaseStats(db);
  if (stats.size_mb > 100) {
    hints.push(`Database size: ${stats.size_mb.toFixed(1)} MB - consider periodic cleanup`);
  }

  // Check WAL mode
  const walMode = db.prepare('PRAGMA journal_mode').get() as { journal_mode: string };
  if (walMode.journal_mode.toUpperCase() !== 'WAL') {
    hints.push("Enable WAL mode for better concurrency (handled automatically)");
  }

  return hints;
}
