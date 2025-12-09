/**
 * Database integrity checks and auto-repair functions
 */

import { Database } from 'bun:sqlite';

export interface IntegrityIssue {
  severity: 'error' | 'warning' | 'info';
  type: string;
  message: string;
  details?: string[];
  fixable: boolean;
  fix?: () => void;
}

/**
 * Check for orphaned vectors (vectors with no corresponding document)
 */
export function checkOrphanedVectors(db: Database): IntegrityIssue | null {
  const orphaned = db.prepare(`
    SELECT DISTINCT cv.hash, COUNT(*) as chunk_count
    FROM content_vectors cv
    LEFT JOIN documents d ON d.hash = cv.hash
    WHERE d.hash IS NULL
    GROUP BY cv.hash
  `).all() as Array<{ hash: string; chunk_count: number }>;

  if (orphaned.length === 0) return null;

  const totalChunks = orphaned.reduce((sum, o) => sum + o.chunk_count, 0);

  return {
    severity: 'warning',
    type: 'orphaned_vectors',
    message: `${orphaned.length} orphaned vector set(s) (${totalChunks} total chunks)`,
    details: orphaned.slice(0, 5).map(o => `Hash: ${o.hash.substring(0, 12)}... (${o.chunk_count} chunks)`),
    fixable: true,
    fix: () => {
      // Check if vectors_vec table exists
      const vecTableExists = db.prepare(`
        SELECT COUNT(*) as count FROM sqlite_master
        WHERE type='table' AND name='vectors_vec'
      `).get() as { count: number };

      for (const { hash } of orphaned) {
        db.prepare(`DELETE FROM content_vectors WHERE hash = ?`).run(hash);
        // Also delete from vec table if it exists
        if (vecTableExists.count > 0) {
          db.prepare(`DELETE FROM vectors_vec WHERE hash_seq LIKE ?`).run(`${hash}_%`);
        }
      }
    },
  };
}

/**
 * Check for partial embeddings (incomplete chunk sequences)
 */
export function checkPartialEmbeddings(db: Database): IntegrityIssue | null {
  const partial = db.prepare(`
    SELECT hash, COUNT(*) as chunk_count, MIN(seq) as min_seq, MAX(seq) as max_seq
    FROM content_vectors
    GROUP BY hash
    HAVING MIN(seq) != 0 OR MAX(seq) != COUNT(*) - 1
  `).all() as Array<{ hash: string; chunk_count: number; min_seq: number; max_seq: number }>;

  if (partial.length === 0) return null;

  return {
    severity: 'error',
    type: 'partial_embeddings',
    message: `${partial.length} document(s) with incomplete chunk sequences`,
    details: partial.slice(0, 5).map(p => `Hash: ${p.hash.substring(0, 12)}... (chunks: ${p.chunk_count}, seq: ${p.min_seq}-${p.max_seq})`),
    fixable: true,
    fix: () => {
      // Check if vectors_vec table exists
      const vecTableExists = db.prepare(`
        SELECT COUNT(*) as count FROM sqlite_master
        WHERE type='table' AND name='vectors_vec'
      `).get() as { count: number };

      for (const { hash} of partial) {
        db.prepare(`DELETE FROM content_vectors WHERE hash = ?`).run(hash);
        if (vecTableExists.count > 0) {
          db.prepare(`DELETE FROM vectors_vec WHERE hash_seq LIKE ?`).run(`${hash}_%`);
        }
      }
    },
  };
}

/**
 * Check for display path collisions (duplicate display_paths)
 */
export function checkDisplayPathCollisions(db: Database): IntegrityIssue | null {
  const collisions = db.prepare(`
    SELECT display_path, COUNT(*) as count, GROUP_CONCAT(filepath) as paths
    FROM documents
    WHERE active = 1 AND display_path != ''
    GROUP BY display_path
    HAVING COUNT(*) > 1
  `).all() as Array<{ display_path: string; count: number; paths: string }>;

  if (collisions.length === 0) return null;

  return {
    severity: 'error',
    type: 'display_path_collisions',
    message: `${collisions.length} display path collision(s) detected`,
    details: collisions.slice(0, 5).map(c => `"${c.display_path}" -> ${c.count} files`),
    fixable: false, // Requires re-indexing
  };
}

/**
 * Check for orphaned documents (documents referencing non-existent collections)
 */
export function checkOrphanedDocuments(db: Database): IntegrityIssue | null {
  const orphaned = db.prepare(`
    SELECT d.id, d.filepath, d.collection_id
    FROM documents d
    LEFT JOIN collections c ON c.id = d.collection_id
    WHERE c.id IS NULL
  `).all() as Array<{ id: number; filepath: string; collection_id: number }>;

  if (orphaned.length === 0) return null;

  return {
    severity: 'error',
    type: 'orphaned_documents',
    message: `${orphaned.length} orphaned document(s) (collection deleted)`,
    details: orphaned.slice(0, 5).map(d => `Doc ${d.id}: ${d.filepath}`),
    fixable: true,
    fix: () => {
      // Deactivate orphaned documents
      for (const { id } of orphaned) {
        db.prepare(`UPDATE documents SET active = 0 WHERE id = ?`).run(id);
      }
    },
  };
}

/**
 * Check FTS index consistency (documents missing from FTS)
 */
export function checkFTSConsistency(db: Database): IntegrityIssue | null {
  const missing = db.prepare(`
    SELECT COUNT(*) as missing_count
    FROM documents d
    LEFT JOIN documents_fts f ON f.rowid = d.id
    WHERE d.active = 1 AND f.rowid IS NULL
  `).get() as { missing_count: number };

  if (missing.missing_count === 0) return null;

  return {
    severity: 'error',
    type: 'fts_inconsistency',
    message: `${missing.missing_count} document(s) missing from FTS index`,
    fixable: true,
    fix: () => {
      // Rebuild FTS index
      db.exec(`INSERT INTO documents_fts(documents_fts) VALUES('rebuild')`);
    },
  };
}

/**
 * Check for stale inactive documents (soft-deleted > 90 days ago)
 */
export function checkStaleDocuments(db: Database): IntegrityIssue | null {
  const stale = db.prepare(`
    SELECT COUNT(*) as stale_count
    FROM documents
    WHERE active = 0
      AND datetime(modified_at) < datetime('now', '-90 days')
  `).get() as { stale_count: number };

  if (stale.stale_count === 0) return null;

  return {
    severity: 'info',
    type: 'stale_documents',
    message: `${stale.stale_count} stale inactive document(s) (>90 days old)`,
    details: ['These documents were soft-deleted over 90 days ago'],
    fixable: false, // Requires explicit cleanup command
  };
}

/**
 * Check for missing vectors in vec table (content_vectors exists but vectors_vec missing)
 */
export function checkMissingVecTableEntries(db: Database): IntegrityIssue | null {
  // Check if vec table exists first
  const vecTableExists = db.prepare(`
    SELECT COUNT(*) as count FROM sqlite_master
    WHERE type='table' AND name='vectors_vec'
  `).get() as { count: number };

  if (vecTableExists.count === 0) return null;

  // Compare counts
  const cvCount = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
  const vecCount = db.prepare(`SELECT COUNT(*) as count FROM vectors_vec`).get() as { count: number };

  if (cvCount.count === vecCount.count) return null;

  const diff = Math.abs(cvCount.count - vecCount.count);

  return {
    severity: 'warning',
    type: 'vec_table_mismatch',
    message: `Vector count mismatch: content_vectors (${cvCount.count}) vs vectors_vec (${vecCount.count})`,
    details: [`Difference: ${diff} vectors`],
    fixable: true,
    fix: () => {
      // Clear both and force re-embedding
      db.exec(`DELETE FROM content_vectors`);
      db.exec(`DELETE FROM vectors_vec`);
    },
  };
}

/**
 * Run all integrity checks
 */
export function runAllIntegrityChecks(db: Database): IntegrityIssue[] {
  const issues: IntegrityIssue[] = [];

  const checks = [
    checkOrphanedVectors,
    checkPartialEmbeddings,
    checkDisplayPathCollisions,
    checkOrphanedDocuments,
    checkFTSConsistency,
    checkMissingVecTableEntries,
    checkStaleDocuments,
  ];

  for (const check of checks) {
    const issue = check(db);
    if (issue) {
      issues.push(issue);
    }
  }

  return issues;
}

/**
 * Auto-fix all fixable issues
 */
export function autoFixIssues(db: Database, issues: IntegrityIssue[]): number {
  let fixed = 0;

  for (const issue of issues) {
    if (issue.fixable && issue.fix) {
      try {
        db.transaction(() => {
          issue.fix!();
        })();
        fixed++;
      } catch (error) {
        console.error(`Failed to fix ${issue.type}: ${error}`);
      }
    }
  }

  return fixed;
}
