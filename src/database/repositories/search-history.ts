/**
 * SearchHistory repository - Data access layer for search_history table
 * All queries use prepared statements to prevent SQL injection
 */

import { Database } from 'bun:sqlite';

export interface SearchHistoryEntry {
  id?: number;
  timestamp: string;
  command: 'search' | 'vsearch' | 'query';
  query: string;
  results_count: number;
  index_name: string;
  created_at?: string;
}

export interface HistoryStats {
  total_searches: number;
  unique_queries: number;
  commands: Record<string, number>;
  indexes: Record<string, number>;
  popular_queries: Array<{ query: string; count: number }>;
}

export class SearchHistoryRepository {
  constructor(private db: Database) {}

  /**
   * Insert a new history entry
   * @param entry - History entry (without id)
   * @returns Inserted entry ID
   */
  insert(entry: Omit<SearchHistoryEntry, 'id' | 'created_at'>): number {
    const stmt = this.db.prepare(`
      INSERT INTO search_history (timestamp, command, query, results_count, index_name)
      VALUES (?, ?, ?, ?, ?)
    `);

    const result = stmt.run(
      entry.timestamp,
      entry.command,
      entry.query,
      entry.results_count,
      entry.index_name
    );

    return Number(result.lastInsertRowid);
  }

  /**
   * Find recent history entries
   * @param limit - Maximum number of entries
   * @returns Array of entries (most recent first)
   */
  findRecent(limit: number = 100): SearchHistoryEntry[] {
    const stmt = this.db.prepare(`
      SELECT id, timestamp, command, query, results_count, index_name, created_at
      FROM search_history
      ORDER BY timestamp DESC
      LIMIT ?
    `);

    return stmt.all(limit) as SearchHistoryEntry[];
  }

  /**
   * Find history entries by date range
   * @param start - Start timestamp (ISO format)
   * @param end - End timestamp (ISO format)
   * @returns Array of entries in range
   */
  findByDateRange(start: string, end: string): SearchHistoryEntry[] {
    const stmt = this.db.prepare(`
      SELECT id, timestamp, command, query, results_count, index_name, created_at
      FROM search_history
      WHERE timestamp >= ? AND timestamp <= ?
      ORDER BY timestamp DESC
    `);

    return stmt.all(start, end) as SearchHistoryEntry[];
  }

  /**
   * Find history entries by command type
   * @param command - Command type
   * @param limit - Maximum entries
   * @returns Array of entries
   */
  findByCommand(command: string, limit: number = 100): SearchHistoryEntry[] {
    const stmt = this.db.prepare(`
      SELECT id, timestamp, command, query, results_count, index_name, created_at
      FROM search_history
      WHERE command = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);

    return stmt.all(command, limit) as SearchHistoryEntry[];
  }

  /**
   * Find history entries by index name
   * @param indexName - Index name
   * @param limit - Maximum entries
   * @returns Array of entries
   */
  findByIndex(indexName: string, limit: number = 100): SearchHistoryEntry[] {
    const stmt = this.db.prepare(`
      SELECT id, timestamp, command, query, results_count, index_name, created_at
      FROM search_history
      WHERE index_name = ?
      ORDER BY timestamp DESC
      LIMIT ?
    `);

    return stmt.all(indexName, limit) as SearchHistoryEntry[];
  }

  /**
   * Get unique queries
   * @param limit - Maximum unique queries
   * @returns Array of unique queries (most recent first)
   */
  getUniqueQueries(limit?: number): string[] {
    let sql = `
      SELECT DISTINCT query
      FROM search_history
      ORDER BY MAX(timestamp) DESC
    `;

    if (limit) {
      sql += ` LIMIT ?`;
    }

    const stmt = this.db.prepare(sql);
    const results = limit
      ? stmt.all(limit) as Array<{ query: string }>
      : stmt.all() as Array<{ query: string }>;

    return results.map(r => r.query);
  }

  /**
   * Get search history statistics
   * @returns Statistics object
   */
  getStats(): HistoryStats {
    // Total searches
    const totalStmt = this.db.prepare(`SELECT COUNT(*) as count FROM search_history`);
    const total = (totalStmt.get() as { count: number }).count;

    // Unique queries
    const uniqueStmt = this.db.prepare(`SELECT COUNT(DISTINCT query) as count FROM search_history`);
    const unique = (uniqueStmt.get() as { count: number }).count;

    // Commands breakdown
    const commandsStmt = this.db.prepare(`
      SELECT command, COUNT(*) as count
      FROM search_history
      GROUP BY command
    `);
    const commandResults = commandsStmt.all() as Array<{ command: string; count: number }>;
    const commands: Record<string, number> = {};
    for (const { command, count } of commandResults) {
      commands[command] = count;
    }

    // Indexes breakdown
    const indexesStmt = this.db.prepare(`
      SELECT index_name, COUNT(*) as count
      FROM search_history
      GROUP BY index_name
    `);
    const indexResults = indexesStmt.all() as Array<{ index_name: string; count: number }>;
    const indexes: Record<string, number> = {};
    for (const { index_name, count } of indexResults) {
      indexes[index_name] = count;
    }

    // Popular queries
    const popularStmt = this.db.prepare(`
      SELECT query, COUNT(*) as count
      FROM search_history
      GROUP BY query
      ORDER BY count DESC
      LIMIT 10
    `);
    const popular_queries = popularStmt.all() as Array<{ query: string; count: number }>;

    return {
      total_searches: total,
      unique_queries: unique,
      commands,
      indexes,
      popular_queries,
    };
  }

  /**
   * Delete old history entries
   * @param olderThanDays - Delete entries older than N days
   * @returns Number of entries deleted
   */
  cleanup(olderThanDays: number): number {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - olderThanDays);

    const stmt = this.db.prepare(`
      DELETE FROM search_history
      WHERE timestamp < ?
    `);

    const result = stmt.run(cutoff.toISOString());
    return result.changes || 0;
  }

  /**
   * Clear all history
   */
  clear(): void {
    const stmt = this.db.prepare(`DELETE FROM search_history`);
    stmt.run();
  }

  /**
   * Get total count
   * @returns Number of entries
   */
  count(): number {
    const stmt = this.db.prepare(`SELECT COUNT(*) as count FROM search_history`);
    return (stmt.get() as { count: number }).count;
  }

  /**
   * Batch insert entries (for migration)
   * @param entries - Array of entries to insert
   * @returns Number of entries inserted
   */
  insertBatch(entries: Array<Omit<SearchHistoryEntry, 'id' | 'created_at'>>): number {
    let inserted = 0;

    this.db.transaction(() => {
      const stmt = this.db.prepare(`
        INSERT INTO search_history (timestamp, command, query, results_count, index_name)
        VALUES (?, ?, ?, ?, ?)
      `);

      for (const entry of entries) {
        stmt.run(
          entry.timestamp,
          entry.command,
          entry.query,
          entry.results_count,
          entry.index_name
        );
        inserted++;
      }
    })();

    return inserted;
  }
}
