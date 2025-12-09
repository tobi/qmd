/**
 * Search history tracking utilities
 * Stores query history (not results) for suggestions and analytics
 */

import { resolve } from 'path';
import { homedir } from 'os';
import { existsSync, appendFileSync, readFileSync } from 'fs';

/**
 * History entry structure
 */
export interface HistoryEntry {
  timestamp: string;
  command: 'search' | 'vsearch' | 'query';
  query: string;
  results_count: number;
  index: string;
}

/**
 * Get path to history file
 * @returns Path to ~/.qmd_history
 */
export function getHistoryPath(): string {
  return resolve(homedir(), '.qmd_history');
}

/**
 * Log a search query to history
 * @param entry - History entry to log
 */
export function logSearch(entry: HistoryEntry): void {
  try {
    const historyPath = getHistoryPath();
    const line = JSON.stringify(entry) + '\n';
    appendFileSync(historyPath, line, 'utf8');
  } catch (error) {
    // Silently fail - history is non-critical
    // Could log to stderr if needed
  }
}

/**
 * Read all history entries
 * @param limit - Maximum number of entries to return (most recent first)
 * @returns Array of history entries
 */
export function readHistory(limit?: number): HistoryEntry[] {
  try {
    const historyPath = getHistoryPath();

    if (!existsSync(historyPath)) {
      return [];
    }

    const content = readFileSync(historyPath, 'utf8');
    const lines = content.trim().split('\n').filter(line => line.length > 0);

    // Parse each line as JSON
    const entries: HistoryEntry[] = [];
    for (const line of lines) {
      try {
        const entry = JSON.parse(line) as HistoryEntry;
        entries.push(entry);
      } catch {
        // Skip malformed lines
        continue;
      }
    }

    // Return most recent first
    entries.reverse();

    // Apply limit if specified
    if (limit && limit > 0) {
      return entries.slice(0, limit);
    }

    return entries;
  } catch (error) {
    return [];
  }
}

/**
 * Get unique queries from history
 * @param limit - Maximum number of unique queries to return
 * @returns Array of unique queries (most recent first)
 */
export function getUniqueQueries(limit?: number): string[] {
  const entries = readHistory();
  const uniqueQueries = new Set<string>();

  for (const entry of entries) {
    uniqueQueries.add(entry.query);
    if (limit && uniqueQueries.size >= limit) {
      break;
    }
  }

  return Array.from(uniqueQueries);
}

/**
 * Get search statistics from history
 * @returns Statistics object
 */
export function getHistoryStats(): {
  total_searches: number;
  unique_queries: number;
  commands: Record<string, number>;
  indexes: Record<string, number>;
  popular_queries: Array<{ query: string; count: number }>;
} {
  const entries = readHistory();

  const commands: Record<string, number> = {};
  const indexes: Record<string, number> = {};
  const queryCounts: Record<string, number> = {};

  for (const entry of entries) {
    // Count by command
    commands[entry.command] = (commands[entry.command] || 0) + 1;

    // Count by index
    indexes[entry.index] = (indexes[entry.index] || 0) + 1;

    // Count query frequency
    queryCounts[entry.query] = (queryCounts[entry.query] || 0) + 1;
  }

  // Sort queries by frequency
  const popular_queries = Object.entries(queryCounts)
    .map(([query, count]) => ({ query, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);

  return {
    total_searches: entries.length,
    unique_queries: Object.keys(queryCounts).length,
    commands,
    indexes,
    popular_queries,
  };
}

/**
 * Clear search history
 */
export function clearHistory(): void {
  try {
    const historyPath = getHistoryPath();
    if (existsSync(historyPath)) {
      // Truncate file
      appendFileSync(historyPath, '', { flag: 'w' });
    }
  } catch (error) {
    throw new Error(`Failed to clear history: ${error}`);
  }
}
