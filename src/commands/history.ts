import { Command, Flags } from '@oclif/core';
import { readHistory, getHistoryStats, clearHistory, getHistoryPath } from '../utils/history.ts';

export default class HistoryCommand extends Command {
  static description = 'Show search history and statistics';

  static examples = [
    '$ qmd history                  # Show recent searches',
    '$ qmd history --limit 20       # Show last 20 searches',
    '$ qmd history --stats          # Show search statistics',
    '$ qmd history --clear          # Clear search history',
  ];

  static flags = {
    limit: Flags.integer({
      description: 'Maximum number of entries to show',
      default: 10,
    }),
    stats: Flags.boolean({
      description: 'Show statistics instead of history',
      default: false,
    }),
    clear: Flags.boolean({
      description: 'Clear search history',
      default: false,
    }),
    json: Flags.boolean({
      description: 'Output as JSON',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(HistoryCommand);

    // Handle clear
    if (flags.clear) {
      try {
        clearHistory();
        this.log('✓ Search history cleared');
        return;
      } catch (error) {
        this.error(`Failed to clear history: ${error}`);
      }
    }

    // Handle stats
    if (flags.stats) {
      const stats = getHistoryStats();

      if (flags.json) {
        this.log(JSON.stringify(stats, null, 2));
        return;
      }

      this.log('');
      this.log('📊 Search History Statistics');
      this.log('━'.repeat(50));
      this.log('');
      this.log(`Total searches: ${stats.total_searches}`);
      this.log(`Unique queries: ${stats.unique_queries}`);
      this.log('');

      this.log('Commands:');
      for (const [cmd, count] of Object.entries(stats.commands)) {
        this.log(`  ${cmd}: ${count}`);
      }
      this.log('');

      this.log('Indexes:');
      for (const [idx, count] of Object.entries(stats.indexes)) {
        this.log(`  ${idx}: ${count}`);
      }
      this.log('');

      if (stats.popular_queries.length > 0) {
        this.log('Popular queries:');
        for (const { query, count } of stats.popular_queries) {
          this.log(`  ${count}× "${query}"`);
        }
      }
      this.log('');

      return;
    }

    // Show history
    const entries = readHistory(flags.limit);

    if (entries.length === 0) {
      this.log('No search history found.');
      this.log(`History location: ${getHistoryPath()}`);
      return;
    }

    if (flags.json) {
      this.log(JSON.stringify(entries, null, 2));
      return;
    }

    this.log('');
    this.log('🔍 Recent Searches');
    this.log('━'.repeat(50));
    this.log('');

    for (const entry of entries) {
      const date = new Date(entry.timestamp);
      const timeStr = date.toLocaleString();
      const cmdIcon = entry.command === 'search' ? '📝' : entry.command === 'vsearch' ? '🔎' : '🎯';

      this.log(`${cmdIcon} ${entry.command}`);
      this.log(`   Query: "${entry.query}"`);
      this.log(`   Results: ${entry.results_count}`);
      this.log(`   Index: ${entry.index}`);
      this.log(`   Time: ${timeStr}`);
      this.log('');
    }

    this.log(`Showing ${entries.length} most recent searches`);
    this.log(`History location: ${getHistoryPath()}`);
    this.log('');
  }
}
