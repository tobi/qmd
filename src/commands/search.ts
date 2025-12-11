import { Command, Flags, Args } from '@oclif/core';
import { existsSync } from 'fs';
import type { OutputOptions } from '../models/types.ts';
import { getDbPath } from '../utils/paths.ts';
import { getDb } from '../database/index.ts';
import { fullTextSearch } from '../services/search.ts';
import { logSearch } from '../utils/history.ts';

export default class SearchCommand extends Command {
  static description = 'Full-text search using BM25';

  static args = {
    query: Args.string({
      description: 'Search query',
      required: true,
    }),
  };

  static flags = {
    index: Flags.string({
      description: 'Index name',
      default: 'default',
    }),
    n: Flags.integer({
      description: 'Number of results',
      default: 10,
    }),
    'min-score': Flags.string({
      description: 'Minimum score threshold',
    }),
    full: Flags.boolean({
      description: 'Show full document content',
      default: false,
    }),
    json: Flags.boolean({
      description: 'Output as JSON',
      default: false,
    }),
    csv: Flags.boolean({
      description: 'Output as CSV',
      default: false,
    }),
    md: Flags.boolean({
      description: 'Output as Markdown',
      default: false,
    }),
    xml: Flags.boolean({
      description: 'Output as XML',
      default: false,
    }),
    files: Flags.boolean({
      description: 'Output file paths only',
      default: false,
    }),
    all: Flags.boolean({
      description: 'Show all results (no limit)',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(SearchCommand);

    const dbPath = getDbPath(flags.index);

    if (!existsSync(dbPath)) {
      this.error(`No index found at: ${dbPath}\nRun: qmd add <files> to create an index`);
    }

    const db = getDb(flags.index);

    try {
      // Determine output format
      let format: 'cli' | 'json' | 'csv' | 'md' | 'xml' | 'files' = 'cli';
      if (flags.json) format = 'json';
      else if (flags.csv) format = 'csv';
      else if (flags.md) format = 'md';
      else if (flags.xml) format = 'xml';
      else if (flags.files) format = 'files';

      const opts: OutputOptions = {
        format,
        full: flags.full,
        limit: flags.all ? 100000 : flags.n,
        minScore: flags['min-score'] ? parseFloat(flags['min-score']) : 0,
        all: flags.all,
      };

      // Search using service
      const fetchLimit = opts.all ? 100000 : Math.max(50, opts.limit * 2);
      const results = await fullTextSearch(db, args.query, fetchLimit);

      // Log search to history
      logSearch({
        timestamp: new Date().toISOString(),
        command: 'search',
        query: args.query,
        results_count: results.length,
        index: flags.index,
      });

      if (results.length === 0) {
        this.log('No results found.');
        return;
      }

      // Output
      this.outputResults(results, args.query, opts);

    } finally {
      db.close();
    }
  }

  private outputResults(results: any[], query: string, opts: OutputOptions): void {
    // Apply filtering
    let filtered = results;

    if (opts.minScore > 0) {
      filtered = filtered.filter(r => r.score >= opts.minScore);
    }

    if (!opts.all) {
      filtered = filtered.slice(0, opts.limit);
    }

    // Output based on format
    if (opts.format === 'json') {
      this.log(JSON.stringify({ query, results: filtered }, null, 2));
    } else if (opts.format === 'csv') {
      this.log('file,title,score,context');
      for (const r of filtered) {
        this.log(`"${r.displayPath}","${r.title}",${r.score},"${r.context || ''}"`);
      }
    } else if (opts.format === 'files') {
      for (const r of filtered) {
        this.log(r.displayPath);
      }
    } else {
      // CLI format
      this.log(`\n🔍 Found ${filtered.length} result${filtered.length === 1 ? '' : 's'}\n`);
      for (const r of filtered) {
        this.log(`📄 ${r.displayPath}`);
        this.log(`   ${r.title}`);
        this.log(`   Score: ${Math.round(r.score * 100)}%`);
        if (r.context) {
          this.log(`   Context: ${r.context}`);
        }
        if (opts.full) {
          this.log(`\n${r.body}\n`);
        }
        this.log('');
      }
    }
  }
}
