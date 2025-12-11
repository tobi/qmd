import { Command, Flags, Args } from '@oclif/core';
import { existsSync } from 'fs';
import type { OutputOptions } from '../models/types.ts';
import { getDbPath } from '../utils/paths.ts';
import { getDb } from '../database/index.ts';
import { vectorSearch } from '../services/search.ts';
import { logSearch } from '../utils/history.ts';
import { getEmbedModel } from '../config/constants.ts';

export default class VSearchCommand extends Command {
  static description = 'Vector similarity search';

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
      default: '0.3',
    }),
    'embed-model': Flags.string({
      description: 'Embedding model',
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
    const { args, flags } = await this.parse(VSearchCommand);

    const dbPath = getDbPath(flags.index);
    if (!existsSync(dbPath)) {
      this.error(`No index found at: ${dbPath}\nRun: qmd add <files> to create an index`);
    }

    const db = getDb(flags.index);

    try {
      const embedModel = getEmbedModel(flags['embed-model']);

      // Determine output format
      let format: 'cli' | 'json' | 'csv' | 'files' = 'cli';
      if (flags.json) format = 'json';
      else if (flags.csv) format = 'csv';
      else if (flags.files) format = 'files';

      const opts: OutputOptions = {
        format,
        full: flags.full,
        limit: flags.all ? 100000 : flags.n,
        minScore: parseFloat(flags['min-score']),
        all: flags.all,
      };

      // Vector search using service
      const fetchLimit = opts.all ? 100000 : Math.max(50, opts.limit * 2);
      const results = await vectorSearch(db, args.query, embedModel, fetchLimit);

      // Log search to history
      logSearch({
        timestamp: new Date().toISOString(),
        command: 'vsearch',
        query: args.query,
        results_count: results.length,
        index: flags.index,
      });

      if (results.length === 0) {
        this.log('No results found.');
        return;
      }

      // Filter and output
      this.outputResults(results, args.query, opts);

    } finally {
      db.close();
    }
  }

  private outputResults(results: any[], query: string, opts: OutputOptions): void {
    // Apply filtering
    let filtered = results.filter(r => r.score >= opts.minScore);

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
        if (r.chunkPos !== undefined) {
          this.log(`   Chunk position: ${r.chunkPos}`);
        }
        if (opts.full) {
          this.log(`\n${r.body}\n`);
        }
        this.log('');
      }
    }
  }
}
