import { Command, Flags, Args } from '@oclif/core';
import { existsSync } from 'fs';
import { getDbPath } from '../utils/paths.ts';
import { getDb } from '../database/index.ts';
import { hybridSearch } from '../services/search.ts';
import { getEmbedModel, getRerankModel } from '../config/constants.ts';
import { logSearch } from '../utils/history.ts';

export default class QueryCommand extends Command {
  static description = 'Hybrid search with RRF fusion and reranking (best quality)';

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
      default: '0',
    }),
    'embed-model': Flags.string({
      description: 'Embedding model',
    }),
    'rerank-model': Flags.string({
      description: 'Reranking model',
    }),
    json: Flags.boolean({
      description: 'Output as JSON',
      default: false,
    }),
    csv: Flags.boolean({
      description: 'Output as CSV',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(QueryCommand);

    const dbPath = getDbPath(flags.index);
    if (!existsSync(dbPath)) {
      this.error(`No index found at: ${dbPath}\nRun: qmd add <files> to create an index`);
    }

    const db = getDb(flags.index);

    try {
      const embedModel = getEmbedModel(flags['embed-model']);
      const rerankModel = getRerankModel(flags['rerank-model']);
      const minScore = parseFloat(flags['min-score'] || '0');

      // Hybrid search using service
      const results = await hybridSearch(db, args.query, embedModel, rerankModel, flags.n * 2);

      // Filter by min score
      const filtered = results.filter(r => r.score >= minScore).slice(0, flags.n);

      // Log search to history
      logSearch({
        timestamp: new Date().toISOString(),
        command: 'query',
        query: args.query,
        results_count: filtered.length,
        index: flags.index,
      });

      if (filtered.length === 0) {
        this.log('No results found.');
        return;
      }

      // Output based on format
      if (flags.json) {
        this.log(JSON.stringify({ query: args.query, results: filtered }, null, 2));
      } else if (flags.csv) {
        this.log('file,title,score,context,snippet');
        for (const r of filtered) {
          const snippet = r.snippet.replace(/"/g, '""');
          this.log(`"${r.file}","${r.title}",${r.score},"${r.context || ''}","${snippet}"`);
        }
      } else {
        // CLI format
        this.log(`\n🔍 Found ${filtered.length} result${filtered.length === 1 ? '' : 's'}\n`);
        for (const r of filtered) {
          this.log(`📄 ${r.file}`);
          this.log(`   ${r.title}`);
          this.log(`   Score: ${Math.round(r.score * 100)}%`);
          if (r.context) {
            this.log(`   Context: ${r.context}`);
          }
          this.log(`   ${r.snippet}`);
          this.log('');
        }
      }

    } finally {
      db.close();
    }
  }
}
