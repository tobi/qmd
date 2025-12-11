import { Command, Flags } from '@oclif/core';
import { existsSync } from 'fs';
import { getDbPath } from '../utils/paths.ts';
import { getDb } from '../database/index.ts';
import { CollectionRepository } from '../database/repositories/index.ts';

export default class StatusCommand extends Command {
  static description = 'Show index status and collections';

  static flags = {
    index: Flags.string({
      description: 'Index name',
      default: 'default',
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(StatusCommand);

    const dbPath = getDbPath(flags.index);

    if (!existsSync(dbPath)) {
      this.log(`No index found at: ${dbPath}`);
      this.log('Run: qmd add <files> to create an index');
      return;
    }

    const db = getDb(flags.index);

    try {
      const collectionRepo = new CollectionRepository(db);

      // Get collections with document counts
      const collections = collectionRepo.findAllWithCounts();

      // Display results
      this.log(`\n📊 Index: ${flags.index}`);
      this.log(`📁 Location: ${dbPath}\n`);

      if (collections.length === 0) {
        this.log('No collections found. Run: qmd add <files>');
        return;
      }

      this.log(`Collections (${collections.length}):`);
      for (const col of collections) {
        this.log(`  ${col.pwd}`);
        this.log(`    Pattern: ${col.glob_pattern}`);
        this.log(`    Documents: ${col.active_count}`);
        this.log(`    Created: ${new Date(col.created_at).toLocaleString()}`);
        this.log('');
      }

      // Calculate totals
      const totalDocs = collections.reduce((sum, col) => sum + col.active_count, 0);
      this.log(`Total: ${totalDocs} documents in ${collections.length} collections`);

    } finally {
      db.close();
    }
  }
}
