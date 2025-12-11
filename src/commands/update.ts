import { Command, Args, Flags } from '@oclif/core';
import { getDb } from '../database/index.ts';
import { indexFiles } from '../services/indexing.ts';
import { CollectionRepository } from '../database/repositories/index.ts';

export default class UpdateCommand extends Command {
  static description = 'Re-index one or all collections';

  static args = {
    collection: Args.string({
      description: 'Collection ID to update (omit to update all)',
      required: false,
    }),
  };

  static flags = {
    index: Flags.string({
      description: 'Index name',
      default: 'default',
    }),
    all: Flags.boolean({
      description: 'Update all collections (same as omitting collection ID)',
      default: false,
    }),
  };

  static examples = [
    '$ qmd update          # Update all collections',
    '$ qmd update --all    # Update all collections (explicit)',
    '$ qmd update 1        # Update collection with ID 1',
  ];

  async run(): Promise<void> {
    const { args, flags } = await this.parse(UpdateCommand);
    const db = getDb(flags.index);

    try {
      const collectionRepo = new CollectionRepository(db);

      // Determine which collections to update
      let collections;
      if (args.collection && !flags.all) {
        // Update specific collection by ID
        const collectionId = parseInt(args.collection, 10);
        if (isNaN(collectionId)) {
          this.error(`Invalid collection ID: ${args.collection}`);
        }

        const collection = collectionRepo.findById(collectionId);
        if (!collection) {
          this.error(`Collection not found: ${collectionId}`);
        }
        collections = [collection];
      } else {
        // Update all collections
        collections = collectionRepo.findAll();
        if (collections.length === 0) {
          this.log('No collections found to update.');
          this.log('Run: qmd add <pattern> to create a collection');
          return;
        }
      }

      this.log(`Updating ${collections.length} collection(s)...\n`);

      let totalIndexed = 0;
      let totalUpdated = 0;
      let totalRemoved = 0;
      let totalUnchanged = 0;
      let failedCollections = 0;

      for (const collection of collections) {
        this.log(`Collection ${collection.id}: ${collection.pwd}`);
        this.log(`  Pattern: ${collection.glob_pattern}`);

        try {
          const stats = await indexFiles(db, collection.glob_pattern, collection.pwd);
          totalIndexed += stats.indexed;
          totalUpdated += stats.updated;
          totalRemoved += stats.removed;
          totalUnchanged += stats.unchanged;

          if (stats.needsEmbedding > 0) {
            this.log(`  ⚠ ${stats.needsEmbedding} hashes need embeddings`);
          }
        } catch (error) {
          this.log(`  ✗ Failed to update: ${error}`);
          failedCollections++;
        }

        this.log('');
      }

      // Summary
      this.log('━'.repeat(50));
      this.log('Summary:');
      this.log(`  Collections updated: ${collections.length - failedCollections}/${collections.length}`);
      this.log(`  Documents indexed: ${totalIndexed} new`);
      this.log(`  Documents updated: ${totalUpdated}`);
      this.log(`  Documents removed: ${totalRemoved}`);
      this.log(`  Documents unchanged: ${totalUnchanged}`);

      if (failedCollections > 0) {
        this.log(`\n⚠ ${failedCollections} collection(s) failed to update`);
      }

      // Check if embeddings needed
      if (totalIndexed > 0 || totalUpdated > 0) {
        this.log("\nRun 'qmd embed' to update embeddings");
      }

    } finally {
      db.close();
    }
  }
}
