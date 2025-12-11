import { Command, Flags } from '@oclif/core';
import { getDb } from '../database/index.ts';
import { cleanup, type CleanupOptions } from '../database/cleanup.ts';

export default class CleanupCommand extends Command {
  static description = 'Permanently delete soft-deleted documents';

  static examples = [
    '<%= config.bin %> <%= command.id %>',
    '<%= config.bin %> <%= command.id %> --older-than=90',
    '<%= config.bin %> <%= command.id %> --dry-run',
    '<%= config.bin %> <%= command.id %> --vacuum',
    '<%= config.bin %> <%= command.id %> --all --vacuum',
  ];

  static flags = {
    'older-than': Flags.integer({
      description: 'Delete documents older than N days (default: 30)',
      default: 30,
    }),
    'dry-run': Flags.boolean({
      description: 'Show what would be deleted without deleting',
      default: false,
    }),
    all: Flags.boolean({
      description: 'Delete ALL inactive documents (ignore age)',
      default: false,
    }),
    vacuum: Flags.boolean({
      description: 'Also cleanup orphaned vectors and vacuum database',
      default: false,
    }),
    index: Flags.string({
      description: 'Index name to cleanup',
      default: 'index',
    }),
    yes: Flags.boolean({
      description: 'Skip confirmation prompt',
      char: 'y',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(CleanupCommand);

    const db = getDb(flags.index);

    // Show warning for dangerous operations
    if (flags.all && !flags['dry-run'] && !flags.yes) {
      this.log('⚠️  WARNING: This will permanently delete ALL inactive documents!');
      this.log('');

      const readline = await import('node:readline/promises');
      const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
      });

      const answer = await rl.question('Are you sure you want to continue? (yes/no): ');
      rl.close();

      if (answer.toLowerCase() !== 'yes') {
        this.log('Cleanup cancelled.');
        return;
      }
    }

    // Prepare options
    const options: CleanupOptions = {
      olderThanDays: flags['older-than'],
      dryRun: flags['dry-run'],
      all: flags.all,
      vacuum: flags.vacuum,
    };

    if (flags['dry-run']) {
      this.log('🔍 Dry run - no changes will be made\n');
    } else {
      this.log('🧹 Cleaning up...\n');
    }

    // Perform cleanup
    const result = cleanup(db, options);

    // Display results
    if (flags['dry-run']) {
      this.log('Would delete:');
    } else {
      this.log('Cleanup complete:');
    }

    this.log(`  Documents: ${result.documents_deleted}`);

    if (flags.vacuum) {
      this.log(`  Vector chunks: ${result.vectors_deleted}`);
      this.log(`  Cache entries: ${result.cache_entries_deleted}`);
    }

    if (result.space_reclaimed_mb > 0) {
      this.log(`  Space reclaimed: ${result.space_reclaimed_mb.toFixed(2)} MB`);
    }

    this.log('');

    if (flags['dry-run'] && result.documents_deleted > 0) {
      this.log('Run without --dry-run to perform cleanup.');
    }

    if (!flags['dry-run'] && result.documents_deleted === 0) {
      this.log('✓ No documents to cleanup.');
    }

    db.close();
  }
}
