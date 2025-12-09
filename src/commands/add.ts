import { Command, Args, Flags } from '@oclif/core';
import { getDb } from '../database/index.ts';
import { indexFiles } from '../services/indexing.ts';
import { getPwd } from '../utils/paths.ts';
import { DEFAULT_GLOB } from '../config/constants.ts';
import { existsSync } from 'fs';

export default class AddCommand extends Command {
  static description = 'Index markdown files';

  static examples = [
    '$ qmd add .                    # Use default **/*.md pattern',
    '$ qmd add "**/*.md"            # Quote glob patterns to prevent shell expansion',
    '$ qmd add "docs/**/*.md"       # Index specific directory',
    '$ qmd add --index work .       # Use named index',
  ];

  static args = {
    pattern: Args.string({
      description: 'Glob pattern (quote to prevent shell expansion, or "." for default **/*.md)',
      default: '.',
    }),
  };

  static flags = {
    index: Flags.string({
      description: 'Index name',
      default: 'default',
    }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(AddCommand);

    // Treat "." as "use default glob in current directory"
    let globPattern = (!args.pattern || args.pattern === ".") ? DEFAULT_GLOB : args.pattern;

    // Detect if pattern looks like a file (possible shell expansion)
    if (globPattern !== DEFAULT_GLOB && !globPattern.includes('*') && !globPattern.includes('?')) {
      // Check if it's an existing file
      if (existsSync(globPattern) && !globPattern.endsWith('/')) {
        this.warn(`Pattern '${globPattern}' looks like a file, not a glob pattern.`);
        this.warn(`Did you forget to quote the pattern?`);
        this.warn(`Example: qmd add "**/*.md" instead of qmd add **/*.md`);
        this.warn('');
        this.warn('Shell expansion may have occurred. Continuing with pattern as-is...');
      }
    }

    const db = getDb(flags.index);

    try {
      await indexFiles(db, globPattern, getPwd());
    } finally {
      db.close();
    }
  }

  // Override error handler to provide helpful message for unexpected args
  protected async catch(err: Error & { oclif?: any }): Promise<any> {
    // Detect "Unexpected argument" error (likely from shell expansion)
    if (err.message?.includes('Unexpected argument')) {
      this.error(
        `Multiple arguments detected. This usually happens when the shell expands your glob pattern.\n\n` +
        `❌ Wrong: qmd add **/*.md\n` +
        `✓ Correct: qmd add "**/*.md"\n\n` +
        `Always quote glob patterns to prevent shell expansion.\n` +
        `Or use: qmd add . (for default **/*.md pattern)`,
        { exit: 2 }
      );
    }

    // Let parent handle other errors
    return super.catch(err);
  }
}
