import { Command, Args, Flags } from '@oclif/core';
import { getDb } from '../database/index.ts';
import { indexFiles } from '../services/indexing.ts';
import { getPwd } from '../utils/paths.ts';
import { DEFAULT_GLOB } from '../config/constants.ts';

export default class AddCommand extends Command {
  static description = 'Index markdown files';

  static args = {
    pattern: Args.string({
      description: 'Glob pattern (or "." for default **/*.md)',
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
    const globPattern = (!args.pattern || args.pattern === ".") ? DEFAULT_GLOB : args.pattern;

    const db = getDb(flags.index);

    try {
      await indexFiles(db, globPattern, getPwd());
    } finally {
      db.close();
    }
  }
}
