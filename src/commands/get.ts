import { Command, Flags, Args } from '@oclif/core';
import { existsSync } from 'fs';
import { homedir } from 'os';
import { getDbPath } from '../utils/paths.ts';
import { getDb } from '../database/index.ts';
import { DocumentRepository, PathContextRepository } from '../database/repositories/index.ts';

export default class GetCommand extends Command {
  static description = 'Retrieve document content by file path';

  static args = {
    file: Args.string({
      description: 'File path (supports "path/file.md:100" for line number)',
      required: true,
    }),
  };

  static flags = {
    index: Flags.string({
      description: 'Index name',
      default: 'default',
    }),
    'from-line': Flags.integer({
      description: 'Start from line number (1-indexed)',
    }),
    'max-lines': Flags.integer({
      description: 'Maximum number of lines to return',
    }),
  };

  async run(): Promise<void> {
    const { args, flags } = await this.parse(GetCommand);

    const dbPath = getDbPath(flags.index);
    if (!existsSync(dbPath)) {
      this.error(`No index found at: ${dbPath}\nRun: qmd add <files> to create an index`);
    }

    const db = getDb(flags.index);

    try {
      const docRepo = new DocumentRepository(db);
      const pathCtxRepo = new PathContextRepository(db);

      // Parse file path with optional line number
      let filepath = args.file;
      let fromLine = flags['from-line'];

      const colonMatch = filepath.match(/:(\d+)$/);
      if (colonMatch && !fromLine) {
        fromLine = parseInt(colonMatch[1], 10);
        filepath = filepath.slice(0, -colonMatch[0].length);
      }

      // Expand tilde
      if (filepath.startsWith("~/")) {
        filepath = homedir() + filepath.slice(1);
      }

      // Find document
      let doc = docRepo.findByFilepath(filepath);

      // Try fuzzy match if exact match fails
      if (!doc) {
        const stmt = db.prepare(`
          SELECT id, filepath, body
          FROM documents
          WHERE filepath LIKE ? AND active = 1
          LIMIT 1
        `);
        doc = stmt.get(`%${filepath}`) as any;
      }

      if (!doc) {
        this.error(`Document not found: ${args.file}`);
      }

      // Get context
      const context = pathCtxRepo.findForPath(doc.filepath);

      // Extract lines if specified
      let output = doc.body;
      if (fromLine !== undefined || flags['max-lines'] !== undefined) {
        const lines = output.split("\n");
        const start = (fromLine || 1) - 1;
        const end = flags['max-lines'] !== undefined ? start + flags['max-lines'] : lines.length;
        output = lines.slice(start, end).join("\n");
      }

      // Display result
      if (context) {
        this.log(`Folder Context: ${context.context}`);
        this.log('---\n');
      }
      this.log(output);

    } finally {
      db.close();
    }
  }
}
