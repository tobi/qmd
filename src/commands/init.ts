import { Command, Flags } from '@oclif/core';
import { existsSync, mkdirSync, writeFileSync } from 'fs';
import { resolve } from 'path';
import { getPwd } from '../utils/paths.ts';
import { getDb } from '../database/index.ts';
import { indexFiles } from '../services/indexing.ts';
import { DEFAULT_GLOB } from '../config/constants.ts';

export default class InitCommand extends Command {
  static description = 'Initialize .qmd/ directory for project-local index';

  static flags = {
    'with-index': Flags.boolean({
      description: 'Index markdown files after initialization',
      default: false,
    }),
    force: Flags.boolean({
      description: 'Overwrite existing .qmd/ directory',
      default: false,
    }),
    config: Flags.boolean({
      description: 'Create config.json with default settings',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(InitCommand);
    const pwd = getPwd();
    const qmdDir = resolve(pwd, '.qmd');

    // Check if .qmd/ already exists
    if (existsSync(qmdDir) && !flags.force) {
      this.log(`✗ .qmd/ directory already exists at: ${qmdDir}`);
      this.log('  Use --force to overwrite');
      return;
    }

    try {
      // Create .qmd/ directory
      if (!existsSync(qmdDir)) {
        mkdirSync(qmdDir, { recursive: true });
        this.log('✓ Created .qmd/ directory');
      } else {
        this.log('✓ .qmd/ directory exists (using --force)');
      }

      // Create .gitignore
      const gitignoreContent = `# QMD Index (generated files)
*.sqlite
*.sqlite-shm
*.sqlite-wal
cache/

# Keep config
!config.json
!.gitignore
`;
      writeFileSync(resolve(qmdDir, '.gitignore'), gitignoreContent);
      this.log('✓ Created .qmd/.gitignore');

      // Optionally create config.json
      if (flags.config) {
        const configContent = {
          embedModel: 'nomic-embed-text',
          rerankModel: 'qwen3-reranker:0.6b-q8_0',
          defaultGlob: '**/*.md',
          excludeDirs: ['node_modules', '.git', 'dist', 'build', '.cache'],
          ollamaUrl: 'http://localhost:11434',
        };
        writeFileSync(
          resolve(qmdDir, 'config.json'),
          JSON.stringify(configContent, null, 2)
        );
        this.log('✓ Created .qmd/config.json');
      }

      // Optionally index files
      if (flags['with-index']) {
        this.log('');
        this.log('Indexing markdown files...');

        // getDb will now automatically use the .qmd/ directory we just created
        const db = getDb('default');

        try {
          const stats = await indexFiles(db, DEFAULT_GLOB, pwd);
          this.log('');
          this.log(`✓ Indexed ${stats.indexed} new documents`);
          if (stats.updated > 0) {
            this.log(`✓ Updated ${stats.updated} documents`);
          }
          if (stats.needsEmbedding > 0) {
            this.log('');
            this.log(`Run 'qmd embed' to generate embeddings (${stats.needsEmbedding} hashes)`);
          }
        } finally {
          db.close();
        }
      }

      this.log('');
      this.log('Ready! Next steps:');
      if (!flags['with-index']) {
        this.log("  1. Run 'qmd add .' to index your markdown files");
      }
      this.log("  2. Run 'qmd search \"query\"' to search");
      this.log("  3. Run 'qmd doctor' to check system health");

    } catch (error) {
      this.error(`Failed to initialize .qmd/ directory: ${error}`);
    }
  }
}
