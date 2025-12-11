import { Command, Flags } from '@oclif/core';
import { getDb, ensureVecTable, getHashesNeedingEmbedding } from '../database/index.ts';
import { embedDocument, chunkDocument } from '../services/embedding.ts';
import { getEmbedModel } from '../config/constants.ts';
import { progress } from '../config/terminal.ts';
import { formatBytes } from '../utils/formatters.ts';

const CHUNK_BYTE_SIZE = 800;

export default class EmbedCommand extends Command {
  static description = 'Generate vector embeddings for indexed documents';

  static flags = {
    index: Flags.string({
      description: 'Index name',
      default: 'default',
    }),
    'embed-model': Flags.string({
      description: 'Embedding model',
    }),
    force: Flags.boolean({
      description: 'Force re-embedding all documents',
      default: false,
    }),
  };

  async run(): Promise<void> {
    const { flags } = await this.parse(EmbedCommand);

    const db = getDb(flags.index);
    const model = getEmbedModel(flags['embed-model']);

    try {
      // If force, clear all vectors
      if (flags.force) {
        this.log('Force re-indexing: clearing all vectors...');
        db.exec(`DELETE FROM content_vectors`);
        db.exec(`DROP TABLE IF EXISTS vectors_vec`);
      }

      // Find unique hashes that need embedding
      const hashesToEmbed = db.prepare(`
        SELECT d.hash, d.body, d.title, MIN(d.display_path) as display_path
        FROM documents d
        LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
        WHERE d.active = 1 AND v.hash IS NULL
        GROUP BY d.hash
      `).all() as { hash: string; body: string; title: string; display_path: string }[];

      if (hashesToEmbed.length === 0) {
        this.log('✓ All content hashes already have embeddings.');
        return;
      }

      this.log(`Embedding ${hashesToEmbed.length} documents with model: ${model}\n`);
      progress.indeterminate();

      let embedded = 0;
      for (const item of hashesToEmbed) {
        if (!item.body) continue;

        // Chunk document
        const chunks = chunkDocument(item.body, CHUNK_BYTE_SIZE).map(c => ({
          text: c.text,
          pos: c.pos,
          title: item.title,
        }));

        // Embed and store
        await embedDocument(db, item.hash, chunks, model);

        embedded++;
        progress.set((embedded / hashesToEmbed.length) * 100);
        process.stderr.write(`\rEmbedding: ${embedded}/${hashesToEmbed.length}        `);
      }

      progress.clear();
      this.log(`\n✓ Embedded ${embedded} documents`);

      // Show summary
      const needsEmbedding = getHashesNeedingEmbedding(db);
      if (needsEmbedding > 0) {
        this.log(`Remaining: ${needsEmbedding} documents still need embeddings`);
      }

    } finally {
      db.close();
    }
  }
}
