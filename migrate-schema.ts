#!/usr/bin/env bun
/**
 * Migrate documents table from collection_id to collection name
 *
 * This script updates the database schema to use collection names
 * instead of collection_id foreign keys, preparing for YAML-based
 * collection management.
 */

import { Database } from "bun:sqlite";
import { join } from "path";
import { homedir } from "os";

const c = {
  reset: "\x1b[0m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  dim: "\x1b[2m",
};

const dbPath = join(homedir(), ".cache", "qmd", "index.sqlite");
console.log(`${c.cyan}Migrating database schema...${c.reset}\n`);
console.log(`Database: ${dbPath}\n`);

const db = new Database(dbPath);

try {
  db.exec("BEGIN TRANSACTION");

  // Step 1: Add collection column to documents
  console.log(`${c.yellow}1. Adding 'collection' column to documents table...${c.reset}`);
  db.exec(`ALTER TABLE documents ADD COLUMN collection TEXT`);
  console.log(`  ${c.green}✓${c.reset} Column added`);

  // Step 2: Populate collection names from collections table
  console.log(`\n${c.yellow}2. Populating collection names...${c.reset}`);
  const result = db.exec(`
    UPDATE documents
    SET collection = (
      SELECT name FROM collections WHERE collections.id = documents.collection_id
    )
    WHERE collection IS NULL
  `);
  console.log(`  ${c.green}✓${c.reset} Updated ${result} rows`);

  // Step 3: Verify no NULL values
  const nullCount = db.query<{ count: number }, []>(
    `SELECT COUNT(*) as count FROM documents WHERE collection IS NULL`
  ).get();

  if (nullCount && nullCount.count > 0) {
    throw new Error(`Found ${nullCount.count} documents with NULL collection names`);
  }
  console.log(`  ${c.green}✓${c.reset} All documents have collection names`);

  // Step 4: Create new documents table without collection_id
  console.log(`\n${c.yellow}3. Creating new documents table...${c.reset}`);
  db.exec(`
    CREATE TABLE documents_new (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      collection TEXT NOT NULL,
      path TEXT NOT NULL,
      title TEXT NOT NULL,
      hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      active INTEGER DEFAULT 1,

      FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
      UNIQUE(collection, path)
    )
  `);
  console.log(`  ${c.green}✓${c.reset} New table created`);

  // Step 5: Copy data
  console.log(`\n${c.yellow}4. Copying data to new table...${c.reset}`);
  db.exec(`
    INSERT INTO documents_new (id, collection, path, title, hash, created_at, modified_at, active)
    SELECT id, collection, path, title, hash, created_at, modified_at, active
    FROM documents
  `);
  const rowCount = db.query<{ count: number }, []>(
    `SELECT COUNT(*) as count FROM documents_new`
  ).get();
  console.log(`  ${c.green}✓${c.reset} Copied ${rowCount?.count} documents`);

  // Step 6: Drop old table and rename new one
  console.log(`\n${c.yellow}5. Replacing old table...${c.reset}`);
  db.exec(`DROP TABLE documents`);
  db.exec(`ALTER TABLE documents_new RENAME TO documents`);
  console.log(`  ${c.green}✓${c.reset} Table replaced`);

  // Step 7: Recreate indices
  console.log(`\n${c.yellow}6. Recreating indices...${c.reset}`);
  db.exec(`CREATE INDEX idx_documents_collection ON documents(collection, active)`);
  db.exec(`CREATE INDEX idx_documents_hash ON documents(hash)`);
  console.log(`  ${c.green}✓${c.reset} Indices created`);

  // Step 8: Update FTS trigger to use collection name
  console.log(`\n${c.yellow}7. Updating FTS trigger...${c.reset}`);
  db.exec(`DROP TRIGGER IF EXISTS documents_ai`);
  db.exec(`
    CREATE TRIGGER documents_ai AFTER INSERT ON documents
    WHEN new.active = 1
    BEGIN
      INSERT INTO documents_fts(rowid, filepath, title, body)
      SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1;
    END
  `);

  db.exec(`DROP TRIGGER IF EXISTS documents_au`);
  db.exec(`
    CREATE TRIGGER documents_au AFTER UPDATE ON documents
    BEGIN
      -- Delete from FTS if no longer active
      DELETE FROM documents_fts WHERE rowid = old.id AND new.active = 0;

      -- Update FTS if still/newly active
      INSERT OR REPLACE INTO documents_fts(rowid, filepath, title, body)
      SELECT
        new.id,
        new.collection || '/' || new.path,
        new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1;
    END
  `);
  console.log(`  ${c.green}✓${c.reset} Triggers updated`);

  // Commit transaction
  db.exec("COMMIT");

  console.log(`\n${c.green}✓ Migration completed successfully!${c.reset}`);

  // Show summary
  const collections = db.query<{ collection: string; count: number }, []>(`
    SELECT collection, COUNT(*) as count
    FROM documents
    WHERE active = 1
    GROUP BY collection
    ORDER BY collection
  `).all();

  console.log(`\n${c.dim}Documents by collection:${c.reset}`);
  for (const coll of collections) {
    console.log(`  ${coll.collection}: ${coll.count} files`);
  }

} catch (error) {
  db.exec("ROLLBACK");
  console.error(`\n${c.yellow}✗ Migration failed:${c.reset} ${error}`);
  console.error(`${c.dim}Database rolled back to previous state${c.reset}`);
  process.exit(1);
} finally {
  db.close();
}
