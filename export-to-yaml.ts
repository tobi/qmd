#!/usr/bin/env bun
/**
 * Export current SQLite collections and contexts to YAML format
 *
 * This script reads from the current database and creates ~/.config/qmd/index.yml
 * Run this once to migrate from database-based to YAML-based configuration.
 */

import { Database } from "bun:sqlite";
import { join } from "path";
import { homedir } from "os";
import { saveConfig, type CollectionConfig, getConfigPath } from "./src/collections";

// Simple colors for output
const c = {
  reset: "\x1b[0m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  dim: "\x1b[2m",
};

// Open the existing database
const dbPath = join(homedir(), ".cache", "qmd", "index.sqlite");
const db = new Database(dbPath, { readonly: true });

console.log(`${c.cyan}Exporting collections from SQLite to YAML...${c.reset}\n`);
console.log(`Database: ${dbPath}`);
console.log(`Output:   ${getConfigPath()}\n`);

// Initialize config
const config: CollectionConfig = {
  global_context: "If you see relevant [[WikiWord]] you can do a search for WikiWord to get more context on the matter",
  collections: {},
};

// Export collections
interface CollectionRow {
  id: number;
  name: string;
  pwd: string;
  glob_pattern: string;
}

const collections = db
  .query<CollectionRow, []>("SELECT id, name, pwd, glob_pattern FROM collections ORDER BY name")
  .all();

console.log(`${c.green}Found ${collections.length} collections:${c.reset}`);

for (const coll of collections) {
  console.log(`  - ${coll.name}`);

  config.collections[coll.name] = {
    path: coll.pwd,
    pattern: coll.glob_pattern,
  };
}

// Export contexts
interface ContextRow {
  collection_id: number;
  collection_name: string;
  path_prefix: string;
  context: string;
}

const contexts = db
  .query<ContextRow, []>(`
    SELECT
      pc.collection_id,
      c.name as collection_name,
      pc.path_prefix,
      pc.context
    FROM path_contexts pc
    JOIN collections c ON pc.collection_id = c.id
    ORDER BY c.name, pc.path_prefix
  `)
  .all();

console.log(`\n${c.green}Found ${contexts.length} contexts:${c.reset}`);

for (const ctx of contexts) {
  const collection = config.collections[ctx.collection_name];
  if (!collection) continue;

  if (!collection.context) {
    collection.context = {};
  }

  collection.context[ctx.path_prefix] = ctx.context;

  // Truncate long contexts for display
  const displayContext = ctx.context.length > 50
    ? ctx.context.substring(0, 50) + "..."
    : ctx.context;

  console.log(`  - ${ctx.collection_name}${ctx.path_prefix}: ${displayContext}`);
}

// Save to YAML
saveConfig(config);

console.log(`\n${c.green}✓ Successfully exported to ${getConfigPath()}${c.reset}`);
console.log(`\n${c.dim}You can now manually edit this file to adjust your collections.${c.reset}`);

db.close();
