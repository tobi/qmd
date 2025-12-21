#!/usr/bin/env bun
import { Database } from "bun:sqlite";
import { Glob, $ } from "bun";
import { parseArgs } from "util";
import * as sqliteVec from "sqlite-vec";
import {
  getPwd,
  getRealPath,
  homedir,
  resolve,
  enableProductionMode,
  searchFTS,
  searchVec,
  reciprocalRankFusion,
  extractSnippet,
  getContextForFile,
  getContextForPath,
  listCollections,
  removeCollection,
  renameCollection,
  findSimilarFiles,
  matchFilesByGlob,
  getHashesNeedingEmbedding,
  getHashesForEmbedding,
  clearAllEmbeddings,
  insertEmbedding,
  getStatus,
  hashContent,
  extractTitle,
  formatDocForEmbedding,
  formatQueryForEmbedding,
  chunkDocument,
  chunkDocumentByTokens,
  clearCache,
  getCacheKey,
  getCachedResult,
  setCachedResult,
  getIndexHealth,
  parseVirtualPath,
  buildVirtualPath,
  isVirtualPath,
  resolveVirtualPath,
  toVirtualPath,
  insertContent,
  insertDocument,
  findActiveDocument,
  updateDocumentTitle,
  updateDocument,
  deactivateDocument,
  getActiveDocumentPaths,
  cleanupOrphanedContent,
  deleteLLMCache,
  deleteInactiveDocuments,
  cleanupOrphanedVectors,
  vacuumDatabase,
  getCollectionsWithoutContext,
  getTopLevelPathsWithoutContext,
  handelize,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_GLOB,
  DEFAULT_MULTI_GET_MAX_BYTES,
  createStore,
  getDefaultDbPath,
} from "./store.js";
import { getDefaultLlamaCpp, disposeDefaultLlamaCpp, type RerankDocument, type ExpandedQuery } from "./llm.js";
import type { SearchResult, RankedResult } from "./store.js";
import {
  formatSearchResults,
  formatDocuments,
  escapeXml,
  escapeCSV,
  type OutputFormat,
} from "./formatter.js";
import {
  getCollection as getCollectionFromYaml,
  listCollections as yamlListCollections,
  addContext as yamlAddContext,
  removeContext as yamlRemoveContext,
  setGlobalContext,
  listAllContexts,
} from "./collections.js";

// Enable production mode - allows using default database path
// Tests must set INDEX_PATH or use createStore() with explicit path
enableProductionMode();

// =============================================================================
// Store/DB lifecycle (no legacy singletons in store.ts)
// =============================================================================

let store: ReturnType<typeof createStore> | null = null;
let storeDbPathOverride: string | undefined;

function getStore(): ReturnType<typeof createStore> {
  if (!store) {
    store = createStore(storeDbPathOverride);
  }
  return store;
}

function getDb(): Database {
  return getStore().db;
}

function closeDb(): void {
  if (store) {
    store.close();
    store = null;
  }
}

function getDbPath(): string {
  return store?.dbPath ?? storeDbPathOverride ?? getDefaultDbPath();
}

function setIndexName(name: string | null): void {
  storeDbPathOverride = name ? getDefaultDbPath(name) : undefined;
  // Reset open handle so next use opens the new index
  closeDb();
}

function ensureVecTable(_db: Database, dimensions: number): void {
  // Store owns the DB; ignore `_db` and ensure vec table on the active store
  getStore().ensureVecTable(dimensions);
}

// Terminal colors (respects NO_COLOR env)
const useColor = !process.env.NO_COLOR && process.stdout.isTTY;
const c = {
  reset: useColor ? "\x1b[0m" : "",
  dim: useColor ? "\x1b[2m" : "",
  bold: useColor ? "\x1b[1m" : "",
  cyan: useColor ? "\x1b[36m" : "",
  yellow: useColor ? "\x1b[33m" : "",
  green: useColor ? "\x1b[32m" : "",
  magenta: useColor ? "\x1b[35m" : "",
  blue: useColor ? "\x1b[34m" : "",
};

// Terminal cursor control
const cursor = {
  hide() { process.stderr.write('\x1b[?25l'); },
  show() { process.stderr.write('\x1b[?25h'); },
};

// Ensure cursor is restored on exit
process.on('SIGINT', () => { cursor.show(); process.exit(130); });
process.on('SIGTERM', () => { cursor.show(); process.exit(143); });

// Terminal progress bar using OSC 9;4 escape sequence
const progress = {
  set(percent: number) {
    process.stderr.write(`\x1b]9;4;1;${Math.round(percent)}\x07`);
  },
  clear() {
    process.stderr.write(`\x1b]9;4;0\x07`);
  },
  indeterminate() {
    process.stderr.write(`\x1b]9;4;3\x07`);
  },
  error() {
    process.stderr.write(`\x1b]9;4;2\x07`);
  },
};

// Format seconds into human-readable ETA
function formatETA(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}


// Check index health and print warnings/tips
function checkIndexHealth(db: Database): void {
  const { needsEmbedding, totalDocs, daysStale } = getIndexHealth(db);

  // Warn if many docs need embedding
  if (needsEmbedding > 0) {
    const pct = Math.round((needsEmbedding / totalDocs) * 100);
    if (pct >= 10) {
      process.stderr.write(`${c.yellow}Warning: ${needsEmbedding} documents (${pct}%) need embeddings. Run 'qmd embed' for better results.${c.reset}\n`);
    } else {
      process.stderr.write(`${c.dim}Tip: ${needsEmbedding} documents need embeddings. Run 'qmd embed' to index them.${c.reset}\n`);
    }
  }

  // Check if most recent document update is older than 2 weeks
  if (daysStale !== null && daysStale >= 14) {
    process.stderr.write(`${c.dim}Tip: Index last updated ${daysStale} days ago. Run 'qmd update' to refresh.${c.reset}\n`);
  }
}

// Compute unique display path for a document
// Always include at least parent folder + filename, add more parent dirs until unique
function computeDisplayPath(
  filepath: string,
  collectionPath: string,
  existingPaths: Set<string>
): string {
  // Get path relative to collection (include collection dir name)
  const collectionDir = collectionPath.replace(/\/$/, '');
  const collectionName = collectionDir.split('/').pop() || '';

  let relativePath: string;
  if (filepath.startsWith(collectionDir + '/')) {
    // filepath is under collection: use collection name + relative path
    relativePath = collectionName + filepath.slice(collectionDir.length);
  } else {
    // Fallback: just use the filepath
    relativePath = filepath;
  }

  const parts = relativePath.split('/').filter(p => p.length > 0);

  // Always include at least parent folder + filename (minimum 2 parts if available)
  // Then add more parent dirs until unique
  const minParts = Math.min(2, parts.length);
  for (let i = parts.length - minParts; i >= 0; i--) {
    const candidate = parts.slice(i).join('/');
    if (!existingPaths.has(candidate)) {
      return candidate;
    }
  }

  // Absolute fallback: use full path (should be unique)
  return filepath;
}

// Rerank documents using node-llama-cpp cross-encoder model
async function rerank(query: string, documents: { file: string; text: string }[], _model: string = DEFAULT_RERANK_MODEL, _db?: Database): Promise<{ file: string; score: number }[]> {
  if (documents.length === 0) return [];

  const total = documents.length;
  process.stderr.write(`Reranking ${total} documents...\n`);
  progress.indeterminate();

  const llm = getDefaultLlamaCpp();
  const rerankDocs: RerankDocument[] = documents.map((doc) => ({
    file: doc.file,
    text: doc.text.slice(0, 4000), // Truncate to context limit
  }));

  const result = await llm.rerank(query, rerankDocs);

  progress.clear();
  process.stderr.write("\n");

  return result.results.map((r) => ({ file: r.file, score: r.score }));
}

function formatTimeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function showStatus(): void {
  const dbPath = getDbPath();
  const db = getDb();

  // Collections are defined in YAML; no duplicate cleanup needed.
  // Collections are defined in YAML; no duplicate cleanup needed.

  // Index size
  let indexSize = 0;
  try {
    const stat = Bun.file(dbPath).size;
    indexSize = stat;
  } catch {}

  // Collections info (from YAML + database stats)
  const collections = listCollections(db);

  // Overall stats
  const totalDocs = db.prepare(`SELECT COUNT(*) as count FROM documents WHERE active = 1`).get() as { count: number };
  const vectorCount = db.prepare(`SELECT COUNT(*) as count FROM content_vectors`).get() as { count: number };
  const needsEmbedding = getHashesNeedingEmbedding(db);

  // Most recent update across all collections
  const mostRecent = db.prepare(`SELECT MAX(modified_at) as latest FROM documents WHERE active = 1`).get() as { latest: string | null };

  console.log(`${c.bold}QMD Status${c.reset}\n`);
  console.log(`Index: ${dbPath}`);
  console.log(`Size:  ${formatBytes(indexSize)}\n`);

  console.log(`${c.bold}Documents${c.reset}`);
  console.log(`  Total:    ${totalDocs.count} files indexed`);
  console.log(`  Vectors:  ${vectorCount.count} embedded`);
  if (needsEmbedding > 0) {
    console.log(`  ${c.yellow}Pending:  ${needsEmbedding} need embedding${c.reset} (run 'qmd embed')`);
  }
  if (mostRecent.latest) {
    const lastUpdate = new Date(mostRecent.latest);
    console.log(`  Updated:  ${formatTimeAgo(lastUpdate)}`);
  }

  // Get all contexts grouped by collection (from YAML)
  const allContexts = listAllContexts();
  const contextsByCollection = new Map<string, { path_prefix: string; context: string }[]>();

  for (const ctx of allContexts) {
    // Group contexts by collection name
    if (!contextsByCollection.has(ctx.collection)) {
      contextsByCollection.set(ctx.collection, []);
    }
    contextsByCollection.get(ctx.collection)!.push({
      path_prefix: ctx.path,
      context: ctx.context
    });
  }

  if (collections.length > 0) {
    console.log(`\n${c.bold}Collections${c.reset}`);
    for (const col of collections) {
      const lastMod = col.last_modified ? formatTimeAgo(new Date(col.last_modified)) : "never";
      const contexts = contextsByCollection.get(col.name) || [];

      console.log(`  ${c.cyan}${col.name}${c.reset} ${c.dim}(qmd://${col.name}/)${c.reset}`);
      console.log(`    ${c.dim}Pattern:${c.reset}  ${col.glob_pattern}`);
      console.log(`    ${c.dim}Files:${c.reset}    ${col.active_count} (updated ${lastMod})`);

      if (contexts.length > 0) {
        console.log(`    ${c.dim}Contexts:${c.reset} ${contexts.length}`);
        for (const ctx of contexts) {
          // Handle both empty string and '/' as root context
          const pathDisplay = (ctx.path_prefix === '' || ctx.path_prefix === '/') ? '/' : `/${ctx.path_prefix}`;
          const contextPreview = ctx.context.length > 60
            ? ctx.context.substring(0, 57) + '...'
            : ctx.context;
          console.log(`      ${c.dim}${pathDisplay}:${c.reset} ${contextPreview}`);
        }
      }
    }

    // Show examples of virtual paths
    console.log(`\n${c.bold}Examples${c.reset}`);
    console.log(`  ${c.dim}# List files in a collection${c.reset}`);
    if (collections.length > 0) {
      console.log(`  qmd ls ${collections[0].name}`);
    }
    console.log(`  ${c.dim}# Get a document${c.reset}`);
    if (collections.length > 0) {
      console.log(`  qmd get qmd://${collections[0].name}/path/to/file.md`);
    }
    console.log(`  ${c.dim}# Search within a collection${c.reset}`);
    if (collections.length > 0) {
      console.log(`  qmd search "query" -c ${collections[0].name}`);
    }
  } else {
    console.log(`\n${c.dim}No collections. Run 'qmd collection add .' to index markdown files.${c.reset}`);
  }

  closeDb();
}

async function updateCollections(): Promise<void> {
  const db = getDb();
  // Collections are defined in YAML; no duplicate cleanup needed.

  // Clear Ollama cache on update
  clearCache(db);

  const collections = listCollections(db);

  if (collections.length === 0) {
    console.log(`${c.dim}No collections found. Run 'qmd collection add .' to index markdown files.${c.reset}`);
    closeDb();
    return;
  }

  // Don't close db here - indexFiles will reuse it and close at the end
  console.log(`${c.bold}Updating ${collections.length} collection(s)...${c.reset}\n`);

  for (let i = 0; i < collections.length; i++) {
    const col = collections[i];
    console.log(`${c.cyan}[${i + 1}/${collections.length}]${c.reset} ${c.bold}${col.name}${c.reset} ${c.dim}(${col.glob_pattern})${c.reset}`);

    // Execute custom update command if specified in YAML
    const yamlCol = getCollectionFromYaml(col.name);
    if (yamlCol?.update) {
      console.log(`${c.dim}    Running update command: ${yamlCol.update}${c.reset}`);
      try {
        const proc = Bun.spawn(["/usr/bin/env", "bash", "-c", yamlCol.update], {
          cwd: col.pwd,
          stdout: "pipe",
          stderr: "pipe",
        });

        const output = await new Response(proc.stdout).text();
        const errorOutput = await new Response(proc.stderr).text();
        const exitCode = await proc.exited;

        if (output.trim()) {
          console.log(output.trim().split('\n').map(l => `    ${l}`).join('\n'));
        }
        if (errorOutput.trim()) {
          console.log(errorOutput.trim().split('\n').map(l => `    ${l}`).join('\n'));
        }

        if (exitCode !== 0) {
          console.log(`${c.yellow}✗ Update command failed with exit code ${exitCode}${c.reset}`);
          process.exit(exitCode);
        }
      } catch (err) {
        console.log(`${c.yellow}✗ Update command failed: ${err}${c.reset}`);
        process.exit(1);
      }
    }

    await indexFiles(col.pwd, col.glob_pattern, col.name);
    console.log("");
  }

  console.log(`${c.green}✓ All collections updated.${c.reset}`);
}

/**
 * Detect which collection (if any) contains the given filesystem path.
 * Returns { collectionId, collectionName, relativePath } or null if not in any collection.
 */
function detectCollectionFromPath(db: Database, fsPath: string): { collectionName: string; relativePath: string } | null {
  const realPath = getRealPath(fsPath);

  // Find collections that this path is under from YAML
  const allCollections = yamlListCollections();

  // Find longest matching path
  let bestMatch: { name: string; path: string } | null = null;
  for (const coll of allCollections) {
    if (realPath.startsWith(coll.path + '/') || realPath === coll.path) {
      if (!bestMatch || coll.path.length > bestMatch.path.length) {
        bestMatch = { name: coll.name, path: coll.path };
      }
    }
  }

  if (!bestMatch) return null;

  // Calculate relative path
  let relativePath = realPath;
  if (relativePath.startsWith(bestMatch.path + '/')) {
    relativePath = relativePath.slice(bestMatch.path.length + 1);
  } else if (relativePath === bestMatch.path) {
    relativePath = '';
  }

  return {
    collectionName: bestMatch.name,
    relativePath
  };
}

async function contextAdd(pathArg: string | undefined, contextText: string): Promise<void> {
  const db = getDb();

  // Handle "/" as global context (applies to all collections)
  if (pathArg === '/') {
    setGlobalContext(contextText);
    console.log(`${c.green}✓${c.reset} Set global context`);
    console.log(`${c.dim}Context: ${contextText}${c.reset}`);
    closeDb();
    return;
  }

  // Resolve path - defaults to current directory if not provided
  let fsPath = pathArg || '.';
  if (fsPath === '.' || fsPath === './') {
    fsPath = getPwd();
  } else if (fsPath.startsWith('~/')) {
    fsPath = homedir() + fsPath.slice(1);
  } else if (!fsPath.startsWith('/') && !fsPath.startsWith('qmd://')) {
    fsPath = resolve(getPwd(), fsPath);
  }

  // Handle virtual paths (qmd://collection/path)
  if (isVirtualPath(fsPath)) {
    const parsed = parseVirtualPath(fsPath);
    if (!parsed) {
      console.error(`${c.yellow}Invalid virtual path: ${fsPath}${c.reset}`);
      process.exit(1);
    }

    const coll = getCollectionFromYaml(parsed.collectionName);
    if (!coll) {
      console.error(`${c.yellow}Collection not found: ${parsed.collectionName}${c.reset}`);
      process.exit(1);
    }

    yamlAddContext(parsed.collectionName, parsed.path, contextText);

    const displayPath = parsed.path
      ? `qmd://${parsed.collectionName}/${parsed.path}`
      : `qmd://${parsed.collectionName}/ (collection root)`;
    console.log(`${c.green}✓${c.reset} Added context for: ${displayPath}`);
    console.log(`${c.dim}Context: ${contextText}${c.reset}`);
    closeDb();
    return;
  }

  // Detect collection from filesystem path
  const detected = detectCollectionFromPath(db, fsPath);
  if (!detected) {
    console.error(`${c.yellow}Path is not in any indexed collection: ${fsPath}${c.reset}`);
    console.error(`${c.dim}Run 'qmd status' to see indexed collections${c.reset}`);
    process.exit(1);
  }

  yamlAddContext(detected.collectionName, detected.relativePath, contextText);

  const displayPath = detected.relativePath ? `qmd://${detected.collectionName}/${detected.relativePath}` : `qmd://${detected.collectionName}/`;
  console.log(`${c.green}✓${c.reset} Added context for: ${displayPath}`);
  console.log(`${c.dim}Context: ${contextText}${c.reset}`);
  closeDb();
}

function contextList(): void {
  const db = getDb();

  const allContexts = listAllContexts();

  if (allContexts.length === 0) {
    console.log(`${c.dim}No contexts configured. Use 'qmd context add' to add one.${c.reset}`);
    closeDb();
    return;
  }

  console.log(`\n${c.bold}Configured Contexts${c.reset}\n`);

  let lastCollection = '';
  for (const ctx of allContexts) {
    if (ctx.collection !== lastCollection) {
      console.log(`${c.cyan}${ctx.collection}${c.reset}`);
      lastCollection = ctx.collection;
    }

    const displayPath = ctx.path ? `  ${ctx.path}` : '  / (root)';
    console.log(`${displayPath}`);
    console.log(`    ${c.dim}${ctx.context}${c.reset}`);
  }

  closeDb();
}

function contextRemove(pathArg: string): void {
  if (pathArg === '/') {
    // Remove global context
    setGlobalContext(undefined);
    console.log(`${c.green}✓${c.reset} Removed global context`);
    return;
  }

  // Handle virtual paths
  if (isVirtualPath(pathArg)) {
    const parsed = parseVirtualPath(pathArg);
    if (!parsed) {
      console.error(`${c.yellow}Invalid virtual path: ${pathArg}${c.reset}`);
      process.exit(1);
    }

    const coll = getCollectionFromYaml(parsed.collectionName);
    if (!coll) {
      console.error(`${c.yellow}Collection not found: ${parsed.collectionName}${c.reset}`);
      process.exit(1);
    }

    const success = yamlRemoveContext(coll.name, parsed.path);

    if (!success) {
      console.error(`${c.yellow}No context found for: ${pathArg}${c.reset}`);
      process.exit(1);
    }

    console.log(`${c.green}✓${c.reset} Removed context for: ${pathArg}`);
    return;
  }

  // Handle filesystem paths
  let fsPath = pathArg;
  if (fsPath === '.' || fsPath === './') {
    fsPath = getPwd();
  } else if (fsPath.startsWith('~/')) {
    fsPath = homedir() + fsPath.slice(1);
  } else if (!fsPath.startsWith('/')) {
    fsPath = resolve(getPwd(), fsPath);
  }

  const db = getDb();
  const detected = detectCollectionFromPath(db, fsPath);
  closeDb();

  if (!detected) {
    console.error(`${c.yellow}Path is not in any indexed collection: ${fsPath}${c.reset}`);
    process.exit(1);
  }

  const success = yamlRemoveContext(detected.collectionName, detected.relativePath);

  if (!success) {
    console.error(`${c.yellow}No context found for: qmd://${detected.collectionName}/${detected.relativePath}${c.reset}`);
    process.exit(1);
  }

  console.log(`${c.green}✓${c.reset} Removed context for: qmd://${detected.collectionName}/${detected.relativePath}`);
}

function contextCheck(): void {
  const db = getDb();

  // Get collections without any context
  const collectionsWithoutContext = getCollectionsWithoutContext(db);

  // Get all collections to check for missing path contexts
  const allCollections = listCollections(db);

  if (collectionsWithoutContext.length === 0 && allCollections.length > 0) {
    // Check if all collections have contexts
    console.log(`\n${c.green}✓${c.reset} ${c.bold}All collections have context configured${c.reset}\n`);
  }

  if (collectionsWithoutContext.length > 0) {
    console.log(`\n${c.yellow}Collections without any context:${c.reset}\n`);

    for (const coll of collectionsWithoutContext) {
      console.log(`${c.cyan}${coll.name}${c.reset} ${c.dim}(${coll.doc_count} documents)${c.reset}`);
      console.log(`  ${c.dim}Suggestion: qmd context add qmd://${coll.name}/ "Description of ${coll.name}"${c.reset}\n`);
    }
  }

  // Check for top-level paths without context within collections that DO have context
  const collectionsWithContext = allCollections.filter(c =>
    !collectionsWithoutContext.some(cwc => cwc.id === c.id)
  );

  let hasPathSuggestions = false;

  for (const coll of collectionsWithContext) {
    const missingPaths = getTopLevelPathsWithoutContext(db, coll.id);

    if (missingPaths.length > 0) {
      if (!hasPathSuggestions) {
        console.log(`${c.yellow}Top-level directories without context:${c.reset}\n`);
        hasPathSuggestions = true;
      }

      console.log(`${c.cyan}${coll.name}${c.reset}`);
      for (const path of missingPaths) {
        console.log(`  ${path}`);
        console.log(`    ${c.dim}Suggestion: qmd context add qmd://${coll.name}/${path} "Description of ${path}"${c.reset}`);
      }
      console.log('');
    }
  }

  if (collectionsWithoutContext.length === 0 && !hasPathSuggestions) {
    console.log(`${c.dim}All collections and major paths have context configured.${c.reset}`);
    console.log(`${c.dim}Use 'qmd context list' to see all configured contexts.${c.reset}\n`);
  }

  closeDb();
}

function getDocument(filename: string, fromLine?: number, maxLines?: number, lineNumbers?: boolean): void {
  const db = getDb();

  // Parse :linenum suffix from filename (e.g., "file.md:100")
  let inputPath = filename;
  const colonMatch = inputPath.match(/:(\d+)$/);
  if (colonMatch && !fromLine) {
    fromLine = parseInt(colonMatch[1], 10);
    inputPath = inputPath.slice(0, -colonMatch[0].length);
  }

  let doc: { collectionName: string; path: string; body: string } | null = null;
  let virtualPath: string;

  // Handle virtual paths (qmd://collection/path)
  if (isVirtualPath(inputPath)) {
    const parsed = parseVirtualPath(inputPath);
    if (!parsed) {
      console.error(`Invalid virtual path: ${inputPath}`);
      closeDb();
      process.exit(1);
    }

    // Try exact match on collection + path
    doc = db.prepare(`
      SELECT d.collection as collectionName, d.path, content.doc as body
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE d.collection = ? AND d.path = ? AND d.active = 1
    `).get(parsed.collectionName, parsed.path) as typeof doc;

    if (!doc) {
      // Try fuzzy match by path ending
      doc = db.prepare(`
        SELECT d.collection as collectionName, d.path, content.doc as body
        FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
        LIMIT 1
      `).get(parsed.collectionName, `%${parsed.path}`) as typeof doc;
    }

    virtualPath = inputPath;
  } else {
    // Try to interpret as collection/path format first (before filesystem path)
    // If path is relative (no / or ~ prefix), check if first component is a collection name
    if (!inputPath.startsWith('/') && !inputPath.startsWith('~')) {
      const parts = inputPath.split('/');
      if (parts.length >= 2) {
        const possibleCollection = parts[0];
        const possiblePath = parts.slice(1).join('/');

        // Check if this collection exists
        const collExists = db.prepare(`
          SELECT 1 FROM documents WHERE collection = ? AND active = 1 LIMIT 1
        `).get(possibleCollection);

        if (collExists) {
          // Try exact match on collection + path
          doc = db.prepare(`
            SELECT d.collection as collectionName, d.path, content.doc as body
            FROM documents d
            JOIN content ON content.hash = d.hash
            WHERE d.collection = ? AND d.path = ? AND d.active = 1
          `).get(possibleCollection, possiblePath) as typeof doc;

          if (!doc) {
            // Try fuzzy match by path ending
            doc = db.prepare(`
              SELECT d.collection as collectionName, d.path, content.doc as body
              FROM documents d
              JOIN content ON content.hash = d.hash
              WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
              LIMIT 1
            `).get(possibleCollection, `%${possiblePath}`) as typeof doc;
          }

          if (doc) {
            virtualPath = buildVirtualPath(doc.collectionName, doc.path);
            // Skip the filesystem path handling below
          }
        }
      }
    }

    // If not found as collection/path, handle as filesystem paths
    if (!doc) {
      let fsPath = inputPath;

      // Expand ~ to home directory
      if (fsPath.startsWith('~/')) {
        fsPath = homedir() + fsPath.slice(1);
      } else if (!fsPath.startsWith('/')) {
        // Relative path - resolve from current directory
        fsPath = resolve(getPwd(), fsPath);
      }
      fsPath = getRealPath(fsPath);

      // Try to detect which collection contains this path
      const detected = detectCollectionFromPath(db, fsPath);

      if (detected) {
        // Found collection - query by collection name + relative path
        doc = db.prepare(`
          SELECT d.collection as collectionName, d.path, content.doc as body
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.collection = ? AND d.path = ? AND d.active = 1
        `).get(detected.collectionName, detected.relativePath) as typeof doc;
      }

      // Fuzzy match by filename (last component of path)
      if (!doc) {
        const filename = inputPath.split('/').pop() || inputPath;
        doc = db.prepare(`
          SELECT d.collection as collectionName, d.path, content.doc as body
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.path LIKE ? AND d.active = 1
          LIMIT 1
        `).get(`%${filename}`) as typeof doc;
      }

      if (doc) {
        virtualPath = buildVirtualPath(doc.collectionName, doc.path);
      } else {
        virtualPath = inputPath;
      }
    }
  }

  if (!doc) {
    console.error(`Document not found: ${filename}`);
    closeDb();
    process.exit(1);
  }

  // Get context for this file
  const context = getContextForPath(db, doc.collectionName, doc.path);

  let output = doc.body;
  const startLine = fromLine || 1;

  // Apply line filtering if specified
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = output.split('\n');
    const start = startLine - 1; // Convert to 0-indexed
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    output = lines.slice(start, end).join('\n');
  }

  // Add line numbers if requested
  if (lineNumbers) {
    output = addLineNumbers(output, startLine);
  }

  // Output context header if exists
  if (context) {
    console.log(`Folder Context: ${context}\n---\n`);
  }
  console.log(output);
  closeDb();
}

// Multi-get: fetch multiple documents by glob pattern or comma-separated list
function multiGet(pattern: string, maxLines?: number, maxBytes: number = DEFAULT_MULTI_GET_MAX_BYTES, format: OutputFormat = "cli"): void {
  const db = getDb();

  // Check if it's a comma-separated list or a glob pattern
  const isCommaSeparated = pattern.includes(',') && !pattern.includes('*') && !pattern.includes('?');

  let files: { filepath: string; displayPath: string; bodyLength: number; collection?: string; path?: string }[];

  if (isCommaSeparated) {
    // Comma-separated list of files (can be virtual paths or relative paths)
    const names = pattern.split(',').map(s => s.trim()).filter(Boolean);
    files = [];
    for (const name of names) {
      let doc: { virtual_path: string; body_length: number; collection: string; path: string } | null = null;

      // Handle virtual paths
      if (isVirtualPath(name)) {
        const parsed = parseVirtualPath(name);
        if (parsed) {
          // Try exact match on collection + path
          doc = db.prepare(`
            SELECT
              'qmd://' || d.collection || '/' || d.path as virtual_path,
              LENGTH(content.doc) as body_length,
              d.collection,
              d.path
            FROM documents d
            JOIN content ON content.hash = d.hash
            WHERE d.collection = ? AND d.path = ? AND d.active = 1
          `).get(parsed.collectionName, parsed.path) as typeof doc;
        }
      } else {
        // Try exact match on path
        doc = db.prepare(`
          SELECT
            'qmd://' || d.collection || '/' || d.path as virtual_path,
            LENGTH(content.doc) as body_length,
            d.collection,
            d.path
          FROM documents d
          JOIN content ON content.hash = d.hash
          WHERE d.path = ? AND d.active = 1
          LIMIT 1
        `).get(name) as typeof doc;

        // Try suffix match
        if (!doc) {
          doc = db.prepare(`
            SELECT
              'qmd://' || d.collection || '/' || d.path as virtual_path,
              LENGTH(content.doc) as body_length,
              d.collection,
              d.path
            FROM documents d
            JOIN content ON content.hash = d.hash
            WHERE d.path LIKE ? AND d.active = 1
            LIMIT 1
          `).get(`%${name}`) as typeof doc;
        }
      }

      if (doc) {
        files.push({
          filepath: doc.virtual_path,
          displayPath: doc.virtual_path,
          bodyLength: doc.body_length,
          collection: doc.collection,
          path: doc.path
        });
      } else {
        console.error(`File not found: ${name}`);
      }
    }
  } else {
    // Glob pattern - matchFilesByGlob now returns virtual paths
    files = matchFilesByGlob(db, pattern).map(f => ({
      ...f,
      collection: undefined,  // Will be fetched later if needed
      path: undefined
    }));
    if (files.length === 0) {
      console.error(`No files matched pattern: ${pattern}`);
      closeDb();
      process.exit(1);
    }
  }

  // Collect results for structured output
  const results: { file: string; displayPath: string; title: string; body: string; context: string | null; skipped: boolean; skipReason?: string }[] = [];

  for (const file of files) {
    // Parse virtual path to get collection info if not already available
    let collection = file.collection;
    let path = file.path;

    if (!collection || !path) {
      const parsed = parseVirtualPath(file.filepath);
      if (parsed) {
        collection = parsed.collectionName;
        path = parsed.path;
      }
    }

    // Get context using collection-scoped function
    const context = collection && path ? getContextForPath(db, collection, path) : null;

    // Check size limit
    if (file.bodyLength > maxBytes) {
      results.push({
        file: file.filepath,
        displayPath: file.displayPath,
        title: file.displayPath.split('/').pop() || file.displayPath,
        body: "",
        context,
        skipped: true,
        skipReason: `File too large (${Math.round(file.bodyLength / 1024)}KB > ${Math.round(maxBytes / 1024)}KB). Use 'qmd get ${file.displayPath}' to retrieve.`,
      });
      continue;
    }

    // Fetch document content using collection and path
    if (!collection || !path) continue;

    const doc = db.prepare(`
      SELECT content.doc as body, d.title
      FROM documents d
      JOIN content ON content.hash = d.hash
      WHERE d.collection = ? AND d.path = ? AND d.active = 1
    `).get(collection, path) as { body: string; title: string } | null;

    if (!doc) continue;

    let body = doc.body;

    // Apply line limit if specified
    if (maxLines !== undefined) {
      const lines = body.split('\n');
      body = lines.slice(0, maxLines).join('\n');
      if (lines.length > maxLines) {
        body += `\n\n[... truncated ${lines.length - maxLines} more lines]`;
      }
    }

    results.push({
      file: file.filepath,
      displayPath: file.displayPath,
      title: doc.title || file.displayPath.split('/').pop() || file.displayPath,
      body,
      context,
      skipped: false,
    });
  }

  closeDb();

  // Output based on format
  if (format === "json") {
    const output = results.map(r => ({
      file: r.displayPath,
      title: r.title,
      ...(r.context && { context: r.context }),
      ...(r.skipped ? { skipped: true, reason: r.skipReason } : { body: r.body }),
    }));
    console.log(JSON.stringify(output, null, 2));
  } else if (format === "csv") {
    const escapeField = (val: string | null): string => {
      if (val === null || val === undefined) return "";
      const str = String(val);
      if (str.includes(",") || str.includes('"') || str.includes("\n")) {
        return `"${str.replace(/"/g, '""')}"`;
      }
      return str;
    };
    console.log("file,title,context,skipped,body");
    for (const r of results) {
      console.log([r.displayPath, r.title, r.context || "", r.skipped ? "true" : "false", r.skipped ? r.skipReason : r.body].map(escapeField).join(","));
    }
  } else if (format === "files") {
    for (const r of results) {
      const ctx = r.context ? `,"${r.context.replace(/"/g, '""')}"` : "";
      const status = r.skipped ? "[SKIPPED]" : "";
      console.log(`${r.displayPath}${ctx}${status ? `,${status}` : ""}`);
    }
  } else if (format === "md") {
    for (const r of results) {
      console.log(`## ${r.displayPath}\n`);
      if (r.title && r.title !== r.displayPath) console.log(`**Title:** ${r.title}\n`);
      if (r.context) console.log(`**Context:** ${r.context}\n`);
      if (r.skipped) {
        console.log(`> ${r.skipReason}\n`);
      } else {
        console.log("```");
        console.log(r.body);
        console.log("```\n");
      }
    }
  } else if (format === "xml") {
    console.log('<?xml version="1.0" encoding="UTF-8"?>');
    console.log("<documents>");
    for (const r of results) {
      console.log("  <document>");
      console.log(`    <file>${escapeXml(r.displayPath)}</file>`);
      console.log(`    <title>${escapeXml(r.title)}</title>`);
      if (r.context) console.log(`    <context>${escapeXml(r.context)}</context>`);
      if (r.skipped) {
        console.log(`    <skipped>true</skipped>`);
        console.log(`    <reason>${escapeXml(r.skipReason || "")}</reason>`);
      } else {
        console.log(`    <body>${escapeXml(r.body)}</body>`);
      }
      console.log("  </document>");
    }
    console.log("</documents>");
  } else {
    // CLI format (default)
    for (const r of results) {
      console.log(`\n${'='.repeat(60)}`);
      console.log(`File: ${r.displayPath}`);
      console.log(`${'='.repeat(60)}\n`);

      if (r.skipped) {
        console.log(`[SKIPPED: ${r.skipReason}]`);
        continue;
      }

      if (r.context) {
        console.log(`Folder Context: ${r.context}\n---\n`);
      }
      console.log(r.body);
    }
  }
}

// List files in virtual file tree
function listFiles(pathArg?: string): void {
  const db = getDb();

  if (!pathArg) {
    // No argument - list all collections
    const yamlCollections = yamlListCollections();

    if (yamlCollections.length === 0) {
      console.log("No collections found. Run 'qmd add .' to index files.");
      closeDb();
      return;
    }

    // Get file counts from database for each collection
    const collections = yamlCollections.map(coll => {
      const stats = db.prepare(`
        SELECT COUNT(*) as file_count
        FROM documents d
        WHERE d.collection = ? AND d.active = 1
      `).get(coll.name) as { file_count: number } | null;

      return {
        name: coll.name,
        file_count: stats?.file_count || 0
      };
    });

    console.log(`${c.bold}Collections:${c.reset}\n`);
    for (const coll of collections) {
      console.log(`  ${c.dim}qmd://${c.reset}${c.cyan}${coll.name}/${c.reset}  ${c.dim}(${coll.file_count} files)${c.reset}`);
    }
    closeDb();
    return;
  }

  // Parse the path argument
  let collectionName: string;
  let pathPrefix: string | null = null;

  if (pathArg.startsWith('qmd://')) {
    // Virtual path format: qmd://collection/path
    const parsed = parseVirtualPath(pathArg);
    if (!parsed) {
      console.error(`Invalid virtual path: ${pathArg}`);
      closeDb();
      process.exit(1);
    }
    collectionName = parsed.collectionName;
    pathPrefix = parsed.path;
  } else {
    // Just collection name or collection/path
    const parts = pathArg.split('/');
    collectionName = parts[0];
    if (parts.length > 1) {
      pathPrefix = parts.slice(1).join('/');
    }
  }

  // Get the collection
  const coll = getCollectionFromYaml(collectionName);
  if (!coll) {
    console.error(`Collection not found: ${collectionName}`);
    console.error(`Run 'qmd ls' to see available collections.`);
    closeDb();
    process.exit(1);
  }

  // List files in the collection with size and modification time
  let query: string;
  let params: any[];

  if (pathPrefix) {
    // List files under a specific path
    query = `
      SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) as size
      FROM documents d
      JOIN content ct ON d.hash = ct.hash
      WHERE d.collection = ? AND d.path LIKE ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.name, `${pathPrefix}%`];
  } else {
    // List all files in the collection
    query = `
      SELECT d.path, d.title, d.modified_at, LENGTH(ct.doc) as size
      FROM documents d
      JOIN content ct ON d.hash = ct.hash
      WHERE d.collection = ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.name];
  }

  const files = db.prepare(query).all(...params) as { path: string; title: string; modified_at: string; size: number }[];

  if (files.length === 0) {
    if (pathPrefix) {
      console.log(`No files found under qmd://${collectionName}/${pathPrefix}`);
    } else {
      console.log(`No files found in collection: ${collectionName}`);
    }
    closeDb();
    return;
  }

  // Calculate max widths for alignment
  const maxSize = Math.max(...files.map(f => formatBytes(f.size).length));

  // Output in ls -l style
  for (const file of files) {
    const sizeStr = formatBytes(file.size).padStart(maxSize);
    const date = new Date(file.modified_at);
    const timeStr = formatLsTime(date);

    // Dim the qmd:// prefix, highlight the filename
    console.log(`${sizeStr}  ${timeStr}  ${c.dim}qmd://${collectionName}/${c.reset}${c.cyan}${file.path}${c.reset}`);
  }

  closeDb();
}

// Format date/time like ls -l
function formatLsTime(date: Date): string {
  const now = new Date();
  const sixMonthsAgo = new Date(now.getTime() - 6 * 30 * 24 * 60 * 60 * 1000);

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const month = months[date.getMonth()];
  const day = date.getDate().toString().padStart(2, ' ');

  // If file is older than 6 months, show year instead of time
  if (date < sixMonthsAgo) {
    const year = date.getFullYear();
    return `${month} ${day}  ${year}`;
  } else {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${month} ${day} ${hours}:${minutes}`;
  }
}

// Collection management commands
function collectionList(): void {
  const db = getDb();
  const collections = listCollections(db);

  if (collections.length === 0) {
    console.log("No collections found. Run 'qmd add .' to create one.");
    closeDb();
    return;
  }

  console.log(`${c.bold}Collections (${collections.length}):${c.reset}\n`);

  for (const coll of collections) {
    const updatedAt = new Date(coll.updated_at);
    const timeAgo = formatTimeAgo(updatedAt);

    console.log(`${c.cyan}${coll.name}${c.reset} ${c.dim}(qmd://${coll.name}/)${c.reset}`);
    console.log(`  ${c.dim}Pattern:${c.reset}  ${coll.glob_pattern}`);
    console.log(`  ${c.dim}Files:${c.reset}    ${coll.active_count}`);
    console.log(`  ${c.dim}Updated:${c.reset}  ${timeAgo}`);
    console.log();
  }

  closeDb();
}

async function collectionAdd(pwd: string, globPattern: string, name?: string): Promise<void> {
  // If name not provided, generate from pwd basename
  if (!name) {
    const parts = pwd.split('/').filter(Boolean);
    name = parts[parts.length - 1] || 'root';
  }

  // Check if collection with this name already exists in YAML
  const existing = getCollectionFromYaml(name);
  if (existing) {
    console.error(`${c.yellow}Collection '${name}' already exists.${c.reset}`);
    console.error(`Use a different name with --name <name>`);
    process.exit(1);
  }

  // Check if a collection with this pwd+glob already exists in YAML
  const allCollections = yamlListCollections();
  const existingPwdGlob = allCollections.find(c => c.path === pwd && c.pattern === globPattern);

  if (existingPwdGlob) {
    console.error(`${c.yellow}A collection already exists for this path and pattern:${c.reset}`);
    console.error(`  Name: ${existingPwdGlob.name} (qmd://${existingPwdGlob.name}/)`);
    console.error(`  Pattern: ${globPattern}`);
    console.error(`\nUse 'qmd update' to re-index it, or remove it first with 'qmd collection remove ${existingPwdGlob.name}'`);
    process.exit(1);
  }

  // Add to YAML config
  const { addCollection } = await import("./collections.js");
  addCollection(name, pwd, globPattern);

  // Create the collection and index files
  console.log(`Creating collection '${name}'...`);
  await indexFiles(pwd, globPattern, name);
  console.log(`${c.green}✓${c.reset} Collection '${name}' created successfully`);
}

function collectionRemove(name: string): void {
  // Check if collection exists in YAML
  const coll = getCollectionFromYaml(name);
  if (!coll) {
    console.error(`${c.yellow}Collection not found: ${name}${c.reset}`);
    console.error(`Run 'qmd collection list' to see available collections.`);
    process.exit(1);
  }

  const db = getDb();
  const result = removeCollection(db, name);
  closeDb();

  console.log(`${c.green}✓${c.reset} Removed collection '${name}'`);
  console.log(`  Deleted ${result.deletedDocs} documents`);
  if (result.cleanedHashes > 0) {
    console.log(`  Cleaned up ${result.cleanedHashes} orphaned content hashes`);
  }
}

function collectionRename(oldName: string, newName: string): void {
  // Check if old collection exists in YAML
  const coll = getCollectionFromYaml(oldName);
  if (!coll) {
    console.error(`${c.yellow}Collection not found: ${oldName}${c.reset}`);
    console.error(`Run 'qmd collection list' to see available collections.`);
    process.exit(1);
  }

  // Check if new name already exists in YAML
  const existing = getCollectionFromYaml(newName);
  if (existing) {
    console.error(`${c.yellow}Collection name already exists: ${newName}${c.reset}`);
    console.error(`Choose a different name or remove the existing collection first.`);
    process.exit(1);
  }

  const db = getDb();
  renameCollection(db, oldName, newName);
  closeDb();

  console.log(`${c.green}✓${c.reset} Renamed collection '${oldName}' to '${newName}'`);
  console.log(`  Virtual paths updated: ${c.cyan}qmd://${oldName}/${c.reset} → ${c.cyan}qmd://${newName}/${c.reset}`);
}

async function indexFiles(pwd?: string, globPattern: string = DEFAULT_GLOB, collectionName?: string): Promise<void> {
  const db = getDb();
  const resolvedPwd = pwd || getPwd();
  const now = new Date().toISOString();
  const excludeDirs = ["node_modules", ".git", ".cache", "vendor", "dist", "build"];

  // Clear Ollama cache on index
  clearCache(db);

  // Collection name must be provided (from YAML)
  if (!collectionName) {
    throw new Error("Collection name is required. Collections must be defined in ~/.config/qmd/index.yml");
  }

  console.log(`Collection: ${resolvedPwd} (${globPattern})`);

  progress.indeterminate();
  const glob = new Glob(globPattern);
  const files: string[] = [];
  for await (const file of glob.scan({ cwd: resolvedPwd, onlyFiles: true, followSymlinks: true })) {
    // Skip node_modules, hidden folders (.*), and other common excludes
    const parts = file.split("/");
    const shouldSkip = parts.some(part =>
      part === "node_modules" ||
      part.startsWith(".") ||
      excludeDirs.includes(part)
    );
    if (!shouldSkip) {
      files.push(file);
    }
  }

  const total = files.length;
  if (total === 0) {
    progress.clear();
    console.log("No files found matching pattern.");
    closeDb();
    return;
  }

  let indexed = 0, updated = 0, unchanged = 0, processed = 0;
  const seenPaths = new Set<string>();
  const startTime = Date.now();

  for (const relativeFile of files) {
    const filepath = getRealPath(resolve(resolvedPwd, relativeFile));
    const path = handelize(relativeFile); // Normalize path for token-friendliness
    seenPaths.add(path);

    const content = await Bun.file(filepath).text();
    const hash = await hashContent(content);
    const title = extractTitle(content, relativeFile);

    // Check if document exists in this collection with this path
    const existing = findActiveDocument(db, collectionName, path);

    if (existing) {
      if (existing.hash === hash) {
        // Hash unchanged, but check if title needs updating
        if (existing.title !== title) {
          updateDocumentTitle(db, existing.id, title, now);
          updated++;
        } else {
          unchanged++;
        }
      } else {
        // Content changed - insert new content hash and update document
        insertContent(db, hash, content, now);
        const stat = await Bun.file(filepath).stat();
        updateDocument(db, existing.id, title, hash,
          stat ? new Date(stat.mtime).toISOString() : now);
        updated++;
      }
    } else {
      // New document - insert content and document
      indexed++;
      insertContent(db, hash, content, now);
      const stat = await Bun.file(filepath).stat();
      insertDocument(db, collectionName, path, title, hash,
        stat ? new Date(stat.birthtime).toISOString() : now,
        stat ? new Date(stat.mtime).toISOString() : now);
    }

    processed++;
    progress.set((processed / total) * 100);
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = processed / elapsed;
    const remaining = (total - processed) / rate;
    const eta = processed > 2 ? ` ETA: ${formatETA(remaining)}` : "";
    process.stderr.write(`\rIndexing: ${processed}/${total}${eta}        `);
  }

  // Deactivate documents in this collection that no longer exist
  const allActive = getActiveDocumentPaths(db, collectionName);
  let removed = 0;
  for (const path of allActive) {
    if (!seenPaths.has(path)) {
      deactivateDocument(db, collectionName, path);
      removed++;
    }
  }

  // Clean up orphaned content hashes (content not referenced by any document)
  const orphanedContent = cleanupOrphanedContent(db);

  // Check if vector index needs updating
  const needsEmbedding = getHashesNeedingEmbedding(db);

  progress.clear();
  console.log(`\nIndexed: ${indexed} new, ${updated} updated, ${unchanged} unchanged, ${removed} removed`);
  if (orphanedContent > 0) {
    console.log(`Cleaned up ${orphanedContent} orphaned content hash(es)`);
  }

  if (needsEmbedding > 0) {
    console.log(`\nRun 'qmd embed' to update embeddings (${needsEmbedding} unique hashes need vectors)`);
  }

  closeDb();
}

function renderProgressBar(percent: number, width: number = 30): string {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  const bar = "█".repeat(filled) + "░".repeat(empty);
  return bar;
}

async function vectorIndex(model: string = DEFAULT_EMBED_MODEL, force: boolean = false): Promise<void> {
  const db = getDb();
  const now = new Date().toISOString();

  // If force, clear all vectors
  if (force) {
    console.log(`${c.yellow}Force re-indexing: clearing all vectors...${c.reset}`);
    clearAllEmbeddings(db);
  }

  // Find unique hashes that need embedding (from active documents)
  const hashesToEmbed = getHashesForEmbedding(db);

  if (hashesToEmbed.length === 0) {
    console.log(`${c.green}✓ All content hashes already have embeddings.${c.reset}`);
    closeDb();
    return;
  }

  // Prepare documents with chunks
  type ChunkItem = { hash: string; title: string; text: string; seq: number; pos: number; tokens: number; bytes: number; displayName: string };
  const allChunks: ChunkItem[] = [];
  let multiChunkDocs = 0;

  // Chunk all documents using actual token counts
  process.stderr.write(`Chunking ${hashesToEmbed.length} documents by token count...\n`);
  for (const item of hashesToEmbed) {
    const encoder = new TextEncoder();
    const bodyBytes = encoder.encode(item.body).length;
    if (bodyBytes === 0) continue; // Skip empty

    const title = extractTitle(item.body, item.path);
    const displayName = item.path;
    const chunks = await chunkDocumentByTokens(item.body);  // Uses actual tokenizer

    if (chunks.length > 1) multiChunkDocs++;

    for (let seq = 0; seq < chunks.length; seq++) {
      allChunks.push({
        hash: item.hash,
        title,
        text: chunks[seq].text,
        seq,
        pos: chunks[seq].pos,
        tokens: chunks[seq].tokens,
        bytes: encoder.encode(chunks[seq].text).length,
        displayName,
      });
    }
  }

  if (allChunks.length === 0) {
    console.log(`${c.green}✓ No non-empty documents to embed.${c.reset}`);
    closeDb();
    return;
  }

  const totalBytes = allChunks.reduce((sum, c) => sum + c.bytes, 0);
  const totalChunks = allChunks.length;
  const totalDocs = hashesToEmbed.length;

  console.log(`${c.bold}Embedding ${totalDocs} documents${c.reset} ${c.dim}(${totalChunks} chunks, ${formatBytes(totalBytes)})${c.reset}`);
  if (multiChunkDocs > 0) {
    console.log(`${c.dim}${multiChunkDocs} documents split into multiple chunks${c.reset}`);
  }
  console.log(`${c.dim}Model: ${model}${c.reset}\n`);

  // Hide cursor during embedding
  cursor.hide();

  // Get embedding dimensions from first chunk
  progress.indeterminate();
  const llm = getDefaultLlamaCpp();
  const firstText = formatDocForEmbedding(allChunks[0].text, allChunks[0].title);
  const firstResult = await llm.embed(firstText);
  if (!firstResult) {
    throw new Error("Failed to get embedding dimensions from first chunk");
  }
  ensureVecTable(db, firstResult.embedding.length);

  let chunksEmbedded = 0, errors = 0, bytesProcessed = 0;
  const startTime = Date.now();

  // Batch embedding for better throughput
  // Process in batches of 32 to balance memory usage and efficiency
  const BATCH_SIZE = 32;

  for (let batchStart = 0; batchStart < allChunks.length; batchStart += BATCH_SIZE) {
    const batchEnd = Math.min(batchStart + BATCH_SIZE, allChunks.length);
    const batch = allChunks.slice(batchStart, batchEnd);

    // Format texts for embedding
    const texts = batch.map(chunk => formatDocForEmbedding(chunk.text, chunk.title));

    try {
      // Batch embed all texts at once
      const embeddings = await llm.embedBatch(texts);

      // Insert each embedding
      for (let i = 0; i < batch.length; i++) {
        const chunk = batch[i];
        const embedding = embeddings[i];

        if (embedding) {
          insertEmbedding(db, chunk.hash, chunk.seq, chunk.pos, new Float32Array(embedding.embedding), model, now);
          chunksEmbedded++;
        } else {
          errors++;
          console.error(`\n${c.yellow}⚠ Error embedding "${chunk.displayName}" chunk ${chunk.seq}${c.reset}`);
        }
        bytesProcessed += chunk.bytes;
      }
    } catch (err) {
      // If batch fails, try individual embeddings as fallback
      for (const chunk of batch) {
        try {
          const text = formatDocForEmbedding(chunk.text, chunk.title);
          const result = await llm.embed(text);
          if (result) {
            insertEmbedding(db, chunk.hash, chunk.seq, chunk.pos, new Float32Array(result.embedding), model, now);
            chunksEmbedded++;
          } else {
            errors++;
          }
        } catch (innerErr) {
          errors++;
          console.error(`\n${c.yellow}⚠ Error embedding "${chunk.displayName}" chunk ${chunk.seq}: ${innerErr}${c.reset}`);
        }
        bytesProcessed += chunk.bytes;
      }
    }

    const percent = (bytesProcessed / totalBytes) * 100;
    progress.set(percent);

    const elapsed = (Date.now() - startTime) / 1000;
    const bytesPerSec = bytesProcessed / elapsed;
    const remainingBytes = totalBytes - bytesProcessed;
    const etaSec = remainingBytes / bytesPerSec;

    const bar = renderProgressBar(percent);
    const percentStr = percent.toFixed(0).padStart(3);
    const throughput = `${formatBytes(bytesPerSec)}/s`;
    const eta = elapsed > 2 ? formatETA(etaSec) : "...";
    const errStr = errors > 0 ? ` ${c.yellow}${errors} err${c.reset}` : "";

    process.stderr.write(`\r${c.cyan}${bar}${c.reset} ${c.bold}${percentStr}%${c.reset} ${c.dim}${chunksEmbedded}/${totalChunks}${c.reset}${errStr} ${c.dim}${throughput} ETA ${eta}${c.reset}   `);
  }

  progress.clear();
  cursor.show();
  const totalTimeSec = (Date.now() - startTime) / 1000;
  const avgThroughput = formatBytes(totalBytes / totalTimeSec);

  console.log(`\r${c.green}${renderProgressBar(100)}${c.reset} ${c.bold}100%${c.reset}                                    `);
  console.log(`\n${c.green}✓ Done!${c.reset} Embedded ${c.bold}${chunksEmbedded}${c.reset} chunks from ${c.bold}${totalDocs}${c.reset} documents in ${c.bold}${formatETA(totalTimeSec)}${c.reset} ${c.dim}(${avgThroughput}/s)${c.reset}`);
  if (errors > 0) {
    console.log(`${c.yellow}⚠ ${errors} chunks failed${c.reset}`);
  }
  closeDb();
}

// Sanitize a term for FTS5: remove punctuation except apostrophes
function sanitizeFTS5Term(term: string): string {
  // Remove all non-alphanumeric except apostrophes (for contractions like "don't")
  return term.replace(/[^\w']/g, '').trim();
}

// Build FTS5 query: phrase-aware with fallback to individual terms
function buildFTS5Query(query: string): string {
  // Sanitize the full query for phrase matching
  const sanitizedQuery = query.replace(/[^\w\s']/g, '').trim();

  const terms = query
    .split(/\s+/)
    .map(sanitizeFTS5Term)
    .filter(term => term.length >= 2); // Skip single chars and empty

  if (terms.length === 0) return "";
  if (terms.length === 1) return `"${terms[0].replace(/"/g, '""')}"`;

  // Strategy: exact phrase OR proximity match OR individual terms
  // Exact phrase matches rank highest, then close proximity, then any term
  const phrase = `"${sanitizedQuery.replace(/"/g, '""')}"`;
  const quotedTerms = terms.map(t => `"${t.replace(/"/g, '""')}"`);

  // FTS5 NEAR syntax: NEAR(term1 term2, distance)
  const nearPhrase = `NEAR(${quotedTerms.join(' ')}, 10)`;
  const orTerms = quotedTerms.join(' OR ');

  // Exact phrase > proximity > any term
  return `(${phrase}) OR (${nearPhrase}) OR (${orTerms})`;
}

// Normalize BM25 score to 0-1 range using sigmoid
function normalizeBM25(score: number): number {
  // BM25 scores are negative in SQLite (lower = better)
  // Typical range: -15 (excellent) to -2 (weak match)
  // Map to 0-1 where higher is better
  const absScore = Math.abs(score);
  // Sigmoid-ish normalization: maps ~2-15 range to ~0.1-0.95
  return 1 / (1 + Math.exp(-(absScore - 5) / 3));
}

function normalizeScores(results: SearchResult[]): SearchResult[] {
  if (results.length === 0) return results;
  const maxScore = Math.max(...results.map(r => r.score));
  const minScore = Math.min(...results.map(r => r.score));
  const range = maxScore - minScore || 1;
  return results.map(r => ({ ...r, score: (r.score - minScore) / range }));
}

// Reciprocal Rank Fusion: combines multiple ranked lists
// RRF score = sum(1 / (k + rank)) across all lists where doc appears
// k=60 is standard, provides good balance between top and lower ranks
export type RankedResult = { file: string; displayPath: string; title: string; body: string; score: number };

function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],  // Weight per result list (default 1.0)
  k: number = 60
): RankedResult[] {
  const scores = new Map<string, { score: number; displayPath: string; title: string; body: string; bestRank: number }>();

  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const results = resultLists[listIdx];
    const weight = weights[listIdx] ?? 1.0;
    for (let rank = 0; rank < results.length; rank++) {
      const doc = results[rank];
      const rrfScore = weight / (k + rank + 1);
      const existing = scores.get(doc.file);
      if (existing) {
        existing.score += rrfScore;
        existing.bestRank = Math.min(existing.bestRank, rank);
      } else {
        scores.set(doc.file, { score: rrfScore, displayPath: doc.displayPath, title: doc.title, body: doc.body, bestRank: rank });
      }
    }
  }

  // Add bonus for best rank: documents that ranked #1-3 in any list get a boost
  // This prevents dilution of exact matches by expansion queries
  return Array.from(scores.entries())
    .map(([file, { score, displayPath, title, body, bestRank }]) => {
      let bonus = 0;
      if (bestRank === 0) bonus = 0.05;  // Ranked #1 somewhere
      else if (bestRank <= 2) bonus = 0.02;  // Ranked top-3 somewhere
      return { file, displayPath, title, body, score: score + bonus };
    })
    .sort((a, b) => b.score - a.score);
}

type OutputOptions = {
  format: OutputFormat;
  full: boolean;
  limit: number;
  minScore: number;
  all?: boolean;
  collection?: string;  // Filter by collection name (pwd suffix match)
  lineNumbers?: boolean; // Add line numbers to output
};

// Highlight query terms in text (skip short words < 3 chars)
function highlightTerms(text: string, query: string): string {
  if (!useColor) return text;
  const terms = query.toLowerCase().split(/\s+/).filter(t => t.length >= 3);
  let result = text;
  for (const term of terms) {
    const regex = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
    result = result.replace(regex, `${c.yellow}${c.bold}$1${c.reset}`);
  }
  return result;
}

// Format score with color based on value
function formatScore(score: number): string {
  const pct = (score * 100).toFixed(0).padStart(3);
  if (!useColor) return `${pct}%`;
  if (score >= 0.7) return `${c.green}${pct}%${c.reset}`;
  if (score >= 0.4) return `${c.yellow}${pct}%${c.reset}`;
  return `${c.dim}${pct}%${c.reset}`;
}

// Shorten directory path for display - relative to $HOME (used for context paths, not documents)
function shortPath(dirpath: string): string {
  const home = homedir();
  if (dirpath.startsWith(home)) {
    return '~' + dirpath.slice(home.length);
  }
  return dirpath;
}

// Add line numbers to text content
function addLineNumbers(text: string, startLine: number = 1): string {
  const lines = text.split('\n');
  return lines.map((line, i) => `${startLine + i}: ${line}`).join('\n');
}

function outputResults(results: { file: string; displayPath: string; title: string; body: string; score: number; context?: string | null; chunkPos?: number; hash?: string; docid?: string }[], query: string, opts: OutputOptions): void {
  const filtered = results.filter(r => r.score >= opts.minScore).slice(0, opts.limit);

  if (filtered.length === 0) {
    console.log("No results found above minimum score threshold.");
    return;
  }

  // Helper to create qmd:// URI from displayPath
  const toQmdPath = (displayPath: string) => `qmd://${displayPath}`;

  if (opts.format === "json") {
    // JSON output for LLM consumption
    const output = filtered.map(row => {
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : undefined);
      let body = opts.full ? row.body : undefined;
      let snippet = !opts.full ? extractSnippet(row.body, query, 300, row.chunkPos).snippet : undefined;
      if (opts.lineNumbers) {
        if (body) body = addLineNumbers(body);
        if (snippet) snippet = addLineNumbers(snippet);
      }
      return {
        ...(docid && { docid: `#${docid}` }),
        score: Math.round(row.score * 100) / 100,
        file: toQmdPath(row.displayPath),
        title: row.title,
        ...(row.context && { context: row.context }),
        ...(body && { body }),
        ...(snippet && { snippet }),
      };
    });
    console.log(JSON.stringify(output, null, 2));
  } else if (opts.format === "files") {
    // Simple docid,score,filepath,context output
    for (const row of filtered) {
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : "");
      const ctx = row.context ? `,"${row.context.replace(/"/g, '""')}"` : "";
      console.log(`#${docid},${row.score.toFixed(2)},${toQmdPath(row.displayPath)}${ctx}`);
    }
  } else if (opts.format === "cli") {
    for (let i = 0; i < filtered.length; i++) {
      const row = filtered[i];
      const { line, snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : undefined);

      // Line 1: filepath with docid
      const path = toQmdPath(row.displayPath);
      // Only show :line if we actually found a term match in the snippet body (exclude header line).
      const snippetBody = snippet.split("\n").slice(1).join("\n").toLowerCase();
      const hasMatch = query.toLowerCase().split(/\s+/).some(t => t.length > 0 && snippetBody.includes(t));
      const lineInfo = hasMatch ? `:${line}` : "";
      const docidStr = docid ? ` ${c.dim}#${docid}${c.reset}` : "";
      console.log(`${c.cyan}${path}${c.dim}${lineInfo}${c.reset}${docidStr}`);

      // Line 2: Title (if available)
      if (row.title) {
        console.log(`${c.bold}Title: ${row.title}${c.reset}`);
      }

      // Line 3: Context (if available)
      if (row.context) {
        console.log(`${c.dim}Context: ${row.context}${c.reset}`);
      }

      // Line 4: Score
      const score = formatScore(row.score);
      console.log(`Score: ${c.bold}${score}${c.reset}`);
      console.log();

      // Snippet with highlighting (diff-style header included)
      let displaySnippet = opts.lineNumbers ? addLineNumbers(snippet, line) : snippet;
      const highlighted = highlightTerms(displaySnippet, query);
      console.log(highlighted);

      // Double empty line between results
      if (i < filtered.length - 1) console.log('\n');
    }
  } else if (opts.format === "md") {
    for (const row of filtered) {
      const heading = row.title || row.displayPath;
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : undefined);
      let content = opts.full ? row.body : extractSnippet(row.body, query, 500, row.chunkPos).snippet;
      if (opts.lineNumbers) {
        content = addLineNumbers(content);
      }
      const docidLine = docid ? `**docid:** \`#${docid}\`\n` : "";
      const contextLine = row.context ? `**context:** ${row.context}\n` : "";
      console.log(`---\n# ${heading}\n${docidLine}${contextLine}\n${content}\n`);
    }
  } else if (opts.format === "xml") {
    for (const row of filtered) {
      const titleAttr = row.title ? ` title="${row.title.replace(/"/g, '&quot;')}"` : "";
      const contextAttr = row.context ? ` context="${row.context.replace(/"/g, '&quot;')}"` : "";
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : "");
      let content = opts.full ? row.body : extractSnippet(row.body, query, 500, row.chunkPos).snippet;
      if (opts.lineNumbers) {
        content = addLineNumbers(content);
      }
      console.log(`<file docid="#${docid}" name="${toQmdPath(row.displayPath)}"${titleAttr}${contextAttr}>\n${content}\n</file>\n`);
    }
  } else {
    // CSV format
    console.log("docid,score,file,title,context,line,snippet");
    for (const row of filtered) {
      const { line, snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
      let content = opts.full ? row.body : snippet;
      if (opts.lineNumbers) {
        content = addLineNumbers(content, line);
      }
      const docid = row.docid || (row.hash ? row.hash.slice(0, 6) : "");
      console.log(`#${docid},${row.score.toFixed(4)},${escapeCSV(toQmdPath(row.displayPath))},${escapeCSV(row.title)},${escapeCSV(row.context || "")},${line},${escapeCSV(content)}`);
    }
  }
}

function search(query: string, opts: OutputOptions): void {
  const db = getDb();

  // Validate collection filter if specified
  let collectionName: string | undefined;
  if (opts.collection) {
    const coll = getCollectionFromYaml(opts.collection);
    if (!coll) {
      console.error(`Collection not found: ${opts.collection}`);
      closeDb();
      process.exit(1);
    }
    collectionName = opts.collection;
  }

  // Use large limit for --all, otherwise fetch more than needed and let outputResults filter
  const fetchLimit = opts.all ? 100000 : Math.max(50, opts.limit * 2);
  // searchFTS accepts collection name as number parameter for legacy reasons (will be fixed in store.ts)
  const results = searchFTS(db, query, fetchLimit, collectionName as any);

  // Add context to results
  const resultsWithContext = results.map(r => ({
    ...r,
    context: getContextForFile(db, r.filepath),
  }));

  closeDb();

  if (resultsWithContext.length === 0) {
    console.log("No results found.");
    return;
  }
  outputResults(resultsWithContext, query, opts);
}

async function vectorSearch(query: string, opts: OutputOptions, model: string = DEFAULT_EMBED_MODEL): Promise<void> {
  const db = getDb();

  // Validate collection filter if specified
  let collectionName: string | undefined;
  if (opts.collection) {
    const coll = getCollectionFromYaml(opts.collection);
    if (!coll) {
      console.error(`Collection not found: ${opts.collection}`);
      closeDb();
      process.exit(1);
    }
    collectionName = opts.collection;
  }

  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) {
    console.error("Vector index not found. Run 'qmd embed' first to create embeddings.");
    closeDb();
    return;
  }

  // Check index health and warn about issues
  checkIndexHealth(db);

  // Expand query using structured output (no lexical for vector-only search)
  const expanded = await expandQueryStructured(query, false);

  // Build list of queries for vector search: original, vectorQuery, and hyde
  const vectorQueries: string[] = [query];
  if (expanded.vectorQuery && expanded.vectorQuery !== query) {
    vectorQueries.push(expanded.vectorQuery);
  }
  if (expanded.hyde && expanded.hyde.length > 20) {
    vectorQueries.push(expanded.hyde);
  }

  process.stderr.write(`${c.dim}Searching ${vectorQueries.length} vector queries...${c.reset}\n`);

  // Collect results from all query variations
  const perQueryLimit = opts.all ? 500 : 20;
  const allResults = new Map<string, { file: string; displayPath: string; title: string; body: string; score: number; hash: string }>();

  for (const q of vectorQueries) {
    const vecResults = await searchVec(db, q, model, perQueryLimit, collectionName as any);
    for (const r of vecResults) {
      const existing = allResults.get(r.filepath);
      if (!existing || r.score > existing.score) {
        allResults.set(r.filepath, { file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score, hash: r.hash });
      }
    }
  }

  // Sort by max score and limit to requested count
  const results = Array.from(allResults.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, opts.limit)
    .map(r => ({ ...r, context: getContextForFile(db, r.file) }));

  closeDb();

  if (results.length === 0) {
    console.log("No results found.");
    return;
  }
  outputResults(results, query, { ...opts, limit: results.length }); // Already limited
}

// Expand query using structured output with JSON schema grammar
async function expandQueryStructured(query: string, includeLexical: boolean = true): Promise<ExpandedQuery> {
  process.stderr.write(`${c.dim}Expanding query...${c.reset}\n`);

  const llm = getDefaultLlamaCpp();
  const expanded = await llm.expandQueryStructured(query, includeLexical);

  // Log the expansion as a tree, starting with original query
  const lines: string[] = [];
  const bothLabel = includeLexical ? ' · (lexical+vector)' : ' · (vector)';
  lines.push(`${c.dim}├─ ${query}${bothLabel}${c.reset}`);

  if (expanded.lexicalQuery && expanded.lexicalQuery !== query) {
    lines.push(`${c.dim}├─ ${expanded.lexicalQuery} · (lexical)${c.reset}`);
  }
  if (expanded.vectorQuery && expanded.vectorQuery !== query) {
    lines.push(`${c.dim}├─ ${expanded.vectorQuery} · (vector)${c.reset}`);
  }
  if (expanded.hyde && expanded.hyde.length > 20) {
    // Truncate hyde to first ~60 chars for display
    const hydePreview = expanded.hyde.length > 60
      ? expanded.hyde.substring(0, 60).replace(/\n/g, ' ') + '...'
      : expanded.hyde.replace(/\n/g, ' ');
    lines.push(`${c.dim}├─ ${hydePreview} · (vector)${c.reset}`);
  }

  // Fix last item to use └─ instead of ├─
  if (lines.length > 0) {
    lines[lines.length - 1] = lines[lines.length - 1].replace('├─', '└─');
  }

  for (const line of lines) {
    process.stderr.write(line + '\n');
  }

  return expanded;
}

async function expandQuery(query: string, _model: string = DEFAULT_QUERY_MODEL, _db?: Database): Promise<string[]> {
  const expanded = await expandQueryStructured(query, true);
  const queries = [query];
  if (expanded.lexicalQuery && expanded.lexicalQuery !== query) queries.push(expanded.lexicalQuery);
  if (expanded.vectorQuery && expanded.vectorQuery !== query) queries.push(expanded.vectorQuery);
  return queries;
}

async function querySearch(query: string, opts: OutputOptions, embedModel: string = DEFAULT_EMBED_MODEL, rerankModel: string = DEFAULT_RERANK_MODEL): Promise<void> {
  const db = getDb();

  // Validate collection filter if specified
  let collectionName: string | undefined;
  if (opts.collection) {
    const coll = getCollectionFromYaml(opts.collection);
    if (!coll) {
      console.error(`Collection not found: ${opts.collection}`);
      closeDb();
      process.exit(1);
    }
    collectionName = opts.collection;
  }

  // Check index health and warn about issues
  checkIndexHealth(db);

  // Run initial BM25 search (will be reused for retrieval)
  const initialFts = searchFTS(db, query, 20, collectionName as any);
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

  // Check if initial results have strong signals (skip expansion if so)
  // Strong signal = top result is strong AND clearly separated from runner-up.
  // This avoids skipping expansion when BM25 has lots of mediocre matches.
  const topScore = initialFts[0]?.score ?? 0;
  const secondScore = initialFts[1]?.score ?? 0;
  const hasStrongSignal = initialFts.length > 0 && topScore >= 0.85 && (topScore - secondScore) >= 0.15;

  let ftsQueries: string[] = [query];
  let vectorQueries: string[] = [query];

  if (hasStrongSignal) {
    // Strong BM25 signal - skip expensive LLM expansion
    process.stderr.write(`${c.dim}Strong BM25 signal (${topScore.toFixed(2)}) - skipping expansion${c.reset}\n`);
    // Still log the "expansion tree" in the same style as vsearch for consistency.
    {
      const lines: string[] = [];
      lines.push(`${c.dim}├─ ${query} · (lexical+vector)${c.reset}`);
      lines[lines.length - 1] = lines[lines.length - 1].replace('├─', '└─');
      for (const line of lines) process.stderr.write(line + '\n');
    }
  } else {
    // Weak signal - expand query for better recall
    const expanded = await expandQueryStructured(query, true);

    if (expanded.lexicalQuery && expanded.lexicalQuery !== query) {
      ftsQueries.push(expanded.lexicalQuery);
    }
    if (expanded.vectorQuery && expanded.vectorQuery !== query) {
      vectorQueries.push(expanded.vectorQuery);
    }
    if (expanded.hyde && expanded.hyde.length > 20) {
      vectorQueries.push(expanded.hyde);
    }
  }

  process.stderr.write(`${c.dim}Searching ${ftsQueries.length} lexical + ${vectorQueries.length} vector queries...${c.reset}\n`);

  // Collect ranked result lists for RRF fusion
  const rankedLists: RankedResult[][] = [];

  // Map to store hash by filepath for final results
  const hashMap = new Map<string, string>();

  // FTS searches with lexical queries (reuse initial search for original query)
  if (initialFts.length > 0) {
    for (const r of initialFts) hashMap.set(r.filepath, r.hash);
    rankedLists.push(initialFts.map(r => ({ file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score })));
  }
  // Run expanded queries (skip first which is original)
  for (const q of ftsQueries.slice(1)) {
    const ftsResults = searchFTS(db, q, 20, collectionName as any);
    if (ftsResults.length > 0) {
      for (const r of ftsResults) hashMap.set(r.filepath, r.hash);
      rankedLists.push(ftsResults.map(r => ({ file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score })));
    }
  }

  // Vector searches with semantic queries + hyde
  if (hasVectors) {
    for (const q of vectorQueries) {
      const vecResults = await searchVec(db, q, embedModel, 20, collectionName as any);
      if (vecResults.length > 0) {
        for (const r of vecResults) hashMap.set(r.filepath, r.hash);
        rankedLists.push(vecResults.map(r => ({ file: r.filepath, displayPath: r.displayPath, title: r.title, body: r.body || "", score: r.score })));
      }
    }
  }

  // Apply Reciprocal Rank Fusion to combine all ranked lists
  // Give 2x weight to original query results (first 2 lists: FTS + vector)
  const weights = rankedLists.map((_, i) => i < 2 ? 2.0 : 1.0);
  const fused = reciprocalRankFusion(rankedLists, weights);
  // Hard cap reranking for latency/cost. We rerank per-document (best chunk only).
  const RERANK_DOC_LIMIT = 40;
  const candidates = fused.slice(0, RERANK_DOC_LIMIT);

  if (candidates.length === 0) {
    console.log("No results found.");
    closeDb();
    return;
  }

  // Rerank multiple chunks per document, then aggregate scores
  // This improves ranking for long documents where keyword-matched chunk isn't always best
  // We only rerank ONE chunk per document (best chunk by a simple keyword heuristic),
  // so we never rerank more than RERANK_DOC_LIMIT items.
  const chunksToRerank: { file: string; text: string; chunkIdx: number }[] = [];
  const docChunkMap = new Map<string, { chunks: { text: string; pos: number }[]; bestIdx: number }>();

  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  for (const c of candidates) {
    const chunks = chunkDocument(c.body);
    if (chunks.length === 0) continue;

    // Choose best chunk by keyword matches; fall back to first chunk.
    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < chunks.length; i++) {
      const chunkLower = chunks[i]!.text.toLowerCase();
      const score = queryTerms.reduce((acc, term) => acc + (chunkLower.includes(term) ? 1 : 0), 0);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    chunksToRerank.push({ file: c.file, text: chunks[bestIdx]!.text, chunkIdx: bestIdx });
    docChunkMap.set(c.file, { chunks, bestIdx });
  }

  // Rerank selected chunks (with caching). One chunk per doc -> one rerank item per doc.
  const reranked = await rerank(
    query,
    chunksToRerank.map(c => ({ file: c.file, text: c.text })),
    rerankModel,
    db
  );

  const aggregatedScores = new Map<string, { score: number; bestChunkIdx: number }>();
  for (const r of reranked) {
    const chunkInfo = docChunkMap.get(r.file);
    aggregatedScores.set(r.file, { score: r.score, bestChunkIdx: chunkInfo?.bestIdx ?? 0 });
  }

  // Blend RRF position score with aggregated reranker score using position-aware weights
  // Top retrieval results get more protection from reranker disagreement
  const candidateMap = new Map(candidates.map(c => [c.file, { displayPath: c.displayPath, title: c.title, body: c.body }]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1])); // 1-indexed rank

  const finalResults = Array.from(aggregatedScores.entries()).map(([file, { score: rerankScore, bestChunkIdx }]) => {
    const rrfRank = rrfRankMap.get(file) || 30;
    // Position-aware blending: top retrieval results preserved more
    // Rank 1-3: 75% RRF, 25% reranker (trust retrieval for exact matches)
    // Rank 4-10: 60% RRF, 40% reranker
    // Rank 11+: 40% RRF, 60% reranker (trust reranker for lower-ranked)
    let rrfWeight: number;
    if (rrfRank <= 3) {
      rrfWeight = 0.75;
    } else if (rrfRank <= 10) {
      rrfWeight = 0.60;
    } else {
      rrfWeight = 0.40;
    }
    const rrfScore = 1 / rrfRank;  // Position-based: 1, 0.5, 0.33...
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * rerankScore;
    const candidate = candidateMap.get(file);
    // Use the best-scoring chunk's text for the body (better for snippets)
    const chunkInfo = docChunkMap.get(file);
    const chunkBody = chunkInfo ? (chunkInfo.chunks[bestChunkIdx]?.text || chunkInfo.chunks[0]!.text) : candidate?.body || "";
    const chunkPos = chunkInfo ? (chunkInfo.chunks[bestChunkIdx]?.pos || 0) : 0;
    return {
      file,
      displayPath: candidate?.displayPath || "",
      title: candidate?.title || "",
      body: chunkBody,
      chunkPos,
      score: blendedScore,
      context: getContextForFile(db, file),
      hash: hashMap.get(file) || "",
    };
  }).sort((a, b) => b.score - a.score);

  // Deduplicate by file (safety net - shouldn't happen but prevents duplicate output)
  const seenFiles = new Set<string>();
  const dedupedResults = finalResults.filter(r => {
    if (seenFiles.has(r.file)) return false;
    seenFiles.add(r.file);
    return true;
  });

  closeDb();
  outputResults(dedupedResults, query, opts);
}

// Parse CLI arguments using util.parseArgs
function parseCLI() {
  const { values, positionals } = parseArgs({
    args: Bun.argv.slice(2), // Skip bun and script path
    options: {
      // Global options
      index: { type: "string" },
      help: { type: "boolean", short: "h" },
      // Search options
      n: { type: "string" },
      "min-score": { type: "string" },
      all: { type: "boolean" },
      full: { type: "boolean" },
      csv: { type: "boolean" },
      md: { type: "boolean" },
      xml: { type: "boolean" },
      files: { type: "boolean" },
      json: { type: "boolean" },
      collection: { type: "string", short: "c" },  // Filter by collection
      // Collection options
      name: { type: "string" },  // collection name
      mask: { type: "string" },  // glob pattern
      // Embed options
      force: { type: "boolean", short: "f" },
      // Update options
      pull: { type: "boolean" },  // git pull before update
      // Get options
      l: { type: "string" },  // max lines
      from: { type: "string" },  // start line
      "max-bytes": { type: "string" },  // max bytes for multi-get
      "line-numbers": { type: "boolean" },  // add line numbers to output
    },
    allowPositionals: true,
    strict: false, // Allow unknown options to pass through
  });

  // Select index name (default: "index")
  if (values.index) {
    setIndexName(values.index);
  }

  // Determine output format
  let format: OutputFormat = "cli";
  if (values.csv) format = "csv";
  else if (values.md) format = "md";
  else if (values.xml) format = "xml";
  else if (values.files) format = "files";
  else if (values.json) format = "json";

  // Default limit: 20 for --files/--json, 5 otherwise
  // --all means return all results (use very large limit)
  const defaultLimit = (format === "files" || format === "json") ? 20 : 5;
  const isAll = values.all || false;

  const opts: OutputOptions = {
    format,
    full: values.full || false,
    limit: isAll ? 100000 : (values.n ? parseInt(values.n, 10) || defaultLimit : defaultLimit),
    minScore: values["min-score"] ? parseFloat(values["min-score"]) || 0 : 0,
    all: isAll,
    collection: values.collection as string | undefined,
    lineNumbers: values["line-numbers"] || false,
  };

  return {
    command: positionals[0] || "",
    args: positionals.slice(1),
    query: positionals.slice(1).join(" "),
    opts,
    values,
  };
}

function showHelp(): void {
  console.log("Usage:");
  console.log("  qmd collection add [path] --name <name> --mask <pattern>  - Create/index collection");
  console.log("  qmd collection list           - List all collections with details");
  console.log("  qmd collection remove <name>  - Remove a collection by name");
  console.log("  qmd collection rename <old> <new>  - Rename a collection");
  console.log("  qmd ls [collection[/path]]    - List collections or files in a collection");
  console.log("  qmd context add [path] \"text\" - Add context for path (defaults to current dir)");
  console.log("  qmd context list              - List all contexts");
  console.log("  qmd context rm <path>         - Remove context");
  console.log("  qmd get <file>[:line] [-l N] [--from N]  - Get document (optionally from line, max N lines)");
  console.log("  qmd multi-get <pattern> [-l N] [--max-bytes N]  - Get multiple docs by glob or comma-separated list");
  console.log("  qmd status                    - Show index status and collections");
  console.log("  qmd update [--pull]           - Re-index all collections (--pull: git pull first)");
  console.log("  qmd embed [-f]                - Create vector embeddings (800 tokens/chunk, 15% overlap)");
  console.log("  qmd cleanup                   - Remove cache and orphaned data, vacuum DB");
  console.log("  qmd search <query>            - Full-text search (BM25)");
  console.log("  qmd vsearch <query>           - Vector similarity search");
  console.log("  qmd query <query>             - Combined search with query expansion + reranking");
  console.log("  qmd mcp                       - Start MCP server (for AI agent integration)");
  console.log("");
  console.log("Global options:");
  console.log("  --index <name>             - Use custom index name (default: index)");
  console.log("");
  console.log("Search options:");
  console.log("  -n <num>                   - Number of results (default: 5, or 20 for --files)");
  console.log("  --all                      - Return all matches (use with --min-score to filter)");
  console.log("  --min-score <num>          - Minimum similarity score");
  console.log("  --full                     - Output full document instead of snippet");
  console.log("  --line-numbers             - Add line numbers to output");
  console.log("  --files                    - Output docid,score,filepath,context (default: 20 results)");
  console.log("  --json                     - JSON output with snippets (default: 20 results)");
  console.log("  --csv                      - CSV output with snippets");
  console.log("  --md                       - Markdown output");
  console.log("  --xml                      - XML output");
  console.log("  -c, --collection <name>    - Filter results to a specific collection");
  console.log("");
  console.log("Multi-get options:");
  console.log("  -l <num>                   - Maximum lines per file");
  console.log("  --max-bytes <num>          - Skip files larger than N bytes (default: 10240)");
  console.log("  --json/--csv/--md/--xml/--files - Output format (same as search)");
  console.log("");
  console.log("Models (auto-downloaded from HuggingFace):");
  console.log("  Embedding: embeddinggemma-300M-Q8_0");
  console.log("  Reranking: qwen3-reranker-0.6b-q8_0");
  console.log("  Generation: Qwen3-0.6B-Q8_0");
  console.log("");
  console.log(`Index: ${getDbPath()}`);
}

// Main CLI - only run if this is the main module
if (import.meta.main) {
const cli = parseCLI();

if (!cli.command || cli.values.help) {
  showHelp();
  process.exit(cli.values.help ? 0 : 1);
}

switch (cli.command) {
  case "context": {
    const subcommand = cli.args[0];
    if (!subcommand) {
      console.error("Usage: qmd context <add|list|check|rm>");
      console.error("");
      console.error("Commands:");
      console.error("  qmd context add [path] \"text\"  - Add context (defaults to current dir)");
      console.error("  qmd context add / \"text\"       - Add global context to all collections");
      console.error("  qmd context list                - List all contexts");
      console.error("  qmd context check               - Check for missing contexts");
      console.error("  qmd context rm <path>           - Remove context");
      process.exit(1);
    }

    switch (subcommand) {
      case "add": {
        if (cli.args.length < 2) {
          console.error("Usage: qmd context add [path] \"text\"");
          console.error("");
          console.error("Examples:");
          console.error("  qmd context add \"Context for current directory\"");
          console.error("  qmd context add . \"Context for current directory\"");
          console.error("  qmd context add /subfolder \"Context for subfolder\"");
          console.error("  qmd context add / \"Global context for all collections\"");
          console.error("");
          console.error("  Using virtual paths:");
          console.error("  qmd context add qmd://journals/ \"Context for entire journals collection\"");
          console.error("  qmd context add qmd://journals/2024 \"Context for 2024 journals\"");
          process.exit(1);
        }

        let pathArg: string | undefined;
        let contextText: string;

        // Check if first arg looks like a path or if it's the context text
        const firstArg = cli.args[1];
        const secondArg = cli.args[2];

        if (secondArg) {
          // Two args: path + context
          pathArg = firstArg;
          contextText = cli.args.slice(2).join(" ");
        } else {
          // One arg: context only (use current directory)
          pathArg = undefined;
          contextText = firstArg;
        }

        await contextAdd(pathArg, contextText);
        break;
      }

      case "list": {
        contextList();
        break;
      }

      case "check": {
        contextCheck();
        break;
      }

      case "rm":
      case "remove": {
        if (cli.args.length < 2) {
          console.error("Usage: qmd context rm <path>");
          console.error("Examples:");
          console.error("  qmd context rm /");
          console.error("  qmd context rm qmd://journals/2024");
          process.exit(1);
        }
        contextRemove(cli.args[1]);
        break;
      }

      default:
        console.error(`Unknown subcommand: ${subcommand}`);
        console.error("Available: add, list, check, rm");
        process.exit(1);
    }
    break;
  }

  case "get": {
    if (!cli.args[0]) {
      console.error("Usage: qmd get <filepath>[:line] [--from <line>] [-l <lines>] [--line-numbers]");
      process.exit(1);
    }
    const fromLine = cli.values.from ? parseInt(cli.values.from as string, 10) : undefined;
    const maxLines = cli.values.l ? parseInt(cli.values.l as string, 10) : undefined;
    getDocument(cli.args[0], fromLine, maxLines, cli.opts.lineNumbers);
    break;
  }

  case "multi-get": {
    if (!cli.args[0]) {
      console.error("Usage: qmd multi-get <pattern> [-l <lines>] [--max-bytes <bytes>] [--json|--csv|--md|--xml|--files]");
      console.error("  pattern: glob (e.g., 'journals/2025-05*.md') or comma-separated list");
      process.exit(1);
    }
    const maxLinesMulti = cli.values.l ? parseInt(cli.values.l as string, 10) : undefined;
    const maxBytes = cli.values["max-bytes"] ? parseInt(cli.values["max-bytes"] as string, 10) : DEFAULT_MULTI_GET_MAX_BYTES;
    multiGet(cli.args[0], maxLinesMulti, maxBytes, cli.opts.format);
    break;
  }

  case "ls": {
    listFiles(cli.args[0]);
    break;
  }

  case "collection": {
    const subcommand = cli.args[0];
    switch (subcommand) {
      case "list": {
        collectionList();
        break;
      }

      case "add": {
        const pwd = cli.args[1] || getPwd();
        const resolvedPwd = pwd === '.' ? getPwd() : getRealPath(resolve(pwd));
        const globPattern = cli.values.mask as string || DEFAULT_GLOB;
        const name = cli.values.name as string | undefined;

        await collectionAdd(resolvedPwd, globPattern, name);
        break;
      }

      case "remove":
      case "rm": {
        if (!cli.args[1]) {
          console.error("Usage: qmd collection remove <name>");
          console.error("  Use 'qmd collection list' to see available collections");
          process.exit(1);
        }
        collectionRemove(cli.args[1]);
        break;
      }

      case "rename":
      case "mv": {
        if (!cli.args[1] || !cli.args[2]) {
          console.error("Usage: qmd collection rename <old-name> <new-name>");
          console.error("  Use 'qmd collection list' to see available collections");
          process.exit(1);
        }
        collectionRename(cli.args[1], cli.args[2]);
        break;
      }

      default:
        console.error(`Unknown subcommand: ${subcommand}`);
        console.error("Available: list, add, remove, rename");
        process.exit(1);
    }
    break;
  }

  case "status":
    showStatus();
    break;

  case "update":
    await updateCollections();
    break;

  case "embed":
    await vectorIndex(DEFAULT_EMBED_MODEL, cli.values.force || false);
    break;

  case "search":
    if (!cli.query) {
      console.error("Usage: qmd search [options] <query>");
      process.exit(1);
    }
    search(cli.query, cli.opts);
    break;

  case "vsearch":
    if (!cli.query) {
      console.error("Usage: qmd vsearch [options] <query>");
      process.exit(1);
    }
    // Default min-score for vector search is 0.3
    if (!cli.values["min-score"]) {
      cli.opts.minScore = 0.3;
    }
    await vectorSearch(cli.query, cli.opts);
    break;

  case "query":
    if (!cli.query) {
      console.error("Usage: qmd query [options] <query>");
      process.exit(1);
    }
    await querySearch(cli.query, cli.opts);
    break;

  case "mcp": {
    const { startMcpServer } = await import("./mcp.js");
    await startMcpServer();
    break;
  }

  case "cleanup": {
    const db = getDb();

    // 1. Clear llm_cache
    const cacheCount = deleteLLMCache(db);
    console.log(`${c.green}✓${c.reset} Cleared ${cacheCount} cached API responses`);

    // 2. Remove orphaned vectors
    const orphanedVecs = cleanupOrphanedVectors(db);
    if (orphanedVecs > 0) {
      console.log(`${c.green}✓${c.reset} Removed ${orphanedVecs} orphaned embedding chunks`);
    } else {
      console.log(`${c.dim}No orphaned embeddings to remove${c.reset}`);
    }

    // 3. Remove inactive documents
    const inactiveDocs = deleteInactiveDocuments(db);
    if (inactiveDocs > 0) {
      console.log(`${c.green}✓${c.reset} Removed ${inactiveDocs} inactive document records`);
    }

    // 4. Vacuum to reclaim space
    vacuumDatabase(db);
    console.log(`${c.green}✓${c.reset} Database vacuumed`);

    closeDb();
    break;
  }

  default:
    console.error(`Unknown command: ${cli.command}`);
    console.error("Run 'qmd --help' for usage.");
    process.exit(1);
}

// Cleanup LlamaCpp instance to prevent NAPI crash on exit
await disposeDefaultLlamaCpp();

} // end if (import.meta.main)
