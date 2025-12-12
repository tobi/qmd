#!/usr/bin/env bun
import { Database } from "bun:sqlite";
import { Glob, $ } from "bun";
import { parseArgs } from "util";
import * as sqliteVec from "sqlite-vec";
import {
  getDb,
  closeDb,
  getDbPath,
  getPwd,
  getRealPath,
  homedir,
  resolve,
  setCustomIndexName,
  searchFTS,
  searchVec,
  reciprocalRankFusion,
  extractSnippet,
  getContextForFile,
  getContextForPath,
  getCollectionIdByName,
  getCollectionByName,
  findSimilarFiles,
  matchFilesByGlob,
  getHashesNeedingEmbedding,
  getDocument as storeGetDocument,
  getMultipleDocuments as storeMultiGetDocuments,
  getStatus,
  hashContent,
  extractTitle,
  formatDocForEmbedding,
  formatQueryForEmbedding,
  chunkDocument,
  ensureVecTable,
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
  OLLAMA_URL,
  DEFAULT_EMBED_MODEL,
  DEFAULT_QUERY_MODEL,
  DEFAULT_RERANK_MODEL,
  DEFAULT_GLOB,
  DEFAULT_MULTI_GET_MAX_BYTES,
} from "./store.js";
import type { SearchResult, RankedResult } from "./store.js";
import {
  formatSearchResults,
  formatDocuments,
  escapeXml,
  escapeCSV,
  type OutputFormat,
} from "./formatter.js";

// Chunking: ~2000 tokens per chunk, ~3 bytes/token = 6KB
const CHUNK_BYTE_SIZE = 6 * 1024;

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

// Auto-pull model if not found
async function ensureModelAvailable(model: string): Promise<void> {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model }),
    });
    if (response.ok) return;
  } catch {
    // Continue to pull attempt
  }

  console.log(`Model ${model} not found. Pulling...`);
  progress.indeterminate();

  const pullResponse = await fetch(`${OLLAMA_URL}/api/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: model, stream: false }),
  });

  if (!pullResponse.ok) {
    progress.error();
    throw new Error(`Failed to pull model ${model}: ${pullResponse.status} - ${await pullResponse.text()}`);
  }

  progress.clear();
  console.log(`Model ${model} pulled successfully.`);
}

async function getEmbedding(text: string, model: string, isQuery: boolean = false, title?: string, retried: boolean = false): Promise<number[]> {
  const input = isQuery ? formatQueryForEmbedding(text) : formatDocForEmbedding(text, title);

  const response = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input }),
  });
  if (!response.ok) {
    const errorText = await response.text();
    if (!retried && (errorText.includes("not found") || errorText.includes("does not exist"))) {
      await ensureModelAvailable(model);
      return getEmbedding(text, model, isQuery, title, true);
    }
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }
  const data = await response.json() as { embeddings: number[][] };
  return data.embeddings[0];
}

// Qwen3-Reranker prompt format (trained for yes/no relevance classification)
const RERANK_SYSTEM = `Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".`;

function formatRerankPrompt(query: string, title: string, doc: string): string {
  return `<Instruct>: Determine if this document from a Shopify knowledge base is relevant to the search query. The query may reference specific Shopify programs, competitions, features, or named concepts (e.g., "Build a Business" competition, "Shop Pay", "Polaris"). Match documents that discuss the queried topic, even if phrasing differs.
<Query>: ${query}
<Document Title>: ${title}
<Document>: ${doc}`;
}

type LogProb = { token: string; logprob: number };
type RerankResponse = {
  response: string;
  logprobs?: LogProb[];
};

function parseRerankResponse(data: RerankResponse): number {
  if (!data.logprobs || data.logprobs.length === 0) {
    throw new Error("Reranker response missing logprobs");
  }

  const firstToken = data.logprobs[0];
  const token = firstToken.token.toLowerCase().trim();
  const confidence = Math.exp(firstToken.logprob);

  if (token === "yes") {
    return confidence;
  }
  if (token === "no") {
    return (1 - confidence) * 0.3;
  }

  throw new Error(`Unexpected reranker token: "${token}"`);
}

async function rerankSingle(prompt: string, model: string, db?: Database, retried: boolean = false): Promise<number> {
  // Use generate with raw template for qwen3-reranker format
  // Include empty <think> tags as per HuggingFace reference implementation
  const fullPrompt = `<|im_start|>system
${RERANK_SYSTEM}<|im_end|>
<|im_start|>user
${prompt}<|im_end|>
<|im_start|>assistant
<think>

</think>

`;

  const requestBody = {
    model,
    prompt: fullPrompt,
    raw: true,
    stream: false,
    logprobs: true,
    options: { num_predict: 1 },
  };

  // Check cache
  const cacheKey = db ? getCacheKey(`${OLLAMA_URL}/api/generate`, requestBody) : "";
  if (db) {
    const cached = getCachedResult(db, cacheKey);
    if (cached) {
      const data = JSON.parse(cached) as RerankResponse;
      return parseRerankResponse(data);
    }
  }

  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorText = await response.text();
    if (!retried && (errorText.includes("not found") || errorText.includes("does not exist"))) {
      await ensureModelAvailable(model);
      return rerankSingle(prompt, model, db, true);
    }
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json() as RerankResponse;

  // Cache the result
  if (db) {
    setCachedResult(db, cacheKey, JSON.stringify(data));
  }

  return parseRerankResponse(data);
}

async function rerank(query: string, documents: { file: string; text: string }[], model: string = DEFAULT_RERANK_MODEL, db?: Database): Promise<{ file: string; score: number }[]> {
  const results: { file: string; score: number }[] = [];
  const total = documents.length;
  const PARALLEL = 5;

  process.stderr.write(`Reranking ${total} documents with ${model} (parallel: ${PARALLEL})...\n`);
  progress.indeterminate();

  // Process in parallel batches
  for (let i = 0; i < documents.length; i += PARALLEL) {
    const batch = documents.slice(i, i + PARALLEL);
    const batchResults = await Promise.all(
      batch.map(async (doc) => {
        try {
          // Extract title from filename for reranker context
          const title = doc.file.split('/').pop()?.replace(/\.md$/, '') || doc.file;
          const prompt = formatRerankPrompt(query, title, doc.text.slice(0, 4000));
          const score = await rerankSingle(prompt, model, db);
          return { file: doc.file, score };
        } catch (err) {
          return { file: doc.file, score: 0 };
        }
      })
    );
    results.push(...batchResults);

    const processed = Math.min(i + PARALLEL, total);
    progress.set((processed / total) * 100);
    process.stderr.write(`\rReranking: ${processed}/${total}`);
  }

  progress.clear();
  process.stderr.write("\n");

  return results.sort((a, b) => b.score - a.score);
}

function getOrCreateCollection(db: Database, pwd: string, globPattern: string, name?: string): number {
  const now = new Date().toISOString();

  // Generate collection name from pwd basename if not provided
  if (!name) {
    const parts = pwd.split('/').filter(Boolean);
    name = parts[parts.length - 1] || 'root';
  }

  // Check if collection with this pwd+glob already exists
  const existing = db.prepare(`SELECT id FROM collections WHERE pwd = ? AND glob_pattern = ?`).get(pwd, globPattern) as { id: number } | null;
  if (existing) return existing.id;

  // Try to insert with generated name
  try {
    const result = db.prepare(`INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`).run(name, pwd, globPattern, now, now);
    return result.lastInsertRowid as number;
  } catch (e) {
    // Name collision - append a unique suffix
    const allCollections = db.prepare(`SELECT name FROM collections WHERE name LIKE ?`).all(`${name}%`) as { name: string }[];
    let suffix = 2;
    let uniqueName = `${name}-${suffix}`;
    while (allCollections.some(c => c.name === uniqueName)) {
      suffix++;
      uniqueName = `${name}-${suffix}`;
    }
    const result = db.prepare(`INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`).run(uniqueName, pwd, globPattern, now, now);
    return result.lastInsertRowid as number;
  }
}

function cleanupDuplicateCollections(db: Database): void {
  // Remove duplicate collections keeping the oldest one
  db.exec(`
    DELETE FROM collections WHERE id NOT IN (
      SELECT MIN(id) FROM collections GROUP BY pwd, glob_pattern
    )
  `);
  // Remove bogus "." glob pattern entries (from earlier bug)
  db.exec(`DELETE FROM collections WHERE glob_pattern = '.'`);
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

  // Cleanup any duplicate collections
  cleanupDuplicateCollections(db);

  // Index size
  let indexSize = 0;
  try {
    const stat = Bun.file(dbPath).size;
    indexSize = stat;
  } catch {}

  // Collections info
  const collections = db.prepare(`
    SELECT c.id, c.name, c.pwd, c.glob_pattern, c.created_at,
           COUNT(d.id) as doc_count,
           SUM(CASE WHEN d.active = 1 THEN 1 ELSE 0 END) as active_count,
           MAX(d.modified_at) as last_modified
    FROM collections c
    LEFT JOIN documents d ON d.collection_id = c.id
    GROUP BY c.id
    ORDER BY c.name
  `).all() as { id: number; name: string; pwd: string; glob_pattern: string; created_at: string; doc_count: number; active_count: number; last_modified: string | null }[];

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

  // Get context counts per collection
  const contextCounts = db.prepare(`
    SELECT collection_id, COUNT(*) as count
    FROM path_contexts
    GROUP BY collection_id
  `).all() as { collection_id: number; count: number }[];
  const contextCountMap = new Map(contextCounts.map(c => [c.collection_id, c.count]));

  if (collections.length > 0) {
    console.log(`\n${c.bold}Collections${c.reset}`);
    for (const col of collections) {
      const lastMod = col.last_modified ? formatTimeAgo(new Date(col.last_modified)) : "never";
      const contextCount = contextCountMap.get(col.id) || 0;

      console.log(`  ${c.cyan}${col.name}${c.reset} ${c.dim}(qmd://${col.name}/)${c.reset}`);
      console.log(`    ${c.dim}Path:${c.reset}     ${col.pwd}`);
      console.log(`    ${c.dim}Pattern:${c.reset}  ${col.glob_pattern}`);
      console.log(`    ${c.dim}Files:${c.reset}    ${col.active_count} (updated ${lastMod})`);
      if (contextCount > 0) {
        console.log(`    ${c.dim}Contexts:${c.reset} ${contextCount}`);
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

// Update display_paths for all documents that have empty display_path
function updateDisplayPaths(db: Database): number {
  // Get all docs with empty display_path, grouped by collection
  const emptyDocs = db.prepare(`
    SELECT d.id, d.filepath, c.pwd
    FROM documents d
    JOIN collections c ON d.collection_id = c.id
    WHERE d.active = 1 AND (d.display_path IS NULL OR d.display_path = '')
  `).all() as { id: number; filepath: string; pwd: string }[];

  if (emptyDocs.length === 0) return 0;

  // Collect existing display_paths
  const existingPaths = new Set<string>(
    (db.prepare(`SELECT display_path FROM documents WHERE active = 1 AND display_path != ''`).all() as { display_path: string }[])
      .map(r => r.display_path)
  );

  const updateStmt = db.prepare(`UPDATE documents SET display_path = ? WHERE id = ?`);
  let updated = 0;

  for (const doc of emptyDocs) {
    const displayPath = computeDisplayPath(doc.filepath, doc.pwd, existingPaths);
    updateStmt.run(displayPath, doc.id);
    existingPaths.add(displayPath);
    updated++;
  }

  return updated;
}

async function updateCollections(): Promise<void> {
  const db = getDb();
  cleanupDuplicateCollections(db);

  // Clear Ollama cache on update
  clearCache(db);

  const collections = db.prepare(`SELECT id, pwd, glob_pattern FROM collections`).all() as { id: number; pwd: string; glob_pattern: string }[];

  if (collections.length === 0) {
    console.log(`${c.dim}No collections found. Run 'qmd add .' to index markdown files.${c.reset}`);
    closeDb();
    return;
  }

  // Update display_paths for any documents missing them (migration)
  const pathsUpdated = updateDisplayPaths(db);
  if (pathsUpdated > 0) {
    console.log(`${c.green}✓${c.reset} Updated ${pathsUpdated} display paths`);
  }

  // Don't close db here - indexFiles will reuse it and close at the end
  console.log(`${c.bold}Updating ${collections.length} collection(s)...${c.reset}\n`);

  for (let i = 0; i < collections.length; i++) {
    const col = collections[i];
    console.log(`${c.cyan}[${i + 1}/${collections.length}]${c.reset} ${c.bold}${col.pwd}${c.reset}`);
    console.log(`${c.dim}    Pattern: ${col.glob_pattern}${c.reset}`);
    // Temporarily set PWD for indexing
    const originalPwd = process.env.PWD;
    process.env.PWD = col.pwd;
    await indexFiles(col.glob_pattern);
    process.env.PWD = originalPwd;
    console.log("");
  }

  console.log(`${c.green}✓ All collections updated.${c.reset}`);
}

/**
 * Detect which collection (if any) contains the given filesystem path.
 * Returns { collectionId, collectionName, relativePath } or null if not in any collection.
 */
function detectCollectionFromPath(db: Database, fsPath: string): { collectionId: number; collectionName: string; relativePath: string } | null {
  const realPath = getRealPath(fsPath);

  // Find collections that this path is under
  const collections = db.prepare(`
    SELECT id, name, pwd
    FROM collections
    WHERE ? LIKE pwd || '/%' OR ? = pwd
    ORDER BY LENGTH(pwd) DESC
    LIMIT 1
  `).get(realPath, realPath) as { id: number; name: string; pwd: string } | null;

  if (!collections) return null;

  // Calculate relative path
  let relativePath = realPath;
  if (relativePath.startsWith(collections.pwd + '/')) {
    relativePath = relativePath.slice(collections.pwd.length + 1);
  } else if (relativePath === collections.pwd) {
    relativePath = '';
  }

  return {
    collectionId: collections.id,
    collectionName: collections.name,
    relativePath
  };
}

async function contextAdd(pathArg: string | undefined, contextText: string): Promise<void> {
  const db = getDb();
  const now = new Date().toISOString();

  // Handle "/" as global/root context (applies to all collections)
  if (pathArg === '/') {
    // Find all collections and add context to each
    const collections = db.prepare(`SELECT id, name FROM collections`).all() as { id: number; name: string }[];
    for (const coll of collections) {
      db.prepare(`
        INSERT INTO path_contexts (collection_id, path_prefix, context, created_at)
        VALUES (?, '', ?, ?)
        ON CONFLICT(collection_id, path_prefix) DO UPDATE SET context = excluded.context
      `).run(coll.id, contextText, now);
    }
    console.log(`${c.green}✓${c.reset} Added global context to ${collections.length} collection(s)`);
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

    const coll = getCollectionByName(db, parsed.collectionName);
    if (!coll) {
      console.error(`${c.yellow}Collection not found: ${parsed.collectionName}${c.reset}`);
      process.exit(1);
    }

    db.prepare(`
      INSERT INTO path_contexts (collection_id, path_prefix, context, created_at)
      VALUES (?, ?, ?, ?)
      ON CONFLICT(collection_id, path_prefix) DO UPDATE SET context = excluded.context
    `).run(coll.id, parsed.path, contextText, now);

    console.log(`${c.green}✓${c.reset} Added context for: qmd://${parsed.collectionName}/${parsed.path || ''}`);
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

  db.prepare(`
    INSERT INTO path_contexts (collection_id, path_prefix, context, created_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(collection_id, path_prefix) DO UPDATE SET context = excluded.context
  `).run(detected.collectionId, detected.relativePath, contextText, now);

  const displayPath = detected.relativePath ? `qmd://${detected.collectionName}/${detected.relativePath}` : `qmd://${detected.collectionName}/`;
  console.log(`${c.green}✓${c.reset} Added context for: ${displayPath}`);
  console.log(`${c.dim}Context: ${contextText}${c.reset}`);
  closeDb();
}

function contextList(): void {
  const db = getDb();

  const contexts = db.prepare(`
    SELECT c.name as collection_name, pc.path_prefix, pc.context
    FROM path_contexts pc
    JOIN collections c ON c.id = pc.collection_id
    ORDER BY c.name, LENGTH(pc.path_prefix) DESC, pc.path_prefix
  `).all() as { collection_name: string; path_prefix: string; context: string }[];

  if (contexts.length === 0) {
    console.log(`${c.dim}No contexts configured. Use 'qmd context add' to add one.${c.reset}`);
    closeDb();
    return;
  }

  console.log(`\n${c.bold}Configured Contexts${c.reset}\n`);

  let lastCollection = '';
  for (const ctx of contexts) {
    if (ctx.collection_name !== lastCollection) {
      console.log(`${c.cyan}${ctx.collection_name}${c.reset}`);
      lastCollection = ctx.collection_name;
    }

    const path = ctx.path_prefix || '/';
    const displayPath = ctx.path_prefix ? `  ${path}` : '  / (root)';
    console.log(`${displayPath}`);
    console.log(`    ${c.dim}${ctx.context}${c.reset}`);
  }

  closeDb();
}

function contextRemove(pathArg: string): void {
  const db = getDb();

  if (pathArg === '/') {
    // Remove all root contexts
    const result = db.prepare(`DELETE FROM path_contexts WHERE path_prefix = ''`).run();
    console.log(`${c.green}✓${c.reset} Removed ${result.changes} global context(s)`);
    closeDb();
    return;
  }

  // Handle virtual paths
  if (isVirtualPath(pathArg)) {
    const parsed = parseVirtualPath(pathArg);
    if (!parsed) {
      console.error(`${c.yellow}Invalid virtual path: ${pathArg}${c.reset}`);
      process.exit(1);
    }

    const coll = getCollectionByName(db, parsed.collectionName);
    if (!coll) {
      console.error(`${c.yellow}Collection not found: ${parsed.collectionName}${c.reset}`);
      process.exit(1);
    }

    const result = db.prepare(`
      DELETE FROM path_contexts
      WHERE collection_id = ? AND path_prefix = ?
    `).run(coll.id, parsed.path);

    if (result.changes === 0) {
      console.error(`${c.yellow}No context found for: ${pathArg}${c.reset}`);
      process.exit(1);
    }

    console.log(`${c.green}✓${c.reset} Removed context for: ${pathArg}`);
    closeDb();
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

  const detected = detectCollectionFromPath(db, fsPath);
  if (!detected) {
    console.error(`${c.yellow}Path is not in any indexed collection: ${fsPath}${c.reset}`);
    process.exit(1);
  }

  const result = db.prepare(`
    DELETE FROM path_contexts
    WHERE collection_id = ? AND path_prefix = ?
  `).run(detected.collectionId, detected.relativePath);

  if (result.changes === 0) {
    console.error(`${c.yellow}No context found for: qmd://${detected.collectionName}/${detected.relativePath}${c.reset}`);
    process.exit(1);
  }

  console.log(`${c.green}✓${c.reset} Removed context for: qmd://${detected.collectionName}/${detected.relativePath}`);
  closeDb();
}

function getDocument(filename: string, fromLine?: number, maxLines?: number): void {
  const db = getDb();

  // Parse :linenum suffix from filename (e.g., "file.md:100")
  let inputPath = filename;
  const colonMatch = inputPath.match(/:(\d+)$/);
  if (colonMatch && !fromLine) {
    fromLine = parseInt(colonMatch[1], 10);
    inputPath = inputPath.slice(0, -colonMatch[0].length);
  }

  let doc: { collectionId: number; collectionName: string; path: string; body: string } | null = null;
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
      SELECT c.id as collectionId, c.name as collectionName, d.path, content.doc as body
      FROM documents d
      JOIN collections c ON c.id = d.collection_id
      JOIN content ON content.hash = d.hash
      WHERE c.name = ? AND d.path = ? AND d.active = 1
    `).get(parsed.collectionName, parsed.path) as typeof doc;

    if (!doc) {
      // Try fuzzy match by path ending
      doc = db.prepare(`
        SELECT c.id as collectionId, c.name as collectionName, d.path, content.doc as body
        FROM documents d
        JOIN collections c ON c.id = d.collection_id
        JOIN content ON content.hash = d.hash
        WHERE c.name = ? AND d.path LIKE ? AND d.active = 1
        LIMIT 1
      `).get(parsed.collectionName, `%${parsed.path}`) as typeof doc;
    }

    virtualPath = inputPath;
  } else {
    // Handle filesystem paths
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
      // Found collection - query by collection_id + relative path
      doc = db.prepare(`
        SELECT c.id as collectionId, c.name as collectionName, d.path, content.doc as body
        FROM documents d
        JOIN collections c ON c.id = d.collection_id
        JOIN content ON content.hash = d.hash
        WHERE c.id = ? AND d.path = ? AND d.active = 1
      `).get(detected.collectionId, detected.relativePath) as typeof doc;
    }

    // Fuzzy match by filename (last component of path)
    if (!doc) {
      const filename = inputPath.split('/').pop() || inputPath;
      doc = db.prepare(`
        SELECT c.id as collectionId, c.name as collectionName, d.path, content.doc as body
        FROM documents d
        JOIN collections c ON c.id = d.collection_id
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

  if (!doc) {
    console.error(`Document not found: ${filename}`);
    closeDb();
    process.exit(1);
  }

  // Get context for this file
  const context = getContextForPath(db, doc.collectionId, doc.path);

  let output = doc.body;

  // Apply line filtering if specified
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = output.split('\n');
    const start = (fromLine || 1) - 1; // Convert to 0-indexed
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    output = lines.slice(start, end).join('\n');
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

  let files: { filepath: string; displayPath: string; bodyLength: number; collectionId?: number; path?: string }[];

  if (isCommaSeparated) {
    // Comma-separated list of files (can be virtual paths or relative paths)
    const names = pattern.split(',').map(s => s.trim()).filter(Boolean);
    files = [];
    for (const name of names) {
      let doc: { virtual_path: string; body_length: number; collection_id: number; path: string } | null = null;

      // Handle virtual paths
      if (isVirtualPath(name)) {
        const parsed = parseVirtualPath(name);
        if (parsed) {
          // Try exact match on collection + path
          doc = db.prepare(`
            SELECT
              'qmd://' || c.name || '/' || d.path as virtual_path,
              LENGTH(content.doc) as body_length,
              d.collection_id,
              d.path
            FROM documents d
            JOIN collections c ON c.id = d.collection_id
            JOIN content ON content.hash = d.hash
            WHERE c.name = ? AND d.path = ? AND d.active = 1
          `).get(parsed.collectionName, parsed.path) as typeof doc;
        }
      } else {
        // Try exact match on path
        doc = db.prepare(`
          SELECT
            'qmd://' || c.name || '/' || d.path as virtual_path,
            LENGTH(content.doc) as body_length,
            d.collection_id,
            d.path
          FROM documents d
          JOIN collections c ON c.id = d.collection_id
          JOIN content ON content.hash = d.hash
          WHERE d.path = ? AND d.active = 1
          LIMIT 1
        `).get(name) as typeof doc;

        // Try suffix match
        if (!doc) {
          doc = db.prepare(`
            SELECT
              'qmd://' || c.name || '/' || d.path as virtual_path,
              LENGTH(content.doc) as body_length,
              d.collection_id,
              d.path
            FROM documents d
            JOIN collections c ON c.id = d.collection_id
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
          collectionId: doc.collection_id,
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
      collectionId: undefined,  // Will be fetched later if needed
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
    let collectionId = file.collectionId;
    let path = file.path;

    if (!collectionId || !path) {
      const parsed = parseVirtualPath(file.displayPath);
      if (parsed) {
        const coll = getCollectionByName(db, parsed.collectionName);
        if (coll) {
          collectionId = coll.id;
          path = parsed.path;
        }
      }
    }

    // Get context using collection-scoped function
    const context = collectionId && path ? getContextForPath(db, collectionId, path) : null;

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

    // Fetch document content - use virtual path to query
    const parsed = parseVirtualPath(file.displayPath);
    if (!parsed) continue;

    const doc = db.prepare(`
      SELECT content.doc as body, d.title
      FROM documents d
      JOIN collections c ON c.id = d.collection_id
      JOIN content ON content.hash = d.hash
      WHERE c.name = ? AND d.path = ? AND d.active = 1
    `).get(parsed.collectionName, parsed.path) as { body: string; title: string } | null;

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
    const collections = db.prepare(`
      SELECT name, COUNT(d.id) as file_count
      FROM collections c
      LEFT JOIN documents d ON d.collection_id = c.id AND d.active = 1
      GROUP BY c.id, c.name
      ORDER BY c.name
    `).all() as { name: string; file_count: number }[];

    if (collections.length === 0) {
      console.log("No collections found. Run 'qmd add .' to index files.");
      closeDb();
      return;
    }

    console.log(`${c.bold}Collections:${c.reset}\n`);
    for (const coll of collections) {
      console.log(`${c.cyan}qmd://${coll.name}/${c.reset} (${coll.file_count} files)`);
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
  const coll = getCollectionByName(db, collectionName);
  if (!coll) {
    console.error(`Collection not found: ${collectionName}`);
    console.error(`Run 'qmd ls' to see available collections.`);
    closeDb();
    process.exit(1);
  }

  // List files in the collection (optionally filtered by path prefix)
  let query: string;
  let params: any[];

  if (pathPrefix) {
    // List files under a specific path
    query = `
      SELECT d.path
      FROM documents d
      WHERE d.collection_id = ? AND d.path LIKE ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.id, `${pathPrefix}%`];
  } else {
    // List all files in the collection
    query = `
      SELECT d.path
      FROM documents d
      WHERE d.collection_id = ? AND d.active = 1
      ORDER BY d.path
    `;
    params = [coll.id];
  }

  const files = db.prepare(query).all(...params) as { path: string }[];

  if (files.length === 0) {
    if (pathPrefix) {
      console.log(`No files found under qmd://${collectionName}/${pathPrefix}`);
    } else {
      console.log(`No files found in collection: ${collectionName}`);
    }
    closeDb();
    return;
  }

  // Output virtual paths
  for (const file of files) {
    console.log(buildVirtualPath(collectionName, file.path));
  }

  closeDb();
}

// Collection management commands
function collectionList(): void {
  const db = getDb();

  const collections = db.prepare(`
    SELECT
      c.id,
      c.name,
      c.pwd,
      c.glob_pattern,
      c.created_at,
      c.updated_at,
      COUNT(d.id) as file_count
    FROM collections c
    LEFT JOIN documents d ON d.collection_id = c.id AND d.active = 1
    GROUP BY c.id
    ORDER BY c.name
  `).all() as {
    id: number;
    name: string;
    pwd: string;
    glob_pattern: string;
    created_at: string;
    updated_at: string;
    file_count: number;
  }[];

  if (collections.length === 0) {
    console.log("No collections found. Run 'qmd add .' to create one.");
    closeDb();
    return;
  }

  console.log(`${c.bold}Collections (${collections.length}):${c.reset}\n`);

  for (const coll of collections) {
    const updatedAt = new Date(coll.updated_at);
    const timeAgo = formatTimeAgo(updatedAt);

    console.log(`${c.cyan}${coll.name}${c.reset}`);
    console.log(`  ${c.dim}Path:${c.reset}     ${coll.pwd}`);
    console.log(`  ${c.dim}Pattern:${c.reset}  ${coll.glob_pattern}`);
    console.log(`  ${c.dim}Files:${c.reset}    ${coll.file_count}`);
    console.log(`  ${c.dim}Updated:${c.reset}  ${timeAgo}`);
    console.log();
  }

  closeDb();
}

async function collectionAdd(pwd: string, globPattern: string, name?: string): Promise<void> {
  const db = getDb();

  // If name not provided, generate from pwd basename
  if (!name) {
    const parts = pwd.split('/').filter(Boolean);
    name = parts[parts.length - 1] || 'root';
  }

  // Check if collection with this name already exists
  const existing = getCollectionByName(db, name);
  if (existing) {
    console.error(`${c.yellow}Collection '${name}' already exists.${c.reset}`);
    console.error(`Use a different name with --name <name>`);
    closeDb();
    process.exit(1);
  }

  // Check if a collection with this pwd+glob already exists
  const existingPwdGlob = db.prepare(`
    SELECT id, name FROM collections WHERE pwd = ? AND glob_pattern = ?
  `).get(pwd, globPattern) as { id: number; name: string } | null;

  if (existingPwdGlob) {
    console.error(`${c.yellow}A collection already exists for this path and pattern:${c.reset}`);
    console.error(`  Name: ${existingPwdGlob.name}`);
    console.error(`  Path: ${pwd}`);
    console.error(`  Pattern: ${globPattern}`);
    console.error(`\nUse 'qmd add ${globPattern}' to update it, or remove it first with 'qmd collection remove ${existingPwdGlob.name}'`);
    closeDb();
    process.exit(1);
  }

  closeDb();

  // Create the collection and index files
  console.log(`Creating collection '${name}'...`);
  await indexFiles(pwd, globPattern, name);
  console.log(`${c.green}✓${c.reset} Collection '${name}' created successfully`);
}

function collectionRemove(name: string): void {
  const db = getDb();

  const coll = getCollectionByName(db, name);
  if (!coll) {
    console.error(`${c.yellow}Collection not found: ${name}${c.reset}`);
    console.error(`Run 'qmd collection list' to see available collections.`);
    closeDb();
    process.exit(1);
  }

  // Get file count
  const fileCount = db.prepare(`
    SELECT COUNT(*) as count FROM documents WHERE collection_id = ? AND active = 1
  `).get(coll.id) as { count: number };

  // Delete documents
  db.prepare(`DELETE FROM documents WHERE collection_id = ?`).run(coll.id);

  // Delete contexts
  db.prepare(`DELETE FROM path_contexts WHERE collection_id = ?`).run(coll.id);

  // Delete collection
  db.prepare(`DELETE FROM collections WHERE id = ?`).run(coll.id);

  // Clean up orphaned content hashes
  const cleanupResult = db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();

  console.log(`${c.green}✓${c.reset} Removed collection '${name}'`);
  console.log(`  Deleted ${fileCount.count} documents`);
  if (cleanupResult.changes > 0) {
    console.log(`  Cleaned up ${cleanupResult.changes} orphaned content hashes`);
  }

  closeDb();
}

function collectionRename(oldName: string, newName: string): void {
  const db = getDb();

  // Check if old collection exists
  const coll = getCollectionByName(db, oldName);
  if (!coll) {
    console.error(`${c.yellow}Collection not found: ${oldName}${c.reset}`);
    console.error(`Run 'qmd collection list' to see available collections.`);
    closeDb();
    process.exit(1);
  }

  // Check if new name already exists
  const existing = getCollectionByName(db, newName);
  if (existing) {
    console.error(`${c.yellow}Collection name already exists: ${newName}${c.reset}`);
    console.error(`Choose a different name or remove the existing collection first.`);
    closeDb();
    process.exit(1);
  }

  // Update the collection name
  db.prepare(`UPDATE collections SET name = ?, updated_at = ? WHERE id = ?`)
    .run(newName, new Date().toISOString(), coll.id);

  console.log(`${c.green}✓${c.reset} Renamed collection '${oldName}' to '${newName}'`);
  console.log(`  Virtual paths updated: ${c.cyan}qmd://${oldName}/${c.reset} → ${c.cyan}qmd://${newName}/${c.reset}`);

  closeDb();
}

async function dropCollection(globPattern: string): Promise<void> {
  const db = getDb();
  const pwd = getPwd();

  const collection = db.prepare(`SELECT id FROM collections WHERE pwd = ? AND glob_pattern = ?`).get(pwd, globPattern) as { id: number } | null;

  if (!collection) {
    // No collection to drop - this is fine, we'll create one during indexing
    return;
  }

  // Delete documents in this collection
  const deleted = db.prepare(`DELETE FROM documents WHERE collection_id = ?`).run(collection.id);

  // Delete the collection
  db.prepare(`DELETE FROM collections WHERE id = ?`).run(collection.id);

  console.log(`Dropped collection: ${pwd} (${globPattern})`);
  console.log(`Removed ${deleted.changes} documents`);
  console.log(`(Vectors kept for potential reuse)`);
  // Don't close db - indexFiles will use it and close at the end
}

async function indexFiles(pwd?: string, globPattern: string = DEFAULT_GLOB, name?: string): Promise<void> {
  const db = getDb();
  const resolvedPwd = pwd || getPwd();
  const now = new Date().toISOString();
  const excludeDirs = ["node_modules", ".git", ".cache", "vendor", "dist", "build"];

  // Clear Ollama cache on index
  clearCache(db);

  // Get or create collection for this (pwd, glob)
  const collectionId = getOrCreateCollection(db, resolvedPwd, globPattern, name);
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

  // Prepared statements for new schema
  const insertContentStmt = db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`);
  const insertDocStmt = db.prepare(`INSERT INTO documents (collection_id, path, title, hash, created_at, modified_at, active) VALUES (?, ?, ?, ?, ?, ?, 1)`);
  const deactivateStmt = db.prepare(`UPDATE documents SET active = 0 WHERE collection_id = ? AND path = ? AND active = 1`);
  const findActiveStmt = db.prepare(`SELECT id, hash, title FROM documents WHERE collection_id = ? AND path = ? AND active = 1`);
  const updateTitleStmt = db.prepare(`UPDATE documents SET title = ?, modified_at = ? WHERE id = ?`);

  let indexed = 0, updated = 0, unchanged = 0, processed = 0;
  const seenPaths = new Set<string>();
  const startTime = Date.now();

  for (const relativeFile of files) {
    const filepath = getRealPath(resolve(resolvedPwd, relativeFile));
    const path = relativeFile; // Use relative path as-is
    seenPaths.add(path);

    const content = await Bun.file(filepath).text();
    const hash = await hashContent(content);
    const title = extractTitle(content, relativeFile);

    // Check if document exists in this collection with this path
    const existing = findActiveStmt.get(collectionId, path) as { id: number; hash: string; title: string } | null;

    if (existing) {
      if (existing.hash === hash) {
        // Hash unchanged, but check if title needs updating
        if (existing.title !== title) {
          updateTitleStmt.run(title, now, existing.id);
          updated++;
        } else {
          unchanged++;
        }
      } else {
        // Content changed - insert new content hash and update document
        insertContentStmt.run(hash, content, now);
        deactivateStmt.run(collectionId, path);
        updated++;
        const stat = await Bun.file(filepath).stat();
        insertDocStmt.run(collectionId, path, title, hash,
          stat ? new Date(stat.birthtime).toISOString() : now,
          stat ? new Date(stat.mtime).toISOString() : now);
      }
    } else {
      // New document - insert content and document
      indexed++;
      insertContentStmt.run(hash, content, now);
      const stat = await Bun.file(filepath).stat();
      insertDocStmt.run(collectionId, path, title, hash,
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
  const allActive = db.prepare(`SELECT path FROM documents WHERE collection_id = ? AND active = 1`).all(collectionId) as { path: string }[];
  let removed = 0;
  for (const row of allActive) {
    if (!seenPaths.has(row.path)) {
      deactivateStmt.run(collectionId, row.path);
      removed++;
    }
  }

  // Clean up orphaned content hashes (content not referenced by any document)
  const cleanupResult = db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
  `).run();
  const orphanedContent = cleanupResult.changes;

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
    db.exec(`DELETE FROM content_vectors`);
    db.exec(`DROP TABLE IF EXISTS vectors_vec`);
  }

  // Find unique hashes that need embedding (from active documents)
  // Use MIN(filepath) to get one representative filepath per hash
  const hashesToEmbed = db.prepare(`
    SELECT d.hash, d.body, MIN(d.filepath) as filepath, MIN(d.display_path) as display_path
    FROM documents d
    LEFT JOIN content_vectors v ON d.hash = v.hash AND v.seq = 0
    WHERE d.active = 1 AND v.hash IS NULL
    GROUP BY d.hash
  `).all() as { hash: string; body: string; filepath: string; display_path: string }[];

  if (hashesToEmbed.length === 0) {
    console.log(`${c.green}✓ All content hashes already have embeddings.${c.reset}`);
    closeDb();
    return;
  }

  // Prepare documents with chunks
  type ChunkItem = { hash: string; title: string; text: string; seq: number; pos: number; bytes: number; displayName: string };
  const allChunks: ChunkItem[] = [];
  let multiChunkDocs = 0;

  for (const item of hashesToEmbed) {
    const encoder = new TextEncoder();
    const bodyBytes = encoder.encode(item.body).length;
    if (bodyBytes === 0) continue; // Skip empty

    const title = extractTitle(item.body, item.filepath);
    const displayName = item.display_path || item.filepath;
    const chunks = chunkDocument(item.body, CHUNK_BYTE_SIZE);

    if (chunks.length > 1) multiChunkDocs++;

    for (let seq = 0; seq < chunks.length; seq++) {
      allChunks.push({
        hash: item.hash,
        title,
        text: chunks[seq].text,
        seq,
        pos: chunks[seq].pos,
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
  const firstEmbedding = await getEmbedding(allChunks[0].text, model, false, allChunks[0].title);
  ensureVecTable(db, firstEmbedding.length);

  const insertVecStmt = db.prepare(`INSERT OR REPLACE INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`);
  const insertContentVectorStmt = db.prepare(`INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, ?, ?, ?, ?)`);

  let chunksEmbedded = 0, errors = 0, bytesProcessed = 0;
  const startTime = Date.now();

  // Insert first chunk
  const firstHashSeq = `${allChunks[0].hash}_${allChunks[0].seq}`;
  insertVecStmt.run(firstHashSeq, new Float32Array(firstEmbedding));
  insertContentVectorStmt.run(allChunks[0].hash, allChunks[0].seq, allChunks[0].pos, model, now);
  chunksEmbedded++;
  bytesProcessed += allChunks[0].bytes;

  for (let i = 1; i < allChunks.length; i++) {
    const chunk = allChunks[i];
    try {
      const embedding = await getEmbedding(chunk.text, model, false, chunk.title);
      const hashSeq = `${chunk.hash}_${chunk.seq}`;
      insertVecStmt.run(hashSeq, new Float32Array(embedding));
      insertContentVectorStmt.run(chunk.hash, chunk.seq, chunk.pos, model, now);
      chunksEmbedded++;
      bytesProcessed += chunk.bytes;
    } catch (err) {
      errors++;
      bytesProcessed += chunk.bytes;
      progress.error();
      console.error(`\n${c.yellow}⚠ Error embedding "${chunk.displayName}" chunk ${chunk.seq}: ${err}${c.reset}`);
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

// Get collection ID by name (matches pwd or glob_pattern suffix)
function getCollectionIdByName(db: Database, name: string): number | null {
  // Search both pwd and glob_pattern columns for the name
  const result = db.prepare(`
    SELECT id FROM collections
    WHERE pwd LIKE ? OR glob_pattern LIKE ?
    ORDER BY LENGTH(pwd) DESC
    LIMIT 1
  `).get(`%${name}%`, `%${name}%`) as { id: number } | null;
  return result?.id || null;
}

// searchFTS and searchVec are now imported from store.ts with updated schema

// Removed duplicate searchFTS and searchVec functions - using store.ts versions instead
async function REMOVED_searchVec(db: Database, query: string, model: string, limit: number = 20, collectionId?: number): Promise<SearchResult[]> {
  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) return [];

  const queryEmbedding = await getEmbedding(query, model, true);
  const queryVec = new Float32Array(queryEmbedding);

  // Join: vectors_vec -> content_vectors -> documents
  // Over-retrieve to handle multiple chunks per document, then dedupe
  let sql = `
    SELECT d.filepath, d.display_path, d.title, d.body, vec.distance, cv.pos
    FROM vectors_vec vec
    JOIN content_vectors cv ON vec.hash_seq = cv.hash || '_' || cv.seq
    JOIN documents d ON d.hash = cv.hash AND d.active = 1
    WHERE vec.embedding MATCH ? AND k = ?
  `;
  if (collectionId !== undefined) {
    sql += ` AND d.collection_id = ${collectionId}`;
  }
  sql += ` ORDER BY vec.distance`;

  const stmt = db.prepare(sql);
  const rawResults = stmt.all(queryVec, limit * 3) as { filepath: string; display_path: string; title: string; body: string; distance: number; pos: number }[];

  // Aggregate chunks per document: max score + small bonus for additional matches
  const byFile = new Map<string, { filepath: string; displayPath: string; title: string; body: string; chunkCount: number; bestPos: number; bestDist: number }>();
  for (const r of rawResults) {
    const existing = byFile.get(r.filepath);
    if (!existing) {
      byFile.set(r.filepath, { filepath: r.filepath, displayPath: r.display_path, title: r.title, body: r.body, chunkCount: 1, bestPos: r.pos, bestDist: r.distance });
    } else {
      existing.chunkCount++;
      if (r.distance < existing.bestDist) {
        existing.bestDist = r.distance;
        existing.bestPos = r.pos;
      }
    }
  }

  // Score = max chunk score + 0.02 bonus per additional chunk (capped at +0.1)
  return Array.from(byFile.values())
    .map(r => {
      const maxScore = 1 / (1 + r.bestDist);
      const bonusChunks = Math.min(r.chunkCount - 1, 5);
      const bonus = bonusChunks * 0.02;
      return {
        file: r.filepath,
        displayPath: r.displayPath,
        title: r.title,
        body: r.body,
        score: maxScore + bonus,
        source: "vec" as const,
        chunkPos: r.bestPos,
      };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, limit);
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
};

// Extract snippet with more context lines for CLI display
function extractSnippetWithContext(body: string, query: string, contextLines = 3, chunkPos?: number): { line: number; snippet: string; hasMatch: boolean } {
  // If chunkPos provided, focus search on that area
  let lineOffset = 0;
  let searchBody = body;
  if (chunkPos && chunkPos > 0) {
    const contextStart = Math.max(0, chunkPos - 200);
    searchBody = body.slice(contextStart);
    if (contextStart > 0) {
      lineOffset = body.slice(0, contextStart).split('\n').length - 1;
    }
  }

  const lines = searchBody.split('\n');
  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  let bestLine = 0, bestScore = -1;

  for (let i = 0; i < lines.length; i++) {
    const lineLower = lines[i].toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      if (lineLower.includes(term)) score++;
    }
    if (score > bestScore) {
      bestScore = score;
      bestLine = i;
    }
  }

  // No query match found - return beginning of chunk area or file
  if (bestScore <= 0) {
    const preview = lines.slice(0, contextLines * 2).join('\n').trim();
    return { line: lineOffset + 1, snippet: preview, hasMatch: false };
  }

  const startLine = Math.max(0, bestLine - contextLines);
  const endLine = Math.min(lines.length, bestLine + contextLines + 1);
  const snippet = lines.slice(startLine, endLine).join('\n').trim();
  return { line: lineOffset + bestLine + 1, snippet, hasMatch: true };
}

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

function outputResults(results: { file: string; displayPath: string; title: string; body: string; score: number; context?: string | null; chunkPos?: number }[], query: string, opts: OutputOptions): void {
  const filtered = results.filter(r => r.score >= opts.minScore).slice(0, opts.limit);

  if (filtered.length === 0) {
    console.log("No results found above minimum score threshold.");
    return;
  }

  if (opts.format === "json") {
    // JSON output for LLM consumption
    const output = filtered.map(row => ({
      score: Math.round(row.score * 100) / 100,
      file: row.displayPath,
      title: row.title,
      ...(row.context && { context: row.context }),
      ...(opts.full && { body: row.body }),
      ...(!opts.full && { snippet: extractSnippet(row.body, query, 300, row.chunkPos).snippet }),
    }));
    console.log(JSON.stringify(output, null, 2));
  } else if (opts.format === "files") {
    // Simple score,filepath,context output
    for (const row of filtered) {
      const ctx = row.context ? `,"${row.context.replace(/"/g, '""')}"` : "";
      console.log(`${row.score.toFixed(2)},${row.displayPath}${ctx}`);
    }
  } else if (opts.format === "cli") {
    for (let i = 0; i < filtered.length; i++) {
      const row = filtered[i];
      const { line, snippet, hasMatch } = extractSnippetWithContext(row.body, query, 2, row.chunkPos);

      // Line 1: filepath
      const path = row.displayPath;
      const lineInfo = hasMatch ? `:${line}` : "";
      console.log(`${c.cyan}${path}${c.dim}${lineInfo}${c.reset}`);

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

      // Snippet with highlighting (no leading | chars for better word wrap)
      const highlighted = highlightTerms(snippet, query);
      console.log(highlighted);

      // Double empty line between results
      if (i < filtered.length - 1) console.log('\n');
    }
  } else if (opts.format === "md") {
    for (const row of filtered) {
      const heading = row.title || row.displayPath;
      if (opts.full) {
        console.log(`---\n# ${heading}\n\n${row.body}\n`);
      } else {
        const { snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
        console.log(`---\n# ${heading}\n\n${snippet}\n`);
      }
    }
  } else if (opts.format === "xml") {
    for (const row of filtered) {
      const titleAttr = row.title ? ` title="${row.title.replace(/"/g, '&quot;')}"` : "";
      if (opts.full) {
        console.log(`<file name="${row.displayPath}"${titleAttr}>\n${row.body}\n</file>\n`);
      } else {
        const { snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
        console.log(`<file name="${row.displayPath}"${titleAttr}>\n${snippet}\n</file>\n`);
      }
    }
  } else {
    // CSV format
    console.log("score,file,title,line,snippet");
    for (const row of filtered) {
      const { line, snippet } = extractSnippet(row.body, query, 500, row.chunkPos);
      const content = opts.full ? row.body : snippet;
      console.log(`${row.score.toFixed(4)},${escapeCSV(row.displayPath)},${escapeCSV(row.title)},${line},${escapeCSV(content)}`);
    }
  }
}

function search(query: string, opts: OutputOptions): void {
  const db = getDb();

  // Resolve collection filter if specified
  let collectionId: number | undefined;
  if (opts.collection) {
    collectionId = getCollectionIdByName(db, opts.collection) ?? undefined;
    if (collectionId === undefined) {
      console.error(`Collection not found: ${opts.collection}`);
      closeDb();
      process.exit(1);
    }
  }

  // Use large limit for --all, otherwise fetch more than needed and let outputResults filter
  const fetchLimit = opts.all ? 100000 : Math.max(50, opts.limit * 2);
  const results = searchFTS(db, query, fetchLimit, collectionId);

  // Add context to results
  const resultsWithContext = results.map(r => ({
    ...r,
    context: getContextForFile(db, r.file),
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

  // Resolve collection filter if specified
  let collectionId: number | undefined;
  if (opts.collection) {
    collectionId = getCollectionIdByName(db, opts.collection) ?? undefined;
    if (collectionId === undefined) {
      console.error(`Collection not found: ${opts.collection}`);
      closeDb();
      process.exit(1);
    }
  }

  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) {
    console.error("Vector index not found. Run 'qmd embed' first to create embeddings.");
    closeDb();
    return;
  }

  // Check index health and warn about issues
  checkIndexHealth(db);

  // Expand query to multiple variations (with caching)
  const queries = await expandQuery(query, DEFAULT_QUERY_MODEL, db);
  process.stderr.write(`Searching with ${queries.length} query variations...\n`);

  // Collect results from all query variations
  // For --all, fetch more results per query
  const perQueryLimit = opts.all ? 500 : 20;
  const allResults = new Map<string, { file: string; displayPath: string; title: string; body: string; score: number }>();

  for (const q of queries) {
    const vecResults = await searchVec(db, q, model, perQueryLimit, collectionId);
    for (const r of vecResults) {
      const existing = allResults.get(r.file);
      if (!existing || r.score > existing.score) {
        allResults.set(r.file, { file: r.file, displayPath: r.displayPath, title: r.title, body: r.body, score: r.score });
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

async function expandQuery(query: string, model: string = DEFAULT_QUERY_MODEL, db?: Database): Promise<string[]> {
  process.stderr.write("Generating query variations...\n");

  const prompt = `You are a search query expander. Given a search query, generate 2 alternative queries that would help find relevant documents.

Rules:
- Use synonyms and related terminology (e.g., "craft" → "craftsmanship", "quality", "excellence")
- Rephrase to capture different angles (e.g., "engineering culture" → "technical excellence", "developer practices")
- Keep proper nouns and named concepts exactly as written (e.g., "Build a Business", "Stripe", "Shopify")
- Each variation should be 3-8 words, natural search terms
- Do NOT just append words like "search" or "find" or "documents"

Query: "${query}"

Output exactly 2 variations, one per line, no numbering or bullets:`;

  const requestBody = {
    model,
    prompt,
    stream: false,
    think: false,
    options: { num_predict: 150 },
  };

  // Check cache
  const cacheDb = db || getDb();
  const cacheKey = getCacheKey(`${OLLAMA_URL}/api/generate`, requestBody);
  const cached = getCachedResult(cacheDb, cacheKey);

  let responseText: string;
  if (cached) {
    responseText = cached;
  } else {
    const response = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      if (errorText.includes("not found") || errorText.includes("does not exist")) {
        await ensureModelAvailable(model);
        if (!db) cacheDb.close();
        return expandQuery(query, model, db);
      }
      if (!db) cacheDb.close();
      return [query];
    }

    const data = await response.json() as { response: string };
    responseText = data.response;
    setCachedResult(cacheDb, cacheKey, responseText);
  }

  if (!db) cacheDb.close();

  const lines = responseText.trim().split('\n')
    .map(l => l.replace(/^[\d\.\-\*\"\s]+/, '').replace(/["\s]+$/, '').trim())
    .filter(l => l.length > 2 && l.length < 100 && !l.startsWith('<') && !l.toLowerCase().includes('variation'))
    .slice(0, 2);

  const allQueries = [query, ...lines];
  process.stderr.write(`${c.dim}Queries: ${allQueries.join(' | ')}${c.reset}\n`);
  return allQueries;
}

async function querySearch(query: string, opts: OutputOptions, embedModel: string = DEFAULT_EMBED_MODEL, rerankModel: string = DEFAULT_RERANK_MODEL): Promise<void> {
  const db = getDb();

  // Resolve collection filter if specified
  let collectionId: number | undefined;
  if (opts.collection) {
    collectionId = getCollectionIdByName(db, opts.collection) ?? undefined;
    if (collectionId === undefined) {
      console.error(`Collection not found: ${opts.collection}`);
      closeDb();
      process.exit(1);
    }
  }

  // Check index health and warn about issues
  checkIndexHealth(db);

  // Expand query to multiple variations (with caching)
  const queries = await expandQuery(query, DEFAULT_QUERY_MODEL, db);
  process.stderr.write(`Searching with ${queries.length} query variations...\n`);

  // Collect ranked result lists for RRF fusion
  const rankedLists: RankedResult[][] = [];
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();

  for (const q of queries) {
    // FTS search - get ranked results
    const ftsResults = searchFTS(db, q, 20, collectionId);
    if (ftsResults.length > 0) {
      rankedLists.push(ftsResults.map(r => ({ file: r.file, displayPath: r.displayPath, title: r.title, body: r.body, score: r.score })));
    }

    // Vector search - get ranked results
    if (hasVectors) {
      const vecResults = await searchVec(db, q, embedModel, 20, collectionId);
      if (vecResults.length > 0) {
        rankedLists.push(vecResults.map(r => ({ file: r.file, displayPath: r.displayPath, title: r.title, body: r.body, score: r.score })));
      }
    }
  }

  // Apply Reciprocal Rank Fusion to combine all ranked lists
  // Give 2x weight to original query results (first 2 lists: FTS + vector)
  const weights = rankedLists.map((_, i) => i < 2 ? 2.0 : 1.0);
  const fused = reciprocalRankFusion(rankedLists, weights);
  const candidates = fused.slice(0, 30); // Over-retrieve for reranking

  if (candidates.length === 0) {
    console.log("No results found.");
    closeDb();
    return;
  }

  // Rerank with the original query (with caching)
  const reranked = await rerank(
    query,
    candidates.map(c => ({ file: c.file, text: c.body })),
    rerankModel,
    db
  );

  // Blend RRF position score with reranker score using position-aware weights
  // Top retrieval results get more protection from reranker disagreement
  const candidateMap = new Map(candidates.map(c => [c.file, { displayPath: c.displayPath, title: c.title, body: c.body }]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1])); // 1-indexed rank

  const finalResults = reranked.map(r => {
    const rrfRank = rrfRankMap.get(r.file) || 30;
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
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * r.score;
    const candidate = candidateMap.get(r.file);
    return {
      file: r.file,
      displayPath: candidate?.displayPath || "",
      title: candidate?.title || "",
      body: candidate?.body || "",
      score: blendedScore,
      context: getContextForFile(db, r.file),
    };
  }).sort((a, b) => b.score - a.score);

  closeDb();
  outputResults(finalResults, query, opts);
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
      // Get options
      l: { type: "string" },  // max lines
      from: { type: "string" },  // start line
      "max-bytes": { type: "string" },  // max bytes for multi-get
    },
    allowPositionals: true,
    strict: false, // Allow unknown options to pass through
  });

  // Set global index name in store
  if (values.index) {
    setCustomIndexName(values.index);
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
  console.log("  qmd update                    - Re-index all collections");
  console.log("  qmd embed [-f]                - Create vector embeddings (chunks ~6KB each)");
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
  console.log("  --files                    - Output score,filepath,context (default: 20 results)");
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
  console.log("Environment:");
  console.log("  OLLAMA_URL                 - Ollama server URL (default: http://localhost:11434)");
  console.log("");
  console.log("Models:");
  console.log(`  Embedding: ${DEFAULT_EMBED_MODEL}`);
  console.log(`  Reranking: ${DEFAULT_RERANK_MODEL}`);
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
      console.error("Usage: qmd context <add|list|rm>");
      console.error("");
      console.error("Commands:");
      console.error("  qmd context add [path] \"text\"  - Add context (defaults to current dir)");
      console.error("  qmd context add / \"text\"       - Add global context to all collections");
      console.error("  qmd context list                - List all contexts");
      console.error("  qmd context rm <path>           - Remove context");
      process.exit(1);
    }

    switch (subcommand) {
      case "add": {
        if (cli.args.length < 2) {
          console.error("Usage: qmd context add [path] \"text\"");
          console.error("Examples:");
          console.error("  qmd context add \"Context for current directory\"");
          console.error("  qmd context add . \"Context for current directory\"");
          console.error("  qmd context add /subfolder \"Context for subfolder\"");
          console.error("  qmd context add / \"Global context for all collections\"");
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
        console.error("Available: add, list, rm");
        process.exit(1);
    }
    break;
  }

  // Legacy alias for backwards compatibility
  case "add-context": {
    console.error(`${c.yellow}Note: 'qmd add-context' is deprecated. Use 'qmd context add' instead.${c.reset}`);
    if (cli.args.length === 0) {
      console.error("Usage: qmd context add [path] \"text\"");
      process.exit(1);
    }
    let pathArg: string | undefined;
    let contextText: string;
    if (cli.args.length === 1) {
      pathArg = undefined;
      contextText = cli.args[0];
    } else {
      pathArg = cli.args[0];
      contextText = cli.args.slice(1).join(" ");
    }
    await contextAdd(pathArg, contextText);
    break;
  }

  case "get": {
    if (!cli.args[0]) {
      console.error("Usage: qmd get <filepath>[:line] [--from <line>] [-l <lines>]");
      process.exit(1);
    }
    const fromLine = cli.values.from ? parseInt(cli.values.from as string, 10) : undefined;
    const maxLines = cli.values.l ? parseInt(cli.values.l as string, 10) : undefined;
    getDocument(cli.args[0], fromLine, maxLines);
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

    // 1. Clear ollama_cache
    const cacheCount = db.prepare(`SELECT COUNT(*) as c FROM ollama_cache`).get() as { c: number };
    db.exec(`DELETE FROM ollama_cache`);
    console.log(`${c.green}✓${c.reset} Cleared ${cacheCount.c} cached API responses`);

    // 2. Remove orphaned vectors (no active document with that hash)
    const orphanedVecs = db.prepare(`
      SELECT COUNT(*) as c FROM content_vectors cv
      WHERE NOT EXISTS (
        SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
      )
    `).get() as { c: number };

    if (orphanedVecs.c > 0) {
      db.exec(`
        DELETE FROM vectors_vec WHERE hash_seq IN (
          SELECT cv.hash || '_' || cv.seq FROM content_vectors cv
          WHERE NOT EXISTS (
            SELECT 1 FROM documents d WHERE d.hash = cv.hash AND d.active = 1
          )
        )
      `);
      db.exec(`
        DELETE FROM content_vectors WHERE hash NOT IN (
          SELECT hash FROM documents WHERE active = 1
        )
      `);
      console.log(`${c.green}✓${c.reset} Removed ${orphanedVecs.c} orphaned embedding chunks`);
    } else {
      console.log(`${c.dim}No orphaned embeddings to remove${c.reset}`);
    }

    // 3. Count inactive documents
    const inactiveDocs = db.prepare(`SELECT COUNT(*) as c FROM documents WHERE active = 0`).get() as { c: number };
    if (inactiveDocs.c > 0) {
      db.exec(`DELETE FROM documents WHERE active = 0`);
      console.log(`${c.green}✓${c.reset} Removed ${inactiveDocs.c} inactive document records`);
    }

    // 4. Vacuum to reclaim space
    db.exec(`VACUUM`);
    console.log(`${c.green}✓${c.reset} Database vacuumed`);

    closeDb();
    break;
  }

  default:
    console.error(`Unknown command: ${cli.command}`);
    console.error("Run 'qmd --help' for usage.");
    process.exit(1);
}
} // end if (import.meta.main)
