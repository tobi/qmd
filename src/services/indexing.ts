/**
 * Document indexing service
 */

import { Database } from 'bun:sqlite';
import { Glob } from 'bun';
import { CollectionRepository, DocumentRepository } from '../database/repositories/index.ts';
import { hashContent } from '../utils/hash.ts';
import { getRealPath, computeDisplayPath, getPwd } from '../utils/paths.ts';
import { formatETA } from '../utils/formatters.ts';
import { progress } from '../config/terminal.ts';
import { getHashesNeedingEmbedding } from '../database/db.ts';
import { shouldAnalyze, analyzeDatabase } from '../database/performance.ts';
import { resolve } from 'path';

/**
 * Extract title from markdown content
 * @param content - File content
 * @param filename - Fallback filename
 * @returns Document title
 */
function extractTitle(content: string, filename: string): string {
  const firstLine = content.split('\n')[0];
  if (firstLine?.startsWith('# ')) {
    return firstLine.slice(2).trim();
  }
  return filename.replace(/\.md$/, '').split('/').pop() || filename;
}

/**
 * Clear Ollama cache
 * @param db - Database instance
 */
function clearCache(db: Database): void {
  db.prepare('DELETE FROM ollama_cache').run();
}

/**
 * Get or create collection
 * @param db - Database instance
 * @param pwd - Working directory
 * @param globPattern - Glob pattern
 * @returns Collection ID
 */
function getOrCreateCollection(db: Database, pwd: string, globPattern: string): number {
  const collectionRepo = new CollectionRepository(db);

  const existing = collectionRepo.findByPwdAndPattern(pwd, globPattern);
  if (existing) {
    return existing.id;
  }

  return collectionRepo.insert(pwd, globPattern);
}

/**
 * Index markdown files
 * @param db - Database instance
 * @param globPattern - Glob pattern for files
 * @param pwd - Working directory (defaults to process.cwd())
 * @returns Indexing statistics
 */
export async function indexFiles(
  db: Database,
  globPattern: string,
  pwd: string = getPwd()
): Promise<{ indexed: number; updated: number; unchanged: number; removed: number; needsEmbedding: number }> {
  const now = new Date().toISOString();
  const excludeDirs = ["node_modules", ".git", ".cache", "vendor", "dist", "build"];

  // Clear Ollama cache on index
  clearCache(db);

  // Get or create collection for this (pwd, glob)
  const collectionId = getOrCreateCollection(db, pwd, globPattern);
  console.log(`Collection: ${pwd} (${globPattern})`);

  progress.indeterminate();
  const glob = new Glob(globPattern);
  const files: string[] = [];
  for await (const file of glob.scan({ cwd: pwd, onlyFiles: true, followSymlinks: true })) {
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
    return { indexed: 0, updated: 0, unchanged: 0, removed: 0, needsEmbedding: 0 };
  }

  const insertStmt = db.prepare(`INSERT INTO documents (collection_id, name, title, hash, filepath, display_path, body, created_at, modified_at, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)`);
  const deactivateStmt = db.prepare(`UPDATE documents SET active = 0 WHERE collection_id = ? AND filepath = ? AND active = 1`);
  const findActiveStmt = db.prepare(`SELECT id, hash, title, display_path FROM documents WHERE collection_id = ? AND filepath = ? AND active = 1`);
  const findActiveAnyCollectionStmt = db.prepare(`SELECT id, collection_id, hash, title, display_path FROM documents WHERE filepath = ? AND active = 1`);
  const updateTitleStmt = db.prepare(`UPDATE documents SET title = ?, modified_at = ? WHERE id = ?`);
  const updateDisplayPathStmt = db.prepare(`UPDATE documents SET display_path = ? WHERE id = ?`);

  // Collect all existing display_paths for uniqueness check
  const existingDisplayPaths = new Set<string>(
    (db.prepare(`SELECT display_path FROM documents WHERE active = 1 AND display_path != ''`).all() as { display_path: string }[])
      .map(r => r.display_path)
  );

  let indexed = 0, updated = 0, unchanged = 0, processed = 0;
  const seenFiles = new Set<string>();
  const startTime = Date.now();

  for (const relativeFile of files) {
    const filepath = getRealPath(resolve(pwd, relativeFile));
    seenFiles.add(filepath);

    const content = await Bun.file(filepath).text();
    const hash = await hashContent(content);
    const name = relativeFile.replace(/\.md$/, "").split("/").pop() || relativeFile;
    const title = extractTitle(content, relativeFile);

    // First check if file exists in THIS collection
    const existing = findActiveStmt.get(collectionId, filepath) as { id: number; hash: string; title: string; display_path: string } | null;

    if (existing) {
      if (existing.hash === hash) {
        // Hash unchanged, but check if title needs updating
        if (existing.title !== title) {
          updateTitleStmt.run(title, now, existing.id);
          updated++;
        } else {
          unchanged++;
        }
        // Update display_path if empty
        if (!existing.display_path) {
          const displayPath = computeDisplayPath(filepath, pwd, existingDisplayPaths);
          updateDisplayPathStmt.run(displayPath, existing.id);
          existingDisplayPaths.add(displayPath);
        }
      } else {
        // Content changed - deactivate old, insert new
        existingDisplayPaths.delete(existing.display_path);
        deactivateStmt.run(collectionId, filepath);
        updated++;
        const stat = await Bun.file(filepath).stat();
        const displayPath = computeDisplayPath(filepath, pwd, existingDisplayPaths);
        insertStmt.run(collectionId, name, title, hash, filepath, displayPath, content, stat ? new Date(stat.birthtime).toISOString() : now, stat ? new Date(stat.mtime).toISOString() : now);
        existingDisplayPaths.add(displayPath);
      }
    } else {
      // Check if file exists in ANY collection (would violate unique constraint)
      const existingAnywhere = findActiveAnyCollectionStmt.get(filepath) as { id: number; collection_id: number; hash: string; title: string; display_path: string } | null;
      if (existingAnywhere) {
        // File already indexed in another collection - skip it
        unchanged++;
      } else {
        indexed++;
        const stat = await Bun.file(filepath).stat();
        const displayPath = computeDisplayPath(filepath, pwd, existingDisplayPaths);
        insertStmt.run(collectionId, name, title, hash, filepath, displayPath, content, stat ? new Date(stat.birthtime).toISOString() : now, stat ? new Date(stat.mtime).toISOString() : now);
        existingDisplayPaths.add(displayPath);
      }
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
  const allActive = db.prepare(`SELECT filepath FROM documents WHERE collection_id = ? AND active = 1`).all(collectionId) as { filepath: string }[];
  let removed = 0;
  for (const row of allActive) {
    if (!seenFiles.has(row.filepath)) {
      deactivateStmt.run(collectionId, row.filepath);
      removed++;
    }
  }

  // Check if vector index needs updating
  const needsEmbedding = getHashesNeedingEmbedding(db);

  progress.clear();
  console.log(`\nIndexed: ${indexed} new, ${updated} updated, ${unchanged} unchanged, ${removed} removed`);

  if (needsEmbedding > 0) {
    console.log(`\nRun 'qmd embed' to update embeddings (${needsEmbedding} unique hashes need vectors)`);
  }

  // Optimize query planner if significant changes were made
  const totalChanges = indexed + updated + removed;
  if (shouldAnalyze(db, totalChanges)) {
    analyzeDatabase(db);
  }

  return { indexed, updated, unchanged, removed, needsEmbedding };
}
