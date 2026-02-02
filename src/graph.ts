/**
 * QMD Graph - Lightweight graph layer for qmd
 *
 * Adds entity and edge storage to qmd's SQLite database,
 * enabling relationship traversal alongside vector/BM25 search.
 */

import { Database } from "bun:sqlite";

// =============================================================================
// Types
// =============================================================================

export interface Entity {
  id: string;
  type: string;
  name: string;
  metadata?: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface Edge {
  id: number;
  source_id: string;
  target_id: string;
  relation: string;
  weight: number;
  metadata?: Record<string, unknown>;
  created_at: string;
}

export interface TraversalResult {
  node_id: string;
  node_type: 'entity' | 'document';
  depth: number;
  path: string[];
  relations: string[];
}

export interface GraphStats {
  entity_count: number;
  edge_count: number;
  entity_types: Record<string, number>;
  relation_types: Record<string, number>;
}

// =============================================================================
// Schema Initialization
// =============================================================================

export function initGraphSchema(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS entities (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      name TEXT NOT NULL,
      metadata TEXT,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)`);

  db.exec(`
    CREATE TABLE IF NOT EXISTS edges (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      source_id TEXT NOT NULL,
      target_id TEXT NOT NULL,
      relation TEXT NOT NULL,
      weight REAL NOT NULL DEFAULT 1.0,
      metadata TEXT,
      created_at TEXT NOT NULL,
      UNIQUE(source_id, target_id, relation)
    )
  `);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation)`);

  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
      id, name, type,
      tokenize='porter unicode61'
    )
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities
    BEGIN
      INSERT INTO entities_fts(rowid, id, name, type)
      VALUES (new.rowid, new.id, new.name, new.type);
    END
  `);
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities
    BEGIN
      DELETE FROM entities_fts WHERE rowid = old.rowid;
    END
  `);
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities
    BEGIN
      DELETE FROM entities_fts WHERE rowid = old.rowid;
      INSERT INTO entities_fts(rowid, id, name, type)
      VALUES (new.rowid, new.id, new.name, new.type);
    END
  `);
}

// =============================================================================
// Entity CRUD
// =============================================================================

function slugify(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
}

export function entityId(type: string, name: string): string {
  return `${type}:${slugify(name)}`;
}

export function createEntity(
  db: Database, type: string, name: string, id?: string, metadata?: Record<string, unknown>
): Entity {
  const eid = id || entityId(type, name);
  const now = new Date().toISOString();
  const metadataJson = metadata ? JSON.stringify(metadata) : null;
  db.prepare(`
    INSERT INTO entities (id, type, name, metadata, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET name = excluded.name, metadata = excluded.metadata, updated_at = excluded.updated_at
  `).run(eid, type, name, metadataJson, now, now);
  return { id: eid, type, name, metadata, created_at: now, updated_at: now };
}

export function getEntity(db: Database, id: string): Entity | null {
  const row = db.prepare(`SELECT * FROM entities WHERE id = ?`).get(id) as {
    id: string; type: string; name: string; metadata: string | null;
    created_at: string; updated_at: string;
  } | null;
  if (!row) return null;
  return { ...row, metadata: row.metadata ? JSON.parse(row.metadata) : undefined };
}

export function listEntities(db: Database, type?: string, limit = 100): Entity[] {
  const query = type
    ? db.prepare(`SELECT * FROM entities WHERE type = ? ORDER BY name LIMIT ?`)
    : db.prepare(`SELECT * FROM entities ORDER BY type, name LIMIT ?`);
  const rows = (type ? query.all(type, limit) : query.all(limit)) as Array<{
    id: string; type: string; name: string; metadata: string | null;
    created_at: string; updated_at: string;
  }>;
  return rows.map(r => ({ ...r, metadata: r.metadata ? JSON.parse(r.metadata) : undefined }));
}

export function searchEntities(db: Database, query: string, limit = 20): Entity[] {
  const rows = db.prepare(`
    SELECT e.* FROM entities e
    JOIN entities_fts fts ON e.rowid = fts.rowid
    WHERE entities_fts MATCH ?
    ORDER BY rank LIMIT ?
  `).all(query, limit) as Array<{
    id: string; type: string; name: string; metadata: string | null;
    created_at: string; updated_at: string;
  }>;
  return rows.map(r => ({ ...r, metadata: r.metadata ? JSON.parse(r.metadata) : undefined }));
}

export function deleteEntity(db: Database, id: string): boolean {
  db.prepare(`DELETE FROM edges WHERE source_id = ? OR target_id = ?`).run(id, id);
  return db.prepare(`DELETE FROM entities WHERE id = ?`).run(id).changes > 0;
}

// =============================================================================
// Edge CRUD
// =============================================================================

export function createEdge(
  db: Database, sourceId: string, targetId: string, relation: string, weight = 1.0, metadata?: Record<string, unknown>
): Edge {
  const now = new Date().toISOString();
  const metadataJson = metadata ? JSON.stringify(metadata) : null;
  const result = db.prepare(`
    INSERT INTO edges (source_id, target_id, relation, weight, metadata, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(source_id, target_id, relation) DO UPDATE SET weight = excluded.weight, metadata = excluded.metadata
  `).run(sourceId, targetId, relation, weight, metadataJson, now);
  return { id: Number(result.lastInsertRowid), source_id: sourceId, target_id: targetId, relation, weight, metadata, created_at: now };
}

export function getEdges(db: Database, nodeId: string, direction: 'outgoing' | 'incoming' | 'both' = 'both', relation?: string): Edge[] {
  let query: string;
  const params: unknown[] = [];
  if (direction === 'outgoing') { query = `SELECT * FROM edges WHERE source_id = ?`; params.push(nodeId); }
  else if (direction === 'incoming') { query = `SELECT * FROM edges WHERE target_id = ?`; params.push(nodeId); }
  else { query = `SELECT * FROM edges WHERE (source_id = ? OR target_id = ?)`; params.push(nodeId, nodeId); }
  if (relation) { query += ` AND relation = ?`; params.push(relation); }
  const rows = db.prepare(query).all(...params) as Array<{
    id: number; source_id: string; target_id: string; relation: string;
    weight: number; metadata: string | null; created_at: string;
  }>;
  return rows.map(r => ({ ...r, metadata: r.metadata ? JSON.parse(r.metadata) : undefined }));
}

export function deleteEdge(db: Database, sourceId: string, targetId: string, relation?: string): number {
  if (relation) return db.prepare(`DELETE FROM edges WHERE source_id = ? AND target_id = ? AND relation = ?`).run(sourceId, targetId, relation).changes;
  return db.prepare(`DELETE FROM edges WHERE source_id = ? AND target_id = ?`).run(sourceId, targetId).changes;
}

// =============================================================================
// Graph Traversal
// =============================================================================

export function traverse(
  db: Database, startId: string, maxDepth = 2, relations?: string[], direction: 'outgoing' | 'incoming' | 'both' = 'both'
): TraversalResult[] {
  const relationFilter = relations?.length ? `AND e.relation IN (${relations.map(() => '?').join(',')})` : '';
  let edgeCondition: string, nextNode: string;
  if (direction === 'outgoing') { edgeCondition = 'e.source_id = t.node_id'; nextNode = 'e.target_id'; }
  else if (direction === 'incoming') { edgeCondition = 'e.target_id = t.node_id'; nextNode = 'e.source_id'; }
  else { edgeCondition = '(e.source_id = t.node_id OR e.target_id = t.node_id)'; nextNode = "CASE WHEN e.source_id = t.node_id THEN e.target_id ELSE e.source_id END"; }

  const query = `
    WITH RECURSIVE traversal(node_id, depth, path, relations) AS (
      SELECT ? as node_id, 0 as depth, json_array(?) as path, json_array() as relations
      UNION ALL
      SELECT ${nextNode}, t.depth + 1, json_insert(t.path, '$[#]', ${nextNode}), json_insert(t.relations, '$[#]', e.relation)
      FROM traversal t JOIN edges e ON ${edgeCondition}
      WHERE t.depth < ? AND ${nextNode} NOT IN (SELECT value FROM json_each(t.path)) ${relationFilter}
    )
    SELECT DISTINCT t.node_id, CASE WHEN EXISTS (SELECT 1 FROM entities WHERE id = t.node_id) THEN 'entity' ELSE 'document' END as node_type, t.depth, t.path, t.relations
    FROM traversal t WHERE t.depth > 0 ORDER BY t.depth, t.node_id
  `;
  const params: unknown[] = [startId, startId, maxDepth];
  if (relations?.length) params.push(...relations);
  const rows = db.prepare(query).all(...params) as Array<{
    node_id: string; node_type: 'entity' | 'document'; depth: number; path: string; relations: string;
  }>;
  return rows.map(r => ({ node_id: r.node_id, node_type: r.node_type, depth: r.depth, path: JSON.parse(r.path), relations: JSON.parse(r.relations) }));
}

export function findPath(db: Database, fromId: string, toId: string, maxDepth = 5): TraversalResult | null {
  const results = traverse(db, fromId, maxDepth, undefined, 'both');
  return results.find(r => r.node_id === toId) || null;
}

export function getGraphStats(db: Database): GraphStats {
  const entityCount = (db.prepare(`SELECT COUNT(*) as cnt FROM entities`).get() as { cnt: number }).cnt;
  const edgeCount = (db.prepare(`SELECT COUNT(*) as cnt FROM edges`).get() as { cnt: number }).cnt;
  const entityTypes = db.prepare(`SELECT type, COUNT(*) as cnt FROM entities GROUP BY type`).all() as Array<{ type: string; cnt: number }>;
  const relationTypes = db.prepare(`SELECT relation, COUNT(*) as cnt FROM edges GROUP BY relation`).all() as Array<{ relation: string; cnt: number }>;
  return {
    entity_count: entityCount, edge_count: edgeCount,
    entity_types: Object.fromEntries(entityTypes.map(r => [r.type, r.cnt])),
    relation_types: Object.fromEntries(relationTypes.map(r => [r.relation, r.cnt])),
  };
}
