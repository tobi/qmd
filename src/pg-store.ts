/**
 * pg-store.ts - PostgreSQL adapter for QMD
 *
 * This module provides database exploration capabilities by:
 * 1. Indexing the database schema as searchable documents
 * 2. Syncing key content tables to the local SQLite index
 * 3. Translating natural language queries to SQL
 *
 * Usage:
 *   const pg = createPgStore("postgresql://user:pass@localhost/db");
 *   await pg.indexSchema();  // Index schema for semantic search
 *   await pg.syncContent();  // Sync messages, bookmarks, etc.
 */

import { Database } from "bun:sqlite";
import { getDefaultLlamaCpp, type LlamaCpp } from "./llm";

// =============================================================================
// Types
// =============================================================================

export type PgConfig = {
  connectionString: string;
  name: string;  // Collection name for this database
};

export type TableInfo = {
  schema: string;
  name: string;
  type: "table" | "view";
  rowCount: number;
  columns: ColumnInfo[];
  primaryKey: string[];
  foreignKeys: ForeignKeyInfo[];
  description?: string;
};

export type ColumnInfo = {
  name: string;
  type: string;
  nullable: boolean;
  defaultValue?: string;
  description?: string;
};

export type ForeignKeyInfo = {
  column: string;
  referencesTable: string;
  referencesColumn: string;
};

export type SchemaDocument = {
  type: "table" | "column" | "relationship";
  path: string;  // e.g., "messages", "messages.sender_contact_id"
  title: string;
  body: string;
  metadata: Record<string, unknown>;
};

export type QueryPlan = {
  sql: string;
  explanation: string;
  tables: string[];
  estimatedRows?: number;
};

export type ContentSyncConfig = {
  table: string;
  contentColumn: string;
  titleColumn?: string;
  timestampColumn?: string;
  idColumn: string;
  joins?: string[];  // SQL JOIN clauses for denormalization
  where?: string;    // Additional WHERE conditions
  limit?: number;
};

// =============================================================================
// Default content sync configurations for the target database
// =============================================================================

export const DEFAULT_CONTENT_SYNC: ContentSyncConfig[] = [
  // Messages - the main content
  {
    table: "messages",
    contentColumn: "body",
    titleColumn: "msgtype",
    timestampColumn: "origin_server_ts",
    idColumn: "message_id",
    joins: [
      "LEFT JOIN contacts c ON messages.sender_contact_id = c.contact_id",
      "LEFT JOIN rooms r ON messages.room_id = r.room_id",
    ],
  },
  // Message text representations (includes OCR, transcripts)
  {
    table: "message_text_representation",
    contentColumn: "text",
    idColumn: "id",
    joins: [
      "JOIN messages m ON message_text_representation.message_id = m.message_id",
    ],
  },
  // Session summaries - AI-generated conversation summaries
  {
    table: "session_summaries",
    contentColumn: "summary",
    titleColumn: "headline",
    timestampColumn: "created_at",
    idColumn: "id",
  },
  // Bookmarks - saved URLs with content
  {
    table: "bookmarks",
    contentColumn: "url",  // URL as searchable content
    idColumn: "bookmark_id",
    joins: [
      "LEFT JOIN bookmark_titles bt ON bookmarks.bookmark_id = bt.bookmark_id",
      "LEFT JOIN processed_contents pc ON bookmarks.bookmark_id = pc.bookmark_id",
    ],
  },
  // Processed bookmark content
  {
    table: "processed_contents",
    contentColumn: "content",
    idColumn: "id",
    joins: [
      "JOIN bookmarks b ON processed_contents.bookmark_id = b.bookmark_id",
    ],
  },
  // Entities (knowledge graph nodes)
  {
    table: "entities",
    contentColumn: "name",
    titleColumn: "type",
    idColumn: "entity_id",
  },
  // Dispatch transcriptions
  {
    table: "dispatch_transcription",
    contentColumn: "text",
    timestampColumn: "created_at",
    idColumn: "id",
  },
  // Dispatch analysis/bulletpoints
  {
    table: "dispatch_bulletpoint",
    contentColumn: "content",
    idColumn: "id",
  },
];

// =============================================================================
// Schema Document Generation
// =============================================================================

/**
 * Generate a searchable document from table metadata.
 * This allows queries like "what tables store user information?" to find contacts, contact_sources, etc.
 */
export function tableToDocument(table: TableInfo): SchemaDocument {
  const columnList = table.columns
    .map(c => `  - ${c.name} (${c.type}${c.nullable ? ", nullable" : ""})${c.description ? `: ${c.description}` : ""}`)
    .join("\n");

  const fkList = table.foreignKeys.length > 0
    ? "\n\nForeign Keys:\n" + table.foreignKeys
        .map(fk => `  - ${fk.column} → ${fk.referencesTable}.${fk.referencesColumn}`)
        .join("\n")
    : "";

  const body = `Table: ${table.schema}.${table.name}
Type: ${table.type}
Rows: ${table.rowCount.toLocaleString()}
${table.description ? `Description: ${table.description}\n` : ""}
Columns:
${columnList}${fkList}

Primary Key: ${table.primaryKey.length > 0 ? table.primaryKey.join(", ") : "(none)"}`;

  return {
    type: "table",
    path: `${table.schema}/${table.name}`,
    title: `${table.name} (${table.rowCount.toLocaleString()} rows)`,
    body,
    metadata: {
      schema: table.schema,
      table: table.name,
      rowCount: table.rowCount,
      columnCount: table.columns.length,
      hasForeignKeys: table.foreignKeys.length > 0,
    },
  };
}

/**
 * Generate documents for important columns (enables "where is email stored?" queries)
 */
export function columnToDocument(table: TableInfo, column: ColumnInfo): SchemaDocument {
  const fks = table.foreignKeys.filter(fk => fk.column === column.name);
  const fkInfo = fks.length > 0
    ? `\nReferences: ${fks.map(fk => `${fk.referencesTable}.${fk.referencesColumn}`).join(", ")}`
    : "";

  const body = `Column: ${table.name}.${column.name}
Type: ${column.type}
Nullable: ${column.nullable ? "yes" : "no"}
${column.defaultValue ? `Default: ${column.defaultValue}\n` : ""}${column.description ? `Description: ${column.description}\n` : ""}${fkInfo}

Part of table ${table.name} which has ${table.rowCount.toLocaleString()} rows.`;

  return {
    type: "column",
    path: `${table.schema}/${table.name}/${column.name}`,
    title: `${table.name}.${column.name}`,
    body,
    metadata: {
      schema: table.schema,
      table: table.name,
      column: column.name,
      dataType: column.type,
      nullable: column.nullable,
    },
  };
}

/**
 * Generate documents for relationships (enables "how are contacts and messages related?" queries)
 */
export function relationshipToDocument(
  fromTable: string,
  fromColumn: string,
  toTable: string,
  toColumn: string
): SchemaDocument {
  const body = `Relationship: ${fromTable}.${fromColumn} → ${toTable}.${toColumn}

This foreign key connects the ${fromTable} table to the ${toTable} table.
Use this to join ${fromTable} with ${toTable}:

  SELECT * FROM ${fromTable}
  JOIN ${toTable} ON ${fromTable}.${fromColumn} = ${toTable}.${toColumn}`;

  return {
    type: "relationship",
    path: `relationships/${fromTable}_${fromColumn}_${toTable}`,
    title: `${fromTable} → ${toTable}`,
    body,
    metadata: {
      fromTable,
      fromColumn,
      toTable,
      toColumn,
    },
  };
}

// =============================================================================
// SQL Query Generation
// =============================================================================

/**
 * System prompt for SQL generation
 */
export const SQL_GENERATION_PROMPT = `You are a PostgreSQL expert. Given a natural language question and database schema context, generate a SQL query to answer the question.

Rules:
1. Use only tables and columns from the provided schema
2. Always include appropriate JOINs when crossing table boundaries
3. Use LIMIT to prevent returning too many rows (default 100)
4. Prefer readable column aliases
5. Handle NULL values appropriately
6. Use aggregate functions (COUNT, SUM, AVG) for statistical questions

Output format:
- First line: Just the SQL query (no markdown, no explanation)
- Blank line
- Then explanation of what the query does

Example:
SELECT c.name, COUNT(m.message_id) as message_count
FROM contacts c
JOIN messages m ON c.contact_id = m.sender_contact_id
GROUP BY c.contact_id, c.name
ORDER BY message_count DESC
LIMIT 20

This query finds the top 20 contacts by message count.`;

/**
 * Generate SQL from natural language using schema context
 */
export async function generateSQL(
  question: string,
  relevantTables: TableInfo[],
  llm: LlamaCpp
): Promise<QueryPlan> {
  // Build schema context for the LLM
  const schemaContext = relevantTables.map(t => {
    const cols = t.columns.map(c => `    ${c.name} ${c.type}${c.nullable ? "" : " NOT NULL"}`).join(",\n");
    const fks = t.foreignKeys.map(fk =>
      `    FOREIGN KEY (${fk.column}) REFERENCES ${fk.referencesTable}(${fk.referencesColumn})`
    ).join(",\n");
    return `-- ${t.name}: ${t.rowCount.toLocaleString()} rows
CREATE TABLE ${t.name} (
${cols}${fks ? ",\n" + fks : ""}
);`;
  }).join("\n\n");

  const prompt = `${SQL_GENERATION_PROMPT}

Database Schema:
${schemaContext}

Question: ${question}

SQL:`;

  const result = await llm.generate(prompt, {
    maxTokens: 500,
    temperature: 0.1,  // Low temperature for deterministic SQL
  });

  // Parse the response
  const lines = result.text.trim().split("\n");
  const sqlLines: string[] = [];
  const explanationLines: string[] = [];
  let inExplanation = false;

  for (const line of lines) {
    if (line.trim() === "" && sqlLines.length > 0) {
      inExplanation = true;
      continue;
    }
    if (inExplanation) {
      explanationLines.push(line);
    } else {
      sqlLines.push(line);
    }
  }

  const sql = sqlLines.join("\n").trim();
  const explanation = explanationLines.join("\n").trim();

  // Extract table names from SQL (simple regex)
  const tableMatches = sql.match(/(?:FROM|JOIN)\s+(\w+)/gi) || [];
  const tables = [...new Set(tableMatches.map(m => m.split(/\s+/)[1]!.toLowerCase()))];

  return {
    sql,
    explanation,
    tables,
  };
}

// =============================================================================
// PgStore Class
// =============================================================================

export type PgStore = {
  config: PgConfig;

  // Schema operations
  getSchema(): Promise<TableInfo[]>;
  getSchemaDocuments(): Promise<SchemaDocument[]>;

  // Query operations
  query(sql: string): Promise<Record<string, unknown>[]>;
  nlQuery(question: string): Promise<{ plan: QueryPlan; results: Record<string, unknown>[] }>;

  // Content sync
  syncToIndex(localDb: Database, configs?: ContentSyncConfig[]): Promise<{ synced: number; tables: string[] }>;

  // Cleanup
  close(): Promise<void>;
};

/**
 * Create a PostgreSQL store for database exploration.
 *
 * Note: This requires the 'postgres' npm package. Install with:
 *   bun add postgres
 */
export async function createPgStore(config: PgConfig): Promise<PgStore> {
  // Dynamic import to avoid requiring postgres when not using pg-store
  const { default: postgres } = await import("postgres");

  const sql = postgres(config.connectionString, {
    max: 10,
    idle_timeout: 20,
  });

  // Cache for schema info
  let schemaCache: TableInfo[] | null = null;

  async function getSchema(): Promise<TableInfo[]> {
    if (schemaCache) return schemaCache;

    // Get all tables with row counts
    const tables = await sql<{ schema_name: string; table_name: string; table_type: string }[]>`
      SELECT
        schemaname as schema_name,
        tablename as table_name,
        'table' as table_type
      FROM pg_tables
      WHERE schemaname = 'public'
      UNION ALL
      SELECT
        schemaname as schema_name,
        viewname as table_name,
        'view' as table_type
      FROM pg_views
      WHERE schemaname = 'public'
      ORDER BY table_name
    `;

    const result: TableInfo[] = [];

    for (const t of tables) {
      // Get row count (approximate for large tables)
      const countResult = await sql`
        SELECT reltuples::bigint as count
        FROM pg_class
        WHERE relname = ${t.table_name}
      `;
      const rowCount = Number(countResult[0]?.count || 0);

      // Get columns
      const columns = await sql<{ column_name: string; data_type: string; is_nullable: string; column_default: string | null }[]>`
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = ${t.table_name}
        ORDER BY ordinal_position
      `;

      // Get primary key
      const pkResult = await sql<{ column_name: string }[]>`
        SELECT a.attname as column_name
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = ${t.table_name}::regclass AND i.indisprimary
      `;

      // Get foreign keys
      const fkResult = await sql<{ column_name: string; foreign_table: string; foreign_column: string }[]>`
        SELECT
          kcu.column_name,
          ccu.table_name AS foreign_table,
          ccu.column_name AS foreign_column
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = ${t.table_name}
      `;

      result.push({
        schema: t.schema_name,
        name: t.table_name,
        type: t.table_type as "table" | "view",
        rowCount,
        columns: columns.map(c => ({
          name: c.column_name,
          type: c.data_type,
          nullable: c.is_nullable === "YES",
          defaultValue: c.column_default || undefined,
        })),
        primaryKey: pkResult.map(pk => pk.column_name),
        foreignKeys: fkResult.map(fk => ({
          column: fk.column_name,
          referencesTable: fk.foreign_table,
          referencesColumn: fk.foreign_column,
        })),
      });
    }

    schemaCache = result;
    return result;
  }

  async function getSchemaDocuments(): Promise<SchemaDocument[]> {
    const schema = await getSchema();
    const docs: SchemaDocument[] = [];

    for (const table of schema) {
      // Add table document
      docs.push(tableToDocument(table));

      // Add column documents for important columns
      for (const col of table.columns) {
        // Index columns that are likely searchable or important
        if (
          col.name.includes("name") ||
          col.name.includes("title") ||
          col.name.includes("body") ||
          col.name.includes("content") ||
          col.name.includes("text") ||
          col.name.includes("email") ||
          col.name.includes("url") ||
          col.type.includes("text") ||
          table.foreignKeys.some(fk => fk.column === col.name)
        ) {
          docs.push(columnToDocument(table, col));
        }
      }

      // Add relationship documents
      for (const fk of table.foreignKeys) {
        docs.push(relationshipToDocument(table.name, fk.column, fk.referencesTable, fk.referencesColumn));
      }
    }

    return docs;
  }

  async function query(sqlQuery: string): Promise<Record<string, unknown>[]> {
    const result = await sql.unsafe(sqlQuery);
    return result as Record<string, unknown>[];
  }

  async function nlQuery(question: string): Promise<{ plan: QueryPlan; results: Record<string, unknown>[] }> {
    const schema = await getSchema();
    const llm = getDefaultLlamaCpp();

    // Find relevant tables by keyword matching (simple approach)
    // In production, use vector search over schema documents
    const questionLower = question.toLowerCase();
    const relevantTables = schema.filter(t => {
      const tableNameMatch = questionLower.includes(t.name.toLowerCase());
      const columnMatch = t.columns.some(c =>
        questionLower.includes(c.name.toLowerCase().replace(/_/g, " "))
      );
      return tableNameMatch || columnMatch;
    });

    // If no tables matched, include high-value tables as context
    const tablesToUse = relevantTables.length > 0
      ? relevantTables
      : schema.filter(t =>
          ["messages", "contacts", "rooms", "bookmarks", "entities", "sessions"].includes(t.name)
        );

    const plan = await generateSQL(question, tablesToUse.slice(0, 10), llm);

    try {
      const results = await query(plan.sql);
      return { plan, results };
    } catch (error) {
      return {
        plan: { ...plan, explanation: `Error: ${error}` },
        results: []
      };
    }
  }

  async function syncToIndex(
    localDb: Database,
    configs: ContentSyncConfig[] = DEFAULT_CONTENT_SYNC
  ): Promise<{ synced: number; tables: string[] }> {
    let totalSynced = 0;
    const syncedTables: string[] = [];

    for (const cfg of configs) {
      try {
        const joins = cfg.joins?.join("\n") || "";
        const where = cfg.where ? `WHERE ${cfg.where}` : "";
        const limit = cfg.limit ? `LIMIT ${cfg.limit}` : "";

        const selectCols = [
          `${cfg.table}.${cfg.idColumn} as id`,
          `${cfg.table}.${cfg.contentColumn} as content`,
          cfg.titleColumn ? `${cfg.table}.${cfg.titleColumn} as title` : "NULL as title",
          cfg.timestampColumn ? `${cfg.table}.${cfg.timestampColumn} as ts` : "NULL as ts",
        ].join(", ");

        const queryStr = `
          SELECT ${selectCols}
          FROM ${cfg.table}
          ${joins}
          ${where}
          ${limit}
        `;

        const rows = await sql.unsafe(queryStr);

        // Insert into local SQLite index
        const insertStmt = localDb.prepare(`
          INSERT OR REPLACE INTO content (hash, doc, indexed_at)
          VALUES (?, ?, datetime('now'))
        `);

        const docStmt = localDb.prepare(`
          INSERT OR REPLACE INTO documents (collection, path, title, hash, created_at, modified_at, active)
          VALUES (?, ?, ?, ?, datetime('now'), datetime('now'), 1)
        `);

        for (const row of rows) {
          if (!row.content) continue;

          const content = String(row.content);
          const hash = Bun.hash(content).toString(16);
          const path = `${cfg.table}/${row.id}`;
          const title = row.title ? String(row.title) : path;

          insertStmt.run(hash, content);
          docStmt.run(config.name, path, title, hash);
          totalSynced++;
        }

        syncedTables.push(cfg.table);
      } catch (error) {
        console.error(`Error syncing ${cfg.table}:`, error);
      }
    }

    return { synced: totalSynced, tables: syncedTables };
  }

  async function close(): Promise<void> {
    await sql.end();
  }

  return {
    config,
    getSchema,
    getSchemaDocuments,
    query,
    nlQuery,
    syncToIndex,
    close,
  };
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Format query results as a readable table
 */
export function formatResults(results: Record<string, unknown>[], maxWidth: number = 120): string {
  if (results.length === 0) return "(no results)";

  const keys = Object.keys(results[0]!);

  // Calculate column widths
  const widths = keys.map(k => {
    const values = results.map(r => String(r[k] ?? "").slice(0, 50));
    return Math.max(k.length, ...values.map(v => v.length));
  });

  // Truncate if total width exceeds max
  const totalWidth = widths.reduce((a, b) => a + b, 0) + (keys.length - 1) * 3;
  if (totalWidth > maxWidth) {
    const scale = maxWidth / totalWidth;
    for (let i = 0; i < widths.length; i++) {
      widths[i] = Math.max(5, Math.floor(widths[i]! * scale));
    }
  }

  // Build output
  const header = keys.map((k, i) => k.slice(0, widths[i]).padEnd(widths[i]!)).join(" | ");
  const separator = widths.map(w => "-".repeat(w)).join("-+-");
  const rows = results.map(r =>
    keys.map((k, i) => {
      const val = String(r[k] ?? "").slice(0, widths[i]);
      return val.padEnd(widths[i]!);
    }).join(" | ")
  );

  return [header, separator, ...rows].join("\n");
}
