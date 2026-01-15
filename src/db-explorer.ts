/**
 * db-explorer.ts - Database exploration CLI commands
 *
 * Provides natural language interface to explore PostgreSQL databases:
 *
 *   qmd db connect <url>     - Connect and index a PostgreSQL database
 *   qmd db schema [table]    - Show schema overview or table details
 *   qmd db search <query>    - Find relevant tables/columns
 *   qmd db ask <question>    - Natural language to SQL
 *   qmd db sync              - Sync content to local search index
 *   qmd db query <sql>       - Execute raw SQL
 */

import { Database } from "bun:sqlite";
import {
  createPgStore,
  formatResults,
  tableToDocument,
  type PgStore,
  type PgConfig,
  type TableInfo,
  type SchemaDocument,
  DEFAULT_CONTENT_SYNC,
} from "./pg-store";
import {
  createStore,
  insertContent,
  insertDocument,
  searchFTS,
  getDefaultDbPath,
  type Store,
} from "./store";
import { getDefaultLlamaCpp } from "./llm";

// =============================================================================
// Configuration
// =============================================================================

type DbConfig = {
  connections: Record<string, PgConfig>;
  activeConnection?: string;
};

const CONFIG_PATH = `${Bun.env.HOME}/.config/qmd/databases.json`;

function loadDbConfig(): DbConfig {
  try {
    const content = Bun.file(CONFIG_PATH).text();
    return JSON.parse(content as unknown as string) as DbConfig;
  } catch {
    return { connections: {} };
  }
}

function saveDbConfig(config: DbConfig): void {
  const dir = CONFIG_PATH.replace(/\/[^/]+$/, "");
  Bun.spawnSync(["mkdir", "-p", dir]);
  Bun.write(CONFIG_PATH, JSON.stringify(config, null, 2));
}

// =============================================================================
// Terminal Colors
// =============================================================================

const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
};

// =============================================================================
// Commands
// =============================================================================

/**
 * Connect to a PostgreSQL database and index its schema
 */
export async function dbConnect(connectionString: string, name?: string): Promise<void> {
  const config = loadDbConfig();

  // Generate name from connection string if not provided
  const dbName = name || connectionString.match(/\/([^/?]+)(\?|$)/)?.[1] || "default";

  console.log(`${c.dim}Connecting to database...${c.reset}`);

  try {
    const pgStore = await createPgStore({
      connectionString,
      name: dbName,
    });

    // Get schema to verify connection
    const schema = await pgStore.getSchema();
    console.log(`${c.green}✓${c.reset} Connected to ${c.cyan}${dbName}${c.reset}`);
    console.log(`${c.dim}  Found ${schema.length} tables${c.reset}`);

    // Save connection
    config.connections[dbName] = { connectionString, name: dbName };
    config.activeConnection = dbName;
    saveDbConfig(config);

    // Index schema documents into local SQLite
    console.log(`\n${c.dim}Indexing schema for semantic search...${c.reset}`);
    const docs = await pgStore.getSchemaDocuments();

    const localDb = new Database(getDefaultDbPath());
    await indexSchemaDocuments(localDb, dbName, docs);
    localDb.close();

    console.log(`${c.green}✓${c.reset} Indexed ${docs.length} schema documents`);
    console.log(`\n${c.bold}Available commands:${c.reset}`);
    console.log(`  qmd db schema              - View database schema`);
    console.log(`  qmd db search <query>      - Find relevant tables`);
    console.log(`  qmd db ask "<question>"    - Natural language query`);
    console.log(`  qmd db sync                - Sync content to search index`);

    await pgStore.close();
  } catch (error) {
    console.error(`${c.red}Error:${c.reset} ${error}`);
    process.exit(1);
  }
}

/**
 * Show database schema overview or details for a specific table
 */
export async function dbSchema(tableName?: string): Promise<void> {
  const config = loadDbConfig();
  const activeConn = config.connections[config.activeConnection || ""];

  if (!activeConn) {
    console.error(`${c.yellow}No active database connection.${c.reset}`);
    console.error(`Run: qmd db connect <postgresql://...>`);
    process.exit(1);
  }

  const pgStore = await createPgStore(activeConn);
  const schema = await pgStore.getSchema();

  if (tableName) {
    // Show specific table
    const table = schema.find(t => t.name.toLowerCase() === tableName.toLowerCase());
    if (!table) {
      console.error(`${c.yellow}Table not found: ${tableName}${c.reset}`);
      console.error(`\nAvailable tables:`);
      for (const t of schema.slice(0, 20)) {
        console.log(`  ${t.name}`);
      }
      if (schema.length > 20) {
        console.log(`  ... and ${schema.length - 20} more`);
      }
      await pgStore.close();
      process.exit(1);
    }

    console.log(`\n${c.bold}${c.cyan}${table.name}${c.reset} ${c.dim}(${table.rowCount.toLocaleString()} rows)${c.reset}\n`);

    // Columns
    console.log(`${c.bold}Columns:${c.reset}`);
    for (const col of table.columns) {
      const pkMark = table.primaryKey.includes(col.name) ? `${c.yellow}PK${c.reset} ` : "";
      const fkMark = table.foreignKeys.find(fk => fk.column === col.name);
      const fkStr = fkMark ? ` ${c.dim}→ ${fkMark.referencesTable}.${fkMark.referencesColumn}${c.reset}` : "";
      const nullStr = col.nullable ? "" : ` ${c.dim}NOT NULL${c.reset}`;
      console.log(`  ${pkMark}${c.cyan}${col.name}${c.reset} ${c.dim}${col.type}${c.reset}${nullStr}${fkStr}`);
    }

    // Foreign keys summary
    if (table.foreignKeys.length > 0) {
      console.log(`\n${c.bold}References:${c.reset}`);
      for (const fk of table.foreignKeys) {
        console.log(`  ${table.name}.${fk.column} → ${c.cyan}${fk.referencesTable}${c.reset}.${fk.referencesColumn}`);
      }
    }

    // Find tables that reference this one
    const referencedBy = schema.filter(t =>
      t.foreignKeys.some(fk => fk.referencesTable === table.name)
    );
    if (referencedBy.length > 0) {
      console.log(`\n${c.bold}Referenced by:${c.reset}`);
      for (const t of referencedBy) {
        const fks = t.foreignKeys.filter(fk => fk.referencesTable === table.name);
        for (const fk of fks) {
          console.log(`  ${c.cyan}${t.name}${c.reset}.${fk.column} → ${table.name}.${fk.referencesColumn}`);
        }
      }
    }

    // Sample query
    console.log(`\n${c.bold}Sample query:${c.reset}`);
    console.log(`${c.dim}  SELECT * FROM ${table.name} LIMIT 5;${c.reset}`);

  } else {
    // Show overview
    console.log(`\n${c.bold}Database: ${c.cyan}${activeConn.name}${c.reset}\n`);
    console.log(`${c.bold}Tables (${schema.length}):${c.reset}\n`);

    // Group by domain (simple heuristic based on prefix)
    const domains = new Map<string, TableInfo[]>();
    for (const t of schema) {
      const prefix = t.name.split("_")[0] || "other";
      if (!domains.has(prefix)) domains.set(prefix, []);
      domains.get(prefix)!.push(t);
    }

    // Sort domains by total rows
    const sortedDomains = [...domains.entries()]
      .sort((a, b) => {
        const aRows = a[1].reduce((sum, t) => sum + t.rowCount, 0);
        const bRows = b[1].reduce((sum, t) => sum + t.rowCount, 0);
        return bRows - aRows;
      });

    for (const [domain, tables] of sortedDomains) {
      const totalRows = tables.reduce((sum, t) => sum + t.rowCount, 0);
      console.log(`${c.bold}${domain}${c.reset} ${c.dim}(${totalRows.toLocaleString()} rows)${c.reset}`);
      for (const t of tables.sort((a, b) => b.rowCount - a.rowCount)) {
        const rowStr = t.rowCount.toLocaleString().padStart(10);
        console.log(`  ${c.cyan}${t.name.padEnd(35)}${c.reset} ${c.dim}${rowStr}${c.reset}`);
      }
      console.log();
    }
  }

  await pgStore.close();
}

/**
 * Search schema for relevant tables/columns using semantic search
 */
export async function dbSearch(query: string): Promise<void> {
  const config = loadDbConfig();
  const activeConn = config.connections[config.activeConnection || ""];

  if (!activeConn) {
    console.error(`${c.yellow}No active database connection.${c.reset}`);
    console.error(`Run: qmd db connect <postgresql://...>`);
    process.exit(1);
  }

  console.log(`${c.dim}Searching schema for: "${query}"${c.reset}\n`);

  // Search the indexed schema documents
  const localDb = new Database(getDefaultDbPath());
  const results = searchFTS(localDb, query, 20, activeConn.name);
  localDb.close();

  if (results.length === 0) {
    console.log(`${c.yellow}No matching tables or columns found.${c.reset}`);
    console.log(`Try broader terms or run 'qmd db schema' to browse.`);
    return;
  }

  console.log(`${c.bold}Found ${results.length} relevant items:${c.reset}\n`);

  for (const r of results) {
    const path = r.displayPath.replace(`${activeConn.name}/`, "");
    const score = Math.round(r.score * 100);

    // Color based on type (inferred from path depth)
    const parts = path.split("/");
    if (parts.length === 2) {
      // Table
      console.log(`${c.cyan}${c.bold}TABLE${c.reset} ${parts[1]} ${c.dim}(${score}%)${c.reset}`);
    } else if (parts.length === 3) {
      // Column
      console.log(`${c.blue}COLUMN${c.reset} ${parts[1]}.${parts[2]} ${c.dim}(${score}%)${c.reset}`);
    } else if (path.startsWith("relationships/")) {
      // Relationship
      console.log(`${c.magenta}RELATIONSHIP${c.reset} ${path.replace("relationships/", "")} ${c.dim}(${score}%)${c.reset}`);
    }

    // Show snippet
    if (r.body) {
      const lines = r.body.split("\n").slice(0, 3);
      for (const line of lines) {
        console.log(`  ${c.dim}${line.slice(0, 80)}${c.reset}`);
      }
    }
    console.log();
  }
}

/**
 * Natural language query - translate question to SQL and execute
 */
export async function dbAsk(question: string, execute: boolean = true): Promise<void> {
  const config = loadDbConfig();
  const activeConn = config.connections[config.activeConnection || ""];

  if (!activeConn) {
    console.error(`${c.yellow}No active database connection.${c.reset}`);
    console.error(`Run: qmd db connect <postgresql://...>`);
    process.exit(1);
  }

  console.log(`${c.dim}Analyzing question...${c.reset}\n`);

  const pgStore = await createPgStore(activeConn);

  try {
    const { plan, results } = await pgStore.nlQuery(question);

    // Show the generated SQL
    console.log(`${c.bold}Generated SQL:${c.reset}`);
    console.log(`${c.cyan}${plan.sql}${c.reset}\n`);

    if (plan.explanation) {
      console.log(`${c.dim}${plan.explanation}${c.reset}\n`);
    }

    if (execute) {
      if (results.length > 0) {
        console.log(`${c.bold}Results (${results.length} rows):${c.reset}\n`);
        console.log(formatResults(results));
      } else {
        console.log(`${c.yellow}No results found.${c.reset}`);
      }
    } else {
      console.log(`${c.dim}Use 'qmd db query' to execute this SQL.${c.reset}`);
    }
  } catch (error) {
    console.error(`${c.red}Error:${c.reset} ${error}`);
  }

  await pgStore.close();
}

/**
 * Execute raw SQL query
 */
export async function dbQuery(sql: string): Promise<void> {
  const config = loadDbConfig();
  const activeConn = config.connections[config.activeConnection || ""];

  if (!activeConn) {
    console.error(`${c.yellow}No active database connection.${c.reset}`);
    console.error(`Run: qmd db connect <postgresql://...>`);
    process.exit(1);
  }

  const pgStore = await createPgStore(activeConn);

  try {
    const results = await pgStore.query(sql);
    if (results.length > 0) {
      console.log(formatResults(results));
    } else {
      console.log(`${c.dim}(no results)${c.reset}`);
    }
  } catch (error) {
    console.error(`${c.red}Error:${c.reset} ${error}`);
  }

  await pgStore.close();
}

/**
 * Sync content from PostgreSQL to local search index
 */
export async function dbSync(): Promise<void> {
  const config = loadDbConfig();
  const activeConn = config.connections[config.activeConnection || ""];

  if (!activeConn) {
    console.error(`${c.yellow}No active database connection.${c.reset}`);
    console.error(`Run: qmd db connect <postgresql://...>`);
    process.exit(1);
  }

  console.log(`${c.dim}Syncing content from ${activeConn.name}...${c.reset}\n`);

  const pgStore = await createPgStore(activeConn);
  const localDb = new Database(getDefaultDbPath());

  try {
    const { synced, tables } = await pgStore.syncToIndex(localDb, DEFAULT_CONTENT_SYNC);

    console.log(`${c.green}✓${c.reset} Synced ${synced.toLocaleString()} documents from ${tables.length} tables:`);
    for (const t of tables) {
      console.log(`  ${c.cyan}${t}${c.reset}`);
    }

    console.log(`\n${c.dim}Run 'qmd embed' to generate embeddings for vector search.${c.reset}`);
  } catch (error) {
    console.error(`${c.red}Error:${c.reset} ${error}`);
  }

  localDb.close();
  await pgStore.close();
}

/**
 * List connected databases
 */
export function dbList(): void {
  const config = loadDbConfig();
  const connections = Object.values(config.connections);

  if (connections.length === 0) {
    console.log(`${c.yellow}No database connections configured.${c.reset}`);
    console.log(`Run: qmd db connect <postgresql://...>`);
    return;
  }

  console.log(`${c.bold}Configured databases:${c.reset}\n`);
  for (const conn of connections) {
    const active = conn.name === config.activeConnection ? ` ${c.green}(active)${c.reset}` : "";
    // Mask password in connection string
    const masked = conn.connectionString.replace(/:([^@]+)@/, ":****@");
    console.log(`  ${c.cyan}${conn.name}${c.reset}${active}`);
    console.log(`    ${c.dim}${masked}${c.reset}`);
  }
}

/**
 * Switch active database connection
 */
export function dbUse(name: string): void {
  const config = loadDbConfig();

  if (!config.connections[name]) {
    console.error(`${c.yellow}Database not found: ${name}${c.reset}`);
    console.error(`Run 'qmd db list' to see available databases.`);
    process.exit(1);
  }

  config.activeConnection = name;
  saveDbConfig(config);
  console.log(`${c.green}✓${c.reset} Switched to database: ${c.cyan}${name}${c.reset}`);
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Index schema documents into local SQLite for semantic search
 */
async function indexSchemaDocuments(
  db: Database,
  collectionName: string,
  docs: SchemaDocument[]
): Promise<void> {
  const now = new Date().toISOString();

  for (const doc of docs) {
    const hash = Bun.hash(doc.body).toString(16);
    const path = `schema/${doc.path}`;

    // Insert content
    db.prepare(`
      INSERT OR REPLACE INTO content (hash, doc, indexed_at)
      VALUES (?, ?, ?)
    `).run(hash, doc.body, now);

    // Insert document
    db.prepare(`
      INSERT OR REPLACE INTO documents (collection, path, title, hash, created_at, modified_at, active)
      VALUES (?, ?, ?, ?, ?, ?, 1)
    `).run(collectionName, path, doc.title, hash, now, now);
  }
}

// =============================================================================
// CLI Handler
// =============================================================================

export async function handleDbCommand(args: string[]): Promise<void> {
  const subcommand = args[0];

  switch (subcommand) {
    case "connect":
      if (!args[1]) {
        console.error("Usage: qmd db connect <postgresql://...> [--name <name>]");
        process.exit(1);
      }
      const nameIdx = args.indexOf("--name");
      const name = nameIdx > 0 ? args[nameIdx + 1] : undefined;
      await dbConnect(args[1], name);
      break;

    case "schema":
      await dbSchema(args[1]);
      break;

    case "search":
      if (!args[1]) {
        console.error("Usage: qmd db search <query>");
        process.exit(1);
      }
      await dbSearch(args.slice(1).join(" "));
      break;

    case "ask":
      if (!args[1]) {
        console.error("Usage: qmd db ask <question>");
        process.exit(1);
      }
      const noExec = args.includes("--no-exec");
      const question = args.slice(1).filter(a => a !== "--no-exec").join(" ");
      await dbAsk(question, !noExec);
      break;

    case "query":
      if (!args[1]) {
        console.error("Usage: qmd db query <sql>");
        process.exit(1);
      }
      await dbQuery(args.slice(1).join(" "));
      break;

    case "sync":
      await dbSync();
      break;

    case "list":
      dbList();
      break;

    case "use":
      if (!args[1]) {
        console.error("Usage: qmd db use <name>");
        process.exit(1);
      }
      dbUse(args[1]);
      break;

    default:
      console.log(`${c.bold}Database Explorer Commands:${c.reset}

  ${c.cyan}qmd db connect <url>${c.reset}    Connect to a PostgreSQL database
  ${c.cyan}qmd db list${c.reset}             List configured databases
  ${c.cyan}qmd db use <name>${c.reset}       Switch active database
  ${c.cyan}qmd db schema [table]${c.reset}   Show schema overview or table details
  ${c.cyan}qmd db search <query>${c.reset}   Find relevant tables/columns
  ${c.cyan}qmd db ask <question>${c.reset}   Natural language to SQL
  ${c.cyan}qmd db query <sql>${c.reset}      Execute raw SQL
  ${c.cyan}qmd db sync${c.reset}             Sync content to search index

${c.bold}Examples:${c.reset}
  qmd db connect postgresql://user:pass@localhost/mydb
  qmd db schema messages
  qmd db search "user email"
  qmd db ask "how many messages did each contact send?"
  qmd db ask "find conversations about machine learning"
`);
  }
}
