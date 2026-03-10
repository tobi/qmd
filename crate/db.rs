//! db.rs - SQLite database initialization and schema management
//!
//! Provides database connection setup, schema migrations, and FTS5 configuration.

use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::{Path, PathBuf};

/// Default database filename
const DEFAULT_INDEX_NAME: &str = "index";

/// Get the default database path: ~/.cache/qmd/index.sqlite
pub fn get_default_db_path(index_name: Option<&str>) -> PathBuf {
    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
        })
        .join("qmd");

    let name = index_name.unwrap_or(DEFAULT_INDEX_NAME);
    cache_dir.join(format!("{name}.sqlite"))
}

/// Open a SQLite database connection with WAL mode and performance settings.
pub fn open_database(path: &Path) -> Result<Connection> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create database directory: {}", parent.display()))?;
    }

    let conn = Connection::open(path)
        .with_context(|| format!("Failed to open database: {}", path.display()))?;

    // Performance settings
    // journal_mode returns a result row, so use query_row
    let _: String = conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
    conn.execute_batch(
        "PRAGMA synchronous = NORMAL;
         PRAGMA cache_size = -64000;
         PRAGMA foreign_keys = ON;",
    )?;

    Ok(conn)
}

/// Run all schema migrations to bring the database up to date.
pub fn run_migrations(conn: &Connection) -> Result<()> {
    // Core tables
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS content (
            hash TEXT PRIMARY KEY,
            doc TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collection TEXT NOT NULL,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            modified_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE CASCADE,
            UNIQUE(collection, path)
        );

        CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection, active);
        CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
        CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path, active);

        CREATE TABLE IF NOT EXISTS llm_cache (
            hash TEXT PRIMARY KEY,
            result TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS content_vectors (
            hash TEXT NOT NULL,
            seq INTEGER NOT NULL DEFAULT 0,
            pos INTEGER NOT NULL DEFAULT 0,
            model TEXT NOT NULL,
            embedded_at TEXT NOT NULL,
            PRIMARY KEY (hash, seq)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            filepath,
            title,
            body,
            tokenize='porter unicode61'
        );
        ",
    )
    .context("Failed to create tables")?;

    // Triggers with SELECT subqueries can't be created via execute_batch
    // in rusqlite (it interprets the SELECT as returning results).
    // Use the raw sqlite3_exec function instead.
    create_trigger_raw(conn,
        "CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
         WHEN NEW.active = 1
         BEGIN
             INSERT INTO documents_fts(rowid, filepath, title, body)
             SELECT NEW.id,
                    NEW.collection || '/' || NEW.path,
                    NEW.title,
                    COALESCE((SELECT doc FROM content WHERE hash = NEW.hash), '');
         END;"
    )?;

    create_trigger_raw(conn,
        "CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents
         BEGIN
             DELETE FROM documents_fts WHERE rowid = OLD.id;
         END;"
    )?;

    create_trigger_raw(conn,
        "CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
         BEGIN
             DELETE FROM documents_fts WHERE rowid = OLD.id;
             INSERT INTO documents_fts(rowid, filepath, title, body)
             SELECT NEW.id,
                    NEW.collection || '/' || NEW.path,
                    NEW.title,
                    CASE WHEN NEW.active = 1
                         THEN COALESCE((SELECT doc FROM content WHERE hash = NEW.hash), '')
                         ELSE ''
                    END
             WHERE NEW.active = 1;
         END;"
    )?;

    Ok(())
}

/// Create a trigger using raw sqlite3_exec to bypass rusqlite's result checking.
fn create_trigger_raw(conn: &Connection, sql: &str) -> Result<()> {
    use std::ffi::CString;
    let c_sql = CString::new(sql).context("Invalid SQL string")?;
    let rc = unsafe {
        rusqlite::ffi::sqlite3_exec(
            conn.handle(),
            c_sql.as_ptr(),
            None,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if rc != rusqlite::ffi::SQLITE_OK {
        anyhow::bail!("Failed to create trigger (sqlite error code {rc})");
    }
    Ok(())
}

/// Ensure the vectors_vec virtual table exists with the given dimensions.
pub fn ensure_vec_table(conn: &Connection, dimensions: usize) -> Result<()> {
    let sql = format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vectors_vec USING vec0(
            hash_seq TEXT PRIMARY KEY,
            embedding float[{dimensions}] distance_metric=cosine
        )"
    );
    conn.execute_batch(&sql)
        .context("Failed to create vectors_vec table")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_db_path() {
        let path = get_default_db_path(None);
        assert!(path.to_string_lossy().contains("qmd"));
        assert!(path.to_string_lossy().ends_with("index.sqlite"));
    }

    #[test]
    fn test_open_and_migrate() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.sqlite");
        let conn = open_database(&db_path).unwrap();
        run_migrations(&conn).unwrap();

        // Verify tables exist
        let count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='documents'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
    }
}
