//! store.rs - Core data access layer
//!
//! Provides document indexing, content hashing, CRUD operations,
//! and the Store abstraction that ties everything together.

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::collections::{self, NamedCollection};
use crate::db;

// =============================================================================
// Types
// =============================================================================

/// A document result from the database
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentResult {
    pub filepath: String,
    pub display_path: String,
    pub title: String,
    pub hash: String,
    pub docid: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    pub body_length: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    pub modified_at: String,
    pub collection: String,
    pub path: String,
}

/// A search result (extends DocumentResult with score)
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub filepath: String,
    pub display_path: String,
    pub title: String,
    pub hash: String,
    pub docid: String,
    pub score: f64,
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_pos: Option<i64>,
}

/// Result for multi-get operations
#[derive(Debug, Clone, serde::Serialize)]
pub struct MultiGetResult {
    pub filepath: String,
    pub display_path: String,
    pub title: String,
    pub body: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    pub skipped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

/// Index status summary
#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexStatus {
    pub total_documents: usize,
    pub needs_embedding: usize,
    pub has_vector_index: bool,
    pub collections: Vec<CollectionStatus>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct CollectionStatus {
    pub name: String,
    pub path: String,
    pub pattern: String,
    pub documents: usize,
    pub last_updated: String,
}

/// Document not found result with similar suggestions
#[derive(Debug)]
pub struct DocumentNotFound {
    pub query: String,
    pub similar_files: Vec<String>,
}

// =============================================================================
// Store
// =============================================================================

/// Main store abstraction wrapping the database connection.
pub struct Store {
    pub conn: Connection,
    pub db_path: PathBuf,
}

impl Store {
    /// Create a new store, opening the database and running migrations.
    pub fn new(db_path: Option<&Path>) -> Result<Self> {
        let path = db_path
            .map(PathBuf::from)
            .unwrap_or_else(|| db::get_default_db_path(None));

        let conn = db::open_database(&path)?;
        db::run_migrations(&conn)?;

        Ok(Store {
            conn,
            db_path: path,
        })
    }

    /// Create a store with a specific index name.
    pub fn with_index_name(index_name: &str) -> Result<Self> {
        let path = db::get_default_db_path(Some(index_name));
        Self::new(Some(&path))
    }

    /// Close the database connection.
    pub fn close(self) -> Result<()> {
        self.conn.close().map_err(|(_, e)| e)?;
        Ok(())
    }
}

// =============================================================================
// Content hashing
// =============================================================================

/// Compute SHA-256 hash of content, returned as hex string.
pub fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Extract docid (first 6 chars) from a hash.
pub fn get_docid(hash: &str) -> String {
    hash[..6.min(hash.len())].to_string()
}

/// Normalize a docid input: strip leading #, quotes, whitespace.
pub fn normalize_docid(input: &str) -> String {
    input
        .trim()
        .trim_matches(|c| c == '#' || c == '"' || c == '\'')
        .to_string()
}

/// Check if a string looks like a docid (6 hex chars, optionally with #).
pub fn is_docid(input: &str) -> bool {
    let normalized = normalize_docid(input);
    normalized.len() == 6 && normalized.chars().all(|c| c.is_ascii_hexdigit())
}

// =============================================================================
// Content operations
// =============================================================================

/// Insert content into the content table (deduplicates by hash).
pub fn insert_content(conn: &Connection, hash: &str, content: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?1, ?2, ?3)",
        params![hash, content, now],
    )?;
    Ok(())
}

/// Insert a document record.
pub fn insert_document(
    conn: &Connection,
    collection: &str,
    path: &str,
    title: &str,
    hash: &str,
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR REPLACE INTO documents (collection, path, title, hash, created_at, modified_at, active)
         VALUES (?1, ?2, ?3, ?4, ?5, ?5, 1)",
        params![collection, path, title, hash, now],
    )?;
    Ok(())
}

/// Find an active document by collection and path.
pub fn find_active_document(
    conn: &Connection,
    collection: &str,
    path: &str,
) -> Result<Option<(i64, String, String)>> {
    let result = conn
        .query_row(
            "SELECT id, hash, title FROM documents WHERE collection = ?1 AND path = ?2 AND active = 1",
            params![collection, path],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .optional()?;
    Ok(result)
}

/// Update an existing document's hash and title.
pub fn update_document(conn: &Connection, doc_id: i64, title: &str, hash: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "UPDATE documents SET title = ?1, hash = ?2, modified_at = ?3 WHERE id = ?4",
        params![title, hash, now, doc_id],
    )?;
    Ok(())
}

/// Deactivate (soft delete) a document.
pub fn deactivate_document(conn: &Connection, collection: &str, path: &str) -> Result<()> {
    conn.execute(
        "UPDATE documents SET active = 0 WHERE collection = ?1 AND path = ?2 AND active = 1",
        params![collection, path],
    )?;
    Ok(())
}

/// Get all active document paths for a collection.
pub fn get_active_document_paths(conn: &Connection, collection: &str) -> Result<Vec<String>> {
    let mut stmt =
        conn.prepare("SELECT path FROM documents WHERE collection = ?1 AND active = 1")?;
    let paths = stmt
        .query_map(params![collection], |row| row.get(0))?
        .collect::<Result<Vec<String>, _>>()?;
    Ok(paths)
}

// =============================================================================
// Document retrieval
// =============================================================================

/// Find a document by docid (first 6 chars of hash).
pub fn find_document_by_docid(
    conn: &Connection,
    docid: &str,
) -> Result<Option<(String, String)>> {
    let docid = normalize_docid(docid);
    let result = conn
        .query_row(
            "SELECT d.collection || '/' || d.path, d.hash
             FROM documents d
             WHERE d.hash LIKE ?1 || '%' AND d.active = 1
             LIMIT 1",
            params![docid],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    Ok(result)
}

/// Get a document's body text.
pub fn get_document_body(conn: &Connection, hash: &str) -> Result<Option<String>> {
    let result = conn
        .query_row(
            "SELECT doc FROM content WHERE hash = ?1",
            params![hash],
            |row| row.get(0),
        )
        .optional()?;
    Ok(result)
}

/// Find a document by filepath (collection/path format).
pub fn find_document(
    conn: &Connection,
    filepath: &str,
    include_body: bool,
) -> Result<Option<DocumentResult>> {
    // Try docid first
    if is_docid(filepath) {
        if let Some((fp, _hash)) = find_document_by_docid(conn, filepath)? {
            return find_document(conn, &fp, include_body);
        }
        return Ok(None);
    }

    // Try virtual path: qmd://collection/path
    let (collection, path) = if let Some(rest) = filepath.strip_prefix("qmd://") {
        if let Some(slash_pos) = rest.find('/') {
            (
                rest[..slash_pos].to_string(),
                rest[slash_pos + 1..].to_string(),
            )
        } else {
            (rest.to_string(), String::new())
        }
    } else if let Some(slash_pos) = filepath.find('/') {
        (
            filepath[..slash_pos].to_string(),
            filepath[slash_pos + 1..].to_string(),
        )
    } else {
        // Bare filename: try suffix match
        return find_document_by_suffix(conn, filepath, include_body);
    };

    let row = conn
        .query_row(
            "SELECT d.id, d.collection, d.path, d.title, d.hash, d.modified_at,
                    COALESCE(LENGTH(c.doc), 0)
             FROM documents d
             LEFT JOIN content c ON c.hash = d.hash
             WHERE d.collection = ?1 AND d.path = ?2 AND d.active = 1",
            params![collection, path],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, usize>(6)?,
                ))
            },
        )
        .optional()?;

    match row {
        Some((_, col, p, title, hash, modified_at, body_length)) => {
            let body = if include_body {
                get_document_body(conn, &hash)?
            } else {
                None
            };
            let docid = get_docid(&hash);
            let display_path = format!("qmd://{col}/{p}");
            let context =
                get_context_for_path(conn, &col, &p)?;

            Ok(Some(DocumentResult {
                filepath: format!("{col}/{p}"),
                display_path,
                title,
                hash,
                docid,
                body,
                body_length,
                context,
                modified_at,
                collection: col,
                path: p,
            }))
        }
        None => {
            // Try suffix match as fallback
            find_document_by_suffix(conn, filepath, include_body)
        }
    }
}

/// Try to find a document by suffix matching on the path.
fn find_document_by_suffix(
    conn: &Connection,
    suffix: &str,
    include_body: bool,
) -> Result<Option<DocumentResult>> {
    let pattern = format!("%{suffix}");
    let row = conn
        .query_row(
            "SELECT d.collection, d.path, d.title, d.hash, d.modified_at,
                    COALESCE(LENGTH(c.doc), 0)
             FROM documents d
             LEFT JOIN content c ON c.hash = d.hash
             WHERE (d.collection || '/' || d.path) LIKE ?1 AND d.active = 1
             LIMIT 1",
            params![pattern],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, usize>(5)?,
                ))
            },
        )
        .optional()?;

    match row {
        Some((col, path, title, hash, modified_at, body_length)) => {
            let body = if include_body {
                get_document_body(conn, &hash)?
            } else {
                None
            };
            let docid = get_docid(&hash);
            let display_path = format!("qmd://{col}/{path}");
            let context = get_context_for_path(conn, &col, &path)?;

            Ok(Some(DocumentResult {
                filepath: format!("{col}/{path}"),
                display_path,
                title,
                hash,
                docid,
                body,
                body_length,
                context,
                modified_at,
                collection: col,
                path,
            }))
        }
        None => Ok(None),
    }
}

/// Get context for a file path by looking up collection contexts.
fn get_context_for_path(
    _conn: &Connection,
    collection: &str,
    path: &str,
) -> Result<Option<String>> {
    // Load collection config and find matching context
    if let Some(named) = collections::get_collection(collection)? {
        if let Some(ctx_map) = &named.collection.context {
            // Find the most specific matching context (longest prefix match)
            let mut best_match: Option<(&str, &str)> = None;
            for (prefix, ctx) in ctx_map {
                let normalized_prefix = prefix.trim_start_matches('/');
                if path.starts_with(normalized_prefix) || prefix == "/" {
                    match best_match {
                        None => best_match = Some((prefix, ctx)),
                        Some((existing_prefix, _)) => {
                            if prefix.len() > existing_prefix.len() {
                                best_match = Some((prefix, ctx));
                            }
                        }
                    }
                }
            }
            return Ok(best_match.map(|(_, ctx)| ctx.to_string()));
        }
    }
    Ok(None)
}

// =============================================================================
// Title extraction
// =============================================================================

/// Extract a title from markdown content. Falls back to filename.
pub fn extract_title(content: &str, filename: &str) -> String {
    // Look for first markdown heading
    for line in content.lines().take(20) {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("# ") {
            return heading.trim().to_string();
        }
    }
    // Fall back to filename without extension
    Path::new(filename)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| filename.to_string())
}

// =============================================================================
// Indexing
// =============================================================================

/// Index all files in a collection.
pub fn index_collection(store: &Store, named: &NamedCollection) -> Result<IndexResult> {
    let files = collections::discover_files(&named.collection)?;
    let mut added = 0usize;
    let mut updated = 0usize;
    let mut unchanged = 0usize;

    let on_disk: HashSet<String> = files
        .iter()
        .map(|f| collections::relative_path(&named.collection, f))
        .collect();

    for file in &files {
        let rel_path = collections::relative_path(&named.collection, file);
        let content = std::fs::read_to_string(file)
            .with_context(|| format!("Failed to read: {}", file.display()))?;
        let hash = hash_content(&content);
        let title = extract_title(&content, &rel_path);

        insert_content(&store.conn, &hash, &content)?;

        match find_active_document(&store.conn, &named.name, &rel_path)? {
            None => {
                insert_document(&store.conn, &named.name, &rel_path, &title, &hash)?;
                added += 1;
            }
            Some((doc_id, existing_hash, existing_title)) => {
                if existing_hash != hash {
                    update_document(&store.conn, doc_id, &title, &hash)?;
                    updated += 1;
                } else if existing_title != title {
                    let now = chrono::Utc::now().to_rfc3339();
                    store.conn.execute(
                        "UPDATE documents SET title = ?1, modified_at = ?2 WHERE id = ?3",
                        params![title, now, doc_id],
                    )?;
                    unchanged += 1;
                } else {
                    unchanged += 1;
                }
            }
        }
    }

    // Deactivate files no longer on disk
    let active_paths = get_active_document_paths(&store.conn, &named.name)?;
    let mut removed = 0usize;
    for path in &active_paths {
        if !on_disk.contains(path) {
            deactivate_document(&store.conn, &named.name, path)?;
            removed += 1;
        }
    }

    Ok(IndexResult {
        added,
        updated,
        unchanged,
        removed,
    })
}

/// Result of an indexing operation.
#[derive(Debug)]
pub struct IndexResult {
    pub added: usize,
    pub updated: usize,
    pub unchanged: usize,
    pub removed: usize,
}

// =============================================================================
// Status
// =============================================================================

/// Get overall index status.
pub fn get_status(store: &Store) -> Result<IndexStatus> {
    let total: usize = store.conn.query_row(
        "SELECT COUNT(*) FROM documents WHERE active = 1",
        [],
        |r| r.get(0),
    )?;

    let needs_embedding: usize = store.conn.query_row(
        "SELECT COUNT(DISTINCT d.hash) FROM documents d
         LEFT JOIN content_vectors cv ON cv.hash = d.hash
         WHERE d.active = 1 AND cv.hash IS NULL",
        [],
        |r| r.get(0),
    )?;

    // Check if vectors_vec table exists
    let has_vector_index: bool = store
        .conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='vectors_vec'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .map(|c| c > 0)
        .unwrap_or(false);

    let collections = get_collection_statuses(store)?;

    Ok(IndexStatus {
        total_documents: total,
        needs_embedding,
        has_vector_index,
        collections,
    })
}

fn get_collection_statuses(store: &Store) -> Result<Vec<CollectionStatus>> {
    let named_collections = collections::list_collections()?;
    let mut statuses = Vec::new();

    for nc in named_collections {
        let count: usize = store.conn.query_row(
            "SELECT COUNT(*) FROM documents WHERE collection = ?1 AND active = 1",
            params![nc.name],
            |r| r.get(0),
        )?;

        let last_updated: String = store
            .conn
            .query_row(
                "SELECT COALESCE(MAX(modified_at), '') FROM documents WHERE collection = ?1 AND active = 1",
                params![nc.name],
                |r| r.get(0),
            )?;

        statuses.push(CollectionStatus {
            name: nc.name,
            path: nc.collection.path,
            pattern: nc.collection.pattern,
            documents: count,
            last_updated,
        });
    }

    Ok(statuses)
}

// =============================================================================
// Embedding management
// =============================================================================

/// Get hashes that need embedding (have content but no vectors).
pub fn get_hashes_for_embedding(conn: &Connection) -> Result<Vec<(String, String, String)>> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT d.hash, c.doc, d.collection || '/' || d.path
         FROM documents d
         JOIN content c ON c.hash = d.hash
         LEFT JOIN content_vectors cv ON cv.hash = d.hash
         WHERE d.active = 1 AND cv.hash IS NULL",
    )?;
    let results = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(results)
}

/// Insert an embedding vector for a content chunk.
pub fn insert_embedding(
    conn: &Connection,
    hash: &str,
    seq: usize,
    pos: usize,
    embedding: &[f32],
    model: &str,
) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    let hash_seq = format!("{hash}_{seq}");

    // Insert chunk metadata
    conn.execute(
        "INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![hash, seq, pos, model, now],
    )?;

    // Insert vector into vec table
    let embedding_blob = embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect::<Vec<u8>>();

    conn.execute(
        "INSERT OR REPLACE INTO vectors_vec (hash_seq, embedding) VALUES (?1, ?2)",
        params![hash_seq, embedding_blob],
    )?;

    Ok(())
}

/// Clear all embeddings.
pub fn clear_all_embeddings(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "DELETE FROM content_vectors;
         DELETE FROM vectors_vec;",
    )?;
    Ok(())
}

// =============================================================================
// Cache operations
// =============================================================================

/// Get a cached LLM result.
pub fn get_cached_result(conn: &Connection, cache_key: &str) -> Result<Option<String>> {
    let result = conn
        .query_row(
            "SELECT result FROM llm_cache WHERE hash = ?1",
            params![cache_key],
            |row| row.get(0),
        )
        .optional()?;
    Ok(result)
}

/// Set a cached LLM result.
pub fn set_cached_result(conn: &Connection, cache_key: &str, result: &str) -> Result<()> {
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?1, ?2, ?3)",
        params![cache_key, result, now],
    )?;
    Ok(())
}

/// Delete all LLM cache entries.
pub fn delete_llm_cache(conn: &Connection) -> Result<usize> {
    let changes = conn.execute("DELETE FROM llm_cache", [])?;
    Ok(changes)
}

// =============================================================================
// Cleanup
// =============================================================================

/// Delete inactive documents permanently.
pub fn delete_inactive_documents(conn: &Connection) -> Result<usize> {
    let changes = conn.execute("DELETE FROM documents WHERE active = 0", [])?;
    Ok(changes)
}

/// Delete orphaned content (no documents reference it).
pub fn cleanup_orphaned_content(conn: &Connection) -> Result<usize> {
    let changes = conn.execute(
        "DELETE FROM content WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)",
        [],
    )?;
    Ok(changes)
}

/// Vacuum the database to reclaim space.
pub fn vacuum_database(conn: &Connection) -> Result<()> {
    conn.execute_batch("VACUUM")?;
    Ok(())
}

// =============================================================================
// Virtual path utilities
// =============================================================================

/// Parse a virtual path like qmd://collection/path into (collection, path).
pub fn parse_virtual_path(vpath: &str) -> Option<(String, String)> {
    let rest = vpath.strip_prefix("qmd://")?;
    if let Some(slash_pos) = rest.find('/') {
        Some((
            rest[..slash_pos].to_string(),
            rest[slash_pos + 1..].to_string(),
        ))
    } else {
        Some((rest.to_string(), String::new()))
    }
}

/// Check if a string is a virtual path.
pub fn is_virtual_path(s: &str) -> bool {
    s.starts_with("qmd://")
}

/// Build a virtual path from collection and path.
pub fn build_virtual_path(collection: &str, path: &str) -> String {
    if path.is_empty() {
        format!("qmd://{collection}/")
    } else {
        format!("qmd://{collection}/{path}")
    }
}

// =============================================================================
// Multi-get
// =============================================================================

/// Default max bytes for multi-get (10KB)
pub const DEFAULT_MULTI_GET_MAX_BYTES: usize = 10_240;

/// Find multiple documents by pattern (glob or comma-separated list).
pub fn find_documents(
    conn: &Connection,
    pattern: &str,
    max_bytes: Option<usize>,
    max_lines: Option<usize>,
) -> Result<(Vec<MultiGetResult>, Vec<String>)> {
    let max_bytes = max_bytes.unwrap_or(DEFAULT_MULTI_GET_MAX_BYTES);
    let mut results = Vec::new();
    let mut errors = Vec::new();

    // Check if it's a comma-separated list
    let items: Vec<&str> = if pattern.contains(',') {
        pattern.split(',').map(|s| s.trim()).collect()
    } else {
        vec![pattern]
    };

    for item in &items {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }

        // Try to find each document
        match find_document(conn, item, true)? {
            Some(doc) => {
                let body = doc.body.unwrap_or_default();
                let body_bytes = body.len();

                if body_bytes > max_bytes {
                    results.push(MultiGetResult {
                        filepath: doc.filepath,
                        display_path: doc.display_path,
                        title: doc.title,
                        body: String::new(),
                        context: doc.context,
                        skipped: true,
                        skip_reason: Some(format!(
                            "File too large ({} bytes, max {})",
                            body_bytes, max_bytes
                        )),
                    });
                } else {
                    let body = if let Some(max) = max_lines {
                        body.lines().take(max).collect::<Vec<_>>().join("\n")
                    } else {
                        body
                    };
                    results.push(MultiGetResult {
                        filepath: doc.filepath,
                        display_path: doc.display_path,
                        title: doc.title,
                        body,
                        context: doc.context,
                        skipped: false,
                        skip_reason: None,
                    });
                }
            }
            None => {
                // If pattern contains glob chars, try glob matching
                if items.len() == 1
                    && (pattern.contains('*') || pattern.contains('?') || pattern.contains('['))
                {
                    return find_documents_by_glob(conn, pattern, max_bytes, max_lines);
                }
                errors.push(format!("Not found: {item}"));
            }
        }
    }

    Ok((results, errors))
}

/// Find documents matching a glob pattern.
fn find_documents_by_glob(
    conn: &Connection,
    pattern: &str,
    max_bytes: usize,
    max_lines: Option<usize>,
) -> Result<(Vec<MultiGetResult>, Vec<String>)> {
    let mut results = Vec::new();
    let errors = Vec::new();

    // Get all active document paths
    let mut stmt = conn.prepare(
        "SELECT d.collection, d.path, d.title, d.hash, c.doc
         FROM documents d
         JOIN content c ON c.hash = d.hash
         WHERE d.active = 1
         ORDER BY d.collection, d.path",
    )?;

    let glob_pattern =
        glob::Pattern::new(pattern).unwrap_or_else(|_| glob::Pattern::new("*").unwrap());

    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;

    for row in rows {
        let (collection, path, title, _hash, body) = row?;
        let filepath = format!("{collection}/{path}");

        if !glob_pattern.matches(&filepath) {
            continue;
        }

        let display_path = format!("qmd://{filepath}");
        let body_bytes = body.len();

        if body_bytes > max_bytes {
            results.push(MultiGetResult {
                filepath,
                display_path,
                title,
                body: String::new(),
                context: None,
                skipped: true,
                skip_reason: Some(format!(
                    "File too large ({body_bytes} bytes, max {max_bytes})"
                )),
            });
        } else {
            let body = if let Some(max) = max_lines {
                body.lines().take(max).collect::<Vec<_>>().join("\n")
            } else {
                body
            };
            results.push(MultiGetResult {
                filepath,
                display_path,
                title,
                body,
                context: None,
                skipped: false,
                skip_reason: None,
            });
        }
    }

    Ok((results, errors))
}
