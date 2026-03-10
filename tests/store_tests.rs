//! Integration tests for the store module — document CRUD, hashing, indexing, and retrieval.

use qmd::db;
use qmd::store::*;

fn setup_db() -> rusqlite::Connection {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    // WAL doesn't work with in-memory, just set up pragmas
    conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
    db::run_migrations(&conn).unwrap();
    conn
}

// =============================================================================
// Hashing
// =============================================================================

#[test]
fn test_hash_content_deterministic() {
    let h1 = hash_content("hello world");
    let h2 = hash_content("hello world");
    assert_eq!(h1, h2);
}

#[test]
fn test_hash_content_different_inputs() {
    let h1 = hash_content("hello");
    let h2 = hash_content("world");
    assert_ne!(h1, h2);
}

#[test]
fn test_hash_content_sha256_length() {
    let h = hash_content("test");
    assert_eq!(h.len(), 64); // SHA-256 hex = 64 chars
}

// =============================================================================
// Docid
// =============================================================================

#[test]
fn test_get_docid() {
    let hash = hash_content("some content");
    let docid = get_docid(&hash);
    assert_eq!(docid.len(), 6);
    assert_eq!(docid, &hash[..6]);
}

#[test]
fn test_normalize_docid() {
    assert_eq!(normalize_docid("#abc123"), "abc123");
    assert_eq!(normalize_docid("  abc123  "), "abc123");
    assert_eq!(normalize_docid("\"abc123\""), "abc123");
    assert_eq!(normalize_docid("'abc123'"), "abc123");
}

#[test]
fn test_is_docid() {
    assert!(is_docid("#abc123"));
    assert!(is_docid("abc123"));
    assert!(is_docid("ABCDEF"));
    assert!(!is_docid("xyz___")); // underscores aren't hex
    assert!(!is_docid("abc12"));  // too short
    assert!(!is_docid("abc1234")); // too long (after normalization)
}

// =============================================================================
// Title extraction
// =============================================================================

#[test]
fn test_extract_title_from_heading() {
    assert_eq!(extract_title("# My Title\n\nBody text", "file.md"), "My Title");
}

#[test]
fn test_extract_title_from_heading_with_whitespace() {
    assert_eq!(extract_title("\n\n# Title Here\n", "file.md"), "Title Here");
}

#[test]
fn test_extract_title_no_heading() {
    assert_eq!(extract_title("No heading here\nJust paragraphs", "notes.md"), "notes");
}

#[test]
fn test_extract_title_h2_not_used() {
    // Only h1 is used for title extraction
    assert_eq!(extract_title("## Section\n\nBody", "readme.md"), "readme");
}

#[test]
fn test_extract_title_filename_without_ext() {
    assert_eq!(extract_title("no heading", "my-notes.md"), "my-notes");
}

// =============================================================================
// Content operations
// =============================================================================

#[test]
fn test_insert_content_and_retrieve() {
    let conn = setup_db();
    let content = "# Hello\n\nWorld";
    let hash = hash_content(content);

    insert_content(&conn, &hash, content).unwrap();

    let body = get_document_body(&conn, &hash).unwrap();
    assert_eq!(body, Some(content.to_string()));
}

#[test]
fn test_insert_content_deduplication() {
    let conn = setup_db();
    let content = "duplicate content";
    let hash = hash_content(content);

    // Insert twice — should not error
    insert_content(&conn, &hash, content).unwrap();
    insert_content(&conn, &hash, content).unwrap();

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM content WHERE hash = ?1", [&hash], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);
}

// =============================================================================
// Document CRUD
// =============================================================================

#[test]
fn test_insert_and_find_document() {
    let conn = setup_db();
    let content = "# Test Doc\n\nSome content here";
    let hash = hash_content(content);

    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "test.md", "Test Doc", &hash).unwrap();

    let doc = find_active_document(&conn, "notes", "test.md").unwrap();
    assert!(doc.is_some());
    let (id, found_hash, title) = doc.unwrap();
    assert!(id > 0);
    assert_eq!(found_hash, hash);
    assert_eq!(title, "Test Doc");
}

#[test]
fn test_update_document() {
    let conn = setup_db();
    let content1 = "# Version 1\n\nFirst version";
    let hash1 = hash_content(content1);
    insert_content(&conn, &hash1, content1).unwrap();
    insert_document(&conn, "notes", "doc.md", "Version 1", &hash1).unwrap();

    let (doc_id, _, _) = find_active_document(&conn, "notes", "doc.md").unwrap().unwrap();

    let content2 = "# Version 2\n\nUpdated version";
    let hash2 = hash_content(content2);
    insert_content(&conn, &hash2, content2).unwrap();
    update_document(&conn, doc_id, "Version 2", &hash2).unwrap();

    let (_, found_hash, title) = find_active_document(&conn, "notes", "doc.md").unwrap().unwrap();
    assert_eq!(found_hash, hash2);
    assert_eq!(title, "Version 2");
}

#[test]
fn test_deactivate_document() {
    let conn = setup_db();
    let content = "# To Delete\n\nContent";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "delete-me.md", "To Delete", &hash).unwrap();

    deactivate_document(&conn, "notes", "delete-me.md").unwrap();

    let doc = find_active_document(&conn, "notes", "delete-me.md").unwrap();
    assert!(doc.is_none());
}

#[test]
fn test_get_active_document_paths() {
    let conn = setup_db();
    for name in &["a.md", "b.md", "c.md"] {
        let content = format!("# {name}\n\nContent");
        let hash = hash_content(&content);
        insert_content(&conn, &hash, &content).unwrap();
        insert_document(&conn, "notes", name, name, &hash).unwrap();
    }

    let paths = get_active_document_paths(&conn, "notes").unwrap();
    assert_eq!(paths.len(), 3);
    assert!(paths.contains(&"a.md".to_string()));
    assert!(paths.contains(&"b.md".to_string()));
    assert!(paths.contains(&"c.md".to_string()));
}

// =============================================================================
// Document retrieval
// =============================================================================

#[test]
fn test_find_document_by_collection_path() {
    let conn = setup_db();
    let content = "# Found\n\nContent";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "found.md", "Found", &hash).unwrap();

    let doc = find_document(&conn, "notes/found.md", true).unwrap();
    assert!(doc.is_some());
    let doc = doc.unwrap();
    assert_eq!(doc.title, "Found");
    assert_eq!(doc.body, Some(content.to_string()));
    assert_eq!(doc.display_path, "qmd://notes/found.md");
}

#[test]
fn test_find_document_by_virtual_path() {
    let conn = setup_db();
    let content = "# Virtual\n\nContent";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "docs", "guide.md", "Virtual", &hash).unwrap();

    let doc = find_document(&conn, "qmd://docs/guide.md", false).unwrap();
    assert!(doc.is_some());
    assert_eq!(doc.unwrap().title, "Virtual");
}

#[test]
fn test_find_document_by_docid() {
    let conn = setup_db();
    let content = "# Docid Test\n\nSome unique content for docid";
    let hash = hash_content(content);
    let docid = get_docid(&hash);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "docid-test.md", "Docid Test", &hash).unwrap();

    let doc = find_document(&conn, &format!("#{docid}"), true).unwrap();
    assert!(doc.is_some());
    assert_eq!(doc.unwrap().title, "Docid Test");
}

#[test]
fn test_find_document_by_suffix() {
    let conn = setup_db();
    let content = "# Suffix\n\nContent for suffix match";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "subfolder/target.md", "Suffix", &hash).unwrap();

    let doc = find_document(&conn, "target.md", false).unwrap();
    assert!(doc.is_some());
    assert_eq!(doc.unwrap().title, "Suffix");
}

#[test]
fn test_find_document_not_found() {
    let conn = setup_db();
    let doc = find_document(&conn, "notes/nonexistent.md", false).unwrap();
    assert!(doc.is_none());
}

// =============================================================================
// FTS5 integration
// =============================================================================

#[test]
fn test_fts5_triggers_insert() {
    let conn = setup_db();
    let content = "# FTS Test\n\nSearchable content about quantum computing";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "fts.md", "FTS Test", &hash).unwrap();

    // Verify FTS index has the content
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH 'quantum'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);
}

#[test]
fn test_fts5_triggers_deactivate() {
    let conn = setup_db();
    let content = "# Delete FTS\n\nSearchable delete target uniqueword789";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "fts-del.md", "Delete FTS", &hash).unwrap();

    // Verify it's searchable
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH 'uniqueword789'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 1);

    // Deactivate triggers FTS deletion via the update trigger
    deactivate_document(&conn, "notes", "fts-del.md").unwrap();

    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM documents_fts WHERE documents_fts MATCH 'uniqueword789'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 0);
}

// =============================================================================
// Content-addressable storage
// =============================================================================

#[test]
fn test_content_addressable_dedup() {
    let conn = setup_db();
    let content = "Shared content between two files";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();

    // Two documents pointing to the same content
    insert_document(&conn, "notes", "file1.md", "File 1", &hash).unwrap();
    insert_document(&conn, "notes", "file2.md", "File 2", &hash).unwrap();

    // Only one content row
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM content", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);

    // But two document rows
    let doc_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM documents WHERE active = 1", [], |r| r.get(0))
        .unwrap();
    assert_eq!(doc_count, 2);
}

// =============================================================================
// Embedding operations
// =============================================================================

#[test]
fn test_get_hashes_for_embedding() {
    let conn = setup_db();
    let content = "# Embed Me\n\nContent for embedding";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "embed.md", "Embed Me", &hash).unwrap();

    let hashes = get_hashes_for_embedding(&conn).unwrap();
    assert_eq!(hashes.len(), 1);
    assert_eq!(hashes[0].0, hash);
}

#[test]
fn test_get_hashes_for_embedding_returns_unembedded() {
    let conn = setup_db();
    let content = "# Embedded\n\nContent with vectors";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "embedded.md", "Embedded", &hash).unwrap();

    // Document without embeddings should be returned
    let hashes = get_hashes_for_embedding(&conn).unwrap();
    assert_eq!(hashes.len(), 1);
    assert_eq!(hashes[0].0, hash);

    // Manually insert content_vectors record to simulate embedding
    let now = chrono::Utc::now().to_rfc3339();
    conn.execute(
        "INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?1, 0, 0, 'test', ?2)",
        rusqlite::params![hash, now],
    ).unwrap();

    // Now should return empty
    assert_eq!(get_hashes_for_embedding(&conn).unwrap().len(), 0);
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn test_empty_content() {
    let conn = setup_db();
    let content = "";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "empty.md", "empty", &hash).unwrap();

    let body = get_document_body(&conn, &hash).unwrap();
    assert_eq!(body, Some(String::new()));
}

#[test]
fn test_unicode_content() {
    let conn = setup_db();
    let content = "# 日本語テスト\n\nこんにちは世界 🌍";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "notes", "unicode.md", "日本語テスト", &hash).unwrap();

    let doc = find_document(&conn, "notes/unicode.md", true).unwrap().unwrap();
    assert_eq!(doc.title, "日本語テスト");
    assert_eq!(doc.body, Some(content.to_string()));
}

#[test]
fn test_large_content() {
    let conn = setup_db();
    let content = "word ".repeat(50_000); // ~250KB
    let hash = hash_content(&content);
    insert_content(&conn, &hash, &content).unwrap();
    insert_document(&conn, "notes", "large.md", "Large", &hash).unwrap();

    let body = get_document_body(&conn, &hash).unwrap();
    assert_eq!(body.unwrap().len(), content.len());
}

#[test]
fn test_special_characters_in_path() {
    let conn = setup_db();
    let content = "# Spaces and stuff\n\nBody";
    let hash = hash_content(content);
    insert_content(&conn, &hash, content).unwrap();
    insert_document(&conn, "my-notes", "sub folder/file (1).md", "Spaces and stuff", &hash).unwrap();

    let doc = find_document(&conn, "my-notes/sub folder/file (1).md", false).unwrap();
    assert!(doc.is_some());
}
