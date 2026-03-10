//! Integration tests for the search module — FTS5 search, snippets, RRF.

use qmd::db;
use qmd::search::*;
use qmd::store::*;

fn setup_db() -> rusqlite::Connection {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
    db::run_migrations(&conn).unwrap();
    conn
}

fn insert_doc(conn: &rusqlite::Connection, collection: &str, path: &str, content: &str) {
    let hash = hash_content(content);
    let title = extract_title(content, path);
    insert_content(conn, &hash, content).unwrap();
    insert_document(conn, collection, path, &title, &hash).unwrap();
}

// =============================================================================
// FTS5 query builder
// =============================================================================

#[test]
fn test_fts5_single_word() {
    assert_eq!(build_fts5_query("hello"), "\"hello\"*");
}

#[test]
fn test_fts5_multiple_words() {
    assert_eq!(
        build_fts5_query("hello world"),
        "\"hello\"* AND \"world\"*"
    );
}

#[test]
fn test_fts5_quoted_phrase() {
    assert_eq!(build_fts5_query("\"exact match\""), "\"exact match\"");
}

#[test]
fn test_fts5_negation() {
    assert_eq!(
        build_fts5_query("hello -world"),
        "\"hello\"* AND NOT \"world\"*"
    );
}

#[test]
fn test_fts5_mixed() {
    assert_eq!(
        build_fts5_query("rust \"memory safe\" -garbage"),
        "\"rust\"* AND \"memory safe\" AND NOT \"garbage\"*"
    );
}

#[test]
fn test_fts5_empty() {
    assert_eq!(build_fts5_query(""), "");
}

// =============================================================================
// FTS5 search integration
// =============================================================================

#[test]
fn test_search_fts_basic() {
    let conn = setup_db();
    insert_doc(&conn, "docs", "rust.md", "# Rust\n\nRust is a systems programming language focused on safety and performance");
    insert_doc(&conn, "docs", "python.md", "# Python\n\nPython is a dynamic scripting language");

    let results = search_fts(&conn, "rust", 10, None).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].display_path.contains("rust.md"));
}

#[test]
fn test_search_fts_no_results() {
    let conn = setup_db();
    insert_doc(&conn, "docs", "test.md", "# Test\n\nSome content");

    let results = search_fts(&conn, "nonexistent_xyzzy_term", 10, None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_search_fts_collection_filter() {
    let conn = setup_db();
    insert_doc(&conn, "notes", "note.md", "# Note\n\nImportant meeting notes about design");
    insert_doc(&conn, "docs", "doc.md", "# Doc\n\nTechnical design document");

    let filter = vec!["notes".to_string()];
    let results = search_fts(&conn, "design", 10, Some(&filter)).unwrap();
    assert_eq!(results.len(), 1);
    assert!(results[0].display_path.contains("notes"));
}

#[test]
fn test_search_fts_scores_normalized() {
    let conn = setup_db();
    insert_doc(&conn, "docs", "test.md", "# Test\n\nSearchable content for testing");

    let results = search_fts(&conn, "test", 10, None).unwrap();
    for r in &results {
        assert!(r.score >= 0.0 && r.score < 1.0, "Score {} not in [0, 1)", r.score);
    }
}

// =============================================================================
// Snippet extraction
// =============================================================================

#[test]
fn test_extract_snippet_short_body() {
    let body = "Short body text";
    let (snippet, line) = extract_snippet(body, "short", 100, None, None, None);
    assert_eq!(snippet, body);
    assert_eq!(line, 1);
}

#[test]
fn test_extract_snippet_empty() {
    let (snippet, line) = extract_snippet("", "query", 100, None, None, None);
    assert_eq!(snippet, "");
    assert_eq!(line, 0);
}

#[test]
fn test_extract_snippet_finds_term() {
    let body = "aaaa ".repeat(100) + "TARGET_WORD " + &"bbbb ".repeat(100);
    let (snippet, _line) = extract_snippet(&body, "TARGET_WORD", 200, None, None, None);
    assert!(snippet.contains("TARGET_WORD"));
}

#[test]
fn test_extract_snippet_with_chunk_pos() {
    let body = "line1\nline2\nline3\nline4\nline5";
    let (_snippet, line) = extract_snippet(body, "query", 10, Some(12), None, None);
    assert_eq!(line, 3); // "line1\nline2\n" has 2 newlines
}

// =============================================================================
// Reciprocal Rank Fusion
// =============================================================================

#[test]
fn test_rrf_single_list() {
    let list = vec![
        make_result("a", 0.9),
        make_result("b", 0.7),
    ];
    let ranked = reciprocal_rank_fusion(&[list], &[1.0], 60.0);
    assert_eq!(ranked.len(), 2);
    assert!(ranked[0].rrf_score > ranked[1].rrf_score);
}

#[test]
fn test_rrf_merges_lists() {
    let list1 = vec![make_result("a", 0.9), make_result("b", 0.5)];
    let list2 = vec![make_result("b", 0.8), make_result("c", 0.6)];

    let ranked = reciprocal_rank_fusion(&[list1, list2], &[1.0, 1.0], 60.0);
    // "b" appears in both lists so should have a higher RRF score than "c"
    let b_score = ranked.iter().find(|r| r.filepath == "b").unwrap().rrf_score;
    let c_score = ranked.iter().find(|r| r.filepath == "c").unwrap().rrf_score;
    assert!(b_score > c_score);
}

#[test]
fn test_rrf_weighted() {
    let list1 = vec![make_result("a", 0.9)];
    let list2 = vec![make_result("a", 0.9)];

    let unweighted = reciprocal_rank_fusion(&[list1.clone(), list2.clone()], &[1.0, 1.0], 60.0);
    let weighted = reciprocal_rank_fusion(&[list1, list2], &[2.0, 1.0], 60.0);

    // With weight 2.0+1.0 vs 1.0+1.0, weighted should have higher score
    assert!(weighted[0].rrf_score > unweighted[0].rrf_score);
}

#[test]
fn test_rrf_empty() {
    let ranked = reciprocal_rank_fusion(&[], &[], 60.0);
    assert!(ranked.is_empty());
}

// =============================================================================
// Hybrid query
// =============================================================================

#[test]
fn test_hybrid_query_basic() {
    let conn = setup_db();
    insert_doc(&conn, "docs", "api.md", "# API Reference\n\nEndpoints for user authentication and authorization");
    insert_doc(&conn, "docs", "readme.md", "# README\n\nProject overview and getting started guide");

    let opts = HybridQueryOptions {
        limit: 5,
        min_score: 0.0,
        collection_filter: None,
        candidate_limit: 40,
        intent: None,
    };
    let results = hybrid_query(&conn, "authentication", &opts).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].display_path.contains("api.md"));
}

#[test]
fn test_hybrid_query_min_score_filter() {
    let conn = setup_db();
    insert_doc(&conn, "docs", "test.md", "# Test\n\nSome weakly matching content about testing");

    let opts = HybridQueryOptions {
        limit: 5,
        min_score: 0.99, // Very high threshold
        collection_filter: None,
        candidate_limit: 40,
        intent: None,
    };
    let results = hybrid_query(&conn, "testing", &opts).unwrap();
    assert!(results.is_empty());
}

// =============================================================================
// Helpers
// =============================================================================

fn make_result(filepath: &str, score: f64) -> SearchResult {
    SearchResult {
        filepath: filepath.to_string(),
        display_path: format!("qmd://c/{filepath}"),
        title: filepath.to_string(),
        hash: hash_content(filepath),
        docid: get_docid(&hash_content(filepath)),
        score,
        source: "test".to_string(),
        body: None,
        context: None,
        chunk_pos: None,
    }
}
