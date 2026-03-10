//! Evaluation tests for search quality.
//!
//! These tests verify that the search pipeline returns relevant results
//! for known queries against a fixed corpus, simulating real-world usage.

use qmd::db;
use qmd::search::*;
use qmd::store::*;

fn setup_corpus() -> rusqlite::Connection {
    let conn = rusqlite::Connection::open_in_memory().unwrap();
    conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();
    db::run_migrations(&conn).unwrap();

    let docs = vec![
        ("notes", "meeting-2025-01-15.md",
         "# Product Planning Meeting\n\nAttendees: Alice, Bob, Carol\n\n## Action Items\n\n- Alice to prepare Q1 roadmap\n- Bob to review authentication RFC\n- Carol to draft user testing plan\n\n## Discussion\n\nWe discussed the upcoming launch timeline and agreed on March 15 as the target date.\nThe team raised concerns about the authentication migration from OAuth1 to OAuth2."),
        ("notes", "meeting-2025-01-22.md",
         "# Engineering Standup\n\nAttendees: Dave, Eve, Frank\n\n## Updates\n\n- Dave: Finished database migration script\n- Eve: Working on new search algorithm using BM25\n- Frank: Fixed memory leak in connection pooling\n\n## Blockers\n\nEve needs access to the production search index for benchmarking."),
        ("docs", "architecture.md",
         "# System Architecture\n\n## Overview\n\nThe system uses a microservices architecture with the following components:\n\n1. **API Gateway** - handles routing and authentication\n2. **Search Service** - full-text search using Elasticsearch\n3. **User Service** - user management and profiles\n4. **Notification Service** - email and push notifications\n\n## Database\n\nWe use PostgreSQL for transactional data and Redis for caching.\nSearch indices are maintained in Elasticsearch with nightly reindexing."),
        ("docs", "api-reference.md",
         "# API Reference\n\n## Authentication\n\nAll API requests require a Bearer token in the Authorization header.\n\n```\nAuthorization: Bearer <token>\n```\n\n## Endpoints\n\n### GET /users/:id\n\nReturns user profile data.\n\n### POST /search\n\nPerforms full-text search across all indexed documents.\n\nParameters:\n- `query` (string, required) - search query\n- `limit` (integer, optional) - max results (default: 10)\n- `offset` (integer, optional) - pagination offset"),
        ("notes", "rust-notes.md",
         "# Rust Learning Notes\n\n## Ownership\n\nRust uses ownership with borrowing to ensure memory safety without garbage collection.\nEvery value has exactly one owner. When the owner goes out of scope, the value is dropped.\n\n## Lifetimes\n\nLifetimes are a way to tell the compiler how long references are valid.\nThe borrow checker uses lifetimes to ensure references don't outlive the data they refer to.\n\n## Error Handling\n\nRust uses Result<T, E> for recoverable errors and panic! for unrecoverable ones."),
        ("docs", "deployment.md",
         "# Deployment Guide\n\n## Prerequisites\n\n- Docker 20.x or later\n- Kubernetes 1.25+\n- Helm 3.x\n\n## Steps\n\n1. Build the container image: `docker build -t app:latest .`\n2. Push to registry: `docker push registry.example.com/app:latest`\n3. Deploy with Helm: `helm upgrade --install app ./charts/app`\n\n## Monitoring\n\nUse Grafana dashboards for monitoring. Alerts are configured in PagerDuty."),
        ("notes", "ideas.md",
         "# Product Ideas\n\n## AI-Powered Search\n\nUse embeddings and semantic search to improve result relevance.\nCombine BM25 keyword matching with vector similarity for hybrid search.\n\n## Collaborative Editing\n\nReal-time collaborative markdown editing with CRDT-based conflict resolution.\n\n## Mobile App\n\nNative iOS and Android apps with offline support and sync."),
    ];

    for (collection, path, content) in docs {
        let hash = hash_content(content);
        let title = extract_title(content, path);
        insert_content(&conn, &hash, content).unwrap();
        insert_document(&conn, collection, path, &title, &hash).unwrap();
    }

    conn
}

// =============================================================================
// Relevance evals: verify top result is correct for known queries
// =============================================================================

#[test]
fn eval_query_authentication() {
    let conn = setup_corpus();
    let results = search_fts(&conn, "authentication", 5, None).unwrap();
    assert!(!results.is_empty(), "Should find results for 'authentication'");
    // api-reference.md or architecture.md should be top results (both discuss auth)
    let top_paths: Vec<&str> = results.iter().map(|r| r.filepath.as_str()).collect();
    assert!(
        top_paths.iter().any(|p| p.contains("api-reference") || p.contains("architecture")),
        "Top results should include auth-related docs, got: {:?}", top_paths
    );
}

#[test]
fn eval_query_meeting_action_items() {
    let conn = setup_corpus();
    let results = search_fts(&conn, "action items roadmap", 5, None).unwrap();
    assert!(!results.is_empty(), "Should find results for 'action items roadmap'");
    assert!(
        results[0].filepath.contains("meeting-2025-01-15"),
        "Top result should be the planning meeting, got: {}", results[0].filepath
    );
}

#[test]
fn eval_query_rust_ownership() {
    let conn = setup_corpus();
    let results = search_fts(&conn, "ownership borrowing memory safety", 5, None).unwrap();
    assert!(!results.is_empty(), "Should find results for Rust concepts");
    assert!(
        results[0].filepath.contains("rust-notes"),
        "Top result should be rust-notes.md, got: {}", results[0].filepath
    );
}

#[test]
fn eval_query_deployment_docker() {
    let conn = setup_corpus();
    let results = search_fts(&conn, "docker kubernetes deployment", 5, None).unwrap();
    assert!(!results.is_empty());
    assert!(
        results[0].filepath.contains("deployment"),
        "Top result should be deployment.md, got: {}", results[0].filepath
    );
}

#[test]
fn eval_query_search_algorithm() {
    let conn = setup_corpus();
    let results = search_fts(&conn, "BM25 search algorithm", 5, None).unwrap();
    assert!(!results.is_empty());
    // Should find either the standup (Eve working on BM25) or ideas (BM25 keyword matching)
    let top_paths: Vec<&str> = results.iter().map(|r| r.filepath.as_str()).collect();
    assert!(
        top_paths.iter().any(|p| p.contains("meeting-2025-01-22") || p.contains("ideas")),
        "Should find BM25-related docs, got: {:?}", top_paths
    );
}

#[test]
fn eval_query_database_migration() {
    let conn = setup_corpus();
    let results = search_fts(&conn, "database migration", 5, None).unwrap();
    assert!(!results.is_empty());
    // Standup mentions "database migration script"
    let top_paths: Vec<&str> = results.iter().map(|r| r.filepath.as_str()).collect();
    assert!(
        top_paths.iter().any(|p| p.contains("meeting-2025-01-22") || p.contains("architecture")),
        "Should find database-related docs, got: {:?}", top_paths
    );
}

// =============================================================================
// Collection filtering evals
// =============================================================================

#[test]
fn eval_collection_filter_notes_only() {
    let conn = setup_corpus();
    let filter = vec!["notes".to_string()];
    let results = search_fts(&conn, "search", 10, Some(&filter)).unwrap();
    for r in &results {
        assert!(
            r.filepath.starts_with("notes/"),
            "All results should be from 'notes' collection, got: {}", r.filepath
        );
    }
}

#[test]
fn eval_collection_filter_docs_only() {
    let conn = setup_corpus();
    let filter = vec!["docs".to_string()];
    let results = search_fts(&conn, "search", 10, Some(&filter)).unwrap();
    for r in &results {
        assert!(
            r.filepath.starts_with("docs/"),
            "All results should be from 'docs' collection, got: {}", r.filepath
        );
    }
}

// =============================================================================
// Hybrid query evals
// =============================================================================

#[test]
fn eval_hybrid_query_limit() {
    let conn = setup_corpus();
    let opts = HybridQueryOptions {
        limit: 2,
        min_score: 0.0,
        collection_filter: None,
        candidate_limit: 40,
        intent: None,
    };
    let results = hybrid_query(&conn, "search", &opts).unwrap();
    assert!(results.len() <= 2, "Should respect limit, got {} results", results.len());
}

#[test]
fn eval_hybrid_query_ordering() {
    let conn = setup_corpus();
    let opts = HybridQueryOptions::default();
    let results = hybrid_query(&conn, "search API endpoint", &opts).unwrap();
    if results.len() >= 2 {
        assert!(
            results[0].score >= results[1].score,
            "Results should be ordered by score: {} >= {}", results[0].score, results[1].score
        );
    }
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn eval_empty_query() {
    let conn = setup_corpus();
    let results = search_fts(&conn, "", 10, None).unwrap();
    assert!(results.is_empty(), "Empty query should return no results");
}

#[test]
fn eval_special_characters() {
    let conn = setup_corpus();
    // Should not crash with special characters
    let results = search_fts(&conn, "C++ & Rust <script>", 10, None).unwrap();
    // May or may not find results, but should not error
    let _ = results;
}

#[test]
fn eval_very_long_query() {
    let conn = setup_corpus();
    let long_query = "word ".repeat(100);
    let results = search_fts(&conn, &long_query, 10, None).unwrap();
    let _ = results; // Should not crash
}
