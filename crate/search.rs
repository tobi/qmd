//! search.rs - Full-text search (BM25 via FTS5) and hybrid search pipeline
//!
//! Provides BM25 keyword search, vector similarity search,
//! Reciprocal Rank Fusion (RRF), and the hybrid query pipeline.

use anyhow::Result;
use rusqlite::{params, Connection};
use std::collections::HashMap;

use crate::store::{SearchResult, get_docid};

// =============================================================================
// FTS5 BM25 Search
// =============================================================================

/// Build an FTS5 query from user input.
/// Supports prefix matching, quoted phrases, and negation.
pub fn build_fts5_query(query: &str) -> String {
    let mut parts = Vec::new();
    let mut chars = query.chars().peekable();
    let mut current_word = String::new();
    let mut in_quotes = false;
    let mut negate = false;

    while let Some(&ch) = chars.peek() {
        match ch {
            '"' => {
                chars.next();
                if in_quotes {
                    // End of quoted phrase
                    if !current_word.is_empty() {
                        let phrase = format!("\"{}\"", current_word);
                        if negate {
                            parts.push(format!("NOT {phrase}"));
                            negate = false;
                        } else {
                            parts.push(phrase);
                        }
                        current_word.clear();
                    }
                    in_quotes = false;
                } else {
                    in_quotes = true;
                }
            }
            '-' if !in_quotes && current_word.is_empty() => {
                chars.next();
                negate = true;
            }
            ' ' | '\t' if !in_quotes => {
                chars.next();
                if !current_word.is_empty() {
                    let term = format!("\"{}\"*", current_word);
                    if negate {
                        parts.push(format!("NOT {term}"));
                        negate = false;
                    } else {
                        parts.push(term);
                    }
                    current_word.clear();
                }
            }
            _ => {
                chars.next();
                current_word.push(ch);
            }
        }
    }

    // Handle remaining word
    if !current_word.is_empty() {
        if in_quotes {
            let phrase = format!("\"{}\"", current_word);
            if negate {
                parts.push(format!("NOT {phrase}"));
            } else {
                parts.push(phrase);
            }
        } else {
            let term = format!("\"{}\"*", current_word);
            if negate {
                parts.push(format!("NOT {term}"));
            } else {
                parts.push(term);
            }
        }
    }

    parts.join(" AND ")
}

/// Search using FTS5 with BM25 scoring.
pub fn search_fts(
    conn: &Connection,
    query: &str,
    limit: usize,
    collection_filter: Option<&[String]>,
) -> Result<Vec<SearchResult>> {
    let fts_query = build_fts5_query(query);
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }

    // BM25 with field weights: filepath=10, title=1, body=1
    let sql = "SELECT
            d.collection, d.path, d.title, d.hash,
            bm25(documents_fts, 10.0, 1.0, 1.0) as score,
            d.modified_at
        FROM documents_fts fts
        JOIN documents d ON d.id = fts.rowid
        LEFT JOIN content c ON c.hash = d.hash
        WHERE documents_fts MATCH ?1
          AND d.active = 1
        ORDER BY score
        LIMIT ?2";

    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![fts_query, limit], |row| {
        let collection: String = row.get(0)?;
        let path: String = row.get(1)?;
        let title: String = row.get(2)?;
        let hash: String = row.get(3)?;
        let raw_score: f64 = row.get(4)?;
        Ok((collection, path, title, hash, raw_score))
    })?;

    let mut results = Vec::new();
    for row in rows {
        let (collection, path, title, hash, raw_score) = row?;

        // Apply collection filter
        if let Some(filter) = collection_filter {
            if !filter.iter().any(|f| f == &collection) {
                continue;
            }
        }

        // Convert BM25 score to [0, 1) range: |x| / (1 + |x|)
        let abs_score = raw_score.abs();
        let normalized_score = abs_score / (1.0 + abs_score);

        let docid = get_docid(&hash);
        let display_path = format!("qmd://{collection}/{path}");
        let filepath = format!("{collection}/{path}");

        // Get body for snippet extraction
        let body: Option<String> = conn
            .query_row(
                "SELECT doc FROM content WHERE hash = ?1",
                params![hash],
                |r| r.get(0),
            )
            .ok();

        results.push(SearchResult {
            filepath,
            display_path,
            title,
            hash,
            docid,
            score: normalized_score,
            source: "fts".to_string(),
            body,
            context: None,
            chunk_pos: None,
        });
    }

    Ok(results)
}

// =============================================================================
// Snippet extraction
// =============================================================================

/// Extract a relevant snippet from document body around matching query terms.
pub fn extract_snippet(
    body: &str,
    query: &str,
    max_len: usize,
    chunk_pos: Option<i64>,
    _from_line: Option<usize>,
    _intent: Option<&str>,
) -> (String, usize) {
    if body.is_empty() {
        return (String::new(), 0);
    }

    if body.len() <= max_len {
        return (body.to_string(), 1);
    }

    // If we have a chunk position, start from there
    if let Some(pos) = chunk_pos {
        let start = (pos as usize).min(body.len());
        let end = (start + max_len).min(body.len());
        let snippet = &body[start..end];
        let line = body[..start].matches('\n').count() + 1;
        return (snippet.to_string(), line);
    }

    // Find the first occurrence of any query term
    let terms: Vec<&str> = query.split_whitespace().collect();
    let body_lower = body.to_lowercase();

    let mut best_pos = 0usize;
    for term in &terms {
        let term_lower = term.to_lowercase();
        if let Some(pos) = body_lower.find(&term_lower) {
            best_pos = pos;
            break;
        }
    }

    // Center the snippet around the match
    let half = max_len / 2;
    let start = best_pos.saturating_sub(half);
    let end = (start + max_len).min(body.len());
    let start = if end == body.len() {
        end.saturating_sub(max_len)
    } else {
        start
    };

    let snippet = &body[start..end];
    let line = body[..start].matches('\n').count() + 1;
    (snippet.to_string(), line)
}

// =============================================================================
// Reciprocal Rank Fusion (RRF)
// =============================================================================

/// A ranked result for RRF
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub filepath: String,
    pub display_path: String,
    pub title: String,
    pub hash: String,
    pub docid: String,
    pub score: f64,
    pub body: Option<String>,
    pub context: Option<String>,
    pub chunk_pos: Option<i64>,
    pub rrf_score: f64,
}

/// Combine multiple ranked result lists using Reciprocal Rank Fusion.
/// RRF score = Σ weight[i] / (k + rank[i] + 1)
pub fn reciprocal_rank_fusion(
    result_lists: &[Vec<SearchResult>],
    weights: &[f64],
    k: f64,
) -> Vec<RankedResult> {
    let k = if k == 0.0 { 60.0 } else { k };

    // Map filepath -> accumulated RRF score and best result
    let mut scores: HashMap<String, (f64, SearchResult)> = HashMap::new();

    for (i, list) in result_lists.iter().enumerate() {
        let weight = weights.get(i).copied().unwrap_or(1.0);
        for (rank, result) in list.iter().enumerate() {
            let rrf_contribution = weight / (k + rank as f64 + 1.0);
            let entry = scores
                .entry(result.filepath.clone())
                .or_insert((0.0, result.clone()));
            entry.0 += rrf_contribution;
            // Keep the result with the higher original score
            if result.score > entry.1.score {
                entry.1 = result.clone();
            }
        }
    }

    let mut ranked: Vec<RankedResult> = scores
        .into_iter()
        .map(|(_, (rrf_score, r))| RankedResult {
            filepath: r.filepath,
            display_path: r.display_path,
            title: r.title,
            hash: r.hash,
            docid: r.docid,
            score: r.score,
            body: r.body,
            context: r.context,
            chunk_pos: r.chunk_pos,
            rrf_score,
        })
        .collect();

    // Sort by RRF score descending
    ranked.sort_by(|a, b| b.rrf_score.partial_cmp(&a.rrf_score).unwrap());
    ranked
}

// =============================================================================
// Hybrid query (placeholder for full pipeline with LLM)
// =============================================================================

/// Options for the hybrid search query.
pub struct HybridQueryOptions {
    pub limit: usize,
    pub min_score: f64,
    pub collection_filter: Option<Vec<String>>,
    pub candidate_limit: usize,
    pub intent: Option<String>,
}

impl Default for HybridQueryOptions {
    fn default() -> Self {
        Self {
            limit: 5,
            min_score: 0.0,
            collection_filter: None,
            candidate_limit: 40,
            intent: None,
        }
    }
}

/// Run the full hybrid search pipeline.
/// For now, this only runs BM25 search. Vector search and LLM reranking
/// will be added when the LLM module is implemented.
pub fn hybrid_query(
    conn: &Connection,
    query: &str,
    options: &HybridQueryOptions,
) -> Result<Vec<SearchResult>> {
    let collection_filter = options.collection_filter.as_deref();

    // Phase 1: BM25 search
    let fts_results = search_fts(conn, query, options.candidate_limit, collection_filter)?;

    // TODO: Phase 2 - Vector search (requires LLM embeddings)
    // TODO: Phase 3 - RRF fusion of BM25 + vector results
    // TODO: Phase 4 - LLM reranking of top candidates
    // TODO: Phase 5 - Score blending

    // For now, just return BM25 results with min_score filter
    let results: Vec<SearchResult> = fts_results
        .into_iter()
        .filter(|r| r.score >= options.min_score)
        .take(options.limit)
        .collect();

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_fts5_query_simple() {
        assert_eq!(
            build_fts5_query("hello world"),
            "\"hello\"* AND \"world\"*"
        );
    }

    #[test]
    fn test_build_fts5_query_quoted() {
        assert_eq!(
            build_fts5_query("\"exact phrase\""),
            "\"exact phrase\""
        );
    }

    #[test]
    fn test_build_fts5_query_negation() {
        assert_eq!(
            build_fts5_query("hello -world"),
            "\"hello\"* AND NOT \"world\"*"
        );
    }

    #[test]
    fn test_rrf_basic() {
        let list1 = vec![SearchResult {
            filepath: "a".to_string(),
            display_path: "qmd://c/a".to_string(),
            title: "A".to_string(),
            hash: "aaaaaa".to_string(),
            docid: "aaaaaa".to_string(),
            score: 0.9,
            source: "fts".to_string(),
            body: None,
            context: None,
            chunk_pos: None,
        }];
        let list2 = vec![SearchResult {
            filepath: "a".to_string(),
            display_path: "qmd://c/a".to_string(),
            title: "A".to_string(),
            hash: "aaaaaa".to_string(),
            docid: "aaaaaa".to_string(),
            score: 0.8,
            source: "vec".to_string(),
            body: None,
            context: None,
            chunk_pos: None,
        }];

        let ranked = reciprocal_rank_fusion(&[list1, list2], &[1.0, 1.0], 60.0);
        assert_eq!(ranked.len(), 1);
        assert!(ranked[0].rrf_score > 0.0);
    }
}
