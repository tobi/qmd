//! Tests for the formatter module — output format detection, line numbers, escaping.

use qmd::formatter::*;
use qmd::store::{MultiGetResult, SearchResult};

// =============================================================================
// OutputFormat
// =============================================================================

#[test]
fn test_output_format_from_flags_default() {
    assert_eq!(OutputFormat::from_flags(false, false, false, false, false), OutputFormat::Cli);
}

#[test]
fn test_output_format_from_flags_json() {
    assert_eq!(OutputFormat::from_flags(true, false, false, false, false), OutputFormat::Json);
}

#[test]
fn test_output_format_from_flags_csv() {
    // CSV takes precedence over JSON
    assert_eq!(OutputFormat::from_flags(true, true, false, false, false), OutputFormat::Csv);
}

#[test]
fn test_output_format_from_flags_files() {
    assert_eq!(OutputFormat::from_flags(false, false, false, false, true), OutputFormat::Files);
}

#[test]
fn test_output_format_default_limit() {
    assert_eq!(OutputFormat::Json.default_limit(), 20);
    assert_eq!(OutputFormat::Files.default_limit(), 20);
    assert_eq!(OutputFormat::Cli.default_limit(), 5);
    assert_eq!(OutputFormat::Csv.default_limit(), 5);
}

// =============================================================================
// Line numbers
// =============================================================================

#[test]
fn test_add_line_numbers() {
    let text = "line one\nline two\nline three";
    let result = add_line_numbers(text, 1);
    assert_eq!(result, "1: line one\n2: line two\n3: line three");
}

#[test]
fn test_add_line_numbers_offset() {
    let text = "first\nsecond";
    let result = add_line_numbers(text, 10);
    assert_eq!(result, "10: first\n11: second");
}

// =============================================================================
// Search result formatting
// =============================================================================

fn make_search_result() -> SearchResult {
    SearchResult {
        filepath: "notes/test.md".to_string(),
        display_path: "qmd://notes/test.md".to_string(),
        title: "Test Doc".to_string(),
        hash: "abcdef1234567890".to_string(),
        docid: "abcdef".to_string(),
        score: 0.85,
        source: "fts".to_string(),
        body: Some("This is the body content of the test document".to_string()),
        context: Some("Test context".to_string()),
        chunk_pos: None,
    }
}

#[test]
fn test_format_search_json() {
    let results = vec![make_search_result()];
    let opts = FormatOptions {
        query: "test".to_string(),
        ..Default::default()
    };
    let output = format_search_results(&results, OutputFormat::Json, &opts);
    assert!(output.contains("\"docid\""));
    assert!(output.contains("#abcdef"));
    assert!(output.contains("qmd://notes/test.md"));
}

#[test]
fn test_format_search_csv() {
    let results = vec![make_search_result()];
    let opts = FormatOptions {
        query: "test".to_string(),
        ..Default::default()
    };
    let output = format_search_results(&results, OutputFormat::Csv, &opts);
    assert!(output.starts_with("docid,score,file,title,context,line,snippet"));
    assert!(output.contains("#abcdef"));
}

#[test]
fn test_format_search_files() {
    let results = vec![make_search_result()];
    let opts = FormatOptions::default();
    let output = format_search_results(&results, OutputFormat::Files, &opts);
    assert!(output.contains("#abcdef"));
    assert!(output.contains("qmd://notes/test.md"));
}

#[test]
fn test_format_search_markdown() {
    let results = vec![make_search_result()];
    let opts = FormatOptions {
        query: "test".to_string(),
        ..Default::default()
    };
    let output = format_search_results(&results, OutputFormat::Md, &opts);
    assert!(output.contains("# Test Doc"));
    assert!(output.contains("`#abcdef`"));
}

#[test]
fn test_format_search_xml() {
    let results = vec![make_search_result()];
    let opts = FormatOptions {
        query: "test".to_string(),
        ..Default::default()
    };
    let output = format_search_results(&results, OutputFormat::Xml, &opts);
    assert!(output.contains("<file docid=\"#abcdef\""));
    assert!(output.contains("title=\"Test Doc\""));
}

#[test]
fn test_format_search_empty() {
    let opts = FormatOptions::default();
    let output = format_search_results(&[], OutputFormat::Cli, &opts);
    assert!(output.contains("No results"));
}

// =============================================================================
// Multi-get document formatting
// =============================================================================

fn make_multiget_result() -> MultiGetResult {
    MultiGetResult {
        filepath: "notes/doc.md".to_string(),
        display_path: "qmd://notes/doc.md".to_string(),
        title: "My Document".to_string(),
        body: "Document body content".to_string(),
        context: None,
        skipped: false,
        skip_reason: None,
    }
}

#[test]
fn test_format_documents_json() {
    let results = vec![make_multiget_result()];
    let output = format_documents(&results, OutputFormat::Json);
    assert!(output.contains("\"file\""));
    assert!(output.contains("qmd://notes/doc.md"));
    assert!(output.contains("Document body content"));
}

#[test]
fn test_format_documents_csv() {
    let results = vec![make_multiget_result()];
    let output = format_documents(&results, OutputFormat::Csv);
    assert!(output.starts_with("file,title,context,skipped,body"));
}

#[test]
fn test_format_documents_files() {
    let results = vec![make_multiget_result()];
    let output = format_documents(&results, OutputFormat::Files);
    assert_eq!(output, "qmd://notes/doc.md");
}

#[test]
fn test_format_documents_markdown() {
    let results = vec![make_multiget_result()];
    let output = format_documents(&results, OutputFormat::Md);
    assert!(output.contains("## qmd://notes/doc.md"));
    assert!(output.contains("Document body content"));
}

#[test]
fn test_format_documents_skipped() {
    let results = vec![MultiGetResult {
        filepath: "notes/big.md".to_string(),
        display_path: "qmd://notes/big.md".to_string(),
        title: "Big File".to_string(),
        body: String::new(),
        context: None,
        skipped: true,
        skip_reason: Some("File too large (15KB > 10KB)".to_string()),
    }];
    let json = format_documents(&results, OutputFormat::Json);
    assert!(json.contains("\"skipped\": true"));
    assert!(json.contains("File too large"));
}

// =============================================================================
// Single document formatting
// =============================================================================

#[test]
fn test_format_document_json() {
    let doc = qmd::store::DocumentResult {
        filepath: "notes/test.md".to_string(),
        display_path: "qmd://notes/test.md".to_string(),
        title: "Test".to_string(),
        hash: "abc123".to_string(),
        docid: "abc123".to_string(),
        body: Some("Body text".to_string()),
        body_length: 9,
        context: None,
        modified_at: "2025-01-01T00:00:00Z".to_string(),
        collection: "notes".to_string(),
        path: "test.md".to_string(),
    };
    let output = format_document(&doc, OutputFormat::Json);
    assert!(output.contains("\"title\": \"Test\""));
    assert!(output.contains("Body text"));
}

#[test]
fn test_format_document_xml() {
    let doc = qmd::store::DocumentResult {
        filepath: "notes/test.md".to_string(),
        display_path: "qmd://notes/test.md".to_string(),
        title: "Test & Title".to_string(),
        hash: "abc123".to_string(),
        docid: "abc123".to_string(),
        body: Some("<html>escaped</html>".to_string()),
        body_length: 20,
        context: None,
        modified_at: "2025-01-01T00:00:00Z".to_string(),
        collection: "notes".to_string(),
        path: "test.md".to_string(),
    };
    let output = format_document(&doc, OutputFormat::Xml);
    // XML escaping
    assert!(output.contains("Test &amp; Title"));
    assert!(output.contains("&lt;html&gt;"));
}
