//! Benchmarks for search and chunking operations.
//!
//! Run with: cargo bench

#![feature(test)]
extern crate test;

use test::Bencher;
use qmd::chunking::{chunk_document, scan_break_points, find_code_fences, find_best_cutoff};
use qmd::search::build_fts5_query;
use qmd::store::hash_content;

// =============================================================================
// Hashing benchmarks
// =============================================================================

#[bench]
fn bench_hash_small(b: &mut Bencher) {
    let content = "# Small document\n\nA short paragraph.";
    b.iter(|| hash_content(content));
}

#[bench]
fn bench_hash_medium(b: &mut Bencher) {
    let content = "word ".repeat(2000); // ~10KB
    b.iter(|| hash_content(&content));
}

#[bench]
fn bench_hash_large(b: &mut Bencher) {
    let content = "word ".repeat(50_000); // ~250KB
    b.iter(|| hash_content(&content));
}

// =============================================================================
// FTS5 query builder benchmarks
// =============================================================================

#[bench]
fn bench_fts5_query_simple(b: &mut Bencher) {
    b.iter(|| build_fts5_query("hello world"));
}

#[bench]
fn bench_fts5_query_complex(b: &mut Bencher) {
    b.iter(|| build_fts5_query("rust \"memory safe\" -garbage collection programming"));
}

// =============================================================================
// Chunking benchmarks
// =============================================================================

fn make_markdown_doc(sections: usize) -> String {
    let mut doc = String::new();
    for i in 0..sections {
        doc.push_str(&format!("## Section {i}\n\n"));
        doc.push_str(&"Lorem ipsum dolor sit amet. ".repeat(50));
        doc.push_str("\n\n");
        if i % 3 == 0 {
            doc.push_str("```python\ndef example():\n    pass\n```\n\n");
        }
    }
    doc
}

#[bench]
fn bench_scan_break_points_small(b: &mut Bencher) {
    let doc = make_markdown_doc(5);
    b.iter(|| scan_break_points(&doc));
}

#[bench]
fn bench_scan_break_points_large(b: &mut Bencher) {
    let doc = make_markdown_doc(100);
    b.iter(|| scan_break_points(&doc));
}

#[bench]
fn bench_find_code_fences(b: &mut Bencher) {
    let doc = make_markdown_doc(50);
    b.iter(|| find_code_fences(&doc));
}

#[bench]
fn bench_find_best_cutoff(b: &mut Bencher) {
    let doc = make_markdown_doc(50);
    let points = scan_break_points(&doc);
    let fences = find_code_fences(&doc);
    let target = doc.len() / 2;
    b.iter(|| find_best_cutoff(&points, target, 800, 2.0, &fences));
}

#[bench]
fn bench_chunk_small_doc(b: &mut Bencher) {
    let doc = make_markdown_doc(5);
    b.iter(|| chunk_document(&doc, None, None, None));
}

#[bench]
fn bench_chunk_medium_doc(b: &mut Bencher) {
    let doc = make_markdown_doc(30);
    b.iter(|| chunk_document(&doc, None, None, None));
}

#[bench]
fn bench_chunk_large_doc(b: &mut Bencher) {
    let doc = make_markdown_doc(200);
    b.iter(|| chunk_document(&doc, None, None, None));
}
