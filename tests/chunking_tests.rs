//! Tests for the chunking module — break points, code fences, chunking.

use qmd::chunking::*;

// =============================================================================
// Break point scanning
// =============================================================================

#[test]
fn test_break_points_h1() {
    let text = "# Title\n\nParagraph";
    let points = scan_break_points(text);
    assert!(points.iter().any(|p| p.score == 100 && p.break_type == "h1"));
}

#[test]
fn test_break_points_h2_h3() {
    let text = "## Section\n\n### Subsection\n\nContent";
    let points = scan_break_points(text);
    assert!(points.iter().any(|p| p.score == 90 && p.break_type == "h2"));
    assert!(points.iter().any(|p| p.score == 80 && p.break_type == "h3"));
}

#[test]
fn test_break_points_code_fence() {
    let text = "Before\n```rust\nfn main() {}\n```\nAfter";
    let points = scan_break_points(text);
    assert!(points.iter().any(|p| p.break_type == "code_fence"));
}

#[test]
fn test_break_points_hr() {
    let text = "Above\n\n---\n\nBelow";
    let points = scan_break_points(text);
    assert!(points.iter().any(|p| p.break_type == "hr"));
}

#[test]
fn test_break_points_blank_lines() {
    let text = "First paragraph\n\nSecond paragraph\n\nThird";
    let points = scan_break_points(text);
    let blank_count = points.iter().filter(|p| p.break_type == "blank").count();
    assert!(blank_count >= 2);
}

#[test]
fn test_break_points_sorted_by_position() {
    let text = "## B\n\n# A\n\n### C";
    let points = scan_break_points(text);
    for w in points.windows(2) {
        assert!(w[0].pos <= w[1].pos, "Break points should be sorted by position");
    }
}

// =============================================================================
// Code fences
// =============================================================================

#[test]
fn test_find_code_fences_single() {
    let text = "text\n```python\ncode\n```\nmore text";
    let fences = find_code_fences(text);
    assert_eq!(fences.len(), 1);
    assert!(fences[0].start < fences[0].end);
}

#[test]
fn test_find_code_fences_multiple() {
    let text = "```\nblock1\n```\n\n```\nblock2\n```";
    let fences = find_code_fences(text);
    assert_eq!(fences.len(), 2);
}

#[test]
fn test_find_code_fences_unclosed() {
    let text = "```\nunclosed code fence";
    let fences = find_code_fences(text);
    assert_eq!(fences.len(), 0); // Unclosed fence = no region
}

#[test]
fn test_find_code_fences_no_fences() {
    let text = "Just regular text\nNo fences here";
    let fences = find_code_fences(text);
    assert_eq!(fences.len(), 0);
}

// =============================================================================
// Best cutoff
// =============================================================================

#[test]
fn test_find_best_cutoff_prefers_headings() {
    let text = "text\n\n## Good Break\n\nmore text\n\nmore more text";
    let points = scan_break_points(text);
    let fences = find_code_fences(text);
    let heading_pos = points.iter().find(|p| p.break_type == "h2").unwrap().pos;

    let cutoff = find_best_cutoff(&points, heading_pos + 5, 100, 2.0, &fences);
    // Should snap to heading position
    assert_eq!(cutoff, heading_pos);
}

#[test]
fn test_find_best_cutoff_avoids_code_fences() {
    let text = "text\n```\ncode here\n```\nafter code\n\nblank line";
    let points = scan_break_points(text);
    let fences = find_code_fences(text);

    // Target inside the code fence
    let fence_middle = (fences[0].start + fences[0].end) / 2;
    let cutoff = find_best_cutoff(&points, fence_middle, 200, 2.0, &fences);
    // Should not cut inside the code fence
    assert!(cutoff <= fences[0].start || cutoff >= fences[0].end,
        "Cutoff {} is inside code fence [{}, {}]", cutoff, fences[0].start, fences[0].end);
}

// =============================================================================
// Document chunking
// =============================================================================

#[test]
fn test_chunk_short_document() {
    let text = "Short document that fits in one chunk";
    let chunks = chunk_document(text, None, None, None);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
    assert_eq!(chunks[0].pos, 0);
}

#[test]
fn test_chunk_exactly_max_chars() {
    let text = "x".repeat(CHUNK_SIZE_CHARS);
    let chunks = chunk_document(&text, None, None, None);
    assert_eq!(chunks.len(), 1);
}

#[test]
fn test_chunk_long_document_produces_multiple() {
    // Create document larger than chunk size
    let section = "## Section\n\nSome paragraph content here.\n\n";
    let text = section.repeat(200); // ~8000 chars
    let chunks = chunk_document(&text, Some(500), Some(50), Some(200));
    assert!(chunks.len() > 1, "Expected multiple chunks, got {}", chunks.len());
}

#[test]
fn test_chunks_cover_entire_document() {
    let text = "word ".repeat(2000); // ~10000 chars
    let chunks = chunk_document(&text, Some(500), Some(50), Some(200));

    // First chunk starts at 0
    assert_eq!(chunks[0].pos, 0);
    // Last chunk reaches end of document
    let last = chunks.last().unwrap();
    assert_eq!(last.pos + last.text.len(), text.len());
}

#[test]
fn test_chunks_have_overlap() {
    let text = "# Section 1\n\n".to_string()
        + &"word ".repeat(500)
        + "\n\n# Section 2\n\n"
        + &"word ".repeat(500);
    let chunks = chunk_document(&text, Some(500), Some(50), Some(200));

    if chunks.len() >= 2 {
        // Check that consecutive chunks overlap
        for w in chunks.windows(2) {
            let end_of_first = w[0].pos + w[0].text.len();
            let start_of_second = w[1].pos;
            assert!(
                start_of_second < end_of_first,
                "Expected overlap: chunk1 ends at {}, chunk2 starts at {}",
                end_of_first,
                start_of_second
            );
        }
    }
}

#[test]
fn test_chunk_empty_document() {
    let chunks = chunk_document("", None, None, None);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "");
}
