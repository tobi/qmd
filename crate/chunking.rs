//! chunking.rs - Smart markdown-aware document chunking
//!
//! Splits documents into chunks for embedding, preferring markdown heading
//! boundaries and avoiding splits inside code fences.

use regex::Regex;
use std::sync::LazyLock;

// =============================================================================
// Constants
// =============================================================================

/// Default chunk size in characters (~900 tokens × 4 chars/token)
pub const CHUNK_SIZE_CHARS: usize = 3600;

/// Default overlap in characters (~135 tokens × 4 chars/token = 15% of chunk)
pub const CHUNK_OVERLAP_CHARS: usize = 540;

/// Search window for finding break points (in characters)
pub const CHUNK_WINDOW_CHARS: usize = 800;

// =============================================================================
// Types
// =============================================================================

/// A potential break point in the document.
#[derive(Debug, Clone)]
pub struct BreakPoint {
    pub pos: usize,
    pub score: u32,
    pub break_type: String,
}

/// A code fence region (start..end).
#[derive(Debug, Clone)]
pub struct CodeFenceRegion {
    pub start: usize,
    pub end: usize,
}

/// A chunk of document text with its position.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub text: String,
    pub pos: usize,
}

// =============================================================================
// Break point scanning
// =============================================================================

static HEADING_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^(#{1,6})\s").unwrap());
static BLANK_LINE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n").unwrap());
static CODE_FENCE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^```").unwrap());
static HR_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^(---+|===+|\*\*\*+)\s*$").unwrap());

/// Scan the document for potential break points, scored by quality.
pub fn scan_break_points(text: &str) -> Vec<BreakPoint> {
    let mut points = Vec::new();

    // Headings (score by level: h1=100, h2=90, h3=80, ...)
    for cap in HEADING_RE.find_iter(text) {
        let hashes = cap.as_str().trim().trim_end_matches(|c: char| !c.is_ascii());
        let hash_count = hashes.chars().take_while(|&c| c == '#').count();
        let score = match hash_count {
            1 => 100,
            2 => 90,
            3 => 80,
            4 => 70,
            5 => 60,
            6 => 50,
            _ => 40,
        };
        // Break point is at the start of the heading line
        let line_start = text[..cap.start()].rfind('\n').map(|p| p + 1).unwrap_or(0);
        points.push(BreakPoint {
            pos: line_start,
            score,
            break_type: format!("h{hash_count}"),
        });
    }

    // Code fence boundaries (score: 80)
    for cap in CODE_FENCE_RE.find_iter(text) {
        let line_start = text[..cap.start()].rfind('\n').map(|p| p + 1).unwrap_or(0);
        points.push(BreakPoint {
            pos: line_start,
            score: 80,
            break_type: "code_fence".to_string(),
        });
    }

    // Horizontal rules (score: 60)
    for cap in HR_RE.find_iter(text) {
        let line_start = text[..cap.start()].rfind('\n').map(|p| p + 1).unwrap_or(0);
        points.push(BreakPoint {
            pos: line_start,
            score: 60,
            break_type: "hr".to_string(),
        });
    }

    // Blank lines (score: 20)
    for cap in BLANK_LINE_RE.find_iter(text) {
        points.push(BreakPoint {
            pos: cap.end(),
            score: 20,
            break_type: "blank".to_string(),
        });
    }

    // Sort by position
    points.sort_by_key(|p| p.pos);
    // Deduplicate by position
    points.dedup_by_key(|p| p.pos);

    points
}

/// Find code fence regions in the text.
pub fn find_code_fences(text: &str) -> Vec<CodeFenceRegion> {
    let mut regions = Vec::new();
    let mut fence_start: Option<usize> = None;

    for cap in CODE_FENCE_RE.find_iter(text) {
        let line_start = text[..cap.start()].rfind('\n').map(|p| p + 1).unwrap_or(0);
        match fence_start {
            None => fence_start = Some(line_start),
            Some(start) => {
                let end = text[cap.end()..].find('\n').map(|p| cap.end() + p + 1).unwrap_or(text.len());
                regions.push(CodeFenceRegion { start, end });
                fence_start = None;
            }
        }
    }

    regions
}

/// Check if a position is inside a code fence.
fn is_inside_code_fence(pos: usize, fences: &[CodeFenceRegion]) -> bool {
    fences.iter().any(|f| pos > f.start && pos < f.end)
}

/// Find the best cutoff position near a target, using break point scoring
/// with distance decay.
pub fn find_best_cutoff(
    break_points: &[BreakPoint],
    target_pos: usize,
    window_chars: usize,
    decay_factor: f64,
    code_fences: &[CodeFenceRegion],
) -> usize {
    let window_start = target_pos.saturating_sub(window_chars);
    let window_end = target_pos + window_chars / 4; // Slight forward bias

    let candidates: Vec<&BreakPoint> = break_points
        .iter()
        .filter(|bp| bp.pos >= window_start && bp.pos <= window_end)
        .filter(|bp| !is_inside_code_fence(bp.pos, code_fences))
        .collect();

    if candidates.is_empty() {
        return target_pos;
    }

    // Score each candidate: break_score * distance_decay
    let mut best_pos = target_pos;
    let mut best_score = 0.0f64;

    for bp in &candidates {
        let distance = (bp.pos as f64 - target_pos as f64).abs();
        let max_dist = window_chars as f64;
        let decay = 1.0 - (distance / max_dist).powf(decay_factor);
        let effective_score = bp.score as f64 * decay;

        if effective_score > best_score {
            best_score = effective_score;
            best_pos = bp.pos;
        }
    }

    best_pos
}

// =============================================================================
// Chunking
// =============================================================================

/// Chunk a document into overlapping segments, preferring markdown boundaries.
pub fn chunk_document(
    content: &str,
    max_chars: Option<usize>,
    overlap_chars: Option<usize>,
    window_chars: Option<usize>,
) -> Vec<Chunk> {
    let max_chars = max_chars.unwrap_or(CHUNK_SIZE_CHARS);
    let overlap_chars = overlap_chars.unwrap_or(CHUNK_OVERLAP_CHARS);
    let window_chars = window_chars.unwrap_or(CHUNK_WINDOW_CHARS);

    if content.len() <= max_chars {
        return vec![Chunk {
            text: content.to_string(),
            pos: 0,
        }];
    }

    let break_points = scan_break_points(content);
    let code_fences = find_code_fences(content);

    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < content.len() {
        let target_end = (start + max_chars).min(content.len());

        if target_end >= content.len() {
            // Last chunk
            chunks.push(Chunk {
                text: content[start..].to_string(),
                pos: start,
            });
            break;
        }

        // Find the best break point near the target end
        let cut_pos = find_best_cutoff(&break_points, target_end, window_chars, 2.0, &code_fences);
        let cut_pos = cut_pos.max(start + max_chars / 2); // Don't cut too short

        chunks.push(Chunk {
            text: content[start..cut_pos].to_string(),
            pos: start,
        });

        // Next chunk starts with overlap
        start = cut_pos.saturating_sub(overlap_chars);
        // Ensure we make progress
        if start <= chunks.last().map(|c| c.pos).unwrap_or(0) {
            start = cut_pos;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_break_points_headings() {
        let text = "# Title\n\nSome text\n\n## Section\n\nMore text";
        let points = scan_break_points(text);
        assert!(points.iter().any(|p| p.score == 100)); // h1
        assert!(points.iter().any(|p| p.score == 90)); // h2
    }

    #[test]
    fn test_chunk_short_document() {
        let text = "Short document";
        let chunks = chunk_document(text, None, None, None);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Short document");
        assert_eq!(chunks[0].pos, 0);
    }

    #[test]
    fn test_chunk_long_document() {
        let text = "# Section 1\n\n".to_string()
            + &"word ".repeat(1000)
            + "\n\n# Section 2\n\n"
            + &"word ".repeat(1000);
        let chunks = chunk_document(&text, Some(500), Some(50), Some(200));
        assert!(chunks.len() > 1);
    }

    #[test]
    fn test_find_code_fences() {
        let text = "before\n```\ncode\n```\nafter";
        let fences = find_code_fences(text);
        assert_eq!(fences.len(), 1);
    }
}
