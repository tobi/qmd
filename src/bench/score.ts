/**
 * Scoring functions for the QMD benchmark harness.
 *
 * Computes precision@k, recall, MRR, and F1 for search results
 * against ground-truth expected files.
 */

/**
 * Normalize a file path for comparison.
 * Strips qmd:// prefix, lowercases, removes leading/trailing slashes.
 */
export function normalizePath(p: string): string {
  if (p.startsWith("qmd://")) {
    // qmd://collection/docs/readme.md → docs/readme.md
    const withoutScheme = p.slice("qmd://".length);
    const slashIdx = withoutScheme.indexOf("/");
    p = slashIdx >= 0 ? withoutScheme.slice(slashIdx + 1) : withoutScheme;
  }
  return p.toLowerCase().replace(/^\/+|\/+$/g, "");
}

/**
 * Check if two paths refer to the same file.
 * Handles different path formats by comparing normalized suffixes.
 */
export function pathsMatch(result: string, expected: string): boolean {
  const nr = normalizePath(result);
  const ne = normalizePath(expected);
  if (nr === ne) return true;
  if (nr.endsWith(ne) || ne.endsWith(nr)) return true;
  return false;
}

type ScoreMetrics = {
  precision_at_k: number;
  recall: number;
  recall_at_1: number;
  recall_at_3: number;
  recall_at_5: number;
  mrr: number;
  f1: number;
  hits_at_k: number;
  matched_files: string[];
  unmatched_expected_files: string[];
};

function hitsWithin(resultFiles: string[], expectedFiles: string[], k: number): number {
  const topKResults = resultFiles.slice(0, k);
  let hits = 0;
  for (const expected of expectedFiles) {
    if (topKResults.some(r => pathsMatch(r, expected))) {
      hits++;
    }
  }
  return hits;
}

/**
 * Score a set of search results against expected files.
 */
export function scoreResults(
  resultFiles: string[],
  expectedFiles: string[],
  topK: number,
): ScoreMetrics {
  // Count hits in top-k
  const hitsAtK = hitsWithin(resultFiles, expectedFiles, topK);

  const matchedFiles: string[] = [];
  const unmatchedExpectedFiles: string[] = [];

  for (const expected of expectedFiles) {
    if (resultFiles.some(r => pathsMatch(r, expected))) {
      matchedFiles.push(expected);
    } else {
      unmatchedExpectedFiles.push(expected);
    }
  }

  // MRR: reciprocal rank of first relevant result
  let mrr = 0;
  for (let i = 0; i < resultFiles.length; i++) {
    if (expectedFiles.some(e => pathsMatch(resultFiles[i]!, e))) {
      mrr = 1 / (i + 1);
      break;
    }
  }

  const denominator = Math.min(topK, expectedFiles.length);
  const precision_at_k = denominator > 0 ? hitsAtK / denominator : 0;
  const recall = expectedFiles.length > 0 ? matchedFiles.length / expectedFiles.length : 0;
  const recall_at_1 = expectedFiles.length > 0 ? hitsWithin(resultFiles, expectedFiles, 1) / expectedFiles.length : 0;
  const recall_at_3 = expectedFiles.length > 0 ? hitsWithin(resultFiles, expectedFiles, 3) / expectedFiles.length : 0;
  const recall_at_5 = expectedFiles.length > 0 ? hitsWithin(resultFiles, expectedFiles, 5) / expectedFiles.length : 0;
  const f1 = precision_at_k + recall > 0
    ? 2 * (precision_at_k * recall) / (precision_at_k + recall)
    : 0;

  return {
    precision_at_k,
    recall,
    recall_at_1,
    recall_at_3,
    recall_at_5,
    mrr,
    f1,
    hits_at_k: hitsAtK,
    matched_files: matchedFiles,
    unmatched_expected_files: unmatchedExpectedFiles,
  };
}
