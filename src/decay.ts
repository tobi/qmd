/**
 * decay.ts - Ebbinghaus forgetting curve memory decay
 *
 * Implements strength scoring based on the Ebbinghaus forgetting curve.
 * Documents lose relevance over time unless actively recalled through searches.
 * The decay rate depends on document category and importance.
 */

export type Category = "strategy" | "fact" | "assumption" | "failure";

export const CATEGORIES: Category[] = ["strategy", "fact", "assumption", "failure"];

/**
 * Base decay rates (lambda) per category.
 * Lower = slower decay = stays relevant longer.
 */
export const BASE_LAMBDA: Record<Category, number> = {
  strategy: 0.10,
  fact: 0.16,
  assumption: 0.20,
  failure: 0.35,
};

/** Documents with strength below this are candidates for pruning. */
export const PRUNE_THRESHOLD = 0.05;

/**
 * Compute the current memory strength of a document using the Ebbinghaus
 * forgetting curve with recall reinforcement.
 *
 * strength = importance × e^(−λ_eff × days) × (1 + recall_count × 0.2)
 * λ_eff = base_lambda × (1 − importance × 0.8)
 *
 * @param importance    Document importance weight (0..1)
 * @param category      Document category (determines base decay rate fallback)
 * @param createdAt     ISO datetime of document creation
 * @param recallCount   Number of times this document has been recalled
 * @param baseLambda    Optional collection-level base lambda (overrides category default)
 * @returns Strength score (higher = more relevant)
 */
export function computeStrength(
  importance: number,
  category: Category,
  createdAt: string,
  recallCount: number,
  now?: number,
  baseLambda?: number
): number {
  const lambda = baseLambda ?? BASE_LAMBDA[category];
  const lambdaEff = lambda * (1 - importance * 0.8);
  const days = Math.max(0, ((now ?? Date.now()) - new Date(createdAt).getTime()) / (1000 * 60 * 60 * 24));
  return importance * Math.exp(-lambdaEff * days) * (1 + recallCount * 0.2);
}

/**
 * Check if a category string is a valid Category.
 */
export function isValidCategory(value: string): value is Category {
  return CATEGORIES.includes(value as Category);
}
