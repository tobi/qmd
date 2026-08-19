/**
 * Deterministic bounds for query-expansion variants.
 *
 * The expansion model can emit dozens of near-identical `hyde` lines. Every
 * retained `vec`/`hyde` variant becomes another sequential embed plus
 * sqlite-vec scan, so an unbounded list turns one query into a multi-minute
 * fan-out. Cache reads share this policy: an oversized record written before
 * the cap existed must not replay that fan-out on every warm repeat.
 */

export const MAX_QUERY_EXPANSIONS = 6;
export const MAX_QUERY_EXPANSIONS_PER_TYPE = 2;

const EXPANSION_TYPES: ReadonlySet<string> = new Set(["lex", "vec", "hyde"]);

/** Trim, collapse internal whitespace, and lowercase for identity comparison. */
export function normalizeExpansionText(text: string): string {
  return text.trim().replace(/\s+/g, " ").toLowerCase();
}

/**
 * The single expansion policy point, shared by fresh model output and cache
 * reads. In stable source order:
 *
 * - drop unknown types and text that normalizes to empty;
 * - drop variants that only restate `originalQuery` — the original is always
 *   searched directly, so such a variant spends a slot on a duplicate list;
 * - deduplicate by `(type, normalized text)`, keeping the first occurrence;
 * - keep at most {@link MAX_QUERY_EXPANSIONS_PER_TYPE} of each type and
 *   {@link MAX_QUERY_EXPANSIONS} in total.
 *
 * The same text under two different types is kept: `lex` routes to FTS while
 * `vec`/`hyde` route to vector search, so they are not redundant.
 */
export function sanitizeQueryExpansions<T extends { type: string }>(
  items: readonly T[],
  getText: (item: T) => string,
  originalQuery: string,
): T[] {
  const original = normalizeExpansionText(originalQuery);
  const kept: T[] = [];
  const seen = new Set<string>();
  const perType = new Map<string, number>();

  for (const item of items) {
    if (kept.length >= MAX_QUERY_EXPANSIONS) break;
    if (!EXPANSION_TYPES.has(item.type)) continue;

    const text = getText(item);
    if (typeof text !== "string") continue;
    const normalized = normalizeExpansionText(text);
    if (!normalized || normalized === original) continue;

    const identity = `${item.type} ${normalized}`;
    if (seen.has(identity)) continue;

    const used = perType.get(item.type) ?? 0;
    if (used >= MAX_QUERY_EXPANSIONS_PER_TYPE) continue;

    seen.add(identity);
    perType.set(item.type, used + 1);
    kept.push(item);
  }

  return kept;
}
