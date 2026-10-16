export type LexicalQueryClause = {
  text: string;
  negated: boolean;
  match: "prefix" | "phrase";
  ftsExpression: string;
};

const CJK_CHAR_PATTERN = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/u;
const CJK_RUN_PATTERN = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]+/gu;
// Split compound queries into adjacent phrase terms for dotted and hyphenated
// lookup (#757, #916). Apostrophes and underscores remain inside expressions;
// FTS5 tokenizes them symmetrically in queries and indexed documents (#305).
const FTS5_SEPARATOR_RUN = /[^\p{L}\p{N}'_]+/u;

/** Space CJK runs so FTS5's `unicode61` tokenizer indexes one character per token. */
export function normalizeCjkForFTS(text: string): string {
  return text.replace(CJK_RUN_PATTERN, run => ` ${Array.from(run).join(" ")} `);
}

/** Keep one QMD FTS term, including apostrophes and underscores for tokenizer symmetry (#305). */
export function sanitizeFTS5Term(term: string): string {
  return term.replace(/[^\p{L}\p{N}'_]/gu, "").toLowerCase();
}

export function containsCjk(text: string): boolean {
  return CJK_CHAR_PATTERN.test(text);
}

/** Parse QMD's lexical query syntax into its source clauses and FTS expressions. */
export function parseLexicalQuery(query: string): LexicalQueryClause[] {
  const clauses: LexicalQueryClause[] = [];
  const source = query.trim();
  let offset = 0;

  while (offset < source.length) {
    while (offset < source.length && /\s/.test(source[offset]!)) offset++;
    if (offset >= source.length) break;

    const negated = source[offset] === "-";
    if (negated) offset++;

    if (source[offset] === '"') {
      const start = ++offset;
      while (offset < source.length && source[offset] !== '"') offset++;
      const text = source.slice(start, offset).trim();
      if (offset < source.length) offset++;

      const sanitized = sanitizeFTS5Phrase(text);
      if (text.length > 0 && sanitized.length > 0) {
        clauses.push({
          text,
          negated,
          match: "phrase",
          ftsExpression: `"${sanitized}"`,
        });
      }
      continue;
    }

    const start = offset;
    while (offset < source.length && !/[\s"]/.test(source[offset]!)) offset++;
    const text = source.slice(start, offset);

    if (containsCjk(text)) {
      const sanitized = sanitizeFTS5Phrase(text);
      if (sanitized.length > 0) {
        clauses.push({
          text,
          negated,
          match: "phrase",
          ftsExpression: `"${sanitized}"`,
        });
      }
      continue;
    }

    const parts = splitFTS5CompoundTerm(text);
    if (parts.length === 0) continue;

    const match = parts.length > 1 ? "phrase" : "prefix";
    clauses.push({
      text,
      negated,
      match,
      ftsExpression: match === "phrase" ? `"${parts.join(" ")}"` : `"${parts[0]}"*`,
    });
  }

  return clauses;
}

function sanitizeFTS5Phrase(phrase: string): string {
  return normalizeCjkForFTS(phrase)
    .split(/\s+/)
    .flatMap(splitFTS5CompoundTerm)
    .join(" ");
}

function splitFTS5CompoundTerm(term: string): string[] {
  return term.split(FTS5_SEPARATOR_RUN).map(sanitizeFTS5Term).filter(Boolean);
}
