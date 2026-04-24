/**
 * lexer.ts - Tokenize qmdx filter expressions.
 *
 * Grammar overview:
 *   expr       = term (("AND" | "OR") term)*
 *   term       = "NOT" term | "(" expr ")" | predicate
 *   predicate  = missing | empty | no | comparison
 *   missing    = "missing:" IDENT
 *   empty      = "empty:" IDENT
 *   no         = "no:" IDENT ("=" VALUE)?
 *   comparison = IDENT OP VALUE
 *   OP         = "=" | "~=" | "~/.../" | "<" | ">"
 *   VALUE      = QUOTED_STRING | BARE_WORD | DATE_DURATION
 */

export type TokenType =
  | "IDENT"
  | "EQ"          // =
  | "FUZZY"       // ~=
  | "REGEX"       // ~/pattern/
  | "LT"          // <
  | "GT"          // >
  | "AND"
  | "OR"
  | "NOT"
  | "LPAREN"
  | "RPAREN"
  | "MISSING"     // "missing:"
  | "EMPTY"       // "empty:"
  | "NO"          // "no:"
  | "STRING"      // quoted or bare value
  | "DURATION"    // e.g. 7d, 30d, 1y
  | "EOF";

export interface Token {
  type: TokenType;
  value: string;
  pos: number;
}

const KEYWORDS: Record<string, TokenType> = {
  AND: "AND",
  OR: "OR",
  NOT: "NOT",
};

export function tokenize(input: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;

  function peek(): string { return input[i] ?? ""; }
  function consume(): string { return input[i++] ?? ""; }
  function skipWS(): void { while (i < input.length && /\s/.test(input[i]!)) i++; }

  while (i < input.length) {
    skipWS();
    if (i >= input.length) break;

    const pos = i;
    const ch = peek();

    // Single-char operators
    if (ch === "(") { consume(); tokens.push({ type: "LPAREN", value: "(", pos }); continue; }
    if (ch === ")") { consume(); tokens.push({ type: "RPAREN", value: ")", pos }); continue; }
    if (ch === "<") { consume(); tokens.push({ type: "LT", value: "<", pos }); continue; }
    if (ch === ">") { consume(); tokens.push({ type: "GT", value: ">", pos }); continue; }

    // "~=value" or "~/regex/"
    if (ch === "~") {
      consume();
      if (peek() === "/") {
        // regex: ~/pattern/
        consume();
        let pat = "";
        while (i < input.length && peek() !== "/") pat += consume();
        if (peek() === "/") consume(); // closing /
        tokens.push({ type: "REGEX", value: pat, pos });
      } else if (peek() === "=") {
        consume();
        tokens.push({ type: "FUZZY", value: "~=", pos });
      } else {
        tokens.push({ type: "FUZZY", value: "~=", pos });
      }
      continue;
    }

    // "=" simple equals
    if (ch === "=") { consume(); tokens.push({ type: "EQ", value: "=", pos }); continue; }

    // Quoted strings
    if (ch === '"' || ch === "'") {
      const quote = consume();
      let str = "";
      while (i < input.length && peek() !== quote) {
        if (peek() === "\\") { consume(); str += consume(); }
        else str += consume();
      }
      if (peek() === quote) consume();
      tokens.push({ type: "STRING", value: str, pos });
      continue;
    }

    // Prefixed predicates: missing:, empty:, no:
    if (input.startsWith("missing:", i)) { i += 8; tokens.push({ type: "MISSING", value: "missing:", pos }); continue; }
    if (input.startsWith("empty:", i))   { i += 6; tokens.push({ type: "EMPTY", value: "empty:", pos }); continue; }
    if (input.startsWith("no:", i))      { i += 3; tokens.push({ type: "NO", value: "no:", pos }); continue; }

    // Identifiers, keywords, bare values
    if (/[\w.\-/]/.test(ch)) {
      let word = "";
      while (i < input.length && /[^\s=~<>()"']/.test(peek()) && !input.startsWith("missing:", i) && !input.startsWith("empty:", i) && !input.startsWith("no:", i)) {
        word += consume();
      }

      const upper = word.toUpperCase();
      if (upper in KEYWORDS) {
        tokens.push({ type: KEYWORDS[upper]!, value: word, pos });
      } else if (/^\d+[dwmy]$/i.test(word)) {
        // Duration literal: 7d, 30d, 1w, 2m, 1y
        tokens.push({ type: "DURATION", value: word, pos });
      } else {
        tokens.push({ type: "IDENT", value: word, pos });
      }
      continue;
    }

    // Skip unknown characters
    consume();
  }

  tokens.push({ type: "EOF", value: "", pos: i });
  return tokens;
}
