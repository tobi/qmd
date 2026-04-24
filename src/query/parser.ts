/**
 * parser.ts - Parse tokenized filter expressions into an AST.
 */

import { tokenize, type Token, type TokenType } from "./lexer.js";

// =============================================================================
// AST node types
// =============================================================================

export type FilterNode =
  | { type: "AND"; left: FilterNode; right: FilterNode }
  | { type: "OR";  left: FilterNode; right: FilterNode }
  | { type: "NOT"; operand: FilterNode }
  | { type: "CMP"; field: string; op: "=" | "~=" | "<" | ">" | "regex"; value: string }
  | { type: "MISSING"; field: string }   // field is absent from frontmatter
  | { type: "EMPTY"; field: string }     // field exists but is empty string
  | { type: "NO"; field: string; value: string | null };  // no:headings, no:level=2

// =============================================================================
// Parser
// =============================================================================

class Parser {
  private tokens: Token[];
  private pos = 0;

  constructor(input: string) {
    this.tokens = tokenize(input);
  }

  private peek(): Token {
    return this.tokens[this.pos] ?? { type: "EOF", value: "", pos: 0 };
  }

  private consume(expected?: TokenType): Token {
    const t = this.tokens[this.pos] ?? { type: "EOF", value: "", pos: 0 };
    if (expected && t.type !== expected) {
      throw new Error(`Expected ${expected} but got ${t.type} ("${t.value}") at position ${t.pos}`);
    }
    this.pos++;
    return t;
  }

  parse(): FilterNode {
    const node = this.parseOr();
    if (this.peek().type !== "EOF") {
      throw new Error(`Unexpected token "${this.peek().value}" at position ${this.peek().pos}`);
    }
    return node;
  }

  private parseOr(): FilterNode {
    let left = this.parseAnd();
    while (this.peek().type === "OR") {
      this.consume("OR");
      const right = this.parseAnd();
      left = { type: "OR", left, right };
    }
    return left;
  }

  private parseAnd(): FilterNode {
    let left = this.parseNot();
    while (this.peek().type === "AND") {
      this.consume("AND");
      const right = this.parseNot();
      left = { type: "AND", left, right };
    }
    return left;
  }

  private parseNot(): FilterNode {
    if (this.peek().type === "NOT") {
      this.consume("NOT");
      return { type: "NOT", operand: this.parseNot() };
    }
    return this.parsePrimary();
  }

  private parsePrimary(): FilterNode {
    const t = this.peek();

    if (t.type === "LPAREN") {
      this.consume("LPAREN");
      const node = this.parseOr();
      this.consume("RPAREN");
      return node;
    }

    if (t.type === "MISSING") {
      this.consume("MISSING");
      const field = this.consume("IDENT").value;
      return { type: "MISSING", field };
    }

    if (t.type === "EMPTY") {
      this.consume("EMPTY");
      const field = this.consume("IDENT").value;
      return { type: "EMPTY", field };
    }

    if (t.type === "NO") {
      this.consume("NO");
      const field = this.consume("IDENT").value;
      let value: string | null = null;
      if (this.peek().type === "EQ") {
        this.consume("EQ");
        value = this.consumeValue();
      }
      return { type: "NO", field, value };
    }

    // Comparison: IDENT OP VALUE
    if (t.type === "IDENT") {
      const field = this.consume("IDENT").value;
      const op = this.peek();

      if (op.type === "EQ") {
        this.consume("EQ");
        return { type: "CMP", field, op: "=", value: this.consumeValue() };
      }
      if (op.type === "FUZZY") {
        this.consume("FUZZY");
        return { type: "CMP", field, op: "~=", value: this.consumeValue() };
      }
      if (op.type === "REGEX") {
        const val = this.consume("REGEX").value;
        return { type: "CMP", field, op: "regex", value: val };
      }
      if (op.type === "LT") {
        this.consume("LT");
        return { type: "CMP", field, op: "<", value: this.consumeValue() };
      }
      if (op.type === "GT") {
        this.consume("GT");
        return { type: "CMP", field, op: ">", value: this.consumeValue() };
      }

      // Bare IDENT with no operator treated as tag=value shorthand
      return { type: "CMP", field: "tag", op: "=", value: field };
    }

    throw new Error(`Unexpected token "${t.value}" (${t.type}) at position ${t.pos}`);
  }

  private consumeValue(): string {
    const t = this.peek();
    if (t.type === "STRING" || t.type === "IDENT" || t.type === "DURATION") {
      return this.consume().value;
    }
    throw new Error(`Expected a value but got ${t.type} at position ${t.pos}`);
  }
}

export function parseFilter(expr: string): FilterNode {
  return new Parser(expr.trim()).parse();
}
