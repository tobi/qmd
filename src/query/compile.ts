/**
 * compile.ts - Compile a FilterNode AST into a parameterized SQL WHERE clause.
 *
 * The primary table alias is "d" (documents). The clause can be embedded in:
 *   SELECT d.* FROM documents d WHERE <clause>
 *
 * Field name mapping:
 *   "tag"         → tags.tag
 *   "section"     → sections.heading (exact or fuzzy)
 *   "content"     → sections.body (fuzzy/regex only)
 *   "level"       → sections.level
 *   "created"     → d.created_at
 *   "modified"    → d.modified_at
 *   "title"       → d.title
 *   "word_count"  → d.word_count
 *   anything else → frontmatter.key = field, frontmatter.value_text op value
 */

import type { FilterNode } from "./parser.js";

export interface CompiledQuery {
  where: string;
  params: (string | number | null)[];
}

// =============================================================================
// Date / duration helpers
// =============================================================================

/** Parse "7d", "2w", "3m", "1y" to an absolute ISO date string. */
function parseDuration(value: string): string | null {
  const m = /^(\d+)([dwmy])$/i.exec(value);
  if (!m) return null;
  const n = parseInt(m[1]!, 10);
  const unit = m[2]!.toLowerCase();
  const now = new Date();
  switch (unit) {
    case "d": now.setDate(now.getDate() - n); break;
    case "w": now.setDate(now.getDate() - n * 7); break;
    case "m": now.setMonth(now.getMonth() - n); break;
    case "y": now.setFullYear(now.getFullYear() - n); break;
  }
  return now.toISOString().slice(0, 10);
}

/** Resolve a value to a date string, either by parsing duration or treating as ISO date. */
function resolveDate(value: string): string {
  return parseDuration(value) ?? value;
}

// =============================================================================
// Compiler
// =============================================================================

type Ctx = { params: (string | number | null)[] };

function param(ctx: Ctx, value: string | number | null): string {
  ctx.params.push(value);
  return "?";
}

const DATE_FIELDS = new Set(["created", "modified", "created_at", "modified_at"]);
const DOC_FIELDS: Record<string, string> = {
  created:    "d.created_at",
  modified:   "d.modified_at",
  created_at: "d.created_at",
  modified_at: "d.modified_at",
  title:      "d.title",
  word_count: "d.word_count",
  path:       "d.path",
  collection: "d.collection",
};

function compileNode(node: FilterNode, ctx: Ctx): string {
  switch (node.type) {
    case "AND": return `(${compileNode(node.left, ctx)} AND ${compileNode(node.right, ctx)})`;
    case "OR":  return `(${compileNode(node.left, ctx)} OR ${compileNode(node.right, ctx)})`;
    case "NOT": return `NOT (${compileNode(node.operand, ctx)})`;

    case "CMP": return compileCmp(node.field, node.op, node.value, ctx);

    case "MISSING":
      // Field absent from frontmatter entirely
      return `NOT EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, node.field)})`;

    case "EMPTY":
      if (node.field === "section" || node.field === "sections") {
        // Files that have at least one section with empty body
        return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND (s.body IS NULL OR s.body = ''))`;
      }
      // Frontmatter field that is present but empty
      return `EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, node.field)} AND (fm.value_text IS NULL OR fm.value_text = ''))`;

    case "NO":
      if (node.field === "headings" || node.field === "heading") {
        // Files with no headings at all
        return `NOT EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.level > 0)`;
      }
      if (node.field === "level") {
        const lvl = node.value ? parseInt(node.value, 10) : null;
        if (lvl !== null) {
          return `NOT EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.level = ${param(ctx, lvl)})`;
        }
        return `NOT EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.level > 0)`;
      }
      // no:tag, no:property — absence of a frontmatter key
      return `NOT EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, node.field)})`;
  }
}

function compileCmp(
  field: string,
  op: "=" | "~=" | "<" | ">" | "regex",
  value: string,
  ctx: Ctx
): string {
  // ---- tag field ----
  if (field === "tag" || field === "tags") {
    switch (op) {
      case "=":  return `EXISTS (SELECT 1 FROM tags t WHERE t.doc_id = d.id AND t.tag = ${param(ctx, value)})`;
      case "~=": return `EXISTS (SELECT 1 FROM tags t WHERE t.doc_id = d.id AND t.tag LIKE ${param(ctx, `%${value}%`)})`;
      case "regex": return `EXISTS (SELECT 1 FROM tags t WHERE t.doc_id = d.id AND t.tag REGEXP ${param(ctx, value)})`;
      default:   throw new Error(`Operator ${op} not supported for tag field`);
    }
  }

  // ---- section heading field ----
  if (field === "section") {
    switch (op) {
      case "=":  return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.heading = ${param(ctx, value)})`;
      case "~=": return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.heading LIKE ${param(ctx, `%${value}%`)})`;
      case "regex": return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.heading REGEXP ${param(ctx, value)})`;
      default:   throw new Error(`Operator ${op} not supported for section field`);
    }
  }

  // ---- content (section body) field ----
  if (field === "content") {
    switch (op) {
      case "~=":    return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.body LIKE ${param(ctx, `%${value}%`)})`;
      case "regex": return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.body REGEXP ${param(ctx, value)})`;
      default:      throw new Error(`Operator ${op} not supported for content field`);
    }
  }

  // ---- section level field ----
  if (field === "level") {
    const lvl = parseInt(value, 10);
    switch (op) {
      case "=": return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.level = ${param(ctx, lvl)})`;
      case "<": return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.level < ${param(ctx, lvl)})`;
      case ">": return `EXISTS (SELECT 1 FROM sections s WHERE s.doc_id = d.id AND s.level > ${param(ctx, lvl)})`;
      default:  throw new Error(`Operator ${op} not supported for level field`);
    }
  }

  // ---- document-level fields (title, path, created, modified, word_count) ----
  if (field in DOC_FIELDS) {
    const col = DOC_FIELDS[field]!;
    const isDate = DATE_FIELDS.has(field);
    const resolvedValue = isDate ? resolveDate(value) : value;

    switch (op) {
      case "=":  return `${col} = ${param(ctx, resolvedValue)}`;
      case "<":  return `${col} < ${param(ctx, resolvedValue)}`;
      case ">":  return `${col} > ${param(ctx, resolvedValue)}`;
      case "~=": return `${col} LIKE ${param(ctx, `%${resolvedValue}%`)}`;
      case "regex": return `${col} REGEXP ${param(ctx, resolvedValue)}`;
    }
  }

  // ---- generic frontmatter key ----
  switch (op) {
    case "=":
      return `EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, field)} AND fm.value_text = ${param(ctx, value)})`;
    case "~=":
      return `EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, field)} AND fm.value_text LIKE ${param(ctx, `%${value}%`)})`;
    case "<":
      return `EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, field)} AND (fm.value_num < ${param(ctx, parseFloat(value))} OR fm.value_date < ${param(ctx, resolveDate(value))}))`;
    case ">":
      return `EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, field)} AND (fm.value_num > ${param(ctx, parseFloat(value))} OR fm.value_date > ${param(ctx, resolveDate(value))}))`;
    case "regex":
      return `EXISTS (SELECT 1 FROM frontmatter fm WHERE fm.doc_id = d.id AND fm.key = ${param(ctx, field)} AND fm.value_text REGEXP ${param(ctx, value)})`;
  }
}

export function compileFilter(node: FilterNode): CompiledQuery {
  const ctx: Ctx = { params: [] };
  const where = compileNode(node, ctx);
  return { where, params: ctx.params };
}

/**
 * Register a simple REGEXP UDF on a better-sqlite3 database.
 * Call this once after opening the DB.
 */
export function registerRegexpUDF(db: { function: (name: string, fn: (...args: unknown[]) => unknown) => void }): void {
  db.function("REGEXP", (pattern: unknown, value: unknown): number => {
    try {
      return new RegExp(String(pattern)).test(String(value ?? "")) ? 1 : 0;
    } catch {
      return 0;
    }
  });
}
