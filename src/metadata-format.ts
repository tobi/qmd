/**
 * QMD Metadata Format - Plain-text rendering of metadata discovery results.
 *
 * Shared by the CLI and the MCP `metadata` tool so both print one shape: a
 * filter line when a filter narrowed the documents, a header per key, a body
 * per type, and footers naming the remainder and the options that reach it
 * whenever a value list or the key list is windowed. The empty states render
 * here too, so the filter line survives them.
 *
 * String values print bare only when the bare form is unambiguous. A value
 * that is empty, padded, contains a quote, a backslash, a control character,
 * a line separator, or a delimiter of the compact `value (count), ...` list,
 * or that reads as a JSON number, boolean, or null, or as the list's
 * remainder tail, is printed as a JSON string, so what the reader sees is
 * exactly the value.
 */

import type {
  ListMetadataResult,
  MetadataKeySummary,
  MetadataKeyTypeSummary,
  MetadataValueCount,
} from "./metadata-store.js";

export interface MetadataFormatColors {
  reset: string;
  dim: string;
  bold: string;
  cyan: string;
}

export interface FormatMetadataOptions {
  /** Show which collections contribute each type on type-split keys. */
  showCollections?: boolean;
  /** How the caller widens or pages the value window, e.g. `--value-limit <n>, --value-offset <n>, or --all-values`. */
  valueWindowHint: string;
  /** How the caller widens or pages the key window, e.g. `--key-limit <n>, --key-offset <n>, or --all-keys`. */
  keyWindowHint: string;
  /** The key offset the caller requested, and how the caller spells that option, e.g. `--key-offset`. */
  keyOffset: number;
  keyOffsetLabel: string;
  /** Printed in place of the key blocks when no metadata entry is in the result at all. */
  emptyMessage: string;
  /** ANSI sequences; omit for plain text. */
  colors?: MetadataFormatColors;
}

const NO_COLORS: MetadataFormatColors = { reset: "", dim: "", bold: "", cyan: "" };

/**
 * Render the whole result: the filter line when a filter is in play, then
 * one block per key separated by blank lines and the key footer when keys
 * were left out of the window, or the one-line empty state when there are no
 * blocks to show.
 */
export function formatMetadataKeySummaries(result: ListMetadataResult, options: FormatMetadataOptions): string {
  const colors = options.colors ?? NO_COLORS;
  const blocks: string[] = [];

  if (result.filteredDocuments !== undefined) {
    blocks.push(`${colors.dim}filter:${colors.reset} ${formatCount(result.filteredDocuments)} of ${formatCount(result.documents)} documents`);
  }

  if (result.totalKeys === 0) {
    blocks.push(`${colors.dim}${options.emptyMessage}${colors.reset}`);
  } else if (result.keys.length === 0) {
    const keyLabel = result.totalKeys === 1 ? "key" : "keys";
    blocks.push(`${colors.dim}No keys at ${options.keyOffsetLabel} ${formatCount(options.keyOffset)}, ${formatCount(result.totalKeys)} ${keyLabel} in total.${colors.reset}`);
  }

  blocks.push(...result.keys.map(summary => formatMetadataKeySummary(summary, result, options)));

  if (result.remainingKeys > 0) {
    blocks.push(`${colors.dim}${formatCount(result.remainingKeys)} more ${result.remainingKeys === 1 ? "key" : "keys"}, use ${options.keyWindowHint}${colors.reset}`);
  }

  return blocks.join("\n\n");
}

/**
 * One key: a header of name, types, coverage, and distinct count, then the
 * body per type. Coverage is measured against the documents the filter
 * admitted when there is one, otherwise against every active document.
 */
function formatMetadataKeySummary(summary: MetadataKeySummary, result: ListMetadataResult, options: FormatMetadataOptions): string {
  const colors = options.colors ?? NO_COLORS;
  const typeLabel = summary.types.map(typeLabelOf).join(" | ");
  const coverage = `${formatCount(summary.documents)} of ${formatCount(result.filteredDocuments ?? result.documents)} documents`;
  const header = [`${colors.cyan}${colors.bold}${summary.key}${colors.reset}`, `${colors.dim}${typeLabel}${colors.reset}`, coverage];
  const lines: string[] = [];

  if (summary.types.length === 1) {
    const typeSummary = summary.types[0]!;
    if (typeSummary.type !== "boolean") header.push(`${formatCount(typeSummary.distinctValues)} distinct`);
    lines.push(header.join("  "), ...formatTypeBody(typeSummary));
  } else {
    lines.push(header.join("  "), ...formatTypeSplit(summary.types, options));
  }

  const remainingValues = summary.types.reduce((sum, typeSummary) => sum + typeSummary.remainingValues, 0);
  if (remainingValues > 0) {
    lines.push(`${colors.dim}${formatCount(remainingValues)} more ${remainingValues === 1 ? "value" : "values"}, use ${options.valueWindowHint}${colors.reset}`);
  }

  return lines.join("\n");
}

export interface FormatMetadataOverviewOptions {
  /** Active, extracted documents declaring at least one key. */
  documentsWithMetadata: number;
  /** Active documents awaiting extraction, mentioned so the coverage reads honestly. */
  pendingMetadata: number;
  /** Command that shows the rest, e.g. `qmd collection metadata notes`. */
  drillDownHint: string;
  colors?: MetadataFormatColors;
}

/**
 * The `Metadata:` section of `collection show`: a coverage line, then the
 * keys in the result's window as aligned rows with a short value preview,
 * then a pointer at the drill-down when keys were left out. Indented to sit
 * under the other `show` fields.
 */
export function formatMetadataOverview(result: ListMetadataResult, options: FormatMetadataOverviewOptions): string {
  const colors = options.colors ?? NO_COLORS;
  const pendingNote = options.pendingMetadata > 0 ? ` (${formatCount(options.pendingMetadata)} pending extraction)` : "";

  if (result.totalKeys === 0) return `  Metadata: none${pendingNote}`;

  const keyLabel = result.totalKeys === 1 ? "key" : "keys";
  const lines = [`  Metadata: ${formatCount(result.totalKeys)} ${keyLabel}, ${formatCount(options.documentsWithMetadata)} of ${formatCount(result.documents)} documents${pendingNote}`];

  const shownKeys = result.keys;
  const keyWidth = Math.max(...shownKeys.map(summary => summary.key.length));
  const typeWidth = Math.max(...shownKeys.map(summary => summary.types.map(typeLabelOf).join(" | ").length));
  const documentsWidth = Math.max(...shownKeys.map(summary => formatCount(summary.documents).length));
  const distinctWidth = Math.max(...shownKeys.map(summary => formatCount(distinctValuesOf(summary)).length));

  for (const summary of shownKeys) {
    const typeLabel = summary.types.map(typeLabelOf).join(" | ");
    const columns = [
      `${colors.cyan}${summary.key.padEnd(keyWidth)}${colors.reset}`,
      `${colors.dim}${typeLabel.padEnd(typeWidth)}${colors.reset}`,
      `${formatCount(summary.documents).padStart(documentsWidth)} ${documentsLabelOf(summary.documents)}`,
      `${formatCount(distinctValuesOf(summary)).padStart(distinctWidth)} distinct`,
    ];
    const preview = formatValuePreview(summary, options.drillDownHint);
    if (preview) columns.push(preview);
    lines.push(`    ${columns.join("  ")}`);
  }

  if (result.remainingKeys > 0) {
    lines.push(`    ${colors.dim}${formatCount(result.remainingKeys)} more ${result.remainingKeys === 1 ? "key" : "keys"}, see '${options.drillDownHint}'${colors.reset}`);
  }

  return lines.join("\n");
}

/**
 * One-line value preview for the overview row. Strings list the window with
 * a trailing ellipsis when truncated, and nothing at all when every value is
 * unique (a value list would be noise). Numbers give the range, booleans the
 * two counts, and a type conflict points at the drill-down.
 */
function formatValuePreview(summary: MetadataKeySummary, drillDownHint: string): string {
  if (summary.types.length > 1) {
    // JSON quoting does not protect an apostrophe inside a shell single quote.
    const match = JSON.stringify({ field: "key", operator: "eq", value: summary.key }).replaceAll("'", "'\\''");
    return `types disagree, see: ${drillDownHint} --match '${match}'`;
  }

  const typeSummary = summary.types[0]!;
  if (typeSummary.type === "boolean") return formatBooleanCounts(typeSummary.values).replace("  ", ", ");
  if (typeSummary.type === "number") {
    const range = typeSummary.range!;
    return `${formatValue(range.min)} to ${formatValue(range.max)}, median ${formatValue(range.median)}`;
  }
  if (typeSummary.distinctValues === typeSummary.documents) return "";

  const preview = typeSummary.values.map(count => `${formatValue(count.value)} (${formatCount(count.documents)})`);
  if (typeSummary.remainingValues > 0) preview.push("...");
  return preview.join(", ");
}

function distinctValuesOf(summary: MetadataKeySummary): number {
  return summary.types.reduce((sum, typeSummary) => sum + typeSummary.distinctValues, 0);
}

/** Body for a key with one type: vertical values for strings, a range for numbers, one line for booleans. */
function formatTypeBody(typeSummary: MetadataKeyTypeSummary): string[] {
  if (typeSummary.type === "boolean") return [`  ${formatBooleanCounts(typeSummary.values)}`];

  if (typeSummary.type === "number") {
    const lines = [`  ${formatRange(typeSummary)}`];
    if (typeSummary.values.length > 0) lines.push(`  ${formatInlineValues(typeSummary).join("  ")}`);
    return lines;
  }

  const valueWidth = Math.max(...typeSummary.values.map(count => formatValue(count.value).length));
  const countWidth = Math.max(...typeSummary.values.map(count => formatCount(count.documents).length));
  return typeSummary.values.map(count => `  ${formatValue(count.value).padEnd(valueWidth)}  ${formatCount(count.documents).padStart(countWidth)}`);
}

/**
 * Body for a key whose documents disagree on type: one line per type with
 * its own document count and a compact value summary, plus the contributing
 * collections when the view spans more than one.
 */
function formatTypeSplit(typeSummaries: MetadataKeyTypeSummary[], options: FormatMetadataOptions): string[] {
  const typeWidth = Math.max(...typeSummaries.map(typeSummary => typeSummary.type.length));
  const documentsWidth = Math.max(...typeSummaries.map(typeSummary => formatCount(typeSummary.documents).length));

  const rows = typeSummaries.map(typeSummary => {
    let valuesSummary: string;
    if (typeSummary.type === "boolean") valuesSummary = formatBooleanCounts(typeSummary.values);
    else if (typeSummary.type === "number") valuesSummary = formatRange(typeSummary);
    else {
      const inlineValues = typeSummary.values.map(count => `${formatValue(count.value)} (${formatCount(count.documents)})`);
      if (typeSummary.remainingValues > 0) inlineValues.push(`${formatCount(typeSummary.remainingValues)} more`);
      valuesSummary = inlineValues.join(", ");
    }
    const row = `  ${typeSummary.type.padEnd(typeWidth)}  ${formatCount(typeSummary.documents).padStart(documentsWidth)} ${documentsLabelOf(typeSummary.documents)}  ${valuesSummary}`;
    return { row, collections: typeSummary.collections.join(", ") };
  });

  if (!options.showCollections) return rows.map(({ row }) => row);

  const rowWidth = Math.max(...rows.map(({ row }) => row.length));
  return rows.map(({ row, collections }) => `${row.padEnd(rowWidth)}  ${collections}`);
}

function typeLabelOf(typeSummary: MetadataKeyTypeSummary): string {
  return typeSummary.multiValued ? `${typeSummary.type}[]` : typeSummary.type;
}

function formatRange(typeSummary: MetadataKeyTypeSummary): string {
  const range = typeSummary.range!;
  return `min ${formatValue(range.min)}  median ${formatValue(range.median)}  max ${formatValue(range.max)}`;
}

/**
 * Numbers enumerate inline as `value (documents)`. When the whole
 * distribution fits it reads in value order; a truncated window keeps the
 * requested order so the most common values stay visible.
 */
function formatInlineValues(typeSummary: MetadataKeyTypeSummary): string[] {
  const values = typeSummary.remainingValues === 0
    ? [...typeSummary.values].sort((a, b) => Number(a.value) - Number(b.value))
    : typeSummary.values;
  return values.map(count => `${formatValue(count.value)} (${formatCount(count.documents)})`);
}

function formatBooleanCounts(values: MetadataValueCount[]): string {
  const trueCount = values.find(count => count.value === true);
  const falseCount = values.find(count => count.value === false);
  const parts: string[] = [];
  if (trueCount) parts.push(`true ${formatCount(trueCount.documents)}`);
  if (falseCount) parts.push(`false ${formatCount(falseCount.documents)}`);
  return parts.join("  ");
}

/** Padded so `doc` and `docs` rows stay column-aligned. */
function documentsLabelOf(documents: number): string {
  return documents === 1 ? "doc " : "docs";
}

function formatValue(value: string | number | boolean): string {
  if (typeof value !== "string") return String(value);
  return isUnambiguousBare(value) ? value : formatQuoted(value);
}

// Controls (C0, DEL, C1) and the line and paragraph separators, none of which
// print as themselves.
const CONTROL_CHARACTERS = /[\u0000-\u001f\u007f-\u009f\u2028\u2029]/u;
// Whitespace other than one interior ASCII space between other characters.
const AMBIGUOUS_WHITESPACE = /^\s|\s$|\s\s|[^\S ]/u;
// The compact list joins `value (count)` items with ", " and ends with a
// remainder tail, so a bare string can neither contain the delimiters nor
// read as a tail. One rule for both layouts, so a value never prints two ways.
const LIST_DELIMITERS = /[,()]/u;
const LIST_TAIL = /^(?:\.\.\.|\d+ more)$/u;

/** True when printing the string as-is cannot be mistaken for another value or for layout. */
function isUnambiguousBare(text: string): boolean {
  if (text === "" || text.includes('"') || text.includes("\\")) return false;
  if (CONTROL_CHARACTERS.test(text) || AMBIGUOUS_WHITESPACE.test(text)) return false;
  if (LIST_DELIMITERS.test(text) || LIST_TAIL.test(text)) return false;
  return !readsAsJsonLiteral(text);
}

/** A string that JSON would parse as something other than a string, e.g. `42`, `true`, `null`, `[1]`. */
function readsAsJsonLiteral(text: string): boolean {
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
}

/** A JSON string, with every character in CONTROL_CHARACTERS escaped, not only the ones JSON.stringify escapes. */
function formatQuoted(text: string): string {
  return JSON.stringify(text).replace(
    /[\u007f-\u009f\u2028\u2029]/gu,
    character => `\\u${character.codePointAt(0)!.toString(16).padStart(4, "0")}`,
  );
}

function formatCount(count: number): string {
  return count.toLocaleString("en-US");
}
