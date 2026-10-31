/**
 * QMD Metadata Filter - Recursive predicate AST, strict runtime validation,
 * and parameterized SQL compilation.
 *
 * The predicate has one canonical, `operator`-discriminated recursive shape
 * shared by every public search surface (CLI, SDK, MCP, HTTP):
 *
 *   { "operator": "and", "operands": [ ... ] }
 *   { "operator": "not", "operand": { ... } }
 *   { "field": "status", "operator": "eq", "value": "published" }
 *   { "field": "topics", "operator": "prefix", "value": "sql", "caseInsensitive": true }
 *
 * A condition tests one field of the record under evaluation, named by `field`,
 * against `value` using `operator`. The grammar evaluates two kinds of record:
 *
 * - A document, whose fields are its metadata keys. This is the filter
 *   (`MetadataFilter`) every search surface accepts, and it compiles to one
 *   subquery per condition over `document_metadata_values`, correlated for
 *   a few candidates or a document set for the corpus (FilterScope).
 * - A metadata entry (one row of that table), whose fields are `key` and
 *   `value`. This is the match (`MetadataMatch`) metadata discovery accepts,
 *   and it compiles to a predicate over one row.
 *
 * Both are `MetadataPredicate`, the one recursive grammar, parameterized by
 * the conditions the record admits.
 *
 * Comparison, membership, and text operators are typed by their operand: a
 * string operand only ever compares against string values, a number against
 * numbers, a boolean against booleans, and a mismatch never matches. Every
 * user value binds as a parameter. Metadata keys and values are data, never
 * SQL.
 */

import type { MetadataScalar, MetadataScalarArray, MetadataValueType } from "./metadata.js";
import { METADATA_LIMITS } from "./metadata.js";

// =============================================================================
// Public types
// =============================================================================

/**
 * The recursive grammar: a condition, or `and`/`or`/`not` over predicates.
 * `Condition` is the set of conditions the record under evaluation admits.
 */
export type MetadataPredicate<Condition> =
  | Condition
  | MetadataPredicateGroup<Condition>
  | MetadataPredicateNegation<Condition>;

export interface MetadataPredicateGroup<Condition> {
  operator: "and" | "or";
  operands: readonly MetadataPredicate<Condition>[];
}

export interface MetadataPredicateNegation<Condition> {
  operator: "not";
  operand: MetadataPredicate<Condition>;
}

/** A predicate over a document, whose fields are its metadata keys. */
export type MetadataFilter = MetadataPredicate<MetadataCondition>;
export type MetadataFilterGroup = MetadataPredicateGroup<MetadataCondition>;
export type MetadataFilterNegation = MetadataPredicateNegation<MetadataCondition>;

/** A predicate over a metadata entry, whose fields are `key` and `value`. */
export type MetadataMatch = MetadataPredicate<MetadataEntryCondition>;

/** The fields of a metadata entry. */
export type MetadataEntryField = "key" | "value";

/**
 * The conditions every record admits, over the fields it has.
 * `caseInsensitive` folds ASCII letters on both sides and is accepted only
 * where the operand is a string or an array of strings.
 */
type SharedMetadataCondition<Field extends string> =
  | { field: Field; operator: "eq" | "ne"; value: MetadataScalar; caseInsensitive?: boolean }
  | { field: Field; operator: "gt" | "gte" | "lt" | "lte"; value: string | number; caseInsensitive?: boolean }
  | { field: Field; operator: "in" | "nin"; value: MetadataScalarArray; caseInsensitive?: boolean }
  | { field: Field; operator: "contains" | "prefix" | "suffix"; value: string; caseInsensitive?: boolean }
  | { field: Field; operator: "type"; value: MetadataValueType };

/**
 * One condition over a document. `all` and `exists` speak about the set of
 * values a document holds under a key, so only a document admits them.
 */
export type MetadataCondition =
  | SharedMetadataCondition<string>
  | { field: string; operator: "all"; value: MetadataScalarArray; caseInsensitive?: boolean }
  | { field: string; operator: "exists"; value: boolean };

/** One condition over a metadata entry: its `key` (a string) or its `value` (typed). */
export type MetadataEntryCondition = SharedMetadataCondition<MetadataEntryField>;

export interface CompiledMetadataFilter {
  sql: string;
  params: (string | number)[];
}

/** Which record a filter is evaluated against. */
export type MetadataRecordType = "document" | "entry";

const ENTRY_FIELDS: ReadonlySet<string> = new Set<MetadataEntryField>(["key", "value"]);

/** Raised by parseMetadataFilter and parseMetadataMatch with the JSON path of the failing node. */
export class MetadataFilterError extends Error {
  readonly path: string;
  /** The message without the `Invalid metadata ... at` prefix. */
  readonly detail: string;

  constructor(path: string, detail: string, recordType: MetadataRecordType = "document") {
    super(`Invalid metadata ${recordType === "entry" ? "match" : "filter"} at ${path}: ${detail}`);
    this.name = "MetadataFilterError";
    this.path = path;
    this.detail = detail;
  }
}

// =============================================================================
// Limits
// =============================================================================

/** Defensive limits for recursive filters from untrusted callers. */
export const METADATA_FILTER_LIMITS = {
  maxDepth: 16,
  maxNodes: 256,
  maxGroupOperands: 32,
  maxMembershipValues: 64,
  maxKeyBytes: METADATA_LIMITS.maxKeyBytes,
  maxStringLength: METADATA_LIMITS.maxStringLength,
} as const;

const GROUP_OPERATORS = new Set(["and", "or"]);
const COMPARISON_OPERATORS = new Set(["eq", "ne", "gt", "gte", "lt", "lte"]);
const ORDERED_OPERATORS = new Set(["gt", "gte", "lt", "lte"]);
const MEMBERSHIP_OPERATORS = new Set(["in", "nin", "all"]);
const TEXT_OPERATORS = new Set(["contains", "prefix", "suffix"]);
const CONDITION_OPERATORS = new Set([...COMPARISON_OPERATORS, ...MEMBERSHIP_OPERATORS, ...TEXT_OPERATORS, "type", "exists"]);
const ALL_OPERATORS = [...GROUP_OPERATORS, "not", ...CONDITION_OPERATORS];
const VALUE_TYPES: readonly MetadataValueType[] = ["string", "number", "boolean"];

// =============================================================================
// Validation
// =============================================================================

type FilterParseState = { nodes: number; recordType: MetadataRecordType };

/** Conditions whose operand can be a string, and so can fold case. */
type CaseFoldableCondition = Exclude<MetadataCondition, { operator: "type" | "exists" }>;

/**
 * Strictly validate an untrusted value as a MetadataFilter.
 * Rejects unknown operators, unknown properties, operator-incompatible values,
 * and inputs exceeding METADATA_FILTER_LIMITS. Canonicalizes membership value
 * arrays by de-duplicating while preserving order.
 */
export function parseMetadataFilter(input: unknown): MetadataFilter {
  const state: FilterParseState = { nodes: 0, recordType: "document" };
  return parseFilterNode(input, "$", 1, state);
}

/**
 * Strictly validate an untrusted value as a match over metadata entries. Same
 * grammar and limits as parseMetadataFilter. A condition's `field` names a field
 * of the entry, `key` or `value`. `exists` and `all` have no meaning for a
 * single entry and are rejected.
 */
export function parseMetadataMatch(input: unknown): MetadataMatch {
  const state: FilterParseState = { nodes: 0, recordType: "entry" };
  try {
    // The walker enforces the entry rules through `state.recordType`, so
    // every condition it returns is a MetadataEntryCondition.
    return parseFilterNode(input, "$", 1, state) as MetadataMatch;
  } catch (error) {
    if (error instanceof MetadataFilterError) throw new MetadataFilterError(error.path, error.detail, "entry");
    throw error;
  }
}

function parseFilterNode(input: unknown, path: string, depth: number, state: FilterParseState): MetadataFilter {
  if (depth > METADATA_FILTER_LIMITS.maxDepth) {
    throw new MetadataFilterError(path, `exceeds maximum nesting depth of ${METADATA_FILTER_LIMITS.maxDepth}`);
  }

  state.nodes += 1;
  if (state.nodes > METADATA_FILTER_LIMITS.maxNodes) {
    throw new MetadataFilterError(path, `exceeds maximum of ${METADATA_FILTER_LIMITS.maxNodes} nodes`);
  }

  if (typeof input !== "object" || input === null || Array.isArray(input)) {
    throw new MetadataFilterError(path, "each filter node must be an object");
  }

  const node = input as Record<string, unknown>;
  const operator = node["operator"];
  if (typeof operator !== "string") {
    throw new MetadataFilterError(path, "missing 'operator' property");
  }

  if (GROUP_OPERATORS.has(operator)) {
    return parseFilterGroup(node, operator as "and" | "or", path, depth, state);
  }
  if (operator === "not") {
    return parseFilterNegation(node, path, depth, state);
  }
  if (CONDITION_OPERATORS.has(operator)) {
    return parseFilterCondition(node, operator, path, state.recordType);
  }

  throw new MetadataFilterError(path, `unknown operator '${operator}' — expected one of: ${ALL_OPERATORS.join(", ")}`);
}

function parseFilterGroup(
  node: Record<string, unknown>,
  operator: "and" | "or",
  path: string,
  depth: number,
  state: FilterParseState,
): MetadataFilterGroup {
  rejectUnknownProperties(node, ["operator", "operands"], path);

  const operands = node["operands"];
  if (!Array.isArray(operands)) {
    throw new MetadataFilterError(path, `'${operator}' requires an 'operands' array`);
  }
  if (operands.length === 0) {
    throw new MetadataFilterError(path, `'${operator}' requires a non-empty 'operands' array`);
  }
  if (operands.length > METADATA_FILTER_LIMITS.maxGroupOperands) {
    throw new MetadataFilterError(path, `'${operator}' exceeds maximum of ${METADATA_FILTER_LIMITS.maxGroupOperands} operands`);
  }

  return {
    operator,
    operands: operands.map((operand, index) =>
      parseFilterNode(operand, `${path}.operands[${index}]`, depth + 1, state)),
  };
}

function parseFilterNegation(
  node: Record<string, unknown>,
  path: string,
  depth: number,
  state: FilterParseState,
): MetadataFilterNegation {
  rejectUnknownProperties(node, ["operator", "operand"], path);

  if (!("operand" in node)) {
    throw new MetadataFilterError(path, "'not' requires exactly one 'operand'");
  }

  return {
    operator: "not",
    operand: parseFilterNode(node["operand"], `${path}.operand`, depth + 1, state),
  };
}

function parseFilterCondition(
  node: Record<string, unknown>,
  operator: string,
  path: string,
  recordType: MetadataRecordType,
): MetadataCondition {
  rejectUnknownProperties(node, ["field", "operator", "value", "caseInsensitive"], path);

  const field = node["field"];
  if (typeof field !== "string" || field.length === 0) {
    throw new MetadataFilterError(path, `'${operator}' requires a non-empty string 'field'`);
  }
  if (Buffer.byteLength(field, "utf-8") > METADATA_FILTER_LIMITS.maxKeyBytes) {
    throw new MetadataFilterError(path, `'field' exceeds ${METADATA_FILTER_LIMITS.maxKeyBytes} bytes`);
  }

  if (recordType === "entry") {
    if (!ENTRY_FIELDS.has(field)) {
      throw new MetadataFilterError(path, `'${field}' is not a field of a metadata entry, expected 'key' or 'value'`);
    }
    if (operator === "exists" || operator === "all") {
      throw new MetadataFilterError(path, `'${operator}' has no meaning for a single metadata entry`);
    }
  }

  if (!("value" in node)) {
    throw new MetadataFilterError(path, `'${operator}' requires a 'value'`);
  }
  const value = node["value"];

  const caseInsensitive = node["caseInsensitive"];
  if (caseInsensitive !== undefined && typeof caseInsensitive !== "boolean") {
    throw new MetadataFilterError(`${path}.caseInsensitive`, "'caseInsensitive' must be a boolean");
  }

  if (operator === "exists") {
    if (typeof value !== "boolean") {
      throw new MetadataFilterError(`${path}.value`, "'exists' requires a boolean value");
    }
    rejectCaseInsensitive(caseInsensitive, operator, path);
    return { field, operator, value };
  }

  if (operator === "type") {
    const valueType = VALUE_TYPES.find(candidate => candidate === value);
    if (!valueType) {
      throw new MetadataFilterError(`${path}.value`, `'type' requires one of: ${VALUE_TYPES.join(", ")}`);
    }
    rejectCaseInsensitive(caseInsensitive, operator, path);
    return { field, operator, value: valueType };
  }

  if (TEXT_OPERATORS.has(operator)) {
    const text = parseScalarValue(value, `${path}.value`);
    if (typeof text !== "string" || text.length === 0) {
      throw new MetadataFilterError(`${path}.value`, `'${operator}' requires a non-empty string value`);
    }
    return withCaseFolding({ field, operator: operator as "contains" | "prefix" | "suffix", value: text }, caseInsensitive, path);
  }

  if (MEMBERSHIP_OPERATORS.has(operator)) {
    const values = parseMembershipValues(value, operator, path);
    return withCaseFolding({ field, operator: operator as "in" | "nin" | "all", value: values }, caseInsensitive, path);
  }

  // Comparison operators: eq, ne, gt, gte, lt, lte.
  const scalar = parseScalarValue(value, `${path}.value`);
  if (ORDERED_OPERATORS.has(operator) && typeof scalar === "boolean") {
    throw new MetadataFilterError(`${path}.value`, `'${operator}' requires a string or number value`);
  }
  return withCaseFolding({ field, operator, value: scalar } as CaseFoldableCondition, caseInsensitive, path);
}

/** Attach `caseInsensitive` when given, which only string operands accept. */
function withCaseFolding(condition: CaseFoldableCondition, caseInsensitive: unknown, path: string): MetadataCondition {
  if (caseInsensitive === undefined) return condition;

  const operand: unknown = condition.value;
  const isText = typeof operand === "string";
  const isTextArray = Array.isArray(operand) && operand.every(element => typeof element === "string");
  if (!isText && !isTextArray) {
    throw new MetadataFilterError(`${path}.caseInsensitive`, "'caseInsensitive' applies to string values only");
  }

  return { ...condition, caseInsensitive: caseInsensitive === true };
}

function rejectCaseInsensitive(caseInsensitive: unknown, operator: string, path: string): void {
  if (caseInsensitive === undefined) return;
  throw new MetadataFilterError(`${path}.caseInsensitive`, `'caseInsensitive' does not apply to '${operator}'`);
}

function parseMembershipValues(value: unknown, operator: string, path: string): MetadataScalarArray {
  if (!Array.isArray(value)) {
    throw new MetadataFilterError(`${path}.value`, `'${operator}' requires an array value`);
  }
  if (value.length === 0) {
    throw new MetadataFilterError(`${path}.value`, `'${operator}' requires a non-empty array value`);
  }
  if (value.length > METADATA_FILTER_LIMITS.maxMembershipValues) {
    throw new MetadataFilterError(`${path}.value`, `'${operator}' exceeds maximum of ${METADATA_FILTER_LIMITS.maxMembershipValues} values`);
  }

  const scalars = value.map((element, index) => parseScalarValue(element, `${path}.value[${index}]`));

  // Narrow each homogeneous case explicitly so the public array union remains
  // precise without discarding type evidence through chained assertions.
  if (scalars.every((scalar): scalar is string => typeof scalar === "string")) {
    return Array.from(new Set(scalars));
  }
  if (scalars.every((scalar): scalar is number => typeof scalar === "number")) {
    return Array.from(new Set(scalars));
  }
  if (scalars.every((scalar): scalar is boolean => typeof scalar === "boolean")) {
    return Array.from(new Set(scalars));
  }
  throw new MetadataFilterError(`${path}.value`, `'${operator}' requires a homogeneous array of one scalar type`);
}

function parseScalarValue(value: unknown, path: string): MetadataScalar {
  if (typeof value === "string") {
    if (value.length > METADATA_FILTER_LIMITS.maxStringLength) {
      throw new MetadataFilterError(path, `string exceeds ${METADATA_FILTER_LIMITS.maxStringLength} characters`);
    }
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new MetadataFilterError(path, "numbers must be finite");
    }
    return value;
  }
  if (typeof value === "boolean") return value;
  throw new MetadataFilterError(path, "expected a string, number, or boolean");
}

function rejectUnknownProperties(node: Record<string, unknown>, allowed: string[], path: string): void {
  for (const property of Object.keys(node)) {
    if (!allowed.includes(property)) {
      throw new MetadataFilterError(path, `unknown property '${property}' — allowed: ${allowed.join(", ")}`);
    }
  }
}

// =============================================================================
// SQL compilation
// =============================================================================

/**
 * The documents a compiled filter is applied to, which decides the SQL each
 * condition takes.
 *
 * - `candidates`: a few documents, as when search checks its results. Each
 *   condition is a correlated `EXISTS` that seeks the document's own rows.
 * - `corpus`: every document, as when discovery counts them. Each condition
 *   is an uncorrelated `IN (SELECT document_id ...)` that SQLite evaluates
 *   once per statement and probes per document, where a correlated predicate
 *   would be re-run for every metadata row the statement joins.
 */
export type FilterScope = "candidates" | "corpus";

/**
 * Compile a validated filter into one parameterized SQL predicate over a
 * documents-table alias (e.g. `d`). All keys and values are bound parameters.
 * The caller is responsible for restricting the surrounding query to active
 * documents with current, error-free metadata extraction.
 */
export function compileMetadataFilter(filter: MetadataFilter, documentsAlias: string, scope: FilterScope = "candidates"): CompiledMetadataFilter {
  const params: (string | number)[] = [];
  const sql = compileFilterNode(filter, { alias: documentsAlias, scope }, params);
  return { sql, params };
}

/** The documents-table alias a filter tests and the scope it is applied to. */
interface FilterTarget {
  alias: string;
  scope: FilterScope;
}

/**
 * Compile a validated match into one parameterized SQL predicate over a
 * `document_metadata_values` row alias (e.g. `mv`). A condition's `field` names
 * the field of the entry its operand is tested against: `key`, which is always
 * a string, or `value`, which is typed by `value_type`. The caller is
 * responsible for scoping the rows to eligible documents.
 */
export function compileMetadataMatch(match: MetadataMatch, valuesAlias: string): CompiledMetadataFilter {
  const params: (string | number)[] = [];
  const sql = compileMatchNode(match, valuesAlias, params);
  return { sql, params };
}

/** The typed columns a condition's operand is tested against. */
interface ValueColumns {
  type: string;
  text: string;
  number: string;
  boolean: string;
}

/** The values of a document, as seen from inside a correlated subquery. */
const DOCUMENT_VALUE_COLUMNS: ValueColumns = {
  type: "mv.value_type",
  text: "mv.text_value",
  number: "mv.number_value",
  boolean: "mv.boolean_value",
};

function compileFilterNode(filter: MetadataFilter, target: FilterTarget, params: (string | number)[]): string {
  const columns = DOCUMENT_VALUE_COLUMNS;

  switch (filter.operator) {
    case "and":
    case "or": {
      const joiner = filter.operator === "and" ? " AND " : " OR ";
      return `(${filter.operands.map(operand => compileFilterNode(operand, target, params)).join(joiner)})`;
    }

    case "not":
      return `NOT ${compileFilterNode(filter.operand, target, params)}`;

    case "exists":
      params.push(filter.field);
      return filter.value
        ? buildConditionSql(target, "byDocument")
        : `NOT ${buildConditionSql(target, "byDocument")}`;

    case "type":
      params.push(filter.field, filter.value);
      return buildConditionSql(target, "byDocument", `${columns.type} = ?`);

    case "eq":
    case "gt":
    case "gte":
    case "lt":
    case "lte": {
      const sqlOperator = { eq: "=", gt: ">", gte: ">=", lt: "<", lte: "<=" }[filter.operator];
      const columnSql = buildOperandColumnSql(columns, filter.value, filter.caseInsensitive);
      params.push(filter.field, bindOperand(filter.value, filter.caseInsensitive));
      return buildConditionSql(
        target,
        filter.operator === "eq" ? equalitySeekOf(filter.caseInsensitive) : "byDocument",
        `${columns.type} = '${valueTypeOf(filter.value)}' AND ${columnSql} ${sqlOperator} ?`,
      );
    }

    case "ne": {
      // Key must have at least one same-type value, and no same-type value
      // may equal the operand. Missing keys and type mismatches do not match.
      const valueType = valueTypeOf(filter.value);
      const columnSql = buildOperandColumnSql(columns, filter.value, filter.caseInsensitive);
      params.push(filter.field);
      const presentSql = buildConditionSql(target, "byDocument", `${columns.type} = '${valueType}'`);
      params.push(filter.field, bindOperand(filter.value, filter.caseInsensitive));
      const equalSql = buildConditionSql(target, equalitySeekOf(filter.caseInsensitive), `${columns.type} = '${valueType}' AND ${columnSql} = ?`);
      return `(${presentSql} AND NOT ${equalSql})`;
    }

    case "in":
    case "nin": {
      const valueType = valueTypeOf(filter.value[0]!);
      const columnSql = buildOperandColumnSql(columns, filter.value[0]!, filter.caseInsensitive);
      const placeholders = filter.value.map(() => "?").join(", ");
      const operands = filter.value.map(element => bindOperand(element, filter.caseInsensitive));

      if (filter.operator === "in") {
        params.push(filter.field, ...operands);
        return buildConditionSql(target, equalitySeekOf(filter.caseInsensitive), `${columns.type} = '${valueType}' AND ${columnSql} IN (${placeholders})`);
      }

      params.push(filter.field);
      const presentSql = buildConditionSql(target, "byDocument", `${columns.type} = '${valueType}'`);
      params.push(filter.field, ...operands);
      const memberSql = buildConditionSql(target, equalitySeekOf(filter.caseInsensitive), `${columns.type} = '${valueType}' AND ${columnSql} IN (${placeholders})`);
      return `(${presentSql} AND NOT ${memberSql})`;
    }

    case "all": {
      const valueType = valueTypeOf(filter.value[0]!);
      const columnSql = buildOperandColumnSql(columns, filter.value[0]!, filter.caseInsensitive);
      const memberSqls = filter.value.map(element => {
        params.push(filter.field, bindOperand(element, filter.caseInsensitive));
        return buildConditionSql(target, equalitySeekOf(filter.caseInsensitive), `${columns.type} = '${valueType}' AND ${columnSql} = ?`);
      });
      return `(${memberSqls.join(" AND ")})`;
    }

    case "contains":
    case "prefix":
    case "suffix": {
      params.push(filter.field);
      const columnSql = buildOperandColumnSql(columns, filter.value, filter.caseInsensitive);
      const textSql = compileTextTestSql(filter.operator, filter.value, columnSql, filter.caseInsensitive, params);
      return buildConditionSql(target, "byDocument", `${columns.type} = 'string' AND ${textSql}`);
    }
  }
}

/**
 * How a `candidates` subquery reaches one document's rows for a key. The
 * covering indexes are (key, <value>, document_id), so a condition that pins
 * the value column with an equality is a single probe by value. Any other
 * condition would walk the key's whole value range once per document (the
 * stats-free planner assumes a range is narrow), so it seeks the document's
 * own rows through the primary key instead. A `corpus` subquery runs once,
 * so it always takes the covering index.
 */
type RowSeek = "byValue" | "byDocument";

/** An equality pins the value column unless it is folded, `lower()` is opaque to the index. */
function equalitySeekOf(caseInsensitive: boolean | undefined): RowSeek {
  return caseInsensitive ? "byDocument" : "byValue";
}

/**
 * One condition over the document's rows for the key, whose parameter the
 * caller binds before `conditionSql`'s. In the `candidates` scope the unary
 * `+` on the key term hides it from the planner, which leaves `document_id`
 * as the only indexable term and makes the seek independent of `sqlite_stat1`.
 */
function buildConditionSql(target: FilterTarget, seek: RowSeek, conditionSql?: string): string {
  if (target.scope === "corpus") {
    const whereSql = conditionSql ? `mv.key = ? AND ${conditionSql}` : "mv.key = ?";
    return `${target.alias}.id IN (SELECT mv.document_id FROM document_metadata_values mv WHERE ${whereSql})`;
  }

  const keyTermSql = seek === "byValue" ? "mv.key = ?" : "+mv.key = ?";
  const whereSql = conditionSql ? `${keyTermSql} AND ${conditionSql}` : keyTermSql;
  return `EXISTS (SELECT 1 FROM document_metadata_values mv WHERE mv.document_id = ${target.alias}.id AND ${whereSql})`;
}

function compileMatchNode(match: MetadataMatch, alias: string, params: (string | number)[]): string {
  switch (match.operator) {
    case "and":
    case "or": {
      const joiner = match.operator === "and" ? " AND " : " OR ";
      return `(${match.operands.map(operand => compileMatchNode(operand, alias, params)).join(joiner)})`;
    }

    case "not":
      return `NOT ${compileMatchNode(match.operand, alias, params)}`;

    default:
      return compileMatchCondition(match, alias, params);
  }
}

/**
 * One condition over one entry. The entry's `key` field is a string with no
 * other typed columns, so a number or boolean operand fails its type guard and
 * matches nothing, the same outcome a type mismatch has in a filter.
 */
function compileMatchCondition(condition: MetadataEntryCondition, alias: string, params: (string | number)[]): string {
  const columns: ValueColumns = condition.field === "key"
    ? { type: "'string'", text: `${alias}.key`, number: "NULL", boolean: "NULL" }
    : { type: `${alias}.value_type`, text: `${alias}.text_value`, number: `${alias}.number_value`, boolean: `${alias}.boolean_value` };

  switch (condition.operator) {
    case "type":
      params.push(condition.value);
      return `${columns.type} = ?`;

    case "eq":
    case "ne":
    case "gt":
    case "gte":
    case "lt":
    case "lte": {
      const sqlOperator = { eq: "=", ne: "<>", gt: ">", gte: ">=", lt: "<", lte: "<=" }[condition.operator];
      const columnSql = buildOperandColumnSql(columns, condition.value, condition.caseInsensitive);
      params.push(bindOperand(condition.value, condition.caseInsensitive));
      return `(${columns.type} = '${valueTypeOf(condition.value)}' AND ${columnSql} ${sqlOperator} ?)`;
    }

    case "in":
    case "nin": {
      const valueType = valueTypeOf(condition.value[0]!);
      const columnSql = buildOperandColumnSql(columns, condition.value[0]!, condition.caseInsensitive);
      const placeholders = condition.value.map(() => "?").join(", ");
      const membership = condition.operator === "in" ? "IN" : "NOT IN";
      params.push(...condition.value.map(element => bindOperand(element, condition.caseInsensitive)));
      return `(${columns.type} = '${valueType}' AND ${columnSql} ${membership} (${placeholders}))`;
    }

    case "contains":
    case "prefix":
    case "suffix": {
      const columnSql = buildOperandColumnSql(columns, condition.value, condition.caseInsensitive);
      const textSql = compileTextTestSql(condition.operator, condition.value, columnSql, condition.caseInsensitive, params);
      return `(${columns.type} = 'string' AND ${textSql})`;
    }
  }
}

/**
 * Substring tests over a string column. Prefix and suffix compare UTF-8 bytes,
 * with the operand's byte length bound from JavaScript: SQLite's text
 * `length()` and `substr()` stop at an embedded NUL, and blobs do not. UTF-8
 * is self-synchronizing, so a byte-prefix (or byte-suffix) of a whole operand
 * is exactly a character-prefix (or -suffix).
 */
function compileTextTestSql(
  operator: "contains" | "prefix" | "suffix",
  text: string,
  columnSql: string,
  caseInsensitive: boolean | undefined,
  params: (string | number)[],
): string {
  const operand = caseInsensitive ? foldAsciiCase(text) : text;

  if (operator === "contains") {
    params.push(operand);
    return `instr(${columnSql}, ?) > 0`;
  }

  params.push(Buffer.byteLength(operand, "utf-8"), operand);
  // SQLite returns NULL for a substring of an empty BLOB. Each primitive
  // must return a boolean so entry matches and their negations partition rows.
  return operator === "prefix"
    ? `COALESCE(substr(CAST(${columnSql} AS BLOB), 1, ?) = CAST(? AS BLOB), 0)`
    : `COALESCE(substr(CAST(${columnSql} AS BLOB), -?) = CAST(? AS BLOB), 0)`;
}

function valueTypeOf(scalar: MetadataScalar): MetadataValueType {
  return typeof scalar as MetadataValueType;
}

/** The typed column an operand compares against, folded when the condition ignores case. */
function buildOperandColumnSql(columns: ValueColumns, scalar: MetadataScalar, caseInsensitive: boolean | undefined): string {
  if (typeof scalar === "number") return columns.number;
  if (typeof scalar === "boolean") return columns.boolean;
  return caseInsensitive ? `lower(${columns.text})` : columns.text;
}

/** Booleans bind as 0/1. Case-insensitive strings bind folded the same way SQLite's `lower()` folds the column. */
function bindOperand(scalar: MetadataScalar, caseInsensitive: boolean | undefined): string | number {
  if (typeof scalar === "boolean") return scalar ? 1 : 0;
  if (typeof scalar === "string" && caseInsensitive) return foldAsciiCase(scalar);
  return scalar;
}

/** SQLite's built-in `lower()` folds ASCII letters only, so the operand is folded over the same range. */
function foldAsciiCase(text: string): string {
  return text.replace(/[A-Z]/g, letter => letter.toLowerCase());
}
