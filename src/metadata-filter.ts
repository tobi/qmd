/**
 * QMD Metadata Filter - Recursive filter AST, strict runtime validation, and
 * parameterized SQL compilation.
 *
 * The filter has one canonical, `operator`-discriminated recursive shape shared
 * by every public search surface (CLI, SDK, MCP, HTTP):
 *
 *   { "operator": "and", "operands": [ ... ] }
 *   { "operator": "not", "operand": { ... } }
 *   { "key": "status", "operator": "eq", "value": "published" }
 *
 * Compilation emits correlated EXISTS/NOT EXISTS subqueries over
 * `document_metadata_values` with every user value bound as a parameter —
 * metadata keys and values are data, never SQL.
 */

import type { MetadataScalar, MetadataScalarArray } from "./metadata.js";
import { METADATA_LIMITS } from "./metadata.js";

// =============================================================================
// Public types
// =============================================================================

export type MetadataFilter =
  | MetadataFilterGroup
  | MetadataFilterNegation
  | MetadataCondition;

export interface MetadataFilterGroup {
  operator: "and" | "or";
  operands: readonly MetadataFilter[];
}

export interface MetadataFilterNegation {
  operator: "not";
  operand: MetadataFilter;
}

export type MetadataCondition =
  | { key: string; operator: "eq" | "ne"; value: MetadataScalar }
  | { key: string; operator: "gt" | "gte" | "lt" | "lte"; value: string | number }
  | { key: string; operator: "in" | "nin" | "all"; value: MetadataScalarArray }
  | { key: string; operator: "exists"; value: boolean };

export interface CompiledMetadataFilter {
  sql: string;
  params: (string | number)[];
}

/** Raised by parseMetadataFilter with the JSON path of the failing node. */
export class MetadataFilterError extends Error {
  readonly path: string;

  constructor(path: string, message: string) {
    super(`Invalid metadata filter at ${path}: ${message}`);
    this.name = "MetadataFilterError";
    this.path = path;
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
const CONDITION_OPERATORS = new Set([...COMPARISON_OPERATORS, ...MEMBERSHIP_OPERATORS, "exists"]);
const ALL_OPERATORS = [...GROUP_OPERATORS, "not", ...CONDITION_OPERATORS];

// =============================================================================
// Validation
// =============================================================================

type FilterParseState = { nodes: number };

/**
 * Strictly validate an untrusted value as a MetadataFilter.
 * Rejects unknown operators, unknown properties, operator-incompatible values,
 * and inputs exceeding METADATA_FILTER_LIMITS. Canonicalizes membership value
 * arrays by de-duplicating while preserving order.
 */
export function parseMetadataFilter(input: unknown): MetadataFilter {
  const state: FilterParseState = { nodes: 0 };
  return parseFilterNode(input, "$", 1, state);
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
    return parseFilterCondition(node, operator, path);
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

function parseFilterCondition(node: Record<string, unknown>, operator: string, path: string): MetadataCondition {
  rejectUnknownProperties(node, ["key", "operator", "value"], path);

  const key = node["key"];
  if (typeof key !== "string" || key.length === 0) {
    throw new MetadataFilterError(path, `'${operator}' requires a non-empty string 'key'`);
  }
  if (Buffer.byteLength(key, "utf-8") > METADATA_FILTER_LIMITS.maxKeyBytes) {
    throw new MetadataFilterError(path, `'key' exceeds ${METADATA_FILTER_LIMITS.maxKeyBytes} bytes`);
  }

  if (!("value" in node)) {
    throw new MetadataFilterError(path, `'${operator}' requires a 'value'`);
  }
  const value = node["value"];

  if (operator === "exists") {
    if (typeof value !== "boolean") {
      throw new MetadataFilterError(`${path}.value`, "'exists' requires a boolean value");
    }
    return { key, operator, value };
  }

  if (MEMBERSHIP_OPERATORS.has(operator)) {
    return {
      key,
      operator: operator as "in" | "nin" | "all",
      value: parseMembershipValues(value, operator, path),
    };
  }

  // Comparison operators: eq, ne, gt, gte, lt, lte.
  const scalar = parseScalarValue(value, `${path}.value`);
  if (ORDERED_OPERATORS.has(operator) && typeof scalar === "boolean") {
    throw new MetadataFilterError(`${path}.value`, `'${operator}' requires a string or number value`);
  }
  return { key, operator, value: scalar } as MetadataCondition;
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

  const elementType = typeof scalars[0];
  if (scalars.some(scalar => typeof scalar !== elementType)) {
    throw new MetadataFilterError(`${path}.value`, `'${operator}' requires a homogeneous array of one scalar type`);
  }

  // Canonicalize: de-duplicate while preserving first-seen order.
  return Array.from(new Set(scalars)) as unknown as MetadataScalarArray;
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
 * Compile a validated filter into one parameterized SQL predicate correlated
 * against a documents-table alias (e.g. `d`). All keys and values are bound
 * parameters. The caller is responsible for restricting the surrounding query
 * to active documents with current, error-free metadata extraction.
 */
export function compileMetadataFilter(filter: MetadataFilter, documentsAlias: string): CompiledMetadataFilter {
  const params: (string | number)[] = [];
  const sql = compileFilterNode(filter, documentsAlias, params);
  return { sql, params };
}

function compileFilterNode(filter: MetadataFilter, alias: string, params: (string | number)[]): string {
  switch (filter.operator) {
    case "and":
    case "or": {
      const joiner = filter.operator === "and" ? " AND " : " OR ";
      return `(${filter.operands.map(operand => compileFilterNode(operand, alias, params)).join(joiner)})`;
    }

    case "not":
      return `NOT ${compileFilterNode(filter.operand, alias, params)}`;

    case "exists":
      params.push(filter.key);
      return filter.value
        ? buildValueExistsSql(alias, "mv.key = ?")
        : `NOT ${buildValueExistsSql(alias, "mv.key = ?")}`;

    case "eq":
    case "gt":
    case "gte":
    case "lt":
    case "lte": {
      const sqlOperator = { eq: "=", gt: ">", gte: ">=", lt: "<", lte: "<=" }[filter.operator];
      params.push(filter.key, bindScalar(filter.value));
      return buildValueExistsSql(
        alias,
        `mv.key = ? AND mv.value_type = '${valueTypeOf(filter.value)}' AND mv.${valueColumnOf(filter.value)} ${sqlOperator} ?`,
      );
    }

    case "ne": {
      // Key must have at least one same-type value, and no same-type value
      // may equal the operand. Missing keys and type mismatches do not match.
      const valueType = valueTypeOf(filter.value);
      params.push(filter.key);
      const presentSql = buildValueExistsSql(alias, `mv.key = ? AND mv.value_type = '${valueType}'`);
      params.push(filter.key, bindScalar(filter.value));
      const equalSql = buildValueExistsSql(
        alias,
        `mv.key = ? AND mv.value_type = '${valueType}' AND mv.${valueColumnOf(filter.value)} = ?`,
      );
      return `(${presentSql} AND NOT ${equalSql})`;
    }

    case "in":
    case "nin": {
      const valueType = valueTypeOf(filter.value[0]!);
      const column = valueColumnOf(filter.value[0]!);
      const placeholders = filter.value.map(() => "?").join(", ");

      if (filter.operator === "in") {
        params.push(filter.key, ...filter.value.map(bindScalar));
        return buildValueExistsSql(alias, `mv.key = ? AND mv.value_type = '${valueType}' AND mv.${column} IN (${placeholders})`);
      }

      params.push(filter.key);
      const presentSql = buildValueExistsSql(alias, `mv.key = ? AND mv.value_type = '${valueType}'`);
      params.push(filter.key, ...filter.value.map(bindScalar));
      const memberSql = buildValueExistsSql(alias, `mv.key = ? AND mv.value_type = '${valueType}' AND mv.${column} IN (${placeholders})`);
      return `(${presentSql} AND NOT ${memberSql})`;
    }

    case "all": {
      const valueType = valueTypeOf(filter.value[0]!);
      const column = valueColumnOf(filter.value[0]!);
      const memberSqls = filter.value.map(element => {
        params.push(filter.key, bindScalar(element));
        return buildValueExistsSql(alias, `mv.key = ? AND mv.value_type = '${valueType}' AND mv.${column} = ?`);
      });
      return `(${memberSqls.join(" AND ")})`;
    }
  }
}

function buildValueExistsSql(alias: string, conditionSql: string): string {
  return `EXISTS (SELECT 1 FROM document_metadata_values mv WHERE mv.document_id = ${alias}.id AND ${conditionSql})`;
}

function valueTypeOf(scalar: MetadataScalar): "string" | "number" | "boolean" {
  return typeof scalar as "string" | "number" | "boolean";
}

function valueColumnOf(scalar: MetadataScalar): "text_value" | "number_value" | "boolean_value" {
  if (typeof scalar === "string") return "text_value";
  if (typeof scalar === "number") return "number_value";
  return "boolean_value";
}

function bindScalar(scalar: MetadataScalar): string | number {
  if (typeof scalar === "boolean") return scalar ? 1 : 0;
  return scalar;
}
