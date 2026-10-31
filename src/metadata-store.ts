/**
 * QMD Metadata Store - Schema, persistence, and batch loading for document
 * metadata.
 *
 * Metadata attaches to document identity (`documents.id`), not content
 * identity: two paths can share one content hash while carrying different
 * metadata. SQLite stays a derived index — metadata is rebuilt from source
 * documents on `qmd update`, never mutated in place.
 *
 * `document_metadata` records extraction state per document (including
 * successful-but-empty extraction), so filtered search can distinguish
 * "extracted with no metadata" from "not yet extracted" and "extraction
 * failed". `document_metadata_values` holds one indexed row per scalar value
 * for filtering.
 */

import * as buffer from "node:buffer";

import type { Database, SQLiteValue } from "./db.js";
import {
  extractDocumentMetadata,
  METADATA_EXTRACTION_VERSION,
  type DocumentMetadata,
  type MetadataExtractionResult,
  type MetadataScalar,
  type MetadataValueType,
} from "./metadata.js";
import {
  compileMetadataFilter,
  compileMetadataMatch,
  parseMetadataMatch,
  type CompiledMetadataFilter,
  type MetadataFilter,
  type MetadataMatch,
} from "./metadata-filter.js";

// =============================================================================
// Schema
// =============================================================================

export function initializeMetadataSchema(db: Database): void {
  db.exec(`
    CREATE TABLE IF NOT EXISTS document_metadata (
      document_id INTEGER PRIMARY KEY,
      metadata_json TEXT NOT NULL DEFAULT '{}',
      extraction_version INTEGER NOT NULL,
      extraction_error TEXT,
      extracted_at TEXT NOT NULL,
      FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    )
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS document_metadata_values (
      document_id INTEGER NOT NULL,
      key TEXT NOT NULL,
      ordinal INTEGER NOT NULL,
      value_type TEXT NOT NULL,
      text_value TEXT,
      number_value REAL,
      boolean_value INTEGER,
      PRIMARY KEY (document_id, key, ordinal),
      FOREIGN KEY (document_id)
        REFERENCES document_metadata(document_id)
        ON DELETE CASCADE,
      CHECK (value_type IN ('string', 'number', 'boolean')),
      CHECK (
        (value_type = 'string' AND text_value IS NOT NULL AND number_value IS NULL AND boolean_value IS NULL)
        OR (value_type = 'number' AND number_value IS NOT NULL AND text_value IS NULL AND boolean_value IS NULL)
        OR (value_type = 'boolean' AND boolean_value IN (0, 1) AND text_value IS NULL AND number_value IS NULL)
      )
    )
  `);

  db.exec(`
    CREATE INDEX IF NOT EXISTS idx_metadata_text_lookup
    ON document_metadata_values(key, text_value, document_id)
    WHERE value_type = 'string'
  `);
  db.exec(`
    CREATE INDEX IF NOT EXISTS idx_metadata_number_lookup
    ON document_metadata_values(key, number_value, document_id)
    WHERE value_type = 'number'
  `);
  db.exec(`
    CREATE INDEX IF NOT EXISTS idx_metadata_boolean_lookup
    ON document_metadata_values(key, boolean_value, document_id)
    WHERE value_type = 'boolean'
  `);
}

// =============================================================================
// Persistence
// =============================================================================

/**
 * Extract and persist metadata for one document, replacing any prior rows.
 *
 * With `onlyIfStale`, extraction is skipped when the document already has a
 * current-version extraction row — the cheap path for unchanged documents
 * during re-index. Returns the extraction result, or null when skipped.
 */
export function syncDocumentMetadata(
  db: Database,
  documentId: number,
  content: string,
  path: string,
  options?: { onlyIfStale?: boolean },
): MetadataExtractionResult | null {
  if (options?.onlyIfStale && isDocumentMetadataCurrent(db, documentId)) return null;

  const extraction = extractDocumentMetadata(content, path);
  replaceDocumentMetadata(db, documentId, extraction);
  return extraction;
}

/**
 * Replace a document's metadata rows atomically. A failed extraction persists
 * empty metadata plus the error, so stale metadata never survives a bad edit.
 */
export function replaceDocumentMetadata(db: Database, documentId: number, extraction: MetadataExtractionResult): void {
  const replace = db.transaction(() => {
    db.prepare(`
      INSERT INTO document_metadata (document_id, metadata_json, extraction_version, extraction_error, extracted_at)
      VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(document_id) DO UPDATE SET
        metadata_json = excluded.metadata_json,
        extraction_version = excluded.extraction_version,
        extraction_error = excluded.extraction_error,
        extracted_at = excluded.extracted_at
    `).run(
      documentId,
      JSON.stringify(extraction.metadata),
      extraction.extractionVersion,
      extraction.error ?? null,
      new Date().toISOString(),
    );

    db.prepare(`DELETE FROM document_metadata_values WHERE document_id = ?`).run(documentId);

    const insertValue = db.prepare(`
      INSERT INTO document_metadata_values (document_id, key, ordinal, value_type, text_value, number_value, boolean_value)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    for (const [key, value] of Object.entries(extraction.metadata)) {
      const scalars = Array.isArray(value) ? value : [value];
      scalars.forEach((scalar, ordinal) => {
        insertValue.run(
          documentId,
          key,
          ordinal,
          typeof scalar,
          typeof scalar === "string" ? scalar : null,
          typeof scalar === "number" ? scalar : null,
          typeof scalar === "boolean" ? (scalar ? 1 : 0) : null,
        );
      });
    }
  });

  replace();
}

function isDocumentMetadataCurrent(db: Database, documentId: number): boolean {
  const row = db.prepare(`SELECT extraction_version FROM document_metadata WHERE document_id = ?`)
    .get(documentId) as { extraction_version: number } | undefined;
  return row?.extraction_version === METADATA_EXTRACTION_VERSION;
}

// =============================================================================
// Queries
// =============================================================================

/**
 * Count active documents without a current, error-free metadata extraction.
 * These documents are excluded from filtered search until `qmd update` runs.
 * Scoped to `collectionNames` when given, otherwise the whole index.
 */
export function countDocumentsPendingMetadata(db: Database, collectionNames?: string[]): number {
  const params: SQLiteValue[] = [METADATA_EXTRACTION_VERSION];
  let sql = `
    SELECT COUNT(*) as c FROM documents d
    WHERE d.active = 1
      AND NOT EXISTS (
        SELECT 1 FROM document_metadata dm
        WHERE dm.document_id = d.id
          AND dm.extraction_version = ?
          AND dm.extraction_error IS NULL
      )`;
  if (collectionNames) {
    sql += ` AND d.collection IN (SELECT value FROM json_each(?))`;
    params.push(JSON.stringify(collectionNames));
  }
  const row = db.prepare(sql).get(...params) as { c: number };
  return row.c;
}

/**
 * Batch-load canonical metadata for a set of result filepaths
 * (`qmd://collection/path`). One query — never per-result lookups.
 */
export function getMetadataByFilepath(db: Database, filepaths: readonly string[]): Map<string, DocumentMetadata> {
  const metadataByFilepath = new Map<string, DocumentMetadata>();
  if (filepaths.length === 0) return metadataByFilepath;

  const placeholders = filepaths.map(() => "?").join(", ");
  const rows = db.prepare(`
    SELECT 'qmd://' || d.collection || '/' || d.path AS filepath, dm.metadata_json
    FROM documents d
    JOIN document_metadata dm ON dm.document_id = d.id
    WHERE d.active = 1
      AND 'qmd://' || d.collection || '/' || d.path IN (${placeholders})
  `).all(...filepaths) as { filepath: string; metadata_json: string }[];

  for (const row of rows) {
    metadataByFilepath.set(row.filepath, parseMetadataJson(row.metadata_json));
  }
  return metadataByFilepath;
}

/** Parse a stored `metadata_json` column value, tolerating absent rows. */
export function parseMetadataJson(metadataJson: string | null | undefined): DocumentMetadata {
  if (!metadataJson) return {};
  try {
    return JSON.parse(metadataJson) as DocumentMetadata;
  } catch {
    return {};
  }
}

// =============================================================================
// Discovery
// =============================================================================

export interface ListMetadataOptions {
  /** Restrict to these collections. Undefined means every collection in the index. */
  collection?: string | string[];
  /**
   * Report only the metadata entries matching this condition. Same grammar as
   * `filter`, evaluated against each entry: a condition's `field` names the
   * entry's `key` or `value`. Undefined reports every entry.
   */
  match?: MetadataMatch;
  /** Count only documents matching this filter. Same AST as search. */
  filter?: MetadataFilter;
  /** Keys reported (default 50). `Infinity` removes the window. */
  keyLimit?: number;
  /** Keys skipped before the window, in report order (default 0). */
  keyOffset?: number;
  /** Values reported per key and type (default 10). `Infinity` removes the window. */
  valueLimit?: number;
  /** Values skipped per key and type before the window, in `sort` order (default 0). */
  valueOffset?: number;
  /** Order of values within a key (default "count"). */
  sort?: "count" | "value";
  /** Drop values held by fewer documents than this (default 1). */
  minCount?: number;
}

export interface ListMetadataResult {
  /** Active documents in scope. */
  documents: number;
  /**
   * Documents in scope that pass `filter`. Present only when a filter was
   * given, and then the denominator for every coverage count.
   */
  filteredDocuments?: number;
  /** Keys with a matching entry, before the key window. */
  totalKeys: number;
  /** The key window, by documents descending then key ascending. */
  keys: MetadataKeySummary[];
  /** Keys after the window: `totalKeys - keyOffset - keys.length`, floored at 0. */
  remainingKeys: number;
}

export interface MetadataKeySummary {
  key: string;
  /** Distinct documents holding a matching entry under any type. */
  documents: number;
  /** One entry per value_type present. Length > 1 is a type conflict. */
  types: MetadataKeyTypeSummary[];
}

export interface MetadataKeyTypeSummary {
  type: MetadataValueType;
  /**
   * True when any document holds more than one value for this key. A
   * one-element array is indistinguishable from a scalar in the index.
   */
  multiValued: boolean;
  /** Distinct documents holding a matching value of this type. */
  documents: number;
  /** Distinct matching values that meet `minCount`. */
  distinctValues: number;
  /** The value window, narrowed by `match` and `minCount`, ordered by `sort`. */
  values: MetadataValueCount[];
  /** Distinct values after the window: `distinctValues - valueOffset - values.length`, floored at 0. */
  remainingValues: number;
  /** Numbers only. Computed over every matching value row, so array elements each count. */
  range?: { min: number; median: number; max: number };
  /** Collections contributing a matching value of this type, ascending. */
  collections: string[];
}

export interface MetadataValueCount {
  value: MetadataScalar;
  documents: number;
}

/** The light form of a key summary for status views: name, coverage, and types, no values. */
export interface MetadataKeyOverview {
  key: string;
  /** Distinct documents declaring the key with any type. */
  documents: number;
  /** By documents descending then name. Length > 1 is a type conflict. */
  types: MetadataValueType[];
}

/** Exact vocabulary size and a bounded key overview, computed together. */
export interface MetadataOverview {
  totalKeys: number;
  keys: MetadataKeyOverview[];
}

export const DEFAULT_METADATA_KEY_LIMIT = 50;
export const DEFAULT_METADATA_VALUE_LIMIT = 10;

/**
 * Application budget for the bound parameters of one discovery statement, under
 * SQLite's default variable limit (32,766) with headroom. A filter and a match
 * each pass the parser's own limits independently, and the match is bound
 * twice (once to rank keys, once to aggregate), so the combination is checked
 * here, before any statement is prepared.
 */
export const METADATA_SQL_BINDING_BUDGET = 30_000;

/** Raised by listMetadata when an option is outside its domain. */
export class MetadataOptionError extends Error {
  readonly option: string;

  constructor(option: string, expected: string, received: unknown) {
    super(`Invalid ${option}: expected ${expected}, received ${formatReceived(received)}`);
    this.name = "MetadataOptionError";
    this.option = option;
  }
}

/** Raised by listMetadata when `filter` and `match` together bind more SQL parameters than the budget. */
export class MetadataBindingBudgetError extends Error {
  readonly bindings: number;
  readonly budget: number;

  constructor(bindings: number, budget: number) {
    super(`filter and match together bind ${bindings} SQL parameters, over the budget of ${budget}. Shorten their membership lists.`);
    this.name = "MetadataBindingBudgetError";
    this.bindings = bindings;
    this.budget = budget;
  }
}

/** The window and ordering options with defaults applied and domains checked. */
interface ResolvedWindow {
  keyLimit: number;
  keyOffset: number;
  valueLimit: number;
  valueOffset: number;
  sort: "count" | "value";
  minCount: number;
}

/**
 * The value rows discovery aggregates over. `withSql` defines the `eligible`
 * CTE (and, once a key window is applied, `selected_keys`); `fromSql` joins
 * `document_metadata_values mv` to them, narrowed by the match when one is
 * set. Each query supplies its own SELECT list, WHERE, and GROUP BY around
 * these two parts.
 */
interface Region {
  withSql: string;
  withParams: SQLiteValue[];
  fromSql: string;
  fromParams: SQLiteValue[];
}

type ValueRow = {
  key: string;
  value_type: MetadataValueType;
  text_value: string | null;
  number_value: number | null;
  boolean_value: number | null;
};

/**
 * Summarize metadata keys, types, and value counts for the documents in
 * scope. Discovery sees exactly what filtering sees: the same extraction gate,
 * active-document rule, and collection scope, so every value reported here is
 * a value an `eq` filter can match.
 *
 * `filter` selects which documents are counted. `match` selects which of their
 * metadata entries are reported, with the same grammar evaluated per entry.
 * Counts are documents, not values: a document with `topics: [a, b]`
 * contributes one to each.
 *
 * The key window limits which keys receive detailed aggregation and the value
 * window limits the values returned per key and type. Exact counts, medians,
 * and ordering still process the relevant rows. These windows do not bound
 * database work, temporary storage, or the contributing collection lists.
 *
 * The report is read inside one deferred transaction, so every count comes
 * from the same database snapshot even while another connection writes.
 */
export function listMetadata(db: Database, options: ListMetadataOptions = {}): ListMetadataResult {
  const window = resolveWindow(options);
  // A match built in code gets the same checks as one parsed from JSON.
  const match = options.match ? compileMetadataMatch(parseMetadataMatch(options.match), "mv") : undefined;
  return db.transaction(() => readMetadataReport(db, options, match, window))();
}

function readMetadataReport(
  db: Database,
  options: ListMetadataOptions,
  match: CompiledMetadataFilter | undefined,
  window: ResolvedWindow,
): ListMetadataResult {
  const collectionNames = options.collection === undefined ? undefined : [options.collection].flat();
  const eligible = buildEligibleCte(collectionNames, options.filter);
  const matchedRegion = buildRegion(eligible, match);
  const keyWindowRegion = buildKeyWindowRegion(matchedRegion, window.keyLimit, window.keyOffset);
  assertBindingBudget(keyWindowRegion, matchedRegion);

  const result: ListMetadataResult = {
    documents: countActiveDocuments(db, collectionNames),
    totalKeys: 0,
    keys: [],
    remainingKeys: 0,
  };
  if (options.filter) result.filteredDocuments = countEligibleDocuments(db, eligible);

  result.totalKeys = countKeys(db, matchedRegion);
  if (result.totalKeys === 0) return result;

  const typeStats = queryTypeStats(db, keyWindowRegion);
  if (typeStats.length === 0) return result;

  // Carry the selected names through the report. MATERIALIZED prevents repeat
  // ranking within one statement, but cannot share it across statements.
  const keyNames = [...new Set(typeStats.map(row => row.key))];
  const region = narrowRegionToKeys(matchedRegion, keyNames);
  const medianByKey = queryNumberMedians(db, region);
  const typeSummaryByKeyType = new Map<string, MetadataKeyTypeSummary>();
  const typeSummariesByKey = new Map<string, MetadataKeyTypeSummary[]>();

  for (const row of typeStats) {
    const typeSummary: MetadataKeyTypeSummary = {
      type: row.value_type,
      multiValued: false,
      documents: row.documents,
      distinctValues: 0,
      values: [],
      remainingValues: 0,
      collections: (JSON.parse(row.collections) as string[]).sort(compareBinary),
    };
    if (row.value_type === "number") {
      typeSummary.range = { min: row.min_value!, median: medianByKey.get(row.key)!, max: row.max_value! };
    }
    typeSummaryByKeyType.set(`${row.key}\0${row.value_type}`, typeSummary);

    const typeSummaries = typeSummariesByKey.get(row.key) ?? [];
    typeSummaries.push(typeSummary);
    typeSummariesByKey.set(row.key, typeSummaries);
  }

  // Array-ness describes the key, not the matched entries, so it is read over
  // every entry of the keys in the window.
  const keyRegion = buildRegion(eligible, buildKeySetPredicate([...typeSummariesByKey.keys()]));
  for (const row of queryTypeArrayness(db, keyRegion)) {
    const typeSummary = typeSummaryByKeyType.get(`${row.key}\0${row.value_type}`);
    if (typeSummary) typeSummary.multiValued = row.multi_valued === 1;
  }

  for (const row of queryValueCounts(db, region, window)) {
    const typeSummary = typeSummaryByKeyType.get(`${row.key}\0${row.value_type}`);
    if (!typeSummary) continue;
    typeSummary.distinctValues = row.distinct_values;
    if (row.in_window) typeSummary.values.push({ value: scalarOf(row), documents: row.documents });
  }

  for (const typeSummary of typeSummaryByKeyType.values()) {
    typeSummary.remainingValues = Math.max(0, typeSummary.distinctValues - window.valueOffset - typeSummary.values.length);
  }

  // Keys arrive in window rank order and stay in it. A document holds a key
  // under exactly one type (arrays are homogeneous), so per-type document
  // counts partition the key's documents.
  for (const [key, typeSummaries] of typeSummariesByKey) {
    typeSummaries.sort(compareTypeSummaries);
    result.keys.push({
      key,
      documents: typeSummaries.reduce((sum, typeSummary) => sum + typeSummary.documents, 0),
      types: typeSummaries,
    });
  }
  result.remainingKeys = Math.max(0, result.totalKeys - window.keyOffset - result.keys.length);

  return result;
}

/**
 * Every collection's key names, coverage, and types, in coverage order,
 * windowed to the first `keyLimit` keys per collection. One GROUP BY, no
 * values: what `collection list`, `status`, and the MCP status tool print so
 * a first look reveals that metadata exists.
 */
export function listMetadataCollectionSummaries(db: Database, keyLimit: number = DEFAULT_METADATA_KEY_LIMIT): Map<string, MetadataOverview> {
  return queryMetadataOverviews(db, buildEligibleCte(undefined, undefined), keyLimit);
}

interface MetadataKeyCoverage {
  collection: string;
  key: string;
  documents: number;
  string_documents: number;
  number_documents: number;
  boolean_documents: number;
  total_keys: number;
}

function queryMetadataOverviews(db: Database, eligible: Region, keyLimit: number): Map<string, MetadataOverview> {
  const region = buildRegion(eligible);
  // Ordinal zero represents a document/key once. Extraction guarantees one
  // homogeneous type per key, so arrays need neither value reads nor DISTINCT.
  const coverages = db.prepare(`
    ${region.withSql},
    key_coverages AS (
      SELECT e.collection AS collection, mv.key, COUNT(*) AS documents,
        SUM(mv.value_type = 'string') AS string_documents,
        SUM(mv.value_type = 'number') AS number_documents,
        SUM(mv.value_type = 'boolean') AS boolean_documents,
        COUNT(*) OVER (PARTITION BY e.collection) AS total_keys,
        ROW_NUMBER() OVER (PARTITION BY e.collection ORDER BY COUNT(*) DESC, mv.key) AS rank
      ${region.fromSql}
      WHERE mv.ordinal = 0
      GROUP BY e.collection, mv.key
    )
    SELECT * FROM key_coverages WHERE rank <= ? OR ? = -1 ORDER BY collection, rank
  `).all(...region.withParams, ...region.fromParams, Number.isFinite(keyLimit) ? keyLimit : -1, Number.isFinite(keyLimit) ? keyLimit : -1) as MetadataKeyCoverage[];

  const overviews = new Map<string, MetadataOverview>();
  for (const coverage of coverages) {
    const overview = overviews.get(coverage.collection) ?? { totalKeys: coverage.total_keys, keys: [] };
    const typeCounts: [MetadataValueType, number][] = [["string", coverage.string_documents], ["number", coverage.number_documents], ["boolean", coverage.boolean_documents]];
    typeCounts.sort((left, right) => right[1] - left[1] || compareBinary(left[0], right[0]));
    overview.keys.push({ key: coverage.key, documents: coverage.documents, types: typeCounts.filter(([, count]) => count > 0).map(([type]) => type) });
    overviews.set(coverage.collection, overview);
  }
  return overviews;
}

function compareTypeSummaries(a: MetadataKeyTypeSummary, b: MetadataKeyTypeSummary): number {
  return b.documents - a.documents || compareBinary(a.type, b.type);
}

/** SQLite BINARY order compares UTF-8 bytes, not JavaScript's UTF-16 code units. */
function compareBinary(left: string, right: string): number {
  return buffer.Buffer.compare(buffer.Buffer.from(left), buffer.Buffer.from(right));
}

/** Distinct metadata keys declared by active, extracted documents in scope. */
export function countMetadataKeys(db: Database, collectionNames?: string[]): number {
  const region = buildRegion(buildEligibleCte(collectionNames, undefined));
  const row = db.prepare(`
    ${region.withSql}
    SELECT COUNT(DISTINCT mv.key) AS c ${region.fromSql} WHERE mv.ordinal = 0
  `).get(...region.withParams, ...region.fromParams) as { c: number };
  return row.c;
}

/** Active, extracted documents in scope that declare at least one metadata key. */
export function countDocumentsWithMetadata(db: Database, collectionNames?: string[]): number {
  const eligible = buildEligibleCte(collectionNames, undefined);
  const row = db.prepare(`
    ${eligible.withSql}
    SELECT COUNT(*) AS c FROM eligible e
    WHERE EXISTS (SELECT 1 FROM document_metadata_values mv WHERE mv.document_id = e.document_id)
  `).get(...eligible.withParams) as { c: number };
  return row.c;
}

// -----------------------------------------------------------------------------
// Options
// -----------------------------------------------------------------------------

function resolveWindow(options: ListMetadataOptions): ResolvedWindow {
  return {
    keyLimit: readLimit(options.keyLimit, "keyLimit", DEFAULT_METADATA_KEY_LIMIT),
    keyOffset: readOffset(options.keyOffset, "keyOffset"),
    valueLimit: readLimit(options.valueLimit, "valueLimit", DEFAULT_METADATA_VALUE_LIMIT),
    valueOffset: readOffset(options.valueOffset, "valueOffset"),
    sort: readSort(options.sort),
    minCount: readPositiveInteger(options.minCount, "minCount", 1),
  };
}

// Safe integers are exactly representable in JavaScript. The drivers may bind
// them as REAL, so SQL arithmetic must cast explicitly when it needs INTEGER.
function readLimit(value: unknown, option: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (value === Infinity) return Infinity;
  if (isSafeInteger(value) && value >= 1) return value;
  throw new MetadataOptionError(option, "a positive integer or Infinity", value);
}

function readOffset(value: unknown, option: string): number {
  if (value === undefined) return 0;
  if (isSafeInteger(value) && value >= 0) return value;
  throw new MetadataOptionError(option, "a non-negative integer", value);
}

function readPositiveInteger(value: unknown, option: string, fallback: number): number {
  if (value === undefined) return fallback;
  if (isSafeInteger(value) && value >= 1) return value;
  throw new MetadataOptionError(option, "a positive integer", value);
}

function isSafeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value);
}

function readSort(value: unknown): "count" | "value" {
  if (value === undefined) return "count";
  if (value === "count" || value === "value") return value;
  throw new MetadataOptionError("sort", "'count' or 'value'", value);
}

function formatReceived(value: unknown): string {
  if (typeof value === "string") return JSON.stringify(value);
  if (typeof value === "number" || typeof value === "boolean" || value === null) return String(value);
  return typeof value;
}

// -----------------------------------------------------------------------------
// Regions
// -----------------------------------------------------------------------------

/**
 * Documents that filtered search can see: active, with a current, error-free
 * extraction, in scope, and passing the filter. Lists bind as one JSON
 * parameter so the SQLite variable limit never applies. The filter is
 * compiled for the corpus: every statement that reads this CTE joins it to
 * metadata rows, and a correlated predicate would be re-run for each of
 * them, where a document set is built once per statement and probed.
 */
function buildEligibleCte(collectionNames: string[] | undefined, filter: MetadataFilter | undefined): Region {
  const withParams: SQLiteValue[] = [];
  let withSql = `
    WITH eligible AS (
      SELECT d.id AS document_id, d.collection
      FROM documents d
      JOIN document_metadata dm ON dm.document_id = d.id
      WHERE d.active = 1
        AND dm.extraction_version = ${METADATA_EXTRACTION_VERSION}
        AND dm.extraction_error IS NULL`;

  if (collectionNames) {
    withSql += `
        AND d.collection IN (SELECT value FROM json_each(?))`;
    withParams.push(JSON.stringify(collectionNames));
  }

  if (filter) {
    const compiledFilter = compileMetadataFilter(filter, "d", "corpus");
    withSql += `
        AND ${compiledFilter.sql}`;
    withParams.push(...compiledFilter.params);
  }

  withSql += `
    )`;

  return { withSql, withParams, fromSql: "", fromParams: [] };
}

/**
 * Join eligible documents' values, narrowed to the entries an optional
 * predicate over `mv` admits. The predicate lives in the JOIN so queries can
 * add their own WHERE.
 */
function buildRegion(eligible: Region, predicate?: CompiledMetadataFilter): Region {
  const region: Region = {
    withSql: eligible.withSql,
    withParams: [...eligible.withParams],
    fromSql: `
    FROM document_metadata_values mv
    JOIN eligible e ON e.document_id = mv.document_id`,
    fromParams: [],
  };

  if (predicate) {
    region.fromSql += `
      AND ${predicate.sql}`;
    region.fromParams.push(...predicate.params);
  }

  return region;
}

/**
 * Narrow a region to one window of its keys, in report order (documents
 * descending, then key in BINARY collation), each carrying its `rank` so the
 * order survives the joins and GROUP BYs that read it. The window is a CTE
 * every later aggregation joins. It is MATERIALIZED because a small LIMIT
 * otherwise invites the planner to inline it as a coroutine and re-run the
 * ranking once per value row. SQLite reads `LIMIT -1` as no limit.
 */
function buildKeyWindowRegion(region: Region, keyLimit: number, keyOffset: number): Region {
  const withSql = `${region.withSql},
    selected_keys AS MATERIALIZED (
      SELECT mv.key, ROW_NUMBER() OVER (ORDER BY COUNT(DISTINCT mv.document_id) DESC, mv.key ASC) AS rank
      ${region.fromSql}
      GROUP BY mv.key
      ORDER BY rank
      LIMIT ? OFFSET ?
    )`;

  return {
    withSql,
    withParams: [...region.withParams, ...region.fromParams, Number.isFinite(keyLimit) ? keyLimit : -1, keyOffset],
    fromSql: `${region.fromSql}
    JOIN selected_keys sk ON sk.key = mv.key`,
    fromParams: [...region.fromParams],
  };
}

/** Reuse the report's selected keys without repeating coverage and ranking. */
function narrowRegionToKeys(region: Region, keyNames: string[]): Region {
  const predicate = buildKeySetPredicate(keyNames);
  return {
    ...region,
    fromSql: `${region.fromSql} AND ${predicate.sql}`,
    fromParams: [...region.fromParams, ...predicate.params],
  };
}

/** Every entry of these keys, bound as one JSON parameter. */
function buildKeySetPredicate(keyNames: string[]): CompiledMetadataFilter {
  return { sql: "mv.key IN (SELECT value FROM json_each(?))", params: [JSON.stringify(keyNames)] };
}

/** Selected names as JSON, minCount, offset, and the two endpoint operands. */
const VALUE_QUERY_BINDINGS = 5;

function assertBindingBudget(keyWindowRegion: Region, matchedRegion: Region): void {
  const bindings = Math.max(
    keyWindowRegion.withParams.length + keyWindowRegion.fromParams.length,
    matchedRegion.withParams.length + matchedRegion.fromParams.length + VALUE_QUERY_BINDINGS,
  );
  if (bindings > METADATA_SQL_BINDING_BUDGET) throw new MetadataBindingBudgetError(bindings, METADATA_SQL_BINDING_BUDGET);
}

// -----------------------------------------------------------------------------
// Queries
// -----------------------------------------------------------------------------

function countActiveDocuments(db: Database, collectionNames: string[] | undefined): number {
  let sql = `SELECT COUNT(*) AS c FROM documents d WHERE d.active = 1`;
  const params: SQLiteValue[] = [];
  if (collectionNames) {
    sql += ` AND d.collection IN (SELECT value FROM json_each(?))`;
    params.push(JSON.stringify(collectionNames));
  }
  const row = db.prepare(sql).get(...params) as { c: number };
  return row.c;
}

function countEligibleDocuments(db: Database, eligible: Region): number {
  const row = db.prepare(`${eligible.withSql} SELECT COUNT(*) AS c FROM eligible`).get(...eligible.withParams) as { c: number };
  return row.c;
}

function countKeys(db: Database, region: Region): number {
  const row = db.prepare(`
    ${region.withSql}
    SELECT COUNT(DISTINCT mv.key) AS c
    ${region.fromSql}
  `).get(...region.withParams, ...region.fromParams) as { c: number };
  return row.c;
}

type TypeStatsRow = { key: string; value_type: MetadataValueType; documents: number; min_value: number | null; max_value: number | null; collections: string };

/**
 * Per key and type over a key-window region, in the window's rank order.
 * Collection names are sorted after reading: aggregate ORDER BY would
 * require SQLite 3.44, newer than QMD's declared Bun minimum.
 */
function queryTypeStats(db: Database, region: Region): TypeStatsRow[] {
  return db.prepare(`
    ${region.withSql}
    SELECT mv.key, mv.value_type,
      COUNT(DISTINCT mv.document_id) AS documents,
      MIN(mv.number_value) AS min_value,
      MAX(mv.number_value) AS max_value,
      json_group_array(DISTINCT e.collection) AS collections
    ${region.fromSql}
    GROUP BY mv.key, mv.value_type
    ORDER BY sk.rank, mv.value_type
  `).all(...region.withParams, ...region.fromParams) as TypeStatsRow[];
}

type TypeArraynessRow = { key: string; value_type: MetadataValueType; multi_valued: number };

function queryTypeArrayness(db: Database, keyRegion: Region): TypeArraynessRow[] {
  return db.prepare(`
    ${keyRegion.withSql}
    SELECT mv.key, mv.value_type, MAX(mv.ordinal) > 0 AS multi_valued
    ${keyRegion.fromSql}
    GROUP BY mv.key, mv.value_type
  `).all(...keyRegion.withParams, ...keyRegion.fromParams) as TypeArraynessRow[];
}

type MiddleValueRow = { key: string; number_value: number };

/**
 * Median over value rows per key: the middle row for odd counts, the midpoint
 * of the two middle rows for even. The midpoint is taken in JavaScript because
 * SQL's AVG sums first and the sum of two finite doubles can overflow.
 */
function queryNumberMedians(db: Database, region: Region): Map<string, number> {
  const middleRows = db.prepare(`
    ${region.withSql}
    SELECT key, number_value
    FROM (
      SELECT mv.key, mv.number_value,
        ROW_NUMBER() OVER (PARTITION BY mv.key ORDER BY mv.number_value) AS position,
        COUNT(*) OVER (PARTITION BY mv.key) AS total
      ${region.fromSql}
      WHERE mv.value_type = 'number'
    )
    WHERE position IN ((total + 1) / 2, (total + 2) / 2)
    ORDER BY key, number_value
  `).all(...region.withParams, ...region.fromParams) as MiddleValueRow[];

  const medianByKey = new Map<string, number>();
  for (const row of middleRows) {
    const lower = medianByKey.get(row.key);
    medianByKey.set(row.key, lower === undefined ? row.number_value : midpointOf(lower, row.number_value));
  }
  return medianByKey;
}

/**
 * The correctly rounded midpoint when the sum is representable, which also
 * keeps adjacent subnormals exact. Halving each side first is reserved for the
 * overflow region, where both operands are far from underflow.
 */
function midpointOf(lower: number, upper: number): number {
  const sum = lower + upper;
  if (Number.isFinite(sum)) return sum / 2;
  return lower / 2 + upper / 2;
}

type ValueCountRow = ValueRow & { documents: number; distinct_values: number; in_window: number };

/**
 * Group distinct values once for both the total and the window. The first
 * row of each partition carries its total even when the window is past the
 * end. Only rows marked in_window become returned values.
 * Only one typed column is non-null per partition, so ordering all three is stable.
 */
function queryValueCounts(db: Database, region: Region, window: ResolvedWindow): ValueCountRow[] {
  const valueOrder = "mv.text_value, mv.number_value, mv.boolean_value";
  const rankOrder = window.sort === "count" ? `COUNT(DISTINCT mv.document_id) DESC, ${valueOrder}` : valueOrder;
  const params = [...region.withParams, ...region.fromParams, window.minCount, window.valueOffset];
  let windowSql = "rank > ?";
  // Both drivers may bind REALs. Cast before adding so the endpoint remains
  // exact even when it exceeds JavaScript's safe-integer range.
  if (Number.isFinite(window.valueLimit)) {
    windowSql += " AND rank <= CAST(? AS INTEGER) + CAST(? AS INTEGER)";
    params.push(window.valueOffset, window.valueLimit);
  }

  return db.prepare(`
    ${region.withSql},
    value_counts AS (
      SELECT mv.key, mv.value_type, mv.text_value, mv.number_value, mv.boolean_value,
        COUNT(DISTINCT mv.document_id) AS documents,
        COUNT(*) OVER (PARTITION BY mv.key, mv.value_type) AS distinct_values,
        ROW_NUMBER() OVER (PARTITION BY mv.key, mv.value_type ORDER BY ${rankOrder}) AS rank
      ${region.fromSql}
      GROUP BY mv.key, mv.value_type, mv.text_value, mv.number_value, mv.boolean_value
      HAVING COUNT(DISTINCT mv.document_id) >= ?
    ),
    value_window AS (
      SELECT *, (${windowSql}) AS in_window FROM value_counts
    )
    SELECT key, value_type, text_value, number_value, boolean_value, documents, distinct_values, in_window
    FROM value_window
    WHERE rank = 1 OR in_window
    ORDER BY key, value_type, rank
  `).all(...params) as ValueCountRow[];
}

function scalarOf(row: ValueRow): MetadataScalar {
  if (row.value_type === "string") return row.text_value!;
  if (row.value_type === "number") return row.number_value!;
  return row.boolean_value === 1;
}
