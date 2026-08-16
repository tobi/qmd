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

import type { Database } from "./db.js";
import {
  extractDocumentMetadata,
  METADATA_EXTRACTION_VERSION,
  type DocumentMetadata,
  type MetadataExtractionResult,
} from "./metadata.js";

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
 */
export function countDocumentsPendingMetadata(db: Database): number {
  const row = db.prepare(`
    SELECT COUNT(*) as c FROM documents d
    WHERE d.active = 1
      AND NOT EXISTS (
        SELECT 1 FROM document_metadata dm
        WHERE dm.document_id = d.id
          AND dm.extraction_version = ?
          AND dm.extraction_error IS NULL
      )
  `).get(METADATA_EXTRACTION_VERSION) as { c: number };
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
