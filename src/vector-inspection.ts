/**
 * Deep structural inspection of QMD's partitioned vector index.
 *
 * The caller supplies QMD's selected model, computed embedding fingerprint,
 * and ordinary pending-embedding counter. Keeping those values caller-owned
 * avoids a store.ts import cycle and keeps getIndexHealth() on its cheap path.
 */
import type { Database, SQLiteValue } from "./db.js";
import {
  VEC_COLLECTION_IDS_TABLE,
  VEC_ROWS_TABLE,
  VEC_TABLE,
  vecInteger,
  vecLayout,
  vecTableReadable,
} from "./vec-layout.js";

export type VectorPartitionState = "absent" | "legacy" | "unreadable" | "checked";

export type VectorIndexInspection = {
  model: string;
  embeddingFingerprint: string;
  /** `checked` means the partition table was readable and its peer scan completed. */
  partitionState: VectorPartitionState;
  activeDocuments: number;
  needsEmbedding: number;
  /** Active hashes whose selected generation has an invalid chunk or position layout. */
  inconsistentChunkLayouts: number;
  requiredPartitionRows: number;
  missingRequiredPartitionRows: number | null;
  inconsistentPeerRows: number | null;
  /**
   * Every active hash has complete recorded chunks for the selected
   * model/fingerprint, and every required chunk has a consistent partition
   * row. Model-byte and vector/content semantic verification belong to the
   * embedding provenance that produced those recorded chunks.
   */
  structurallyReady: boolean;
};

type RequiredPartitionRow = {
  rowid: number | bigint | null;
  collectionId: SQLiteValue;
};

type MappingRow = {
  rowid: number | bigint;
  collectionId: SQLiteValue;
  registeredCollectionId: SQLiteValue;
  recordedChunkHash: string | null;
};

type VectorRow = {
  rowid: number | bigint;
};

function normalizeInteger(value: SQLiteValue): bigint | undefined {
  if (typeof value === "number") {
    return Number.isSafeInteger(value) ? BigInt(value) : undefined;
  }
  if (typeof value === "bigint") return value;
  return undefined;
}

function sameInteger(left: SQLiteValue, right: SQLiteValue): boolean {
  const normalizedLeft = normalizeInteger(left);
  const normalizedRight = normalizeInteger(right);
  return normalizedLeft !== undefined && normalizedRight !== undefined && normalizedLeft === normalizedRight;
}

function countActiveDocuments(db: Database): number {
  return (db.prepare(`SELECT COUNT(*) AS count FROM documents WHERE active = 1`).get() as { count: number }).count;
}

/**
 * Count active hashes whose selected chunks differ from the exact sequence
 * 0..total_chunks-1, disagree on total_chunks, have invalid recorded position
 * order, or coexist with chunks from a different model/fingerprint. This
 * validates position shape without claiming that a position lies within the
 * current source body. Hashes with no vectors stay represented by the ordinary
 * needsEmbedding count.
 */
function countInconsistentChunkLayouts(
  db: Database,
  model: string,
  embeddingFingerprint: string,
): number {
  const row = db.prepare(`
    WITH active_hashes AS (
      SELECT DISTINCT hash
      FROM documents
      WHERE active = 1
    ),
    selected_chunks AS (
      SELECT
        cv.hash,
        cv.seq,
        cv.pos,
        LAG(cv.pos) OVER (PARTITION BY cv.hash ORDER BY cv.seq) AS previous_pos
      FROM content_vectors cv
      JOIN active_hashes ah ON ah.hash = cv.hash
      WHERE cv.model = ? AND cv.embed_fingerprint = ?
    ),
    sequence_checks AS (
      SELECT
        hash,
        SUM(CASE WHEN
          typeof(seq) != 'integer'
          OR seq < 0
          OR seq > 9007199254740991
        THEN 1 ELSE 0 END) AS invalid_sequences
      FROM selected_chunks
      GROUP BY hash
    ),
    position_checks AS (
      SELECT
        hash,
        SUM(CASE WHEN
          typeof(pos) != 'integer'
          OR pos < 0
          OR pos > 9007199254740991
          OR (seq = 0 AND pos != 0)
          OR (seq > 0 AND (previous_pos IS NULL OR pos <= previous_pos))
        THEN 1 ELSE 0 END) AS invalid_positions
      FROM selected_chunks
      GROUP BY hash
    ),
    all_chunk_counts AS (
      SELECT ah.hash, COUNT(cv.seq) AS stored_count
      FROM active_hashes ah
      LEFT JOIN content_vectors cv ON cv.hash = ah.hash
      GROUP BY ah.hash
    ),
    selected_chunk_layouts AS (
      SELECT
        ah.hash,
        COUNT(cv.seq) AS selected_count,
        COUNT(cv.total_chunks) AS total_count,
        MIN(cv.seq) AS min_seq,
        MAX(cv.seq) AS max_seq,
        MIN(cv.total_chunks) AS min_total,
        MAX(cv.total_chunks) AS max_total
      FROM active_hashes ah
      LEFT JOIN content_vectors cv
        ON cv.hash = ah.hash
        AND cv.model = ?
        AND cv.embed_fingerprint = ?
      GROUP BY ah.hash
    )
    SELECT COUNT(*) AS count
    FROM all_chunk_counts all_chunks
    JOIN selected_chunk_layouts selected ON selected.hash = all_chunks.hash
    LEFT JOIN sequence_checks sequences ON sequences.hash = all_chunks.hash
    LEFT JOIN position_checks positions ON positions.hash = all_chunks.hash
    WHERE
      all_chunks.stored_count != selected.selected_count
      OR COALESCE(sequences.invalid_sequences, 0) > 0
      OR COALESCE(positions.invalid_positions, 0) > 0
      OR (
        selected.selected_count > 0
        AND (
          selected.total_count != selected.selected_count
          OR selected.min_total != selected.max_total
          OR selected.max_total < 1
          OR selected.selected_count != selected.max_total
          OR selected.min_seq != 0
          OR selected.max_seq != selected.max_total - 1
        )
      )
  `).get(model, embeddingFingerprint, model, embeddingFingerprint) as { count: number };
  return row.count;
}

/**
 * Required collection-partition rows for chunks recorded under the selected
 * model and fingerprint. Active hashes with absent or incomplete current
 * chunks are represented by needsEmbedding, supplied by the caller.
 */
function iterateRequiredPartitionRows(
  db: Database,
  model: string,
  embeddingFingerprint: string,
): IterableIterator<RequiredPartitionRow> {
  return db.prepare(`
    SELECT vr.id AS rowid, ci.id AS collectionId
    FROM content_vectors cv
    JOIN (
      SELECT DISTINCT hash, collection
      FROM documents
      WHERE active = 1
    ) d ON d.hash = cv.hash
    LEFT JOIN ${VEC_COLLECTION_IDS_TABLE} ci ON ci.name = d.collection
    LEFT JOIN ${VEC_ROWS_TABLE} vr
      ON vr.hash = cv.hash
      AND vr.seq = cv.seq
      AND vr.collection_id = ci.id
    WHERE cv.model = ? AND cv.embed_fingerprint = ?
  `).iterate<RequiredPartitionRow>(model, embeddingFingerprint);
}

function inspectReadablePartitions(
  db: Database,
  requiredRows: Iterable<RequiredPartitionRow>,
): { requiredPartitionRows: number; missingRequiredPartitionRows: number; inconsistentPeerRows: number } {
  const vectorByRowid = db.prepare(`SELECT collection_id AS collectionId FROM ${VEC_TABLE} WHERE rowid = ?`);
  let requiredPartitionRows = 0;
  let missingRequiredPartitionRows = 0;

  for (const row of requiredRows) {
    requiredPartitionRows++;
    if (row.rowid === null || row.collectionId === null) {
      missingRequiredPartitionRows++;
      continue;
    }
    const vector = vectorByRowid.get<{ collectionId: SQLiteValue }>(vecInteger(row.rowid));
    if (!vector || !sameInteger(vector.collectionId, row.collectionId)) {
      missingRequiredPartitionRows++;
    }
  }

  // sqlite-vec virtual tables stay outside relational JOINs. Each direction is
  // streamed independently and compared through an indexed rowid probe.
  let inconsistentPeerRows = 0;
  const mappings = db.prepare(`
    SELECT
      vr.id AS rowid,
      vr.collection_id AS collectionId,
      ci.id AS registeredCollectionId,
      cv.hash AS recordedChunkHash
    FROM ${VEC_ROWS_TABLE} vr
    LEFT JOIN ${VEC_COLLECTION_IDS_TABLE} ci ON ci.id = vr.collection_id
    LEFT JOIN content_vectors cv ON cv.hash = vr.hash AND cv.seq = vr.seq
    ORDER BY vr.id
  `);
  for (const mapping of mappings.iterate<MappingRow>()) {
    const vector = vectorByRowid.get<{ collectionId: SQLiteValue }>(vecInteger(mapping.rowid));
    if (
      !vector ||
      !sameInteger(vector.collectionId, mapping.collectionId) ||
      mapping.registeredCollectionId === null ||
      mapping.recordedChunkHash === null
    ) {
      inconsistentPeerRows++;
    }
  }

  const mappingByRowid = db.prepare(`SELECT collection_id AS collectionId FROM ${VEC_ROWS_TABLE} WHERE id = ?`);
  const vectors = db.prepare(`SELECT rowid FROM ${VEC_TABLE}`);
  for (const vector of vectors.iterate<VectorRow>()) {
    const mapping = mappingByRowid.get<{ collectionId: SQLiteValue }>(vector.rowid);
    // A mapped mismatch was counted during the mapping pass. This pass adds
    // vec0-only rowids, preserving one count per rowid with constant heap use.
    if (!mapping) inconsistentPeerRows++;
  }

  return { requiredPartitionRows, missingRequiredPartitionRows, inconsistentPeerRows };
}

function countRequiredPartitionRows(requiredRows: Iterable<RequiredPartitionRow>): number {
  let count = 0;
  for (const _row of requiredRows) count++;
  return count;
}

function isStructurallyReady(
  partitionState: VectorPartitionState,
  activeDocuments: number,
  needsEmbedding: number,
  inconsistentChunkLayouts: number,
  missingRequiredPartitionRows: number | null,
  inconsistentPeerRows: number | null,
): boolean {
  if (partitionState === "legacy" || partitionState === "unreadable") return false;
  if (
    needsEmbedding > 0 ||
    inconsistentChunkLayouts > 0 ||
    missingRequiredPartitionRows !== 0 ||
    inconsistentPeerRows !== 0
  ) return false;
  return partitionState === "checked" || activeDocuments === 0;
}

function inspectVectorIndexInSnapshot(
  db: Database,
  model: string,
  embeddingFingerprint: string,
  countNeedsEmbedding: () => number,
): VectorIndexInspection {
  const needsEmbedding = countNeedsEmbedding();
  const activeDocuments = countActiveDocuments(db);
  const inconsistentChunkLayouts = countInconsistentChunkLayouts(db, model, embeddingFingerprint);
  const layout = vecLayout(db);

  let partitionState: VectorPartitionState;
  let requiredPartitionRowCount: number;
  let missingRequiredPartitionRows: number | null;
  let inconsistentPeerRows: number | null;

  if (layout.kind === "legacy") {
    partitionState = "legacy";
    requiredPartitionRowCount = countRequiredPartitionRows(
      iterateRequiredPartitionRows(db, model, embeddingFingerprint),
    );
    missingRequiredPartitionRows = null;
    inconsistentPeerRows = null;
  } else if (layout.kind === "none") {
    partitionState = "absent";
    requiredPartitionRowCount = countRequiredPartitionRows(
      iterateRequiredPartitionRows(db, model, embeddingFingerprint),
    );
    missingRequiredPartitionRows = requiredPartitionRowCount;
    const mappingCount = db.prepare(`SELECT COUNT(*) AS count FROM ${VEC_ROWS_TABLE}`).get() as { count: number };
    inconsistentPeerRows = mappingCount.count;
  } else if (!vecTableReadable(db, layout)) {
    partitionState = "unreadable";
    requiredPartitionRowCount = countRequiredPartitionRows(
      iterateRequiredPartitionRows(db, model, embeddingFingerprint),
    );
    missingRequiredPartitionRows = null;
    inconsistentPeerRows = null;
  } else {
    partitionState = "checked";
    ({
      requiredPartitionRows: requiredPartitionRowCount,
      missingRequiredPartitionRows,
      inconsistentPeerRows,
    } = inspectReadablePartitions(db, iterateRequiredPartitionRows(db, model, embeddingFingerprint)));
  }

  return {
    model,
    embeddingFingerprint,
    partitionState,
    activeDocuments,
    needsEmbedding,
    inconsistentChunkLayouts,
    requiredPartitionRows: requiredPartitionRowCount,
    missingRequiredPartitionRows,
    inconsistentPeerRows,
    structurallyReady: isStructurallyReady(
      partitionState,
      activeDocuments,
      needsEmbedding,
      inconsistentChunkLayouts,
      missingRequiredPartitionRows,
      inconsistentPeerRows,
    ),
  };
}

/**
 * Inspect the selected embedding generation and the physical vec0/mapping
 * relationship in one SQLite snapshot. Work scales with active chunk records,
 * required partition rows, and physical mapping/vector rows. Publication gates
 * and explicit health checks call this diagnostic. getIndexHealth() serves the
 * ordinary status path.
 */
export function inspectVectorIndex(
  db: Database,
  model: string,
  embeddingFingerprint: string,
  countNeedsEmbedding: () => number,
): VectorIndexInspection {
  const runInspection = db.transaction(() => inspectVectorIndexInSnapshot(
    db,
    model,
    embeddingFingerprint,
    countNeedsEmbedding,
  ));
  return runInspection();
}
