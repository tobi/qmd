/**
 * Centroid expansion (Rocchio / pseudo-relevance feedback).
 *
 * Flow:
 *   Q = user query
 *   1. Initial keyword + vector retrieval fused into one ranking (RRF)
 *   2. Take the top 3 files of that ranking as theme representatives
 *   3. Embed their winning chunk texts
 *   4. Average the embeddings into one centroid vector
 *   5. Vector-search the centroid for 20 nearby files
 *   6. Fuse the centroid hits back into the ranking as one more list
 *
 * Why it works:
 *   Top results may share vocabulary of the query but not
 *   the underlying theme. Their collective embedding points
 *   toward the thematic cluster, e.g. query "async task
 *   cancellation": top-3 are about generic future/cancel
 *   tokens. Their mean pulls docs about cooperative
 *   cancellation tokens that don't mention the query terms.
 *
 * Stored-vector path:
 *   Reuse the chunk embeddings already stored in the vector index
 *   for the winning files. If all are present, build the centroid
 *   directly from them, avoiding re-embedding and a model load.
 *   Fall back to embedding only when stored vectors are missing
 *   or were made by a different model version.
 */

export type CentroidConfig = {
  enabled: boolean;
  topKForCentroid: number; // how many top RRF results to form centroid (3)
  vectorTopK: number; // how many results to fetch with centroid vector (20)
  weight: number; // RRF weight for centroid list (1.0)
};

export const DEFAULT_CENTROID_CONFIG: CentroidConfig = {
  enabled: false,
  topKForCentroid: 3,
  vectorTopK: 20,
  weight: 1.0,
};

export function computeCentroid(embeddings: number[][]): Float32Array | null {
  if (embeddings.length === 0) return null;
  const dim = embeddings[0]!.length;
  if (dim === 0) return null;

  if (embeddings.length === 1) {
    const vec = embeddings[0] as number[];
    let norm = 0;
    for (let j = 0; j < dim; j++) norm += (vec[j] ?? 0) * (vec[j] ?? 0);
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return null;
    const result = new Float32Array(dim);
    for (let j = 0; j < dim; j++) result[j] = (vec[j] ?? 0) / norm;
    return result;
  }

  const mean = new Float64Array(dim);
  let validCount = 0;
  for (const emb of embeddings) {
    if (emb.length !== dim) continue;
    validCount++;
    for (let j = 0; j < dim; j++) { mean[j] = (mean[j] ?? 0) + (emb[j] ?? 0); }
  }
  if (validCount === 0) return null;
  for (let j = 0; j < dim; j++) { mean[j] = ((mean[j] ?? 0) as number) / validCount; }

  let norm = 0;
  for (let j = 0; j < dim; j++) norm += ((mean[j] ?? 0) as number) * ((mean[j] ?? 0) as number);
  norm = Math.sqrt(norm);
  if (norm < 1e-12) return null;

  const result = new Float32Array(dim);
  for (let j = 0; j < dim; j++) result[j] = ((mean[j] ?? 0) as number) / norm
  return result;
}

/**
 * Compute centroid from already-loaded Float32Arrays (stored-vector path:
 * embeddings come from the vector index directly).
 */
export function computeCentroidFromFloat32(embeddings: Float32Array[]): Float32Array | null {
  if (embeddings.length === 0) return null;
  const dim = embeddings[0]!.length;
  if (dim === 0) return null;

  if (embeddings.length === 1) {
    const vec = embeddings[0] as Float32Array;
    let norm = 0;
    for (let j = 0; j < dim; j++) norm += (vec[j] ?? 0) * (vec[j] ?? 0);
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return null;
    const result = new Float32Array(dim);
    for (let j = 0; j < dim; j++) result[j] = (vec[j] ?? 0) / norm;
    return result;
  }

  const mean = new Float64Array(dim);
  for (const emb of embeddings) {
    if (emb.length !== dim) continue;
    for (let j = 0; j < dim; j++) { mean[j] = (mean[j] ?? 0) + (emb[j] ?? 0); }
  }
  const n = embeddings.length;
  for (let j = 0; j < dim; j++) { mean[j] = ((mean[j] ?? 0) as number) / n; }

  let norm = 0;
  for (let j = 0; j < dim; j++) norm += ((mean[j] ?? 0) as number) * ((mean[j] ?? 0) as number);
  norm = Math.sqrt(norm);
  if (norm < 1e-12) return null;

  const result = new Float32Array(dim);
  for (let j = 0; j < dim; j++) result[j] = ((mean[j] ?? 0) as number) / norm
  return result;
}
