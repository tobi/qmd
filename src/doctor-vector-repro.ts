export const DOCTOR_VECTOR_REPRO_THRESHOLD = 0.0001;
// A 100-chunk full-corpus sample from the shared llama.cpp/ROCm service
// produced singleton-vs-stored cosine distances up to 0.000691 (p99 0.000635)
// for the same fingerprint and vector semantics. Keep this remote-only ceiling
// below the materially-different test case while leaving local reproducibility
// at the original strict threshold.
export const DOCTOR_REMOTE_BATCH_DRIFT_THRESHOLD = 0.001;

export type DoctorVectorDistanceClassification = "match" | "remote-drift" | "mismatch";

export type DoctorVectorSampleResult = {
  ok: boolean;
  details: string;
  forceRebuild: boolean;
};

export function classifyDoctorVectorDistance(
  distance: number,
  remoteEmbedding: boolean,
): DoctorVectorDistanceClassification {
  if (!Number.isFinite(distance)) return "mismatch";
  if (distance <= DOCTOR_VECTOR_REPRO_THRESHOLD) return "match";
  if (remoteEmbedding && distance <= DOCTOR_REMOTE_BATCH_DRIFT_THRESHOLD) return "remote-drift";
  return "mismatch";
}

export function buildDoctorVectorSampleResult(
  sampleCount: number,
  mismatches: string[],
  remoteDrifts: string[],
): DoctorVectorSampleResult {
  if (mismatches.length > 0) {
    return {
      ok: false,
      details: `${mismatches.length}/${sampleCount} sampled chunks differ from stored vectors (${mismatches[0]}). Rebuild with \`qmd embed --force\``,
      forceRebuild: true,
    };
  }

  if (remoteDrifts.length > 0) {
    return {
      ok: true,
      details: `${remoteDrifts.length}/${sampleCount} sampled remote chunks show bounded batch-shape drift (${remoteDrifts[0]}); fingerprint and vector semantics match, so no rebuild is needed`,
      forceRebuild: false,
    };
  }

  return {
    ok: true,
    details: `${sampleCount} sampled ${sampleCount === 1 ? "chunk" : "chunks"} reproduce stored vectors`,
    forceRebuild: false,
  };
}
