import { describe, expect, test } from "vitest";
import {
  buildDoctorVectorSampleResult,
  classifyDoctorVectorDistance,
} from "../src/doctor-vector-repro.js";

describe("doctor vector reproducibility classification", () => {
  test("keeps the strict threshold for local embeddings", () => {
    expect(classifyDoctorVectorDistance(0.000228, false)).toBe("mismatch");
  });

  test("classifies calibrated remote batch-shape variance as drift", () => {
    expect(classifyDoctorVectorDistance(0.000691, true)).toBe("remote-drift");
  });

  test("still rejects materially different remote vectors", () => {
    expect(classifyDoctorVectorDistance(0.002, true)).toBe("mismatch");
  });

  test("accepts vectors within the strict reproducibility threshold", () => {
    expect(classifyDoctorVectorDistance(0.00005, false)).toBe("match");
    expect(classifyDoctorVectorDistance(0.00005, true)).toBe("match");
  });

  test("rejects invalid distances for every backend", () => {
    expect(classifyDoctorVectorDistance(Number.NaN, false)).toBe("mismatch");
    expect(classifyDoctorVectorDistance(Number.POSITIVE_INFINITY, true)).toBe("mismatch");
  });

  test("bounded remote drift is healthy and does not recommend a rebuild", () => {
    const result = buildDoctorVectorSampleResult(3, [], ["abc123_1: stored vector distance 0.000691"]);
    expect(result.ok).toBe(true);
    expect(result.forceRebuild).toBe(false);
    expect(result.details).toContain("bounded batch-shape drift");
    expect(result.details).toContain("no rebuild is needed");
    expect(result.details).not.toContain("embed --force");
  });

  test("material mismatch fails and recommends a forced rebuild", () => {
    const result = buildDoctorVectorSampleResult(3, ["abc123_1: stored vector distance 0.002000"], []);
    expect(result.ok).toBe(false);
    expect(result.forceRebuild).toBe(true);
    expect(result.details).toContain("qmd embed --force");
  });
});
