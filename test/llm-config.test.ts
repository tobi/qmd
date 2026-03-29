import { describe, expect, test } from "vitest";
import { resolveRequestedGpuBackend } from "../src/llm.js";

describe("resolveRequestedGpuBackend", () => {
  test("defaults to auto when no env is set", () => {
    expect(resolveRequestedGpuBackend({})).toBe("auto");
  });

  test("respects explicit CPU-only env values", () => {
    expect(resolveRequestedGpuBackend({ QMD_GPU: "false" })).toBe(false);
    expect(resolveRequestedGpuBackend({ QMD_GPU: "cpu" })).toBe(false);
    expect(resolveRequestedGpuBackend({ NODE_LLAMA_CPP_GPU: "0" })).toBe(false);
    expect(resolveRequestedGpuBackend({ NODE_LLAMA_CPP_GPU: "off" })).toBe(false);
  });

  test("accepts explicit backend overrides", () => {
    expect(resolveRequestedGpuBackend({ QMD_GPU: "cuda" })).toBe("cuda");
    expect(resolveRequestedGpuBackend({ QMD_GPU: "metal" })).toBe("metal");
    expect(resolveRequestedGpuBackend({ QMD_GPU: "vulkan" })).toBe("vulkan");
  });

  test("treats unrecognized values as auto", () => {
    expect(resolveRequestedGpuBackend({ QMD_GPU: "true" })).toBe("auto");
    expect(resolveRequestedGpuBackend({ QMD_GPU: "auto" })).toBe("auto");
    expect(resolveRequestedGpuBackend({ QMD_GPU: "maybe" })).toBe("auto");
  });
});
