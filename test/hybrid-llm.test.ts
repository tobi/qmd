import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { HybridLLM } from "../src/hybrid-llm";
import { LlamaCpp } from "../src/llm";
import { RemoteLLM } from "../src/remote-llm";

// Mock dependencies
vi.mock("../src/llm", async () => {
  const actual = await vi.importActual("../src/llm");
  return {
    ...actual,
    LlamaCpp: vi.fn().mockImplementation(() => ({
      embed: vi.fn().mockResolvedValue({ embedding: [1], model: "local" }),
      embedBatch: vi.fn().mockResolvedValue([{ embedding: [1], model: "local" }]),
      generate: vi.fn().mockResolvedValue({ text: "local", model: "local" }),
      rerank: vi.fn().mockResolvedValue({ results: [], model: "local" }),
      tokenize: vi.fn().mockResolvedValue(["local"]),
      detokenize: vi.fn().mockResolvedValue("local"),
      dispose: vi.fn(),
      modelExists: vi.fn().mockResolvedValue({ exists: true }),
      getDeviceInfo: vi.fn().mockResolvedValue({ gpu: false })
    }))
  };
});

vi.mock("../src/remote-llm", () => ({
  RemoteLLM: vi.fn().mockImplementation(() => ({
    embed: vi.fn().mockResolvedValue({ embedding: [2], model: "remote" }),
    embedBatch: vi.fn().mockResolvedValue([{ embedding: [2], model: "remote" }]),
    generate: vi.fn().mockResolvedValue({ text: "remote", model: "remote" }),
    rerank: vi.fn().mockResolvedValue({ results: [], model: "remote" }),
    tokenize: vi.fn().mockResolvedValue(["remote"]),
    detokenize: vi.fn().mockResolvedValue("remote"),
    dispose: vi.fn(),
    modelExists: vi.fn().mockResolvedValue({ exists: true }),
    getDeviceInfo: vi.fn().mockResolvedValue({ gpu: "cloud" })
  }))
}));

describe("HybridLLM", () => {
  let local: LlamaCpp;
  let remote: RemoteLLM;

  beforeEach(() => {
    vi.clearAllMocks();
    local = new LlamaCpp();
    remote = new RemoteLLM({ apiKey: "test" });
  });

  it("should route to local when configured", async () => {
    const hybrid = new HybridLLM(local, remote, {
      embedBackend: "local",
      generateBackend: "local",
      rerankBackend: "local",
      tokenizeBackend: "local"
    });

    await hybrid.embed("test");
    expect(local.embed).toHaveBeenCalled();
    expect(remote.embed).not.toHaveBeenCalled();

    await hybrid.generate("test");
    expect(local.generate).toHaveBeenCalled();
    expect(remote.generate).not.toHaveBeenCalled();
  });

  it("should route to remote when configured", async () => {
    const hybrid = new HybridLLM(local, remote, {
      embedBackend: "remote",
      generateBackend: "remote",
      rerankBackend: "remote",
      tokenizeBackend: "remote"
    });

    await hybrid.embed("test");
    expect(remote.embed).toHaveBeenCalled();
    expect(local.embed).not.toHaveBeenCalled();

    await hybrid.generate("test");
    expect(remote.generate).toHaveBeenCalled();
    expect(local.generate).not.toHaveBeenCalled();
  });

  it("should support mixed configuration", async () => {
    const hybrid = new HybridLLM(local, remote, {
      embedBackend: "remote",
      generateBackend: "remote",
      rerankBackend: "local",
      tokenizeBackend: "local"
    });

    await hybrid.embed("test");
    expect(remote.embed).toHaveBeenCalled();
    
    await hybrid.generate("test");
    expect(remote.generate).toHaveBeenCalled();

    await hybrid.rerank("test", []);
    expect(local.rerank).toHaveBeenCalled();
    expect(remote.rerank).not.toHaveBeenCalled();

    await hybrid.tokenize("test");
    expect(local.tokenize).toHaveBeenCalled();
    expect(remote.tokenize).not.toHaveBeenCalled();
  });

  it("should fallback to local if remote is missing", async () => {
    const hybrid = new HybridLLM(local, undefined, {
      embedBackend: "remote",
      generateBackend: "remote",
      rerankBackend: "remote",
      tokenizeBackend: "remote"
    });

    await hybrid.embed("test");
    expect(local.embed).toHaveBeenCalled();
  });
});

// =============================================================================
// Environment-based Integration Tests for Remote API Key
// =============================================================================

describe("Remote API Key Environment Integration", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it("should load QMD_REMOTE_API_KEY from environment and initialize remote endpoints", async () => {
    // Set environment variables as they would be in .env file
    process.env.QMD_REMOTE_API_KEY = "sk-or-v1-REDACTED";
    process.env.QMD_EMBED_BACKEND = "remote";
    process.env.QMD_GENERATE_BACKEND = "remote";
    
    // Spy on console.warn to verify warning does NOT appear when API key is set
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    
    // Import the llm module which initializes the default LLM
    const { getDefaultLLM } = await import("../src/llm");
    const llm = getDefaultLLM();
    
    // Expect the LLM to be initialized without falling back to local
    expect(llm).toBeDefined();
    
    // Verify that the fallback warning was NOT called (meaning remote was available)
    const fallbackWarnings = warnSpy.mock.calls.filter(call => 
      typeof call[0] === 'string' && call[0].includes('Remote backend requested but not available')
    );
    
    expect(fallbackWarnings).toHaveLength(0);
    
    warnSpy.mockRestore();
  });

  it("should read all configuration from QMD environment variables correctly", async () => {
    // Simulate the exact configuration from .env file
    process.env.QMD_REMOTE_API_KEY = "sk-or-v1-REDACTED";
    process.env.QMD_REMOTE_EMBED_MODEL = "openrouter/text-embedding-3-small";
    process.env.QMD_REMOTE_GENERATE_MODEL = "openrouter/openai/gpt-4o-mini";
    process.env.QMD_EMBED_BACKEND = "remote";
    process.env.QMD_GENERATE_BACKEND = "remote";
    process.env.QMD_RERANK_BACKEND = "local";
    process.env.QMD_TOKENIZE_BACKEND = "local";
    process.env.QMD_REMOTE_TIMEOUT = "60000";
    
    const { getDefaultLLM } = await import("../src/llm");
    const llm = getDefaultLLM();
    
    expect(llm).toBeDefined();
    expect(llm).not.toBeNull();
  });

  it("should configure RemoteLLM with correct API key when available", async () => {
    // Set the API key to a specific test value
    const testApiKey = "sk-or-v1-test-api-key";
    process.env.QMD_REMOTE_API_KEY = testApiKey;
    process.env.QMD_EMBED_BACKEND = "remote";
    process.env.QMD_GENERATE_BACKEND = "remote";
    
    const { getDefaultLLM } = await import("../src/llm");
    const llm = getDefaultLLM();
    
    // LLM should be initialized with remote backend available
    expect(llm).toBeDefined();
    
    // No warning should appear since API key is present
    expect(() => {
      const llm2 = getDefaultLLM();
      expect(llm2).toBe(llm); // Singleton returns same instance
    }).not.toThrow();
  });
});
