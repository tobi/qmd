#!/usr/bin/env bun
/**
 * benchmark-embed.ts - Embedding speed & cost benchmark
 *
 * Samples 64 files from dig_chat/outcome and measures wall-clock time
 * and estimated cost for each model (OpenAI, Gemini, local GGUF).
 *
 * Usage:
 *   source ~/.oh-my-zsh/custom/apikey.env
 *   bun scripts/benchmark-embed.ts
 *
 * Required env vars:
 *   OPENAI_API_KEY   - for OpenAI models
 *   GEMINI_API_KEY   - for Gemini models
 */

import { readdirSync, readFileSync, statSync } from "fs";
import { join } from "path";
import { LlamaCpp, DEFAULT_EMBED_MODEL_URI, DEFAULT_MODEL_CACHE_DIR } from "../src/llm";

// =============================================================================
// Config
// =============================================================================

const OUTCOME_DIR = "/Users/jaesolshin/Documents/GitHub/dig_chat/outcome";
const SAMPLE_SIZE = 64;
const API_BATCH_SIZE = 32;
const LOCAL_BATCH_SIZE = 8;
const MAX_CHARS = 700;

interface ModelConfig {
  label: string;
  apiType: "openai" | "gemini" | "local";
  baseUrl?: string;
  model?: string;
  costPer1MTokens: number | null;
  dims: number | null;
}

const MODELS: ModelConfig[] = [
  {
    label: "embeddinggemma-300M (local)",
    apiType: "local",
    costPer1MTokens: 0,
    dims: 768,
  },
  {
    label: "text-embedding-3-small",
    apiType: "openai",
    baseUrl: "https://api.openai.com/v1",
    model: "text-embedding-3-small",
    costPer1MTokens: 0.020,
    dims: 1536,
  },
  {
    label: "text-embedding-3-large",
    apiType: "openai",
    baseUrl: "https://api.openai.com/v1",
    model: "text-embedding-3-large",
    costPer1MTokens: 0.130,
    dims: 3072,
  },
  {
    label: "gemini-embedding-001",
    apiType: "gemini",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta",
    model: "gemini-embedding-001",
    costPer1MTokens: null,
    dims: null,
  },
  {
    label: "gemini-embedding-2-preview",
    apiType: "gemini",
    baseUrl: "https://generativelanguage.googleapis.com/v1beta",
    model: "gemini-embedding-2-preview",
    costPer1MTokens: null,
    dims: null,
  },
];

// =============================================================================
// File sampling
// =============================================================================

function sampleFiles(dir: string, n: number): string[] {
  let files: string[] = [];
  try {
    files = readdirSync(dir)
      .filter((f) => f.endsWith(".md"))
      .map((f) => join(dir, f));
  } catch {
    console.error(`Cannot read directory: ${dir}`);
    process.exit(1);
  }

  if (files.length === 0) {
    console.error("No .md files found in", dir);
    process.exit(1);
  }

  const sized = files
    .map((f) => {
      try { return { path: f, size: statSync(f).size }; }
      catch { return { path: f, size: 0 }; }
    })
    .filter((f) => f.size > 0)
    .sort((a, b) => a.size - b.size);

  const third = Math.floor(sized.length / 3);
  const perBucket = Math.floor(n / 3);
  const remainder = n - perBucket * 3;

  const pick = (arr: typeof sized, count: number) => {
    if (arr.length <= count) return arr;
    const step = arr.length / count;
    return Array.from({ length: count }, (_, i) => arr[Math.floor(i * step)]!);
  };

  return [
    ...pick(sized.slice(0, third), perBucket),
    ...pick(sized.slice(third, third * 2), perBucket),
    ...pick(sized.slice(third * 2), perBucket + remainder),
  ].map((f) => f.path);
}

// =============================================================================
// Text extraction & token estimation
// =============================================================================

function extractChunk(content: string, maxChars = MAX_CHARS): string {
  const parts = content.split("---");
  const body = parts.length >= 3 ? parts.slice(2).join("---") : content;
  return body.trim().slice(0, maxChars);
}

function estimateTokens(texts: string[]): number {
  const totalChars = texts.reduce((s, t) => s + t.length, 0);
  return Math.ceil(totalChars / 3); // conservative for mixed Korean/English
}

// =============================================================================
// API calls
// =============================================================================

async function embedOpenAI(
  texts: string[], baseUrl: string, model: string, apiKey: string
): Promise<{ dims: number }> {
  const resp = await fetch(`${baseUrl}/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({ model, input: texts }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`OpenAI error ${resp.status}: ${err.slice(0, 200)}`);
  }
  const data = await resp.json() as { data: { index: number; embedding: number[] }[] };
  return { dims: data.data[0]?.embedding.length ?? 0 };
}

async function embedGeminiBatch(
  texts: string[], baseUrl: string, model: string, apiKey: string
): Promise<{ dims: number }> {
  const modelId = model.startsWith("models/") ? model : `models/${model}`;
  const url = `${baseUrl}/${modelId}:batchEmbedContents?key=${apiKey}`;
  const requests = texts.map((text) => ({
    model: modelId,
    content: { parts: [{ text }] },
  }));
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requests }),
  });
  if (!resp.ok) {
    const err = await resp.text();
    throw new Error(`Gemini error ${resp.status}: ${err.slice(0, 200)}`);
  }
  const data = await resp.json() as { embeddings: { values: number[] }[] };
  return { dims: data.embeddings[0]?.values.length ?? 0 };
}

// =============================================================================
// Benchmark runner
// =============================================================================

interface BenchResult {
  label: string;
  totalMs: number;
  perChunkMs: number;
  estimatedTokens: number;
  estimatedCost: number | null;
  dims: number;
  error?: string;
}

async function benchmarkModel(
  model: ModelConfig,
  chunks: string[],
  llamaCpp: LlamaCpp
): Promise<BenchResult> {
  const openaiKey = process.env.OPENAI_API_KEY ?? "";
  const geminiKey = process.env.GEMINI_API_KEY ?? "";

  if (model.apiType === "openai" && !openaiKey) {
    return errorResult(model, "OPENAI_API_KEY not set");
  }
  if (model.apiType === "gemini" && !geminiKey) {
    return errorResult(model, "GEMINI_API_KEY not set");
  }

  const estimatedTokens = estimateTokens(chunks);
  const start = performance.now();
  let actualDims = model.dims ?? 0;
  const batchSize = model.apiType === "local" ? LOCAL_BATCH_SIZE : API_BATCH_SIZE;

  try {
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);

      if (model.apiType === "local") {
        const results = await llamaCpp.embedBatch(batch);
        const first = results.find((r: typeof results[number]) => r !== null);
        if (first) actualDims = first.embedding.length;
      } else if (model.apiType === "openai") {
        const { dims } = await embedOpenAI(batch, model.baseUrl!, model.model!, openaiKey);
        if (dims > 0) actualDims = dims;
      } else {
        const { dims } = await embedGeminiBatch(batch, model.baseUrl!, model.model!, geminiKey);
        if (dims > 0) actualDims = dims;
      }
    }
  } catch (err) {
    return {
      label: model.label,
      totalMs: Math.round(performance.now() - start),
      perChunkMs: 0,
      estimatedTokens,
      estimatedCost: null,
      dims: actualDims,
      error: err instanceof Error ? err.message : String(err),
    };
  }

  const totalMs = Math.round(performance.now() - start);
  const estimatedCost =
    model.costPer1MTokens !== null
      ? (estimatedTokens / 1_000_000) * model.costPer1MTokens
      : null;

  return {
    label: model.label,
    totalMs,
    perChunkMs: Math.round(totalMs / chunks.length),
    estimatedTokens,
    estimatedCost,
    dims: actualDims,
  };
}

function errorResult(model: ModelConfig, error: string): BenchResult {
  return { label: model.label, totalMs: 0, perChunkMs: 0, estimatedTokens: 0, estimatedCost: null, dims: model.dims ?? 0, error };
}

// =============================================================================
// Output
// =============================================================================

function pad(s: string, width: number, right = false): string {
  return right ? s.padStart(width) : s.padEnd(width);
}

function formatCost(cost: number | null): string {
  if (cost === null) return "free";
  if (cost === 0) return "$0 (local)";
  return `$${cost.toFixed(6)}`;
}

function printTable(results: BenchResult[], chunkCount: number): void {
  console.log(`\n[Embedding Benchmark] ${chunkCount} chunks from dig_chat/outcome\n`);

  const header = [
    pad("Model", 32),
    pad("Time(ms)", 10, true),
    pad("Per chunk", 11, true),
    pad("~Tokens", 9, true),
    pad("~Cost", 14, true),
    pad("Dims", 6, true),
  ].join(" | ");

  const sep = "-".repeat(header.length);
  console.log(header);
  console.log(sep);

  for (const r of results) {
    if (r.error) {
      console.log(
        [pad(r.label, 32), pad("SKIP", 10, true), pad("-", 11, true), pad("-", 9, true), pad("-", 14, true), pad("-", 6, true)].join(" | ") +
        `  (${r.error})`
      );
    } else {
      console.log(
        [
          pad(r.label, 32),
          pad(r.totalMs.toLocaleString(), 10, true),
          pad(`${r.perChunkMs}ms`, 11, true),
          pad(r.estimatedTokens.toLocaleString(), 9, true),
          pad(formatCost(r.estimatedCost), 14, true),
          pad(String(r.dims), 6, true),
        ].join(" | ")
      );
    }
  }

  console.log(sep);
  console.log();
}

// =============================================================================
// Main
// =============================================================================

async function main(): Promise<void> {
  console.log("Sampling files...");
  const filePaths = sampleFiles(OUTCOME_DIR, SAMPLE_SIZE);
  const chunks = filePaths
    .map((p) => { try { return extractChunk(readFileSync(p, "utf-8")); } catch { return ""; } })
    .filter((c) => c.length > 10);
  console.log(`Sampled ${filePaths.length} files → ${chunks.length} non-empty chunks.\n`);

  // Initialize local LlamaCpp (lazy-loads on first embed call)
  const llamaCpp = new LlamaCpp({
    embedModel: DEFAULT_EMBED_MODEL_URI,
    modelCacheDir: DEFAULT_MODEL_CACHE_DIR,
    disposeModelsOnInactivity: false,
  });

  const results: BenchResult[] = [];

  for (const model of MODELS) {
    process.stdout.write(`Testing ${model.label}... `);
    const result = await benchmarkModel(model, chunks, llamaCpp);
    if (result.error) {
      console.log(`SKIPPED (${result.error})`);
    } else {
      console.log(`done in ${result.totalMs.toLocaleString()}ms`);
    }
    results.push(result);
  }

  await llamaCpp.dispose();

  printTable(results, chunks.length);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
