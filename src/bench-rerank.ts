#!/usr/bin/env bun
/**
 * QMD Reranker Benchmark
 *
 * Measures reranking performance across different configurations.
 * Reports device, parallelism, memory, VRAM, and throughput.
 *
 * Usage:
 *   bun src/bench-rerank.ts              # full benchmark
 *   bun src/bench-rerank.ts --quick      # quick smoke test (10 docs, 1 iteration)
 *   bun src/bench-rerank.ts --docs 100   # custom doc count
 */

import {
  getLlama,
  getLlamaGpuTypes,
  resolveModelFile,
  LlamaLogLevel,
  type Llama,
  type LlamaModel,
} from "node-llama-cpp";
import { homedir } from "os";
import { join } from "path";
import { cpus } from "os";

// ============================================================================
// Config
// ============================================================================

const RERANK_MODEL = "hf:ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf";
const MODEL_CACHE = join(homedir(), ".cache", "qmd", "models");
const CONTEXT_SIZE = 2048;

const args = process.argv.slice(2);
const quick = args.includes("--quick");
const docsIdx = args.indexOf("--docs");
const DOC_COUNT = docsIdx >= 0 ? parseInt(args[docsIdx + 1]!) : (quick ? 10 : 40);
const ITERATIONS = quick ? 1 : 3;
const PARALLEL_CONFIGS = quick ? [1, 4] : [1, 2, 4, 8];

// ============================================================================
// Test data — realistic-ish chunks of varying length
// ============================================================================

const QUERY = "How do AI agents work and what are their limitations?";

function generateDocs(n: number): string[] {
  const templates = [
    "Artificial intelligence agents are software systems that perceive their environment and take actions to achieve goals. They use techniques like reinforcement learning, planning, and natural language processing to operate autonomously.",
    "The transformer architecture, introduced in 2017, revolutionized natural language processing. Self-attention mechanisms allow models to weigh the importance of different parts of input sequences when generating outputs.",
    "Machine learning models require careful evaluation to avoid overfitting. Cross-validation, holdout sets, and metrics like precision, recall, and F1 score help assess generalization performance.",
    "Retrieval-augmented generation combines information retrieval with language models. Documents are embedded into vector spaces, retrieved based on query similarity, and used as context for generation.",
    "Neural network training involves forward propagation, loss computation, and backpropagation. Optimizers like Adam and SGD adjust weights to minimize the loss function over training iterations.",
    "Large language models exhibit emergent capabilities at scale, including few-shot learning, chain-of-thought reasoning, and instruction following. These properties were not explicitly trained for.",
    "Embedding models convert text into dense vector representations that capture semantic meaning. Similar texts produce similar vectors, enabling efficient similarity search and clustering.",
    "Autonomous agents face challenges including hallucination, lack of grounding, limited planning horizons, and difficulty with multi-step reasoning. Safety and alignment remain open research problems.",
    "The attention mechanism computes query-key-value interactions to determine which parts of the input are most relevant. Multi-head attention allows the model to attend to different representation subspaces.",
    "Fine-tuning adapts a pre-trained model to specific tasks using domain-specific data. Techniques like LoRA reduce the number of trainable parameters while maintaining performance.",
  ];
  return Array.from({ length: n }, (_, i) => templates[i % templates.length]!);
}

// ============================================================================
// Helpers
// ============================================================================

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getMemUsage(): { rss: number; heapUsed: number } {
  const m = process.memoryUsage();
  return { rss: m.rss, heapUsed: m.heapUsed };
}

function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2;
}

// ============================================================================
// Benchmark runner
// ============================================================================

interface BenchResult {
  parallelism: number;
  contextSize: number;
  flashAttention: boolean;
  times: number[];       // ms per run
  medianMs: number;
  docsPerSec: number;
  vramPerContext: number; // bytes
  totalVram: number;      // bytes
  peakRss: number;        // bytes
}

async function benchmarkConfig(
  model: LlamaModel,
  llama: Llama,
  docs: string[],
  parallelism: number,
  flash: boolean,
): Promise<BenchResult> {
  // Measure VRAM before
  const vramBefore = llama.gpu ? await llama.getVramState() : null;
  const rssBefore = getMemUsage().rss;

  // Create contexts. On CPU, split threads evenly across contexts.
  const cpuThreads = !llama.gpu ? Math.floor(llama.cpuMathCores / parallelism) : 0;
  const contexts = [];
  for (let i = 0; i < parallelism; i++) {
    try {
      contexts.push(await model.createRankingContext({
        contextSize: CONTEXT_SIZE,
        flashAttention: flash,
        ...(cpuThreads > 0 ? { threads: cpuThreads } : {}),
      }));
    } catch {
      if (contexts.length === 0) {
        // Try without flash
        contexts.push(await model.createRankingContext({
          contextSize: CONTEXT_SIZE,
          ...(cpuThreads > 0 ? { threads: cpuThreads } : {}),
        }));
      }
      break;
    }
  }
  const actualParallelism = contexts.length;

  // Measure VRAM after context creation
  const vramAfter = llama.gpu ? await llama.getVramState() : null;
  const vramUsed = vramBefore && vramAfter ? vramAfter.used - vramBefore.used : 0;
  const vramPerCtx = actualParallelism > 0 ? vramUsed / actualParallelism : 0;

  // Warm up
  await contexts[0]!.rankAll(QUERY, docs.slice(0, 2));

  // Benchmark iterations
  const times: number[] = [];
  let peakRss = getMemUsage().rss;

  for (let iter = 0; iter < ITERATIONS; iter++) {
    const chunkSize = Math.ceil(docs.length / actualParallelism);

    const t0 = performance.now();
    const allScores = await Promise.all(
      Array.from({ length: actualParallelism }, (_, i) => {
        const chunk = docs.slice(i * chunkSize, (i + 1) * chunkSize);
        return chunk.length > 0 ? contexts[i]!.rankAll(QUERY, chunk) : Promise.resolve([]);
      })
    );
    const elapsed = performance.now() - t0;
    times.push(elapsed);

    // Verify scores are valid
    const flat = allScores.flat();
    if (flat.some(s => s < 0 || s > 1 || isNaN(s))) {
      throw new Error("Invalid scores detected");
    }

    const currentRss = getMemUsage().rss;
    if (currentRss > peakRss) peakRss = currentRss;
  }

  // Cleanup
  for (const ctx of contexts) await ctx.dispose();

  const med = median(times);
  return {
    parallelism: actualParallelism,
    contextSize: CONTEXT_SIZE,
    flashAttention: flash,
    times,
    medianMs: med,
    docsPerSec: (docs.length / med) * 1000,
    vramPerContext: vramPerCtx,
    totalVram: vramUsed,
    peakRss,
  };
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════════");
  console.log("  QMD Reranker Benchmark");
  console.log("═══════════════════════════════════════════════════════════════\n");

  // Detect GPU
  const gpuTypes = await getLlamaGpuTypes();
  const preferred = (["cuda", "metal", "vulkan"] as const).find(g => gpuTypes.includes(g));

  let llama: Llama;
  let gpuLabel: string;
  if (preferred) {
    try {
      llama = await getLlama({ gpu: preferred, logLevel: LlamaLogLevel.error });
      gpuLabel = `${preferred}`;
    } catch {
      llama = await getLlama({ gpu: false, logLevel: LlamaLogLevel.error });
      gpuLabel = "cpu (gpu init failed)";
    }
  } else {
    llama = await getLlama({ gpu: false, logLevel: LlamaLogLevel.error });
    gpuLabel = "cpu";
  }

  // System info
  const cpuInfo = cpus();
  const cpuModel = cpuInfo[0]?.model || "unknown";
  const cpuCount = cpuInfo.length;

  console.log("System");
  console.log(`  CPU:       ${cpuModel}`);
  console.log(`  Cores:     ${cpuCount} (${llama.cpuMathCores} math)`);
  console.log(`  Device:    ${gpuLabel}`);

  if (llama.gpu) {
    const gpuNames = await llama.getGpuDeviceNames();
    const counts = new Map<string, number>();
    for (const name of gpuNames) counts.set(name, (counts.get(name) || 0) + 1);
    const devStr = Array.from(counts.entries())
      .map(([name, n]) => n > 1 ? `${n}× ${name}` : name).join(", ");
    console.log(`  GPU:       ${devStr}`);
    const vram = await llama.getVramState();
    console.log(`  VRAM:      ${formatBytes(vram.total)} total, ${formatBytes(vram.free)} free`);
  }

  console.log(`  RAM:       ${formatBytes(getMemUsage().rss)} RSS at start`);

  // Load model
  console.log(`\nModel`);
  console.log(`  URI:       ${RERANK_MODEL}`);
  const modelPath = await resolveModelFile(RERANK_MODEL, MODEL_CACHE);
  const vramPreModel = llama.gpu ? await llama.getVramState() : null;
  const model = await llama.loadModel({ modelPath });
  const vramPostModel = llama.gpu ? await llama.getVramState() : null;
  const modelVram = vramPreModel && vramPostModel ? vramPostModel.used - vramPreModel.used : 0;
  console.log(`  Params:    ${model.trainContextSize} train ctx`);
  if (modelVram > 0) console.log(`  VRAM:      ${formatBytes(modelVram)} (model weights)`);

  // Generate test docs
  const docs = generateDocs(DOC_COUNT);
  console.log(`\nBenchmark`);
  console.log(`  Documents: ${DOC_COUNT}`);
  console.log(`  Ctx size:  ${CONTEXT_SIZE}`);
  console.log(`  Iterations:${ITERATIONS}`);
  console.log(`  Query:     "${QUERY.slice(0, 50)}..."`);

  // Run benchmarks
  const results: BenchResult[] = [];

  for (const p of PARALLEL_CONFIGS) {
    if (!llama.gpu && p > 1) {
      // CPU: only test if we have enough cores (at least 4 per context)
      if (llama.cpuMathCores < p * 4) {
        console.log(`\n  [${p} ctx] skipped (need ${p * 4} cores, have ${llama.cpuMathCores})`);
        continue;
      }
    }

    // Test with flash attention
    process.stdout.write(`\n  [${p} ctx, flash] running...`);
    try {
      const r = await benchmarkConfig(model, llama, docs, p, true);
      results.push(r);
      process.stdout.write(` ${r.medianMs.toFixed(0)}ms (${r.docsPerSec.toFixed(1)} docs/s)\n`);
    } catch (e: any) {
      process.stdout.write(` failed: ${e.message}\n`);
      // Try without flash
      process.stdout.write(`  [${p} ctx, no flash] running...`);
      try {
        const r = await benchmarkConfig(model, llama, docs, p, false);
        results.push(r);
        process.stdout.write(` ${r.medianMs.toFixed(0)}ms (${r.docsPerSec.toFixed(1)} docs/s)\n`);
      } catch (e2: any) {
        process.stdout.write(` failed: ${e2.message}\n`);
      }
    }
  }

  // Summary table
  console.log("\n═══════════════════════════════════════════════════════════════");
  console.log("  Results");
  console.log("═══════════════════════════════════════════════════════════════\n");

  const header = "  Ctx  Flash  Median    Docs/s   VRAM/ctx   Total VRAM  Peak RSS";
  const sep    = "  ───  ─────  ──────    ──────   ────────   ──────────  ────────";
  console.log(header);
  console.log(sep);

  const baseline = results[0]?.medianMs ?? 1;
  for (const r of results) {
    const speedup = baseline / r.medianMs;
    const speedupStr = r === results[0] ? "      " : `(${speedup.toFixed(1)}×)`;
    console.log(
      `  ${String(r.parallelism).padStart(3)}  ` +
      `${r.flashAttention ? " yes " : "  no "}  ` +
      `${r.medianMs.toFixed(0).padStart(5)}ms  ` +
      `${r.docsPerSec.toFixed(1).padStart(6)}  ` +
      `${formatBytes(r.vramPerContext).padStart(8)}  ` +
      `${formatBytes(r.totalVram).padStart(10)}  ` +
      `${formatBytes(r.peakRss).padStart(8)}  ` +
      speedupStr
    );
  }

  // Best config
  if (results.length > 0) {
    const best = results.reduce((a, b) => a.docsPerSec > b.docsPerSec ? a : b);
    console.log(`\n  Best: ${best.parallelism} contexts, flash=${best.flashAttention}`);
    console.log(`        ${best.medianMs.toFixed(0)}ms for ${DOC_COUNT} docs (${best.docsPerSec.toFixed(1)} docs/s)`);
    if (best.totalVram > 0) console.log(`        ${formatBytes(best.totalVram)} VRAM`);
  }

  console.log("");
  await model.dispose();
  await llama.dispose();
}

main().catch(console.error);
