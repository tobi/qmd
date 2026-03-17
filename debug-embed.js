#!/usr/bin/env node
import { getDefaultLlamaCpp, pullModels, DEFAULT_EMBED_MODEL_URI } from "./dist/llm.js";

console.log("Starting debug...");
console.log("Model URI:", DEFAULT_EMBED_MODEL_URI);

console.log("\n1. Pulling models (if needed)...");
try {
  await pullModels([DEFAULT_EMBED_MODEL_URI]);
  console.log("   ✓ Models ready");
} catch (e) {
  console.error("   ✗ Error pulling models:", e);
  process.exit(1);
}

console.log("\n2. Getting LlamaCpp instance...");
try {
  const llm = getDefaultLlamaCpp();
  console.log("   ✓ Got LlamaCpp instance");

  console.log("\n3. Creating embedding session...");
  const session = await llm.getEmbeddingSession();
  console.log("   ✓ Got session");

  console.log("\n4. Testing embedding...");
  const result = await session.embed("Hello world test");
  console.log("   ✓ Success! Embedding dimensions:", result.embedding.length);

  console.log("\nAll tests passed!");
} catch (e) {
  console.error("   ✗ Error:", e);
  console.error(e.stack);
  process.exit(1);
}
