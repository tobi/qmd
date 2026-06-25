import "dotenv/config";
import { getDefaultLLM } from "../src/llm.js";
import { HybridLLM } from "../src/hybrid-llm.js";

console.log("--- QMD Configuration Debug ---");

// 1. Check Environment Variable
const apiKey = process.env.QMD_REMOTE_API_KEY;
if (apiKey) {
  console.log(`✅ QMD_REMOTE_API_KEY found: ${apiKey.substring(0, 8)}...`);
} else {
  console.log("❌ QMD_REMOTE_API_KEY not found in environment.");
}

// 2. Check Loaded LLM
const llm = getDefaultLLM();
console.log(`LLM Class: ${llm.constructor.name}`);

if (llm instanceof HybridLLM) {
  console.log("✅ HybridLLM initialized.");
  
  // Check backend configuration
  console.log("Backend Preferences:");
  console.log(`- Embed: ${process.env.QMD_EMBED_BACKEND || 'default'}`);
  console.log(`- Generate: ${process.env.QMD_GENERATE_BACKEND || 'default'}`);
  console.log(`- Rerank: ${process.env.QMD_RERANK_BACKEND || 'default'}`);
  console.log(`- Tokenize: ${process.env.QMD_TOKENIZE_BACKEND || 'default'}`);
} else {
  console.log("⚠️ Default LLM is not HybridLLM (unexpected).");
}

console.log("-------------------------------");
