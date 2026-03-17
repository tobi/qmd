#!/usr/bin/env node
// 直接运行 embed 流程，带详细日志
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

console.log('=== QMD Embed Debug ===');
console.log('Working directory:', process.cwd());

// 先测试最基本的 - 直接导入并尝试初始化
console.log('\n[1] Loading store...');
try {
  const { createStore, getHashesNeedingEmbedding, clearAllEmbeddings, generateEmbeddings } = await import('./dist/store.js');
  console.log('   ✓ Store loaded');

  console.log('\n[2] Opening database...');
  const store = createStore();
  console.log('   ✓ Database opened');

  console.log('\n[3] Checking pending embeddings...');
  const pending = getHashesNeedingEmbedding(store.db);
  console.log(`   → Pending hashes: ${pending}`);

  if (pending === 0) {
    console.log('\nNo embeddings needed!');
    process.exit(0);
  }

  console.log('\n[4] Loading LLM module...');
  const { getDefaultLlamaCpp } = await import('./dist/llm.js');
  console.log('   ✓ LLM module loaded');

  console.log('\n[5] Getting LlamaCpp instance (this may take a while)...');
  const llm = getDefaultLlamaCpp();
  console.log('   ✓ LlamaCpp instance created');

  console.log('\n[6] Starting embedding generation...');
  const result = await generateEmbeddings(store, {
    force: true,
    onProgress: (info) => {
      console.log(`   Progress: ${info.bytesProcessed}/${info.totalBytes} bytes (${Math.round(info.bytesProcessed/info.totalBytes*100)}%)`);
    }
  });

  console.log('\n[7] Done! Result:', result);

  store.close();
} catch (e) {
  console.error('\n✗ ERROR:', e);
  console.error(e.stack);
  process.exit(1);
}
