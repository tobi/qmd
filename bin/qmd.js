#!/usr/bin/env node
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawnSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Resolve the package directory
const pkgDir = path.resolve(__dirname, '..');
const scriptPath = path.join(pkgDir, 'dist', 'cli', 'qmd.js');

// Detect runtime based on lockfiles - native modules are compiled for specific runtime
const hasPackageLock = fs.existsSync(path.join(pkgDir, 'package-lock.json'));
const hasBunLock = fs.existsSync(path.join(pkgDir, 'bun.lock')) || fs.existsSync(path.join(pkgDir, 'bun.lockb'));

let runtime;
if (hasBunLock) {
  // If we have a bun.lock, always use bun to avoid ABI mismatches
  runtime = 'bun';
} else if (hasPackageLock) {
  runtime = 'node';
} else {
  // Fall back to current runtime
  runtime = process.execPath;
}

// Disable GPU on Windows to avoid Vulkan crashes
const env = { ...process.env };
if (process.platform === 'win32') {
  env.GGML_NO_VULKAN = '1';
  env.GGML_NO_CUDA = '1';
  env.GGML_NO_OPENCL = '1';
  env.GGML_USE_CPU = '1';
}

// Spawn the process synchronously
const result = spawnSync(runtime, [scriptPath, ...process.argv.slice(2)], {
  stdio: 'inherit',
  shell: process.platform === 'win32',
  env
});

process.exit(result.status || 0);
