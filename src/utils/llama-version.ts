import { existsSync, readFileSync, writeFileSync } from "fs";
import { dirname, join, resolve } from "path";
import { fileURLToPath } from "url";

/**
 * Check if QMD_LLAMA_RELEASE env var is set.
 * If so, patch node-llama-cpp's binariesGithubRelease.json to use that release.
 * This allows users to force QMD to use a specific version of llama.cpp
 * (e.g., "master" for latest features, or a specific commit hash).
 */
export function patchLlamaReleaseIfNeeded(): void {
  const release = process.env.QMD_LLAMA_RELEASE;
  if (!release) return;

  try {
    // Resolve the path to node-llama-cpp's config file
    // We look for it in the node_modules relative to the current module
    const currentDir = dirname(fileURLToPath(import.meta.url));
    // qmd/src/utils -> qmd/node_modules
    const projectRoot = resolve(currentDir, "..", "..");
    const nodeLlamaCppPath = join(projectRoot, "node_modules", "node-llama-cpp");
    const configPath = join(nodeLlamaCppPath, "llama", "binariesGithubRelease.json");

    if (!existsSync(configPath)) {
      // Fallback: try to find it in the package structure if installed differently
      // This is a best-effort patch
      return;
    }

    const currentConfig = JSON.parse(readFileSync(configPath, "utf-8"));
    if (currentConfig.release !== release) {
      process.stderr.write(`[QMD] Updating llama.cpp release from "${currentConfig.release}" to "${release}"...\n`);
      currentConfig.release = release;
      writeFileSync(configPath, JSON.stringify(currentConfig, null, 4));
      process.stderr.write("[QMD] llama.cpp release updated. The new version will be downloaded on next run.\n");
    }
  } catch (e) {
    process.stderr.write(`[QMD] Warning: Failed to patch llama.cpp release: ${e instanceof Error ? e.message : String(e)}\n`);
  }
}
