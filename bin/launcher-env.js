const METAL_MODEL_COMMANDS = new Set(["query", "mcp", "vsearch", "embed", "doctor"]);

/**
 * Seed safe Darwin defaults before the CLI imports node-llama-cpp.
 *
 * Model-backed commands retain automatic Metal selection. Residency sets stay
 * enabled on affected Apple Silicon runtimes unless the operator explicitly
 * selects a different residency behavior.
 */
export function applyDarwinRuntimeDefaults(command, env = process.env, platform = process.platform) {
  if (platform !== "darwin" || !METAL_MODEL_COMMANDS.has(command)) return;

  if (env.QMD_METAL_KEEP_RESIDENCY === undefined && env.GGML_METAL_NO_RESIDENCY === undefined) {
    env.QMD_METAL_KEEP_RESIDENCY = "1";
  }
}
