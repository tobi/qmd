/**
 * LLM configuration management
 *
 * This module manages the YAML-based LLM configuration at ~/.config/qmd/llm.yml.
 */

import { existsSync, mkdirSync, readFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import YAML from "yaml";

export type LLMProvider = "local" | "openai";

export type OpenAIConfig = {
  base_url?: string;
  host?: string;
  port?: number;
  protocol?: "http" | "https";
  api_key?: string;
  responses?: {
    rerank?: boolean;
  };
  models?: {
    embed?: string;
    generate?: string;
    rerank?: string;
  };
  temperatures?: {
    generate?: number;
    rerank?: number;
  };
  timeout_ms?: number;
};

export type LLMConfig = {
  provider?: LLMProvider;
  openai?: OpenAIConfig;
};

function getConfigDir(): string {
  if (process.env.QMD_CONFIG_DIR) {
    return process.env.QMD_CONFIG_DIR;
  }
  return join(homedir(), ".config", "qmd");
}

export function getLLMConfigPath(): string {
  return join(getConfigDir(), "llm.yml");
}

function ensureConfigDir(): void {
  const configDir = getConfigDir();
  if (!existsSync(configDir)) {
    mkdirSync(configDir, { recursive: true });
  }
}

export function loadLLMConfig(): LLMConfig {
  const configPath = getLLMConfigPath();
  if (!existsSync(configPath)) {
    return { provider: "local" };
  }

  try {
    const content = readFileSync(configPath, "utf-8");
    const config = YAML.parse(content) as LLMConfig;
    return config ?? { provider: "local" };
  } catch (error) {
    throw new Error(`Failed to parse ${configPath}: ${error}`);
  }
}

export function ensureLLMConfigDir(): void {
  ensureConfigDir();
}
