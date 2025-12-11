/**
 * Ollama API client service
 * Handles communication with local Ollama server
 */

import { OLLAMA_URL } from '../config/constants.ts';
import { progress } from '../config/terminal.ts';

/**
 * Check if a model is available, pull if not
 * @param model - Model name
 */
export async function ensureModelAvailable(model: string): Promise<void> {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: model }),
    });
    if (response.ok) return;
  } catch {
    // Continue to pull attempt
  }

  console.log(`Model ${model} not found. Pulling...`);
  progress.indeterminate();

  const pullResponse = await fetch(`${OLLAMA_URL}/api/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: model, stream: false }),
  });

  if (!pullResponse.ok) {
    progress.error();
    throw new Error(`Failed to pull model ${model}: ${pullResponse.status} - ${await pullResponse.text()}`);
  }

  progress.clear();
  console.log(`Model ${model} pulled successfully.`);
}

/**
 * Get embedding vector from Ollama
 * @param text - Text to embed
 * @param model - Embedding model name
 * @param isQuery - True if query (vs document)
 * @param title - Document title (for documents)
 * @param retried - Internal retry flag
 * @returns Embedding vector
 */
export async function getEmbedding(
  text: string,
  model: string,
  isQuery: boolean = false,
  title?: string,
  retried: boolean = false
): Promise<number[]> {
  const input = isQuery ? formatQueryForEmbedding(text) : formatDocForEmbedding(text, title);

  const response = await fetch(`${OLLAMA_URL}/api/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, input }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    if (!retried && (errorText.includes("not found") || errorText.includes("does not exist"))) {
      await ensureModelAvailable(model);
      return getEmbedding(text, model, isQuery, title, true);
    }
    throw new Error(`Ollama API error: ${response.status} - ${errorText}`);
  }

  const data = await response.json() as { embeddings: number[][] };
  return data.embeddings[0];
}

/**
 * Generate text completion from Ollama
 * @param model - Model name
 * @param prompt - Prompt text
 * @param raw - Use raw mode (no template)
 * @param logprobs - Include log probabilities
 * @param numPredict - Max tokens to generate
 * @returns Response data
 */
export async function generateCompletion(
  model: string,
  prompt: string,
  raw: boolean = false,
  logprobs: boolean = false,
  numPredict?: number
): Promise<any> {
  const requestBody: any = {
    model,
    prompt,
    raw,
    stream: false,
    logprobs,
  };

  if (numPredict !== undefined) {
    requestBody.options = { num_predict: numPredict };
  }

  const response = await fetch(`${OLLAMA_URL}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.status} - ${await response.text()}`);
  }

  return response.json();
}

/**
 * Format query for embedding (query-focused)
 * @param query - Search query
 * @returns Formatted query
 */
function formatQueryForEmbedding(query: string): string {
  return `search_query: ${query}`;
}

/**
 * Format document for embedding
 * @param text - Document text
 * @param title - Document title
 * @returns Formatted document
 */
function formatDocForEmbedding(text: string, title?: string): string {
  if (title) {
    return `search_document: ${title}\n\n${text}`;
  }
  return `search_document: ${text}`;
}
