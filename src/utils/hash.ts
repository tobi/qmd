/**
 * Hash utility functions for content hashing
 */

/**
 * Generate SHA-256 hash of content
 * @param content - String content to hash
 * @returns Hexadecimal hash string
 */
export async function hashContent(content: string): Promise<string> {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(content);
  return hash.digest("hex");
}

/**
 * Generate a cache key from URL and body
 * Used for caching Ollama API calls
 * @param url - API URL
 * @param body - Request body object
 * @returns Hexadecimal cache key
 */
export function getCacheKey(url: string, body: object): string {
  const hash = new Bun.CryptoHasher("sha256");
  hash.update(url);
  hash.update(JSON.stringify(body));
  return hash.digest("hex");
}
