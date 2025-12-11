/**
 * Test utilities for validation behavior
 */

/**
 * Enable strict validation mode for testing
 * In strict mode, validation errors throw exceptions
 */
export function enableStrictValidation(): void {
  process.env.STRICT_VALIDATION = 'true';
}

/**
 * Disable strict validation mode for testing
 * In non-strict mode, validation errors are logged but don't throw
 */
export function disableStrictValidation(): void {
  delete process.env.STRICT_VALIDATION;
}

/**
 * Run a test with strict validation enabled
 * Automatically restores previous state after test
 */
export function withStrictValidation(fn: () => void | Promise<void>): () => void | Promise<void> {
  return async () => {
    const original = process.env.STRICT_VALIDATION;
    enableStrictValidation();
    try {
      await fn();
    } finally {
      if (original !== undefined) {
        process.env.STRICT_VALIDATION = original;
      } else {
        disableStrictValidation();
      }
    }
  };
}
