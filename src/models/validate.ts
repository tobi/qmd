/**
 * Validation utilities for runtime type checking
 *
 * Provides helpers for validating data with Zod schemas in repositories
 */

import type { ZodSchema, ZodError } from 'zod';

/**
 * Validation error with details
 */
export class ValidationError extends Error {
  constructor(
    message: string,
    public readonly errors: Array<{ path: string; message: string }>
  ) {
    super(message);
    this.name = 'ValidationError';
  }

  /**
   * Format error for display
   */
  toString(): string {
    const details = this.errors
      .map(e => `  - ${e.path}: ${e.message}`)
      .join('\n');
    return `${this.message}\n${details}`;
  }
}

/**
 * Parse and validate data with a Zod schema
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @param context - Context for error messages (e.g., "Document from database")
 * @returns Validated data
 * @throws ValidationError if validation fails
 */
export function validate<T>(
  schema: ZodSchema<T>,
  data: unknown,
  context: string = 'Data'
): T {
  const result = schema.safeParse(data);

  if (!result.success) {
    const errors = result.error.issues.map(issue => ({
      path: issue.path.join('.'),
      message: issue.message,
    }));

    throw new ValidationError(
      `${context} validation failed`,
      errors
    );
  }

  return result.data;
}

/**
 * Validate data without throwing, returning errors
 * @param schema - Zod schema to validate against
 * @param data - Data to validate
 * @returns Validation result with data or errors
 */
export function validateSafe<T>(
  schema: ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: Array<{ path: string; message: string }> } {
  const result = schema.safeParse(data);

  if (!result.success) {
    return {
      success: false,
      errors: result.error.issues.map(issue => ({
        path: issue.path.join('.'),
        message: issue.message,
      })),
    };
  }

  return {
    success: true,
    data: result.data,
  };
}

/**
 * Validate array of items
 * @param schema - Zod schema for single item
 * @param data - Array to validate
 * @param context - Context for error messages
 * @returns Validated array
 * @throws ValidationError if any item fails validation
 */
export function validateArray<T>(
  schema: ZodSchema<T>,
  data: unknown[],
  context: string = 'Array item'
): T[] {
  const validated: T[] = [];
  const errors: Array<{ index: number; path: string; message: string }> = [];

  for (let i = 0; i < data.length; i++) {
    const result = schema.safeParse(data[i]);

    if (!result.success) {
      for (const issue of result.error.issues) {
        errors.push({
          index: i,
          path: issue.path.join('.'),
          message: issue.message,
        });
      }
    } else {
      validated.push(result.data);
    }
  }

  if (errors.length > 0) {
    const errorDetails = errors.map(e =>
      ({ path: `[${e.index}]${e.path ? '.' + e.path : ''}`, message: e.message })
    );

    throw new ValidationError(
      `${context} validation failed`,
      errorDetails
    );
  }

  return validated;
}

/**
 * Check if strict validation mode is enabled
 * If STRICT_VALIDATION=true, validation errors throw
 * Otherwise, validation errors are logged but don't throw
 */
function isStrictValidation(): boolean {
  return process.env.STRICT_VALIDATION === 'true';
}

/**
 * Validate with optional strict mode
 * In strict mode: throws on error
 * In non-strict mode: logs error and returns original data
 */
export function validateOptional<T>(
  schema: ZodSchema<T>,
  data: unknown,
  context: string = 'Data'
): T {
  if (isStrictValidation()) {
    return validate(schema, data, context);
  }

  const result = schema.safeParse(data);

  if (!result.success) {
    // Log validation errors but don't throw
    console.warn(`[Validation Warning] ${context}:`);
    for (const issue of result.error.issues) {
      console.warn(`  - ${issue.path.join('.')}: ${issue.message}`);
    }
    return data as T;
  }

  return result.data;
}
