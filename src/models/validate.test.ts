/**
 * Tests for validation utilities
 */

import { describe, test, expect } from 'bun:test';
import { z } from 'zod';
import {
  validate,
  validateSafe,
  validateArray,
  validateOptional,
  ValidationError,
} from './validate.ts';
import { enableStrictValidation, disableStrictValidation } from '../../tests/fixtures/helpers/test-validation.ts';

const TestSchema = z.object({
  id: z.number().positive(),
  name: z.string().min(1),
  email: z.string().email(),
});

describe('validate', () => {
  test('validates correct data', () => {
    const data = { id: 1, name: 'John', email: 'john@example.com' };
    const result = validate(TestSchema, data);

    expect(result).toEqual(data);
  });

  test('throws ValidationError for invalid data', () => {
    const data = { id: -1, name: '', email: 'invalid' };

    expect(() => {
      validate(TestSchema, data, 'User');
    }).toThrow(ValidationError);
  });

  test('ValidationError includes field details', () => {
    const data = { id: -1, name: '', email: 'invalid' };

    try {
      validate(TestSchema, data, 'User');
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError);
      const ve = error as ValidationError;
      expect(ve.errors.length).toBeGreaterThan(0);
      expect(ve.errors[0].path).toBeTruthy();
      expect(ve.errors[0].message).toBeTruthy();
    }
  });
});

describe('validateSafe', () => {
  test('returns success for valid data', () => {
    const data = { id: 1, name: 'John', email: 'john@example.com' };
    const result = validateSafe(TestSchema, data);

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data).toEqual(data);
    }
  });

  test('returns errors for invalid data', () => {
    const data = { id: -1, name: '', email: 'invalid' };
    const result = validateSafe(TestSchema, data);

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.errors.length).toBeGreaterThan(0);
    }
  });
});

describe('validateArray', () => {
  test('validates array of valid items', () => {
    const data = [
      { id: 1, name: 'John', email: 'john@example.com' },
      { id: 2, name: 'Jane', email: 'jane@example.com' },
    ];

    const result = validateArray(TestSchema, data);
    expect(result).toEqual(data);
  });

  test('throws with details for invalid items', () => {
    const data = [
      { id: 1, name: 'John', email: 'john@example.com' },
      { id: -1, name: '', email: 'invalid' }, // Invalid
      { id: 3, name: 'Bob', email: 'bob@example.com' },
    ];

    expect(() => {
      validateArray(TestSchema, data);
    }).toThrow(ValidationError);

    try {
      validateArray(TestSchema, data);
    } catch (error) {
      const ve = error as ValidationError;
      // Should indicate which array index failed
      expect(ve.errors.some(e => e.path.includes('[1]'))).toBe(true);
    }
  });
});

describe('validateOptional', () => {
  test('validates in strict mode when STRICT_VALIDATION=true', () => {
    const originalEnv = process.env.STRICT_VALIDATION;
    enableStrictValidation();

    const data = { id: -1, name: '', email: 'invalid' };

    expect(() => {
      validateOptional(TestSchema, data);
    }).toThrow();

    // Restore original state
    if (originalEnv) process.env.STRICT_VALIDATION = originalEnv;
    else disableStrictValidation();
  });

  test('returns data without throwing in non-strict mode', () => {
    const originalEnv = process.env.STRICT_VALIDATION;
    disableStrictValidation();

    const data = { id: -1, name: '', email: 'invalid' };

    // Should not throw, but log warnings
    const result = validateOptional(TestSchema, data);
    expect(result).toBeDefined();

    // Restore original state
    if (originalEnv) process.env.STRICT_VALIDATION = originalEnv;
    else disableStrictValidation();
  });
});

describe('ValidationError', () => {
  test('formats error message with details', () => {
    const error = new ValidationError('Test failed', [
      { path: 'id', message: 'Must be positive' },
      { path: 'email', message: 'Invalid email' },
    ]);

    const str = error.toString();
    expect(str).toContain('Test failed');
    expect(str).toContain('id: Must be positive');
    expect(str).toContain('email: Invalid email');
  });
});
