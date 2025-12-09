/**
 * Tests for Add Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect } from 'bun:test';
import AddCommand from './add.ts';

describe('AddCommand', () => {
  test('is an oclif Command', () => {
    expect(AddCommand).toBeDefined();
    expect(AddCommand.description).toBe('Index markdown files');
  });

  test('has pattern argument', () => {
    expect(AddCommand.args).toBeDefined();
    expect(AddCommand.args.pattern).toBeDefined();
    expect(AddCommand.args.pattern.description).toContain('Glob pattern');
  });

  test('has index flag', () => {
    expect(AddCommand.flags).toBeDefined();
    expect(AddCommand.flags.index).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new AddCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
