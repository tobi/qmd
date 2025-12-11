/**
 * Tests for Get Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect } from 'bun:test';
import GetCommand from './get.ts';

describe('GetCommand', () => {
  test('is an oclif Command', () => {
    expect(GetCommand).toBeDefined();
    expect(GetCommand.description).toContain('document');
  });

  test('has file argument', () => {
    expect(GetCommand.args).toBeDefined();
    expect(GetCommand.args.file).toBeDefined();
  });

  test('has index flag', () => {
    expect(GetCommand.flags).toBeDefined();
    expect(GetCommand.flags.index).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new GetCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
