/**
 * Tests for VSearch Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect } from 'bun:test';
import VSearchCommand from './vsearch.ts';

describe('VSearchCommand', () => {
  test('is an oclif Command', () => {
    expect(VSearchCommand).toBeDefined();
    expect(VSearchCommand.description).toContain('similarity');
  });

  test('has query argument', () => {
    expect(VSearchCommand.args).toBeDefined();
    expect(VSearchCommand.args.query).toBeDefined();
  });

  test('has output flags', () => {
    expect(VSearchCommand.flags).toBeDefined();
    expect(VSearchCommand.flags.json).toBeDefined();
    expect(VSearchCommand.flags.n).toBeDefined();
  });

  test('has index flag', () => {
    expect(VSearchCommand.flags.index).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new VSearchCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
