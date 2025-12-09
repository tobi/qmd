/**
 * Tests for VSearch Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect } from 'bun:test';
import VSearchCommand from './vsearch.ts';

describe('VSearchCommand', () => {
  test('is an oclif Command', () => {
    expect(VSearchCommand).toBeDefined();
    expect(VSearchCommand.description).toContain('Vector search');
  });

  test('has query argument', () => {
    expect(VSearchCommand.args).toBeDefined();
    expect(VSearchCommand.args.query).toBeDefined();
  });

  test('has output flags', () => {
    expect(VSearchCommand.flags).toBeDefined();
    expect(VSearchCommand.flags.format).toBeDefined();
    expect(VSearchCommand.flags.limit).toBeDefined();
  });

  test('has model and index flags', () => {
    expect(VSearchCommand.flags.model).toBeDefined();
    expect(VSearchCommand.flags.index).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new VSearchCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
