/**
 * Tests for Query Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect } from 'bun:test';
import QueryCommand from './query.ts';

describe('QueryCommand', () => {
  test('is an oclif Command', () => {
    expect(QueryCommand).toBeDefined();
    expect(QueryCommand.description).toContain('Hybrid search');
  });

  test('has query argument', () => {
    expect(QueryCommand.args).toBeDefined();
    expect(QueryCommand.args.query).toBeDefined();
  });

  test('has output flags', () => {
    expect(QueryCommand.flags).toBeDefined();
    expect(QueryCommand.flags.json).toBeDefined();
    expect(QueryCommand.flags.n).toBeDefined();
  });

  test('has index flag', () => {
    expect(QueryCommand.flags.index).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new QueryCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
