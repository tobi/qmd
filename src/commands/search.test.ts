/**
 * Tests for Search Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect } from 'bun:test';
import SearchCommand from './search.ts';

describe('SearchCommand', () => {
  test('is an oclif Command', () => {
    expect(SearchCommand).toBeDefined();
    expect(SearchCommand.description).toContain('Full-text search');
  });

  test('has query argument', () => {
    expect(SearchCommand.args).toBeDefined();
    expect(SearchCommand.args.query).toBeDefined();
  });

  test('has output flags', () => {
    expect(SearchCommand.flags).toBeDefined();
    expect(SearchCommand.flags.json).toBeDefined();
    expect(SearchCommand.flags.n).toBeDefined();
  });

  test('has index flag', () => {
    expect(SearchCommand.flags.index).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new SearchCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
