/**
 * Tests for Embed Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect } from 'bun:test';
import EmbedCommand from './embed.ts';

describe('EmbedCommand', () => {
  test('is an oclif Command', () => {
    expect(EmbedCommand).toBeDefined();
    expect(EmbedCommand.description).toContain('Generate embeddings');
  });

  test('has index flag', () => {
    expect(EmbedCommand.flags).toBeDefined();
    expect(EmbedCommand.flags.index).toBeDefined();
  });

  test('has model flag', () => {
    expect(EmbedCommand.flags.model).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new EmbedCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
