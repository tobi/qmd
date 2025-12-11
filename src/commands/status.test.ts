/**
 * Tests for Status Command
 * Target coverage: 60%+ (integration-style)
 */

import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { Database } from 'bun:sqlite';
import StatusCommand from './status.ts';
import { createTestDb, cleanupDb } from '../../tests/fixtures/helpers/test-db.ts';
import { getDbPath } from '../utils/paths.ts';
import { writeFileSync, unlinkSync, existsSync } from 'fs';

describe('StatusCommand', () => {
  let db: Database;
  let testDbPath: string;

  beforeEach(() => {
    testDbPath = getDbPath('test-status');
    db = createTestDb();
  });

  afterEach(() => {
    cleanupDb(db);
    if (existsSync(testDbPath)) {
      unlinkSync(testDbPath);
    }
  });

  test('is an oclif Command', () => {
    expect(StatusCommand).toBeDefined();
    expect(StatusCommand.description).toBe('Show index status and collections');
  });

  test('has index flag', () => {
    expect(StatusCommand.flags).toBeDefined();
    expect(StatusCommand.flags.index).toBeDefined();
  });

  test('can be instantiated', () => {
    const command = new StatusCommand([], {} as any);
    expect(command).toBeDefined();
  });
});
