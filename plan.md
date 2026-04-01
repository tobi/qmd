# Ebbinghaus Memory Decay Implementation Plan

## Phase 1: Core Module
- [ ] Create `src/decay.ts` with decay math (computeStrength, types, constants)

## Phase 2: Database Schema
- [ ] Add migration in `initializeDatabase()` (store.ts) for new columns on `documents` table:
  - `importance REAL DEFAULT 0.5`
  - `recall_count INTEGER DEFAULT 0`
  - `last_recalled_at TEXT`
  - `category TEXT DEFAULT 'fact'`

## Phase 3: Store Integration
- [ ] Add decay-related DB accessor functions in `store.ts`
  - `getDecayStats()` — count by category, avg strength, prune candidates
  - `setImportance()` — set importance for a document by path
  - `setCategory()` — set category for a document by path
  - `recordRecall()` — increment recall_count and update last_recalled_at
  - `pruneLowStrength()` — delete documents below PRUNE_THRESHOLD
  - `getDocumentDecayInfo()` — get decay columns for a document
- [ ] Add new methods to Store type and createStore()
- [ ] Integrate decay strength into search scoring (after RRF fusion, multiply by strength)
- [ ] Record recalls when search results are returned

## Phase 4: CLI Commands
- [ ] Add `qmd decay` command — show decay statistics
- [ ] Add `qmd decay --run` — prune low-strength documents
- [ ] Add `qmd importance set <path> <value>` command
- [ ] Add `qmd category set <path> <category>` command
- [ ] Update help text with new commands
- [ ] Export new functions from store.ts for CLI imports

## Phase 5: Build & Test
- [ ] Run `bun run build` to compile TypeScript
- [ ] Verify `qmd --help` shows new commands
- [ ] Run existing tests to check for regressions

## Phase 6: Commit
- [ ] Create clean commit with message: `feat: add Ebbinghaus memory decay with recall scoring`
