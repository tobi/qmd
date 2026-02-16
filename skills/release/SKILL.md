---
name: release
description: Manage releases for this project. Validates changelog, installs git hooks, and cuts releases. Use when user says "/release", "release 1.0.5", "cut a release", or asks about the release process. NOT auto-invoked by the model.
disable-model-invocation: true
---

# Release

Cut a release, validate the changelog, and ensure git hooks are installed.

## Usage

`/release 1.0.5` or `/release patch` (bumps patch from current version).

## Process

When the user triggers `/release <version>`:

1. **Gather context** — run `skills/release/scripts/release-context.sh <version>`.
   This silently installs git hooks and prints everything needed: version info,
   working directory status, commits since last release, files changed, current
   `[Unreleased]` content, and the previous release entry for style reference.

2. **Commit outstanding work** — if the context shows staged, modified, or
   untracked files that belong in this release, commit them first. Use the
   /commit skill or make well-formed commits directly.

3. **Write the changelog** — if `[Unreleased]` is empty, write it now using
   the commits and file changes from the context output. Follow the changelog
   standard below. Re-run the context script after committing if needed.

4. **Cut the release** — run `scripts/release.sh <version>`. This renames
   `[Unreleased]` → `[X.Y.Z] - date`, inserts a fresh `[Unreleased]`,
   bumps `package.json`, commits, and tags.

5. **Show the final changelog** — print the full `[Unreleased]` +
   minor series rollup via `scripts/extract-changelog.sh <version>`.
   Ask the user to confirm before pushing.

6. **Push** — after explicit confirmation, run `git push origin main --tags`.

7. **Watch CI** — after the push, start a background dispatch to watch the
   publish workflow. Use `interactive_shell` in dispatch mode with:
   ```
   gh run watch $(gh run list --workflow=publish.yml --limit=1 --json databaseId --jq '.[0].databaseId') --exit-status
   ```
   The agent will be notified when CI completes and should report the result.

If any step fails, stop and explain. Never force-push or skip validation.

## Changelog Standard

The changelog lives in `CHANGELOG.md` and follows [Keep a Changelog](https://keepachangelog.com/) conventions.

### Heading format

- `## [Unreleased]` — accumulates entries between releases
- `## [X.Y.Z] - YYYY-MM-DD` — released versions

### Structure of a release entry

Each version entry has two parts:

**1. Highlights (optional, 1-4 sentences of prose)**

Immediately after the version heading, before any `###` section. The elevator
pitch — what would you tell someone in 30 seconds? Only for significant
releases; skip for small patches.

```markdown
## [1.1.0] - 2026-03-01

QMD now runs on both Node.js and Bun, with up to 2.7x faster reranking
through parallel contexts. GPU auto-detection replaces the unreliable
`gpu: "auto"` with explicit CUDA/Metal/Vulkan probing.
```

**2. Detailed changelog (`### Changes` and `### Fixes`)**

```markdown
### Changes

- Runtime: support Node.js (>=22) alongside Bun. The `qmd` wrapper
  auto-detects a suitable install via PATH. #149 (thanks @igrigorik)
- Performance: parallel embedding & reranking — up to 2.7x faster on
  multi-core machines.

### Fixes

- Prevent VRAM waste from duplicate context creation during concurrent
  `embedBatch` calls. #152 (thanks @jkrems)
```

### Writing guidelines

- **Explain the why, not just the what.** The changelog is for users.
- **Include numbers.** "2.7x faster", "17x less memory".
- **Group by theme, not by file.** "Performance" not "Changes to llm.ts".
- **Don't list every commit.** Aggregate related changes.
- **Credit contributors:** end bullets with `#NNN (thanks @username)` for
  external PRs. No need to credit the repo owner.

### What not to include

- Internal refactors with no user-visible effect
- Dependency bumps (unless fixing a user-facing bug)
- CI/tooling changes (unless affecting the release artifact)
- Test additions (unless validating a fix worth mentioning)

## GitHub Release Notes

Each GitHub release includes the full changelog for the **minor series** back
to x.x.0. The `scripts/extract-changelog.sh` script handles this, and the
publish workflow (`publish.yml`) calls it to populate the GitHub release.

## Git Hooks

The pre-push hook (`scripts/pre-push`) blocks `v*` tag pushes unless:

1. `package.json` version matches the tag
2. `CHANGELOG.md` has a `## [X.Y.Z] - date` entry for the version
3. CI passed on GitHub (warns in non-interactive shells, blocks in terminals)

Hooks are installed silently by the context script. They can also be installed
manually via `skills/release/scripts/install-hooks.sh` or automatically via
`bun install` (prepare script).
