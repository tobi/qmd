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

1. **Install hooks** — run `scripts/install-hooks.sh` (idempotent)
2. **Check for unstaged work** — run `git status`. If there are valuable
   unstaged/uncommitted changes that belong in this release, commit them
   first (use the commit skill or make well-formed commits directly).
3. **Validate changelog** — check if `## [Unreleased]` has content.
   If empty or missing, **write the changelog now** (see below).
4. **Preview** — show the user what will be released (unreleased content +
   minor series rollup via `scripts/extract-changelog.sh`)
5. **Ask for confirmation** — do NOT proceed without explicit user approval
6. **Run `scripts/release.sh <version>`** — renames `[Unreleased]`, bumps
   version, commits, tags
7. **Remind** — tell the user to `git push origin main --tags`

If any step fails, stop and explain. Never force-push or skip validation.

### Writing the changelog from git history

If `[Unreleased]` is empty when the release is triggered, populate it before
continuing:

1. Find the last release tag: `git describe --tags --abbrev=0`
2. Review commits since then: `git log <tag>..HEAD --oneline`
3. Read the diffs for anything non-obvious: `git log <tag>..HEAD --stat`
4. Write changelog entries following the standard below — group by theme,
   explain the why, include numbers, credit contributors.
5. Add optional prose highlights (1-4 sentences) if the release warrants it.
6. Write the entries into `## [Unreleased]` in CHANGELOG.md.
7. Commit the changelog update, then continue with the release.

## Changelog Standard

The changelog lives in `CHANGELOG.md` and follows [Keep a Changelog](https://keepachangelog.com/) conventions.

### Heading format

- `## [Unreleased]` — accumulates entries between releases
- `## [X.Y.Z] - YYYY-MM-DD` — released versions

The release script renames `[Unreleased]` → `[X.Y.Z] - date` and inserts a
fresh empty `[Unreleased]` section automatically.

### Structure of a release entry

Each version entry has two parts:

**1. Highlights (optional, 1-4 sentences of prose)**

Immediately after the version heading, before any `###` section. This is the
elevator pitch — what would you tell someone in 30 seconds? Only include for
releases with significant changes. Skip for small patches.

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
to x.x.0. Releasing v1.2.3 includes entries for 1.2.3, 1.2.2, 1.2.1, and
1.2.0. The `scripts/extract-changelog.sh` script handles this, and the
publish workflow (`publish.yml`) calls it to populate the GitHub release.

## Git Hooks

The pre-push hook (`scripts/pre-push`) blocks `v*` tag pushes unless:

1. `package.json` version matches the tag
2. `CHANGELOG.md` has a `## [X.Y.Z] - date` entry for the version
3. CI passed on GitHub for the tagged commit

Run `skills/release/scripts/install-hooks.sh` to install (also runs
automatically via `bun install` prepare script).

## Scripts

- [`scripts/install-hooks.sh`](scripts/install-hooks.sh) — install/update git hooks
- Project scripts used during release:
  - `scripts/release.sh` — rename [Unreleased], bump version, commit, tag
  - `scripts/extract-changelog.sh` — extract minor series notes for GitHub release
  - `scripts/pre-push` — pre-push validation hook
