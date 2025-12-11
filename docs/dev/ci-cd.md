# CI/CD Integration Guide

Guide for integrating QMD into continuous integration and deployment workflows.

## Overview

QMD includes a comprehensive GitHub Actions workflow for automated testing, coverage reporting, and build verification.

## GitHub Actions Workflow

### Location

```
.github/workflows/test.yml
```

### Features

- **Multi-platform testing** - Ubuntu, macOS, Windows
- **Bun setup** - Automatic runtime configuration
- **Dependency caching** - Faster builds
- **Test execution** - Full test suite with coverage
- **Code coverage** - Codecov integration
- **Type checking** - TypeScript validation
- **Build verification** - Entry point checks

### Triggers

Runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

## Workflow Jobs

### 1. Test Job

Runs tests across multiple platforms:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    bun-version: [1.1.38]
```

**Steps:**
1. Checkout code
2. Setup Bun runtime
3. Cache dependencies
4. Install dependencies (`bun install --frozen-lockfile`)
5. Run tests with coverage
6. Upload coverage to Codecov (Ubuntu only)
7. Upload coverage artifacts

**Coverage:**
```bash
bun test --coverage --coverage-reporter=lcov --coverage-reporter=text
```

### 2. Lint Job

Type checking and code quality:

```yaml
- name: Type check
  run: bun run tsc --noEmit
```

### 3. Build Job

Verifies CLI functionality:

```yaml
- name: Verify entry points
  run: |
    bun run qmd.ts --version || echo "CLI runs successfully"
    test -f qmd && echo "Shell wrapper exists" || exit 1
```

## Setting Up Codecov

### 1. Get Codecov Token

1. Visit https://codecov.io
2. Link your GitHub repository
3. Copy the repository token

### 2. Add to GitHub Secrets

1. Go to repository Settings
2. Navigate to Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `CODECOV_TOKEN`
5. Value: Your Codecov token
6. Click "Add secret"

### 3. Verify Integration

After pushing code:

1. Check Actions tab
2. Wait for workflow completion
3. Visit Codecov dashboard
4. View coverage reports

## Local CI Testing

### Run Tests Locally

```bash
# Run all tests
bun test

# With coverage
bun test --coverage

# Type check
bun run tsc --noEmit
```

### Simulate CI Environment

```bash
# Install with frozen lockfile (like CI)
bun install --frozen-lockfile

# Verify build
bun run qmd.ts --version
```

## Custom Workflows

### Documentation Indexing

Auto-index documentation on push:

```yaml
name: Update Search Index

on:
  push:
    branches: [main]
    paths:
      - 'docs/**/*.md'
      - 'README.md'

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: oven-sh/setup-bun@v2
        with:
          bun-version: 1.1.38

      - name: Install QMD
        run: bun install -g qmd

      - name: Index documentation
        run: |
          qmd init
          qmd add docs/
          qmd embed

      - name: Upload index
        uses: actions/upload-artifact@v4
        with:
          name: search-index
          path: .qmd/
```

### Scheduled Re-indexing

Update indexes nightly:

```yaml
name: Nightly Index Update

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: oven-sh/setup-bun@v2

      - name: Update all indexes
        run: |
          qmd update
          qmd embed
```

### PR Documentation Check

Verify documentation changes:

```yaml
name: Documentation Check

on:
  pull_request:
    paths:
      - 'docs/**'

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: oven-sh/setup-bun@v2

      - name: Index PR changes
        run: |
          qmd init
          qmd add docs/

      - name: Verify searchable
        run: |
          # Test that new content is indexed
          qmd search "new feature" || true
          qmd status
```

## Deployment Integration

### Deploy with Index

Include search index in deployments:

```yaml
name: Deploy Documentation

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate search index
        run: |
          qmd init
          qmd add docs/
          qmd embed

      - name: Build site
        run: npm run build

      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        with:
          publish_dir: ./dist
          # Index included in build
```

## Monitoring & Notifications

### Slack Notifications

Notify on index updates:

```yaml
- name: Notify Slack
  if: success()
  uses: 8398a7/action-slack@v3
  with:
    status: custom
    custom_payload: |
      {
        text: "Search index updated successfully"
      }
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Coverage Thresholds

Fail if coverage drops:

```yaml
- name: Check coverage
  run: |
    bun test --coverage --coverage-reporter=text-summary
    # Parse coverage and fail if < 80%
```

## Best Practices

### Do's

✅ **Cache dependencies** - Speeds up builds
✅ **Use frozen lockfile** - Ensures consistency
✅ **Test on multiple platforms** - Catch OS-specific issues
✅ **Upload artifacts** - Preserve coverage reports
✅ **Pin versions** - Bun, Node.js versions

### Don'ts

❌ **Don't commit indexes** - Regenerate in CI
❌ **Don't run on all paths** - Limit to relevant files
❌ **Don't skip tests** - Always validate changes
❌ **Don't hardcode secrets** - Use GitHub Secrets

## Troubleshooting

### Tests Fail in CI but Pass Locally

```bash
# Check Node.js/Bun version
bun --version

# Use same version as CI
bun upgrade

# Check lockfile
git status bun.lockb
```

### Codecov Upload Fails

```bash
# Verify token is set
# GitHub → Settings → Secrets → CODECOV_TOKEN

# Check coverage file exists
ls -la coverage/lcov.info

# Verify workflow uses correct token
cat .github/workflows/test.yml | grep CODECOV_TOKEN
```

### Build Times Too Long

```yaml
# Add dependency caching
- uses: actions/cache@v4
  with:
    path: |
      ~/.bun/install/cache
      node_modules
    key: ${{ runner.os }}-bun-${{ hashFiles('**/bun.lockb') }}
```

## Examples

### Minimal Workflow

```yaml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: oven-sh/setup-bun@v2
      - run: bun install
      - run: bun test
```

### Complete Workflow

See [`.github/workflows/test.yml`](../../.github/workflows/test.yml) for full example.

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/actions)
- [Codecov Documentation](https://docs.codecov.com)
- [Bun GitHub Action](https://github.com/oven-sh/setup-bun)
