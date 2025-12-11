# QMD Build Process

## TL;DR: Compilation Doesn't Work

❌ **`bun build --compile` does NOT work for QMD**
✅ **Use the existing shell wrapper approach**

## What We Tested

### Test 1: Compiled Binary (`bun run build`)
```bash
bun build bin/run --compile --outfile builds/qmd
```

**Result:**
- ✅ Binary created (101MB)
- ✅ Runs without errors (exit code 0)
- ❌ **Produces NO output**
- ❌ All commands fail silently

**Issue:** Bun's `--compile` flag doesn't properly handle oclif's dynamic imports and output streams.

### Test 2: Bundled Version (`bun run build:bundle`)
```bash
bun build bin/run --target bun --outdir builds
```

**Result:**
- ❌ Doesn't properly bundle the code
- Creates only a stub file with asset reference
- oclif's dynamic command loading isn't compatible with Bun's bundler

## Why Compilation Fails

1. **oclif uses dynamic imports** for command discovery
2. **stdout/stderr handling** in compiled Bun binaries has issues
3. **Bun's bundler** can't properly handle oclif's architecture
4. **Not specifically a sqlite-vec issue** - the entire binary doesn't work

## Working Solution: Current Setup

The existing setup works perfectly:

```
./qmd (shell wrapper)
  ↓
bin/run (Bun script)
  ↓
@oclif/core
  ↓
src/commands/*
```

**Benefits:**
- ✅ All features work (including sqlite-vec)
- ✅ Fast startup time
- ✅ Easy debugging
- ✅ Simple updates

## Distribution Options

### Option 1: Require Bun on Target Machine (Recommended)
```bash
# On target machine:
curl -fsSL https://bun.sh/install | bash
git clone <repo>
cd qmd
bun install
bun link  # or use ./qmd directly
```

**Pros:**
- Everything works
- Bun installs in seconds
- sqlite-vec works perfectly

### Option 2: Docker Container
```dockerfile
FROM oven/bun:1
WORKDIR /app
COPY . .
RUN bun install
ENTRYPOINT ["./qmd"]
```

**Pros:**
- Portable across machines
- No Bun installation needed on host
- Consistent environment

### Option 3: Package Manager Installation
Publish to npm/bun registry with Bun as peer dependency.

## Build Scripts

Available in `package.json`:

```bash
bun run build         # Compile binary (doesn't work)
bun run build:bundle  # Bundle code (doesn't work)
```

These are included for testing/experimentation, but **not recommended for production use**.

## Conclusion

**For now, stick with the shell wrapper approach.** It's simple, works perfectly, and requires minimal dependencies (just Bun).

If Bun improves `--compile` support for oclif in the future, we can revisit this.

## Testing Results

- **Compiled binary**: Exits with code 0 but produces no output
- **Regular wrapper**: Full output with formatting ✓
- **sqlite-vec**: Works in regular setup, untestable in compiled binary (no output at all)

---

**Last Updated:** 2025-12-11
**Bun Version:** 1.3.0
**oclif Version:** 4.8.0
