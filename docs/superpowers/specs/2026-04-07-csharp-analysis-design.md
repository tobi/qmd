# QMD C# Analysis Design

Date: 2026-04-07
Status: Proposed
Scope: `qmd` repository only

## Goal

Add C# support to QMD's AST-aware chunking pipeline while preserving QMD's existing architecture:

- Node/Bun remains the primary runtime.
- `--chunk-strategy auto` remains the user-facing entry point.
- C# support degrades gracefully:
  - Roslyn enhanced analysis -> tree-sitter baseline -> regex fallback.

The design must improve chunk quality for `.cs` files immediately, while creating a path for richer C#-specific structure understanding without turning QMD into a .NET-first tool.

## Non-Goals

- Do not redesign QMD into a multi-runtime primary application.
- Do not require .NET to use QMD on non-C# projects.
- Do not introduce full semantic indexing, cross-reference search, inheritance graphs, or call graphs in the first implementation plan.
- Do not change the existing CLI contract beyond improving what happens under `--chunk-strategy auto`.
- Do not include `hydra-p4`-specific ingestion or collection setup in this design.

## Problem Statement

QMD currently supports AST-aware chunking for TypeScript, JavaScript, Python, Go, and Rust. For unsupported file types, including C#, QMD falls back to regex-only chunking.

Regex chunking works, but it is weak for large C# codebases that contain:

- long type declarations with attributes and modifiers
- partial classes and partial structs
- records and file-scoped namespaces
- source-generator-heavy code
- ECS and DI code with many small declaration blocks

In those codebases, regex chunking often splits files at positions that do not align with declaration boundaries. This harms retrieval quality because semantically related code is separated across chunks.

At the same time, QMD's current AST layer is intentionally lightweight. It is used to improve breakpoints for chunking, not to perform compiler-grade language analysis. That constraint is useful and should remain true for the baseline.

## Design Summary

QMD will support C# through a layered model:

1. Regex fallback
2. Tree-sitter C# baseline
3. Optional Roslyn enhanced sidecar

All three layers are hidden behind the existing `--chunk-strategy auto` behavior.

The baseline layer makes `.cs` files first-class participants in AST-aware chunking with low integration cost and no .NET requirement.

The enhanced layer adds C#-specific structure and metadata through a separate Roslyn sidecar process. It is optional, auto-detected, and safe to ignore when unavailable.

## Architecture

### Layer 1: Regex fallback

This is the current behavior and remains the final safety net.

Responsibilities:

- ensure indexing and query chunking always complete
- preserve current behavior on unsupported or failing analysis paths

Constraints:

- never becomes C#-specific
- no new behavior beyond existing fallback semantics

### Layer 2: Tree-sitter C# baseline

This layer extends the existing AST chunking architecture in `src/ast.ts`.

Responsibilities:

- detect `.cs` files
- load `tree-sitter-c-sharp`
- produce structural breakpoints from syntax nodes
- merge those breakpoints into the existing chunking pipeline

Properties:

- no .NET dependency
- same lifecycle as other QMD AST languages
- same graceful failure contract as the current AST implementation

This is the minimum guaranteed structured experience for C#.

### Layer 3: Roslyn enhanced sidecar

This layer is an optional C#-only enhancement.

Responsibilities:

- produce more reliable C# declaration boundaries in cases where syntax-only analysis is weak
- emit symbol metadata that can later be used for chunk labeling or retrieval enhancements

Properties:

- not embedded in the Node process
- invoked as a child process
- can fail or be absent without affecting QMD correctness

This layer is an additive enhancement, not a replacement for the baseline.

## User Experience

The user experience remains:

```bash
qmd embed --chunk-strategy auto
qmd query "inventory operation flow" --chunk-strategy auto
```

For `.cs` files:

- if tree-sitter C# is available, baseline C# AST chunking is used
- if the Roslyn sidecar is also available, enhanced C# signals are added automatically
- if either layer fails, QMD degrades silently to the next lower layer with low-noise warnings

No new mandatory CLI switch is introduced for the first implementation plan.

Optional future debug switches may be added later if needed, but they are not part of this design.

## C# Baseline Breakpoint Model

The baseline layer should extract only declaration-level breakpoints that materially improve chunking.

### Baseline node coverage

Map the following tree-sitter C# nodes into QMD breakpoints:

- `using_directive`
- `namespace_declaration`
- `file_scoped_namespace_declaration`
- `class_declaration`
- `struct_declaration`
- `interface_declaration`
- `record_declaration`
- `enum_declaration`
- `constructor_declaration`
- `method_declaration`

### Baseline breakpoint types

Use QMD-style normalized breakpoint categories:

- `ast:import`
- `ast:namespace`
- `ast:type`
- `ast:enum`
- `ast:ctor`
- `ast:method`

### Baseline scores

Keep the scoring system aligned with QMD's existing AST score model:

- namespace: `100`
- type: `100`
- ctor: `90`
- method: `90`
- enum: `80`
- import: `60`

This preserves compatibility with the current cutoff selection logic and avoids a special case for C#.

### Baseline exclusions

Do not include the following in the first baseline:

- property declarations
- field declarations
- local functions
- attribute lists as standalone breakpoints
- using aliases and other subforms as separate categories

Reason:

- property and field boundaries are often too dense and increase the risk of over-fragmenting chunk selection
- local functions are useful but not necessary for the first stage
- attribute handling is better solved by the Roslyn enhanced layer

## Roslyn Enhanced Signal Model

The enhanced layer is responsible for C#-specific structural precision, not for replacing the entire chunking algorithm.

### Enhanced responsibilities

The first enhanced version should provide:

1. breakpoint refinement
2. symbol metadata

### Breakpoint refinement

Roslyn can improve positions for declarations that are awkward in syntax-only extraction, such as:

- attributed declarations
- partial declarations with modifiers and long headers
- record declarations with primary constructors
- nested type declarations
- declarations where the useful chunk boundary should start before a syntax node that tree-sitter may anchor too narrowly

Enhanced breakpoints should be emitted with `roslyn:*` type names, for example:

- `roslyn:type`
- `roslyn:method`
- `roslyn:ctor`
- `roslyn:namespace`

QMD should merge them using the same existing merge behavior used for AST breakpoints today.

### Symbol metadata

The first enhanced symbol model should include:

- `name`
- `kind`
- `line`
- `containerName`
- `signature`
- `modifiers`

Supported `kind` values in the first version:

- class
- struct
- interface
- record
- enum
- constructor
- method
- property

This metadata is primarily for future retrieval and formatting improvements. It is not required for chunking correctness.

### Enhanced exclusions

The first Roslyn enhanced version explicitly does not include:

- full semantic graph export
- project-wide reference resolution
- inheritance or call graph indexing
- analyzer diagnostics indexing
- symbol cross-linking across files

These would turn the sidecar into a much larger system and are out of scope for the first implementation plan.

## Sidecar Execution Model

### Process model

The Roslyn layer should be implemented as an on-demand CLI sidecar process in the first iteration.

Why:

- easier to implement and debug
- avoids service lifetime and port management complexity
- fits the current QMD usage model, where AST analysis is an indexing/query-time enhancement rather than a persistent language service

### Invocation flow

For `.cs` files under `--chunk-strategy auto`:

1. QMD computes regex breakpoints.
2. QMD computes tree-sitter C# breakpoints when available.
3. QMD checks whether the Roslyn sidecar is available.
4. If available, QMD invokes it for the file.
5. QMD merges enhanced breakpoints into the baseline set.
6. QMD continues chunking with the merged result set.

If sidecar execution fails at any point, QMD continues with the baseline result.

### Availability model

The sidecar is optional.

Detection should occur lazily on first C# usage, not at QMD startup.

QMD should support sidecar discovery through one of:

- explicit config path
- environment variable
- executable available on `PATH`

The implementation plan should choose one primary discovery path and one fallback path, not many parallel mechanisms.

## Sidecar Protocol

Use a simple JSON-over-stdio protocol for the first version.

### Request

```json
{
  "version": 1,
  "language": "csharp",
  "filePath": "src/Foo.cs",
  "content": "using System; ...",
  "features": {
    "breakpoints": true,
    "symbols": true
  }
}
```

### Response

```json
{
  "version": 1,
  "language": "csharp",
  "breakpoints": [
    { "pos": 120, "score": 100, "type": "roslyn:type" },
    { "pos": 420, "score": 90, "type": "roslyn:method" }
  ],
  "symbols": [
    {
      "name": "InventoryOperationRuntimeService",
      "kind": "class",
      "line": 72,
      "containerName": "Bokura.Hydra.World.Client.Session.Inventory",
      "signature": "public sealed class InventoryOperationRuntimeService",
      "modifiers": ["public", "sealed"]
    }
  ],
  "diagnostics": []
}
```

### Protocol requirements

- versioned from the start
- deterministic JSON schema
- zero dependency on long-lived transport
- robust to unknown future fields

### Compatibility policy

- QMD accepts matching major versions only
- protocol mismatch causes downgrade, not hard failure

## Failure and Degradation Policy

Failure handling is a core design requirement.

### Degradation chain

- Roslyn enhanced failure -> baseline tree-sitter C#
- baseline tree-sitter C# failure -> regex
- regex remains the final fallback

### Failure classes

QMD must handle these as non-fatal:

- grammar load failure
- parse failure
- sidecar executable missing
- sidecar timeout
- sidecar non-zero exit
- malformed sidecar JSON
- protocol version mismatch

### Logging policy

Warnings should be low-noise and rate-limited:

- one warning per process per failure class
- no per-file spam in large repositories

This keeps QMD usable in large indexing runs.

## Caching Strategy

### Existing cache

Tree-sitter language and query caches remain unchanged in principle.

### New sidecar caches

Add two sidecar-related caches:

1. capability cache
   - is sidecar available
   - what protocol version does it support

2. file result cache
   - keyed by content hash + feature set + sidecar version

This prevents repeated sidecar work during the same process when the same file content is analyzed more than once.

### Cache invalidation

The file result cache must invalidate when any of the following changes:

- file content hash
- sidecar version
- requested feature set
- protocol major version

## Testing Strategy

Testing should be split across three layers.

### Node-side tests

Add tests for:

- `.cs` language detection
- C# baseline breakpoint extraction
- breakpoint score mapping
- merge behavior between regex, tree-sitter, and Roslyn breakpoints
- sidecar unavailable fallback
- sidecar timeout fallback
- sidecar malformed response fallback
- protocol version mismatch fallback

### Sidecar tests

Add .NET tests for:

- class, struct, interface, enum, record
- file-scoped namespace
- constructor and method extraction
- attributed declarations
- partial declarations
- nested declarations
- stable line and position reporting
- JSON output shape

### End-to-end samples

Add fixtures that resemble real large C# codebases with:

- partial types
- source generator code
- ECS or DI-heavy declarations
- long declaration headers

The purpose is to verify that important declaration units remain intact more often than under regex-only chunking.

## Phased Delivery

### Phase 1: Baseline C# AST chunking

Deliver:

- `.cs` language detection
- `tree-sitter-c-sharp` integration
- C# query definitions
- C# tests
- docs updates

Success criteria:

- `.cs` files participate in `--chunk-strategy auto`
- C# chunk quality improves without any .NET dependency

### Phase 2: Roslyn sidecar with enhanced breakpoints

Deliver:

- sidecar CLI
- JSON protocol v1
- auto-detection and fallback
- enhanced breakpoint merge

Success criteria:

- QMD uses enhanced C# breakpoints automatically when sidecar is available
- absence of sidecar does not break indexing or querying

### Phase 3: Symbol metadata

Deliver:

- symbol extraction in sidecar
- QMD-side symbol handling API
- tests and caching improvements

Success criteria:

- QMD can retain symbol metadata for future retrieval or formatting use
- symbol extraction remains optional and non-fatal

## Implementation Boundaries

The implementation plan derived from this design should remain focused enough for a single feature track if it is split by phase.

Recommended first implementation target:

- complete Phase 1
- scaffold protocol and interfaces for Phase 2 without fully implementing all future capabilities

This keeps the first delivery reviewable and upstream-friendly.

## Risks

### Risk: Roslyn sidecar scope expands too quickly

If the sidecar starts requiring full solution restoration, project graph reconstruction, and semantic compilation for all scenarios, the implementation will become much larger than the chunking problem requires.

Mitigation:

- keep first sidecar input file-oriented
- restrict first output to breakpoints and metadata
- avoid project-wide semantic promises in the first plan

### Risk: Over-fragmented C# breakpoints

Adding too many C# node types can reduce chunk quality by making cuts too eager.

Mitigation:

- keep baseline query narrow
- avoid properties and fields in the first baseline
- validate with large real-world fixtures

### Risk: Cross-platform packaging complexity

Optional multi-runtime tooling can become painful if installation and discovery are vague.

Mitigation:

- make sidecar optional
- choose a single simple discovery mechanism first
- treat sidecar absence as normal, not exceptional

### Risk: User confusion about what "C# support" means

Users may assume semantic search over full C# program structure when the first version primarily improves chunking.

Mitigation:

- document baseline vs enhanced clearly
- describe Roslyn as an optional enhancement layer
- avoid overstating semantic capabilities

## Acceptance Criteria

This design is considered successfully implemented when:

- QMD supports `.cs` under `--chunk-strategy auto`
- baseline C# support works with no .NET requirement
- optional Roslyn enhancement can be auto-detected and used without new mandatory CLI switches
- all failure modes degrade safely
- tests cover language detection, breakpoint extraction, fallback behavior, and protocol handling
- docs explain the layered behavior clearly

## Recommendation

Adopt this design and implement it in phases, starting with baseline tree-sitter C# support.

Reason:

- it delivers immediate user value
- it aligns with QMD's current architecture
- it preserves Node/Bun as the primary runtime
- it creates a safe path toward deeper C# understanding without forcing the entire project into compiler-tooling complexity on day one
