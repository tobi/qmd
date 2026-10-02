# QMD interface

QMD is a command-line search tool, library, and MCP server. This change has no
graphical surface. Existing command output and structured result formats are
the observed interface contracts; see README.md and test/cli.test.ts.

The doctor vector check samples stored chunks and compares freshly generated
embeddings with their saved vectors. Its sampling must preserve active-document,
model, and fingerprint filters without expanding full document bodies across
all candidate chunks. No visual or typography changes are part of this work.
The saved character position identifies the passage to re-embed; a stored
sequence number is the vector key and may differ from today's chunk ordering.
