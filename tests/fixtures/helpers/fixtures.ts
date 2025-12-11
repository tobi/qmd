/**
 * Test fixtures and sample data
 * Provides reusable test data for various test scenarios
 */

/**
 * Sample markdown documents for testing
 */
export const sampleDocs = {
  simple: '# Simple Document\n\nThis is a simple test document.',

  withCode: `# Code Example

Here's some code:

\`\`\`javascript
const x = 1;
console.log(x);
\`\`\`

And some more text.`,

  withLinks: `# Links Example

Check out [this link](https://example.com).

And an [internal link](./other-doc.md).`,

  long: '# Long Document\n\n' + 'Lorem ipsum dolor sit amet. '.repeat(500),

  unicode: `# Unicode Test

Japanese: 日本語
Emoji: 🎉 🚀 ✨
Accents: café, naïve, résumé`,

  empty: '',

  onlyTitle: '# Title Only',

  malformed: `# Incomplete Markdown

[broken link](

Unclosed code block:
\`\`\`javascript
const x = 1;
`,

  withHeadings: `# Main Title

## Subsection 1

Content here.

## Subsection 2

More content.

### Sub-subsection

Even more content.`,

  multipleHashTitles: `# First Title

Some content.

# Second Title (should not be extracted as title)

More content.`,
};

/**
 * Sample embeddings (mocked vector data)
 */
export const sampleEmbeddings = {
  /** 128-dimensional embedding (nomic-embed-text default) */
  dim128: Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.1)),

  /** 256-dimensional embedding */
  dim256: Array.from({ length: 256 }, (_, i) => Math.cos(i * 0.1)),

  /** 1024-dimensional embedding */
  dim1024: Array.from({ length: 1024 }, (_, i) => (i % 2 === 0 ? 0.1 : -0.1)),

  /** Zero vector */
  zero128: Array.from({ length: 128 }, () => 0),

  /** Unit vector (all ones) */
  ones128: Array.from({ length: 128 }, () => 1),

  /** Random-like vector */
  random128: Array.from({ length: 128 }, (_, i) => Math.sin(i * 0.123) * Math.cos(i * 0.456)),
};

/**
 * SQL injection attack payloads for security testing
 * CRITICAL: All repository tests must verify these are handled safely
 */
export const sqlInjectionPayloads = [
  // Classic SQL injection
  "'; DROP TABLE documents; --",
  "' OR 1=1 --",
  "' OR '1'='1",
  "admin'--",
  "' OR 'a'='a",

  // UNION-based injection
  "' UNION SELECT * FROM users --",
  "' UNION SELECT NULL, username, password FROM users --",

  // Comment-based injection
  "' /*",
  "*/ OR 1=1 --",

  // Blind SQL injection
  "' AND 1=1 --",
  "' AND 1=2 --",

  // Time-based blind injection
  "'; WAITFOR DELAY '00:00:05' --",
  "' OR SLEEP(5) --",

  // Stacked queries
  "1; DELETE FROM collections WHERE 1=1 --",
  "1; UPDATE documents SET active=0 --",

  // Boolean-based injection
  "' AND '1'='1",
  "' AND '1'='2",

  // Escaped quotes
  "\\'",
  "\\' OR 1=1 --",

  // Hex encoding
  "0x27",
  "0x27 OR 1=1 --",
];

/**
 * Sample file paths for testing path operations
 */
export const samplePaths = {
  absolute: '/home/user/projects/qmd/docs/README.md',
  relative: './docs/README.md',
  withSpaces: '/home/user/My Documents/notes.md',
  withUnicode: '/home/user/文档/日本語.md',
  deeply_nested: '/home/user/projects/qmd/docs/api/endpoints/search/vector.md',
  windowsStyle: 'C:\\Users\\user\\Documents\\notes.md',
};

/**
 * Sample search queries
 */
export const sampleQueries = {
  simple: 'test query',
  multiWord: 'search for documents',
  withOperators: 'test AND query OR example',
  quoted: '"exact phrase"',
  wildcard: 'test*',
  fts5Operators: 'test NEAR query',
  empty: '',
  veryLong: 'word '.repeat(100).trim(),
};

/**
 * Sample collection data
 */
export const sampleCollections = [
  {
    pwd: '/home/user/projects/qmd',
    glob_pattern: '**/*.md',
    created_at: '2024-01-01T00:00:00.000Z',
  },
  {
    pwd: '/home/user/docs',
    glob_pattern: 'docs/**/*.md',
    created_at: '2024-01-02T00:00:00.000Z',
  },
  {
    pwd: '/tmp/test',
    glob_pattern: '*.md',
    created_at: '2024-01-03T00:00:00.000Z',
  },
];

/**
 * Sample document data
 */
export const sampleDocuments = [
  {
    name: 'readme',
    title: 'README',
    hash: 'hash_readme_123',
    filepath: '/home/user/projects/qmd/README.md',
    display_path: 'README',
    body: sampleDocs.simple,
  },
  {
    name: 'architecture',
    title: 'Architecture Guide',
    hash: 'hash_arch_456',
    filepath: '/home/user/projects/qmd/docs/ARCHITECTURE.md',
    display_path: 'docs/ARCHITECTURE',
    body: sampleDocs.withHeadings,
  },
  {
    name: 'api',
    title: 'API Documentation',
    hash: 'hash_api_789',
    filepath: '/home/user/projects/qmd/docs/api/API.md',
    display_path: 'docs/api/API',
    body: sampleDocs.withCode,
  },
];

/**
 * Sample search results for testing ranking algorithms
 */
export const sampleSearchResults = [
  {
    file: '/path/to/doc1.md',
    displayPath: 'doc1',
    title: 'Document 1',
    body: 'Content 1',
    score: 0.9,
    source: 'fts' as const,
  },
  {
    file: '/path/to/doc2.md',
    displayPath: 'doc2',
    title: 'Document 2',
    body: 'Content 2',
    score: 0.8,
    source: 'fts' as const,
  },
  {
    file: '/path/to/doc3.md',
    displayPath: 'doc3',
    title: 'Document 3',
    body: 'Content 3',
    score: 0.7,
    source: 'vector' as const,
  },
];

/**
 * Sample reranking responses (for mocking Ollama reranker)
 */
export const sampleRerankingResponses = {
  yes: 'yes',
  no: 'no',
  yesWithConfidence: 'yes (95% confidence)',
  uncertain: 'maybe',
  invalid: 'not a valid response',
};

/**
 * Edge cases for various functions
 */
export const edgeCases = {
  emptyString: '',
  whitespace: '   \n\t  ',
  nullByte: '\0',
  veryLongString: 'x'.repeat(100000),
  specialChars: '!@#$%^&*()_+-=[]{}|;:\'",.<>?/`~',
  unicode: '你好世界 🌍 مرحبا العالم',
  newlines: 'line1\nline2\r\nline3\rline4',
  tabs: 'col1\tcol2\tcol3',
};

/**
 * Timing constants for performance testing
 */
export const performanceThresholds = {
  /** Maximum time for hash computation (ms) */
  hashComputation: 10,

  /** Maximum time for FTS search (ms) */
  ftsSearch: 100,

  /** Maximum time for vector search (ms) */
  vectorSearch: 200,

  /** Maximum time for embedding (ms, mocked) */
  embeddingMocked: 50,

  /** Maximum time for reranking (ms, mocked) */
  rerankingMocked: 100,
};
