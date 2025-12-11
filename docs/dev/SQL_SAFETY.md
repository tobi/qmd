# SQL Injection Prevention Guidelines

## Rules for All Database Code

### ✅ **ALWAYS USE**: Prepared Statements with Parameters

```typescript
// ✅ CORRECT - Parameter binding
const result = db.prepare("SELECT * FROM documents WHERE hash = ?").get(hash);

// ✅ CORRECT - Named parameters
const result = db.prepare("SELECT * FROM documents WHERE hash = $hash AND active = $active")
  .get({ $hash: hash, $active: 1 });

// ✅ CORRECT - Multiple parameters
const results = db.prepare("SELECT * FROM documents WHERE collection_id = ? AND filepath LIKE ?")
  .all(collectionId, `%${filename}%`);
```

### ❌ **NEVER USE**: String Concatenation or Template Literals

```typescript
// ❌ WRONG - SQL injection vulnerability!
const result = db.prepare(`SELECT * FROM documents WHERE hash = '${hash}'`).get();

// ❌ WRONG - Even with backticks
const result = db.prepare(`SELECT * FROM documents WHERE hash = ${hash}`).get();

// ❌ WRONG - String concatenation
const sql = "SELECT * FROM documents WHERE hash = '" + hash + "'";
const result = db.prepare(sql).get();
```

## Exception: Schema Creation Only

Schema creation (CREATE TABLE, CREATE INDEX) can use string interpolation **only** when:
1. Values come from constants (not user input)
2. Used during initialization
3. Properly validated

```typescript
// ✅ OK - Using constant for table name during init
const dimensions = 384; // constant
db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${dimensions}])`);

// ❌ NEVER - User input in schema
db.exec(`CREATE TABLE ${userTableName} (...)`); // DANGEROUS!
```

## Common Patterns

### Pattern 1: Single Row Fetch
```typescript
function getDocumentByHash(db: Database, hash: string): Document | null {
  return db.prepare("SELECT * FROM documents WHERE hash = ? AND active = 1")
    .get(hash) as Document | null;
}
```

### Pattern 2: Multiple Results
```typescript
function getDocumentsByCollection(db: Database, collectionId: number): Document[] {
  return db.prepare("SELECT * FROM documents WHERE collection_id = ? AND active = 1")
    .all(collectionId) as Document[];
}
```

### Pattern 3: INSERT/UPDATE
```typescript
function insertDocument(db: Database, doc: Partial<Document>): void {
  db.prepare(`
    INSERT INTO documents (collection_id, filepath, hash, title, body, modified_at)
    VALUES (?, ?, ?, ?, ?, ?)
  `).run(
    doc.collection_id,
    doc.filepath,
    doc.hash,
    doc.title,
    doc.body,
    doc.modified_at
  );
}
```

### Pattern 4: Dynamic WHERE Clauses (Safe)
```typescript
function searchDocuments(db: Database, filters: { hash?: string; collectionId?: number }): Document[] {
  const conditions: string[] = ["active = 1"];
  const params: any[] = [];

  if (filters.hash) {
    conditions.push("hash = ?");
    params.push(filters.hash);
  }

  if (filters.collectionId) {
    conditions.push("collection_id = ?");
    params.push(filters.collectionId);
  }

  const sql = `SELECT * FROM documents WHERE ${conditions.join(" AND ")}`;
  return db.prepare(sql).all(...params) as Document[];
}
```

### Pattern 5: LIKE Queries (Safe)
```typescript
function searchByFilename(db: Database, pattern: string): Document[] {
  // User input goes in parameter, not in SQL string
  return db.prepare("SELECT * FROM documents WHERE filepath LIKE ? AND active = 1")
    .all(`%${pattern}%`) as Document[];
}
```

## FTS5 Queries (Special Case)

FTS5 has its own syntax but still uses parameters:

```typescript
// ✅ CORRECT
function searchFTS(db: Database, query: string): SearchResult[] {
  // Sanitize FTS5 query syntax first
  const sanitized = sanitizeFTS5Query(query);

  // Use parameter binding
  return db.prepare("SELECT * FROM documents_fts WHERE documents_fts MATCH ?")
    .all(sanitized) as SearchResult[];
}

function sanitizeFTS5Query(query: string): string {
  // Remove FTS5 special chars that could break query structure
  // But this is query syntax sanitization, NOT SQL injection prevention
  return query.replace(/[:"(){}[\]^]/g, ' ').trim();
}
```

## Testing for SQL Injection

Test with malicious inputs:

```typescript
// These should NOT cause errors or unexpected behavior
const maliciousInputs = [
  "'; DROP TABLE documents; --",
  "' OR 1=1 --",
  "admin'--",
  "' UNION SELECT * FROM users --",
];

for (const input of maliciousInputs) {
  const result = db.prepare("SELECT * FROM documents WHERE hash = ?").get(input);
  // Should safely return no results, not execute malicious SQL
}
```

## Code Review Checklist

Before merging any database code:

- [ ] All queries use prepared statements with `?` or `$param` placeholders
- [ ] No string concatenation or template literals in SQL queries
- [ ] No user input directly in SQL strings
- [ ] FTS5 queries use parameter binding
- [ ] Dynamic queries build parameter arrays
- [ ] Tested with malicious inputs

## Resources

- Bun SQLite docs: https://bun.sh/docs/api/sqlite
- SQLite prepared statements: https://www.sqlite.org/c3ref/prepare.html
- OWASP SQL Injection: https://owasp.org/www-community/attacks/SQL_Injection
