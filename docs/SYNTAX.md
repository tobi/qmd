# QMD Query Syntax

QMD queries are structured documents with typed sub-queries. Each line specifies a search type and query text.

## Grammar

```ebnf
query_document = { line } ;
line           = [ type ":" ] text newline ;
type           = "lex" | "vec" | "hyde" | "expand" ;
text           = quoted_phrase | plain_text ;
quoted_phrase  = '"' { character } '"' ;
plain_text     = { character } ;
newline        = "\n" ;
```

## Query Types

| Type | Method | Description |
|------|--------|-------------|
| `lex` | BM25 | Keyword search with exact matching |
| `vec` | Vector | Semantic similarity search |
| `hyde` | Vector | Hypothetical document embedding |
| `expand` | LLM | Auto-expand into lex/vec/hyde via local model |

## Default Behavior

A query without any type prefix is treated as `expand:` — it gets passed to the query expansion model which generates lex, vec, and hyde variations automatically.

```
# These are equivalent:
how does authentication work
expand: how does authentication work
```

## Lex Query Syntax

Lex queries support special syntax for precise keyword matching:

```ebnf
lex_query   = { lex_term } ;
lex_term    = negation | phrase | word ;
negation    = "-" ( phrase | word ) ;
phrase      = '"' { character } '"' ;
word        = { letter | digit | "'" } ;
```

| Syntax | Meaning | Example |
|--------|---------|---------|
| `word` | Prefix match | `perf` matches "performance" |
| `"phrase"` | Exact phrase | `"rate limiter"` |
| `-word` | Exclude term | `-sports` |
| `-"phrase"` | Exclude phrase | `-"test data"` |

### Examples

```
lex: CAP theorem consistency
lex: "machine learning" -"deep learning"
lex: auth -oauth -saml
```

## Vec Query Syntax

Vec queries are natural language questions. No special syntax — just write what you're looking for.

```
vec: how does the rate limiter handle burst traffic
vec: what is the tradeoff between consistency and availability
```

## Hyde Query Syntax

Hyde queries are hypothetical answer passages (50-100 words). Write what you expect the answer to look like.

```
hyde: The rate limiter uses a sliding window algorithm with a 60-second window. When a client exceeds 100 requests per minute, subsequent requests return 429 Too Many Requests.
```

## Multi-Line Queries

Combine multiple query types for best results. First query gets 2x weight in fusion.

```
lex: rate limiter algorithm
vec: how does rate limiting work in the API
hyde: The API implements rate limiting using a token bucket algorithm...
```

## Expand Queries

Use `expand:` to leverage the local query expansion model. Limited to one per query document.

```
expand: error handling best practices
```

This generates lex, vec, and hyde variations automatically. Useful when you don't know the exact terms.

## Constraints

- Maximum one `expand:` query per document
- `lex` syntax (`-term`, `"phrase"`) only works in lex queries
- Empty lines are ignored
- Leading/trailing whitespace is trimmed

## MCP/HTTP API

The `query` tool accepts a query document:

```json
{
  "q": "lex: CAP theorem\nvec: consistency vs availability",
  "collections": ["docs"],
  "limit": 10
}
```

Or structured format:

```json
{
  "searches": [
    { "type": "lex", "query": "CAP theorem" },
    { "type": "vec", "query": "consistency vs availability" }
  ]
}
```

## CLI

```bash
# Single line (implicit expand)
qmd query "how does auth work"

# Multi-line with types
qmd query $'lex: auth token\nvec: how does authentication work'

# Structured
qmd query $'lex: keywords\nvec: question\nhyde: hypothetical answer...'
```
