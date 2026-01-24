# QMD Query Expansion Scoring

## Goal

Transform a random typed query into a great set of retrieval-optimized expansions.

**Input:** `"auth config"`
**Output:**
```
lex: authentication configuration
lex: auth settings setup
vec: how to configure authentication settings
vec: authentication configuration options
hyde: Authentication can be configured by setting the AUTH_SECRET environment variable and enabling the auth middleware in your application's config file.
```

## Output Format

| Prefix | Purpose | Required | Count |
|--------|---------|----------|-------|
| `lex:` | BM25 keyword variations (shorter, keyword-focused) | Yes | 1-3 |
| `vec:` | Semantic reformulations (natural language) | Yes | 1-3 |
| `hyde:` | Hypothetical document passage | Optional | 0-1 |

## Scoring Criteria

### 1. Format Compliance (0-30 points)

| Criterion | Points | Deduction |
|-----------|--------|-----------|
| Has at least one `lex:` line | +10 | -10 if missing |
| Has at least one `vec:` line | +10 | -10 if missing |
| All lines have valid prefix (`lex:`, `vec:`, `hyde:`) | +10 | -5 per invalid line |
| No garbage/prose outside of prefixed lines | - | -10 if present |

### 2. Diversity & Coverage (0-30 points)

| Criterion | Points | Deduction |
|-----------|--------|-----------|
| 2+ different types present (lex + vec) | +10 | -10 if only one type |
| 2+ total expansions | +5 | -5 if only one |
| Multiple lex: lines are diverse (edit distance > 3) | +5 | -2 per duplicate pair |
| Multiple vec: lines are diverse (edit distance > 5) | +5 | -2 per duplicate pair |
| lex/vec not identical to original query | +5 | -5 per line that equals query |

### 3. Hyde Quality (0-20 points, optional bonus)

| Criterion | Points | Deduction |
|-----------|--------|-----------|
| Hyde present and well-formed | +5 | - |
| Hyde is concise (50-200 chars) | +5 | -3 if too short, -5 if too long |
| Hyde has no newlines | +5 | -5 if contains newlines |
| Hyde has no excessive repetition | +5 | -3 if word repeats 3+ times |

### 4. Content Quality (0-20 points)

| Criterion | Points | Deduction |
|-----------|--------|-----------|
| Expansions are relevant to query | +10 | Subjective |
| Lex lines are keyword-focused (shorter) | +5 | -2 if lex is longer than vec |
| Vec lines are natural language (complete phrases) | +5 | -2 if vec is just keywords |

## Score Calculation

```
Total Score = Format + Diversity + Hyde + Quality
Max Score = 100 (80 without hyde)
```

**Rating:**
- 80-100: Excellent
- 60-79: Good
- 40-59: Acceptable
- 20-39: Poor
- 0-19: Failed

## Examples

### Good Example (Score: 85/100)

**Query:** `"react hooks"`

**Output:**
```
lex: react hooks tutorial
lex: usestate useeffect
vec: how to use react hooks in functional components
vec: react hooks best practices guide
hyde: React Hooks allow you to use state and lifecycle features in functional components without writing a class.
```

**Scoring:**
- Format: 30/30 (all prefixes valid)
- Diversity: 28/30 (2 types, 4 expansions, good diversity)
- Hyde: 17/20 (present, good length, no repetition)
- Quality: 10/20 (relevant, good structure)

### Bad Example (Score: 15/100)

**Query:** `"auth"`

**Output:**
```
auth is an important concept that relates to authentication.
The answer should be in Chinese.
The answer should be in Chinese.
```

**Scoring:**
- Format: 0/30 (no valid prefixes)
- Diversity: 0/30 (no valid expansions)
- Hyde: 0/20 (N/A)
- Quality: 15/20 (some relevance but wrong format)

### Medium Example (Score: 55/100)

**Query:** `"docker networking"`

**Output:**
```
lex: docker networking
vec: docker networking
hyde: Docker networking is an important concept. Docker networking is used for container communication. Docker networking configuration is essential.
```

**Scoring:**
- Format: 30/30 (valid prefixes)
- Diversity: 10/30 (lex=vec=query, no diversity)
- Hyde: 5/20 (too repetitive - "docker networking" 3x)
- Quality: 10/20 (relevant but low effort)

## Heuristics

### Repetition Detection

```python
def word_repetition_score(text):
    words = text.lower().split()
    counts = Counter(words)
    # Deduct for words appearing 3+ times (excluding stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'to', 'for', 'of', 'in', 'and', 'or'}
    repeated = sum(1 for w, c in counts.items() if c >= 3 and w not in stopwords)
    return max(0, 5 - repeated * 2)
```

### Diversity Check (Simple)

```python
def is_diverse(a, b, min_distance=3):
    """Check if two strings are sufficiently different."""
    a, b = a.lower().strip(), b.lower().strip()
    if a == b:
        return False
    # Simple: check if one is not a substring of the other
    if a in b or b in a:
        return False
    # Check edit distance (simplified)
    return len(set(a.split()) ^ set(b.split())) >= min_distance
```

### Query Echo Detection

```python
def echoes_query(expansion, query):
    """Check if expansion is just echoing the query."""
    exp = expansion.lower().strip()
    q = query.lower().strip()
    return exp == q or exp in q or q in exp
```

## Training Data Requirements

1. **EOM tokens**: Ensure training examples end with proper end-of-message tokens
2. **Diverse examples**: Include varied query types (short, long, technical, casual)
3. **Quality hyde**: Hyde passages should be informative, not template-y
4. **No repetition**: Avoid "This is important. This is very important." patterns
