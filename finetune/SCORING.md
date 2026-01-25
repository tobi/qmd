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
| Base relevance | +5 | Subjective |
| Lex lines preserve key terms from query | +5 | -5 if lex is generic |
| Lex lines are keyword-focused (shorter) | +5 | -2 if lex is longer than vec |
| Vec lines are natural language (complete phrases) | +5 | -2 if vec is just keywords |

### 5. Named Entity Preservation (0-20 points, CRITICAL)

Named entities are proper nouns, brand names, technical terms, and acronyms that MUST appear in lex queries. This prevents generic expansions that lose the specific topic.

| Criterion | Points | Deduction |
|-----------|--------|-----------|
| All lex lines contain at least one entity | +15 | - |
| Some lex lines contain entities | +5 | - |
| NO lex lines contain entities | - | **-30 HEAVY PENALTY** |
| Generic filler phrases in lex | - | -15 per phrase |
| Entities also in vec lines | +5 | - |

**Named Entity Detection:**
- All-caps acronyms: `TDS`, `API`, `GPU`, `AWS`
- Capitalized proper nouns: `React`, `Docker`, `Kubernetes`
- Technical terms: `node.js`, `C++`, `.NET`
- CamelCase: `JavaScript`, `TypeScript`
- Compound names: `TDS motorsports` → both words are entities

**Generic Filler Phrases (BANNED in lex):**
- "find information about"
- "search for", "look up"
- "get information", "learn about"
- "details about", "guide to"

**Examples:**

| Query | Bad Lex (Score: 0.30) | Good Lex (Score: 1.00) |
|-------|----------------------|------------------------|
| `who is TDS motorsports` | `lex: find information about` | `lex: TDS motorsports history` |
| | `lex: company details` | `lex: TDS motorsports founders` |
| `how to use React hooks` | `lex: programming tutorial` | `lex: React hooks tutorial` |
| | `lex: how to code` | `lex: useEffect useState hooks` |

**Key Rule**: If a query mentions a specific entity (brand, product, technology), EVERY lex line should include that entity or a direct variation of it.

## Score Calculation

```
Total Score = Format + Diversity + Hyde + Quality + Entity
Max Score = 120 (100 without hyde)
Normalized = Total / Max (0.0 - 1.0)
```

**Rating:**
- 0.80-1.00: Excellent
- 0.60-0.79: Good
- 0.40-0.59: Acceptable
- 0.20-0.39: Poor
- 0.00-0.19: Failed

**Note:** Entity score can go negative, heavily penalizing outputs that miss named entities.

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

### Named Entity Extraction

```python
KEY_TERM_STOPWORDS = {'what', 'is', 'how', 'to', 'the', 'a', 'an', 'in', 'on', 'for', 'of',
                      'and', 'or', 'with', 'my', 'your', 'do', 'does', 'can', 'i', 'me', 'we',
                      'who', 'where', 'when', 'why', 'which', 'find', 'get', 'show', 'tell'}

def extract_named_entities(query: str) -> set:
    """Extract named entities using simple heuristics."""
    entities = set()
    words = query.split()
    prev_was_entity = False

    for i, word in enumerate(words):
        clean = word.strip('.,!?:;()[]"\'')
        if not clean:
            prev_was_entity = False
            continue

        is_entity = False

        # All-caps acronyms: TDS, API, GPU
        if clean.isupper() and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True
        # Capitalized proper nouns (not first word)
        elif i > 0 and clean[0].isupper() and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True
        # Technical terms: node.js, C++
        elif any(c in clean for c in '.+-#@') and len(clean) >= 2:
            entities.add(clean.lower())
            is_entity = True
        # CamelCase: JavaScript
        elif len(clean) > 1 and any(c.isupper() for c in clean[1:]) and clean[0].isupper():
            entities.add(clean.lower())
            is_entity = True
        # Word following an entity (compound names: TDS motorsports)
        elif prev_was_entity and clean.lower() not in KEY_TERM_STOPWORDS:
            entities.add(clean.lower())
            is_entity = True

        prev_was_entity = is_entity

    return entities
```

### Generic Phrase Detection

```python
GENERIC_LEX_PHRASES = {
    'find information about', 'search for', 'look up', 'get information',
    'learn about', 'information on', 'details about', 'find out about',
    'what is', 'how to', 'guide to', 'help with'
}

def lex_is_generic(lex_line: str) -> bool:
    """Check if lex line is a useless generic filler."""
    lex_lower = lex_line.lower().strip()
    for phrase in GENERIC_LEX_PHRASES:
        if phrase in lex_lower:
            # Check if there's specific content beyond the generic phrase
            remaining = lex_lower
            for word in phrase.split():
                remaining = remaining.replace(word, '', 1).strip()
            if len(remaining) < 3:  # Nothing specific left
                return True
    return False
```

## Training Data Requirements

1. **EOM tokens**: Ensure training examples end with proper end-of-message tokens
2. **Diverse examples**: Include varied query types (short, long, technical, casual)
3. **Quality hyde**: Hyde passages should be informative, not template-y
4. **No repetition**: Avoid "This is important. This is very important." patterns
