# QMD Training Data - Balanced Distribution Summary

## Overview

The training data has been rebalanced to reduce excessive tech focus while maintaining adequate technical coverage for QMD's use case. The new distribution emphasizes diverse life topics while keeping tech at a reasonable 15%.

## Distribution Comparison

### Before (Original Data)
```
Technical:        ~50% ████████████████████████████████████████
How-to:           ~45% █████████████████████████████████████
What-is:          ~40% █████████████████████████████████
Other:            ~15% ████████████
Short queries:    10%  ████████
Temporal:         1.6% █
Named entities:   3.4% ██
```

### After (Balanced Approach)
```
Category                    Percentage
────────────────────────────────────────
Health & Wellness           12%  █████████
Finance & Business          12%  █████████
Technology                  15%  ███████████
Home & Garden               10%  ████████
Food & Cooking              10%  ████████
Travel & Geography          10%  ████████
Hobbies & Crafts            10%  ████████
Education & Learning         8%  ██████
Arts & Culture               8%  ██████
Lifestyle & Relationships    5%  ████
────────────────────────────────────────
Short queries (1-2 words):  20%
Temporal (2025/2026):       15%
Named entities:            10%+
```

## Key Improvements

### 1. Category Diversity

**New Non-Tech Categories Added:**
- **Health & Wellness**: Meditation, fitness, nutrition, mental health
- **Finance & Business**: Budgeting, investing, career, entrepreneurship  
- **Home & Garden**: DIY, repairs, cleaning, gardening, organization
- **Food & Cooking**: Recipes, techniques, meal planning, nutrition
- **Travel & Geography**: Travel planning, destinations, geography facts
- **Hobbies & Crafts**: Photography, art, music, woodworking, knitting
- **Education & Learning**: Study techniques, languages, online courses
- **Arts & Culture**: Art history, music, film, theater, literature
- **Lifestyle & Relationships**: Habits, relationships, parenting, minimalism

### 2. Temporal Queries (2025/2026)

Updated to use current era years for recency queries:
- "latest research 2026"
- "Shopify updates 2025"  
- "what changed in React 2026"
- "AI developments 2025"

This ensures the model learns to handle queries from the current time period.

### 3. Short Query Coverage

Expanded from 47 to 144+ short keywords across all categories:
- Tech: auth, config, api, cache, deploy
- Health: meditate, hydrate, stretch, exercise
- Finance: budget, save, invest, taxes
- Home: clean, organize, repair, garden
- Food: cook, bake, recipe, meal
- Travel: travel, pack, passport, hotel
- Hobbies: photo, draw, paint, knit, guitar
- Education: study, learn, course, exam
- Arts: art, music, film, dance
- Life: habit, routine, organize, parent

## Usage

### Quick Start - Use Balanced Data

```bash
cd finetune

# Add 500 balanced examples
cat data/qmd_expansion_balanced.jsonl >> data/qmd_expansion_v2.jsonl

# Prepare with enhanced short query templates  
uv run dataset/prepare_data.py --add-short 2

# Train
uv run train.py sft --config configs/sft.yaml
```

### Generate Fresh Data with Claude API

```bash
# Set API key
export ANTHROPIC_API_KEY=your_key

# Generate 300 balanced examples
uv run dataset/generate_data.py --count 300 \
  --output data/qmd_expansion_fresh.jsonl

# Analyze distribution
uv run dataset/analyze_data.py --input data/qmd_expansion_fresh.jsonl

# Prepare for training
uv run dataset/prepare_data.py --input data/qmd_expansion_fresh.jsonl
```

### Generate Even More Balanced Examples

```bash
# Generate 500 life-focused examples (15% tech)
uv run dataset/generate_balanced.py

# Or generate 265 additional diverse examples
uv run dataset/generate_diverse.py
```

## File Summary

### Modified Files:
- `dataset/generate_data.py` - Added category weights (15% tech), 2025/2026 dates
- `dataset/prepare_data.py` - Expanded SHORT_QUERIES from 47→144, templates 5→16

### New Files:
- `dataset/generate_balanced.py` - Life-focused generator (500 examples)
- `dataset/generate_diverse.py` - Philosophy/History/Geography/Trivia generator (265 examples)
- `dataset/analyze_data.py` - Dataset analysis and quality reporting
- `DATA_IMPROVEMENTS.md` - Detailed improvement documentation

### Generated Data:
- `data/qmd_expansion_balanced.jsonl` - 500 balanced examples
- `data/qmd_expansion_diverse_addon.jsonl` - 265 diverse examples

## Expected Benefits

1. **Better Short Query Handling**: 20% coverage vs 10% before
2. **Named Entity Preservation**: 10%+ coverage vs 3.4% before  
3. **Temporal Understanding**: 15% with 2025/2026 vs 1.6% before
4. **Domain Diversity**: 10 categories vs tech-only before
5. **Life-Document Search**: Better at searching personal notes on health, finance, hobbies

## Next Steps

1. Merge balanced examples into training set
2. Retrain model with improved distribution
3. Evaluate using `evals/queries.txt`
4. Monitor scores on temporal/named-entity/short queries
5. Iterate based on results

---

Generated: 2026-01-30
