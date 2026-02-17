# LFM2.5-1.2B Training Data Preparation Summary

## Source Data
- **Input**: `qmd_expansion_v3.jsonl` (964,544 bytes, 1,498 entries)
- **Date**: Generated from cleaned QMD dataset v3

## Conversion Process
- **Script**: `convert_to_chatml.py`
- **Format**: Converted to ChatML format for LFM2.5
- **Split**: 90% train / 10% validation
- **Shuffle**: Applied with seed=42 for reproducibility

## Output Files
- **Train set**: `train.jsonl` (913K, 1,348 entries)
- **Validation set**: `val.jsonl` (101K, 150 entries)

## Data Quality Verification
- **Success rate**: 100% (no format issues detected)
- **ChatML format**: All entries properly formatted
- **Required components**: All entries contain lex, vec, and hyde expansions

## Data Statistics
### Training Set (1,348 entries)
- Query length: 6-65 chars (avg: 29.3)
- Response length: 307-777 chars (avg: 539.5)

### Validation Set (150 entries)  
- Query length: 2-56 chars (avg: 28.5)
- Response length: 342-762 chars (avg: 536.4)

## ChatML Format Structure
```
<|startoftext|><|im_start|>user
Expand this search query: {original_query}<|im_end|>
<|im_start|>assistant
lex: {lexical_expansion_1}
lex: {lexical_expansion_2}
...
vec: {vector_expansion_1}
vec: {vector_expansion_2}
...
hyde: {hypothetical_document}
<|im_end|>
```

## Verification
- Format validation: ✅ PASSED
- Content completeness: ✅ PASSED  
- File integrity: ✅ PASSED
- Ready for LFM2.5 training: ✅ YES

**Generated**: $(date)
**Conversion time**: ~2 seconds
**Data ready for fine-tuning**