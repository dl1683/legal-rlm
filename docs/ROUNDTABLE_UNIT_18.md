# Roundtable Unit 18: Document Type Priority

## Unit Goal
Prioritize important document types in search results.

## Success Criteria
1. [x] DOCUMENT_PRIORITY weights dictionary
2. [x] get_document_priority() function
3. [x] Integrated into search scoring

## Changes Made

### search.py - Priority Weights
| Document Type | Priority | Category |
|---------------|----------|----------|
| contract | 1.5 | Core legal |
| agreement | 1.5 | Core legal |
| complaint | 1.4 | Core legal |
| judgment | 1.5 | Core legal |
| order | 1.4 | Core legal |
| amendment | 1.4 | Core legal |
| exhibit | 1.3 | Supporting |
| declaration | 1.3 | Supporting |
| affidavit | 1.3 | Supporting |
| motion | 1.3 | Supporting |
| memo | 1.2 | Supporting |
| report | 1.2 | Supporting |
| letter | 1.1 | Supporting |
| email | 0.9 | Correspondence |
| correspondence | 0.9 | Correspondence |
| note | 0.8 | Informal |
| draft | 0.8 | Informal |

### search.py - _score_results()
Added Factor 7: Document type priority
- Multiplies score by document priority weight
- Core legal documents boosted up to 1.5x
- Informal documents reduced to 0.8x

### Key Code
```python
DOCUMENT_PRIORITY = {
    "contract": 1.5,
    "agreement": 1.5,
    "email": 0.9,
    "draft": 0.8,
    ...
}

def get_document_priority(filename: str) -> float:
    filename_lower = filename.lower()
    for doc_type, priority in DOCUMENT_PRIORITY.items():
        if doc_type in filename_lower:
            return priority
    return 1.0

# In _score_results()
doc_priority = get_document_priority(hit.filename)
score *= doc_priority
```

## Review Notes
- Legal documents prioritized over correspondence
- Draft documents deprioritized
- Multiplicative factor ensures significant impact
- Default priority (1.0) for unknown types

## Next Unit
Unit 19: Cross-Reference Detection
