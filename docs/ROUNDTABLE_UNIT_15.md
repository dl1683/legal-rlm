# Roundtable Unit 15: Confidence Scoring

## Unit Goal
Calculate investigation confidence based on evidence quality.

## Success Criteria
1. [x] get_confidence_score() method
2. [x] Multi-factor scoring (citations, verification, docs, entities, facts)
3. [x] Confidence levels (high, medium, low, insufficient)
4. [x] Integrated into summary

## Changes Made

### state.py
| Method | Description |
|--------|-------------|
| get_confidence_score() | Calculate confidence score 0-100 |
| get_summary() | Now includes confidence in output |

### Confidence Factors
| Factor | Max Points | Calculation |
|--------|------------|-------------|
| Citations | 25 | count * 2.5 (max at 10) |
| Verification | 25 | verified_rate * 25 |
| Documents | 20 | count * 4 (max at 5) |
| Entities | 15 | count * 3 (max at 5) |
| Facts | 15 | count * 1.5 (max at 10) |

### Confidence Levels
| Level | Score Range |
|-------|-------------|
| high | >= 80 |
| medium | >= 50 |
| low | >= 25 |
| insufficient | < 25 |

### Key Code
```python
def get_confidence_score(self) -> dict[str, Any]:
    factors = {
        "citations": min(citation_count * 2.5, 25),
        "verification": verified_rate * 25,
        "documents": min(docs_read * 4, 20),
        "entities": min(entity_count * 3, 15),
        "facts": min(fact_count * 1.5, 15),
    }
    total_score = sum(factors.values())

    return {
        "score": total_score,
        "level": level,
        "factors": factors,
    }
```

## Review Notes
- Confidence helps users understand investigation quality
- Multi-factor approach provides balanced assessment
- Verification rate rewards citing verifiable sources
- Factor breakdown enables improvement guidance

## Next Unit
Unit 16: Adaptive Depth
