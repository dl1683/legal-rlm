# Roundtable Unit 12: Investigation Summary

## Unit Goal
Add comprehensive summary methods for investigation results.

## Success Criteria
1. [x] get_summary() returns structured dict
2. [x] get_summary_text() returns human-readable text
3. [x] Includes all key metrics
4. [x] Shows top entities and facts

## Changes Made

### state.py Methods
| Method | Description |
|--------|-------------|
| get_summary() | Returns structured summary dict |
| get_summary_text() | Returns formatted text summary |

### Summary Contents
| Section | Fields |
|---------|--------|
| Basic | id, query, status, duration |
| Metrics | docs_read, searches, citations, leads, entities, facts, depth |
| Hypothesis | Current hypothesis |
| Top Entities | Name, type, mentions (top 5) |
| Key Facts | First 10 accumulated facts |

### Key Code
```python
def get_summary(self) -> dict[str, Any]:
    return {
        "id": self.id,
        "query": self.query,
        "status": self.status,
        "duration_seconds": self.duration_seconds,
        "metrics": {
            "documents_read": self.documents_read,
            "citations": len(self.citations),
            "verified_citations": verification_stats["verified"],
            ...
        },
        "hypothesis": self.hypothesis,
        "top_entities": [...],
        "key_facts": [...],
    }
```

## Review Notes
- Summary provides quick overview of investigation
- Both structured (JSON-friendly) and text formats
- Integrates with verification and lead statistics
- Useful for UI display and logging

## Next Unit
Unit 13: Synthesis Enhancement
