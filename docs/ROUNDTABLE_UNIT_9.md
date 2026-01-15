# Roundtable Unit 9: Deduplication

## Unit Goal
Prevent duplicate citations, leads, and facts from polluting investigation results.

## Success Criteria
1. [x] Citation deduplication (same doc, page, text)
2. [x] Lead deduplication with similarity threshold
3. [x] Fact deduplication with word overlap
4. [x] Engine updated to use deduplication methods

## Changes Made

### state.py
| Method | Description |
|--------|-------------|
| add_citation() | Returns None if duplicate found |
| add_lead() | Returns None if similar lead exists, updates priority if higher |
| _string_similarity() | Word overlap Jaccard similarity |
| add_fact() | Adds fact if not duplicate |
| add_facts() | Batch add with deduplication |

### Deduplication Logic

**Citations:**
- Normalized text comparison (lowercase, collapsed whitespace)
- Same document + same page + 100-char prefix match = duplicate

**Leads:**
- Word overlap similarity using Jaccard index
- 80% similarity threshold
- Higher priority updates existing lead priority

**Facts:**
- 70% word overlap threshold
- Lower threshold allows related but distinct facts

### Key Code
```python
def add_lead(self, description, source, priority=0.5):
    desc_normalized = " ".join(description.lower().split())
    for existing in self.leads:
        existing_normalized = " ".join(existing.description.lower().split())
        if self._string_similarity(desc_normalized, existing_normalized) > 0.8:
            if priority > existing.priority:
                existing.priority = priority
            return None  # Duplicate
    ...

@staticmethod
def _string_similarity(s1, s2):
    words1, words2 = set(s1.split()), set(s2.split())
    return len(words1 & words2) / len(words1 | words2)
```

## Review Notes
- Deduplication prevents citation/fact inflation
- Similarity-based matching handles minor variations
- Priority updates ensure best lead version is kept
- Engine updated to use new deduplication API

## Next Unit
Unit 10: Entity Extraction
