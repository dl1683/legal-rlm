# Roundtable Unit 19: Cross-Reference Detection

## Unit Goal
Detect and track when documents reference each other.

## Success Criteria
1. [x] CrossReference dataclass
2. [x] add_cross_reference() with deduplication
3. [x] detect_cross_references() for automatic detection
4. [x] get_cross_reference_graph() for visualization

## Changes Made

### state.py - CrossReference Class
| Field | Description |
|-------|-------------|
| source_doc | Document containing the reference |
| target_doc | Document being referenced |
| reference_text | Context around the reference |
| page | Page number of reference |
| confidence | 0-1 confidence score |

### state.py - Methods
| Method | Description |
|--------|-------------|
| add_cross_reference() | Add reference with deduplication |
| detect_cross_references() | Auto-detect references in text |
| get_cross_reference_graph() | Get doc->docs adjacency list |

### Reference Patterns Detected
- "see exhibit", "per exhibit", "attached as exhibit"
- "referenced in", "as stated in", "according to"
- "see attached", "per the", "pursuant to"

### Confidence Scoring
| Condition | Confidence |
|-----------|------------|
| Document name found | 0.5 |
| Reference pattern nearby | 0.9 |

### Key Code
```python
def detect_cross_references(self, text, source_doc, known_docs, page=None):
    for doc_name in known_docs:
        if doc_base in text_lower:
            # Check for reference patterns
            confidence = 0.5
            for pattern in ref_patterns:
                if pattern in text_lower[idx-100:idx+100]:
                    confidence = 0.9
                    break
            self.add_cross_reference(...)

def get_cross_reference_graph(self):
    graph = {}
    for ref in self.cross_references:
        if ref.source_doc not in graph:
            graph[ref.source_doc] = []
        graph[ref.source_doc].append(ref.target_doc)
    return graph
```

## Review Notes
- Cross-references help identify related documents
- Graph structure enables document relationship analysis
- Confidence scoring filters low-quality matches
- Deduplication prevents redundant entries

## Next Unit
Unit 20: Timeline Extraction
