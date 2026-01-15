# Roundtable Unit 8: Search Result Ranking

## Unit Goal
Improve search result relevance with multi-factor scoring.

## Success Criteria
1. [x] SearchHit enhanced with match_count, exact_match fields
2. [x] context_richness property for context quality scoring
3. [x] _score_results() multi-factor ranking method
4. [x] Exact match detection in _search_file()

## Changes Made

### search.py - SearchHit
| Field | Description |
|-------|-------------|
| match_count | Number of matches in the line |
| exact_match | Whether the full query phrase was found |
| context_richness | Property calculating context quality score |

### search.py - Scoring Factors
| Factor | Weight | Description |
|--------|--------|-------------|
| Match density | 0.1-0.5 | More matches in file = higher relevance |
| Page spread | 0.05-0.3 | Matches across pages = document-wide relevance |
| Position | 0.0-0.2 | Earlier in document = slightly boosted |
| Context richness | varies | More context = better quality |
| Exact match | 0.5 | Bonus for exact phrase match |
| Match length | 0.2 | Longer matches = more specific |

### Key Code
```python
def _score_results(self, hits: list[SearchHit], query: str):
    for hit in hits:
        score = 1.0
        score += min(file_counts[hit.file_path] * 0.1, 0.5)  # Density
        score += min(page_spread * 0.05, 0.3)  # Spread
        score += position_factor * 0.2  # Position
        score += hit.context_richness  # Context
        if hit.exact_match: score += 0.5  # Exact
        if match_words >= query_words: score += 0.2  # Length
        hit.score = score
```

## Review Notes
- Multi-factor ranking provides better result ordering
- Exact phrase matches are prioritized
- Documents with widespread matches score higher
- Position boost helps surface key findings early

## Next Unit
Unit 9: Deduplication
