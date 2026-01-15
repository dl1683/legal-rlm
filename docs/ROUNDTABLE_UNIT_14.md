# Roundtable Unit 14: Progress Metrics & Callbacks

## Unit Goal
Add progress tracking and callbacks during investigation.

## Success Criteria
1. [x] api_calls and estimated_tokens metrics
2. [x] get_progress() with percentage estimate
3. [x] on_progress callback in RLMEngine
4. [x] Progress emitted with each step

## Changes Made

### state.py
| Field/Method | Description |
|--------------|-------------|
| api_calls | Count of API calls made |
| estimated_tokens | Estimated token usage |
| get_progress() | Returns progress dict with percentage |
| increment_api_calls() | Update API metrics |

### Progress Calculation
| Factor | Weight | Description |
|--------|--------|-------------|
| Leads investigated | 40% | % of leads processed |
| Citations | 30% | Citations vs target (10) |
| Depth | 20% | Max depth vs target (3) |
| Documents | 10% | Docs read vs target (5) |

### engine.py
| Change | Description |
|--------|-------------|
| on_progress callback | New constructor parameter |
| _emit_progress() | Helper to emit progress |
| _emit_step() | Now also emits progress |

### Key Code
```python
def get_progress(self) -> dict[str, Any]:
    progress = 0
    lead_progress = (leads_investigated / total_leads) * 40
    citation_progress = min(citations / 10, 1.0) * 30
    depth_progress = min(max_depth / 3, 1.0) * 20
    doc_progress = min(docs_read / 5, 1.0) * 10

    return {
        "progress_percent": min(progress, 100),
        "status": self.status,
        "elapsed_seconds": elapsed,
        ...
    }
```

## Review Notes
- Progress provides UI feedback during long investigations
- Percentage is an estimate based on typical investigation patterns
- Callbacks enable real-time progress updates

## Next Unit
Unit 15: Confidence Scoring
