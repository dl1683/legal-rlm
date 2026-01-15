# Roundtable Unit 16: Adaptive Depth Control

## Unit Goal
Dynamically adjust investigation depth based on progress and complexity.

## Success Criteria
1. [x] Config options for adaptive depth
2. [x] _calculate_effective_depth() method
3. [x] _should_continue_investigation() method
4. [x] Integrated into investigation loop

## Changes Made

### engine.py - RLMConfig
| Option | Default | Description |
|--------|---------|-------------|
| adaptive_depth | True | Enable/disable adaptive depth |
| min_depth | 2 | Minimum depth even for simple queries |
| depth_citation_threshold | 15 | Stop early if enough citations |

### engine.py - Methods
| Method | Description |
|--------|-------------|
| _calculate_effective_depth() | Calculates depth based on progress |
| _should_continue_investigation() | Decides if investigation should stop |

### Depth Adjustment Logic
| Condition | Adjustment |
|-----------|------------|
| >= 15 citations | Reduce by 2 |
| Confidence >= 70 | Reduce by 1 |
| > 10 pending leads | Increase by 1 (cap 7) |

### Termination Criteria
| Condition | Action |
|-----------|--------|
| Below min_depth | Continue |
| Confidence >= 80 + 10 citations | Stop |
| No high-priority leads | Stop |

### Key Code
```python
def _calculate_effective_depth(self, state):
    base_depth = self.config.max_depth

    if len(state.citations) >= self.config.depth_citation_threshold:
        return max(self.config.min_depth, base_depth - 2)

    confidence = state.get_confidence_score()
    if confidence["score"] >= 70:
        return max(self.config.min_depth, base_depth - 1)

    return base_depth

def _should_continue_investigation(self, state):
    if state.max_depth_reached < self.config.min_depth:
        return True
    confidence = state.get_confidence_score()
    if confidence["score"] >= 80 and len(state.citations) >= 10:
        return False
    ...
```

## Review Notes
- Adaptive depth prevents over-investigation of simple queries
- Confidence-based termination saves resources
- Minimum depth ensures thorough investigation
- Dynamic adjustment based on evidence accumulation

## Next Unit
Unit 17: Query Analysis
