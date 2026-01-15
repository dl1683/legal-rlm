# Roundtable Unit 21: Contradiction Detection

## Unit Goal
Track potential contradictions between documents or statements.

## Success Criteria
1. [x] Contradiction dataclass
2. [x] add_contradiction() method
3. [x] get_contradictions_by_severity() filter
4. [x] get_contradictions_formatted() display

## Changes Made

### state.py - Contradiction Class
| Field | Description |
|-------|-------------|
| id | Unique identifier |
| statement1 | First conflicting statement |
| source1 | Source of first statement |
| statement2 | Second conflicting statement |
| source2 | Source of second statement |
| contradiction_type | factual, date, amount, claim |
| severity | high, medium, low |
| notes | Additional notes |

### state.py - Methods
| Method | Description |
|--------|-------------|
| add_contradiction() | Add new contradiction |
| get_contradictions_by_severity() | Filter by severity |
| get_contradictions_formatted() | Human-readable output |

### Contradiction Types
| Type | Description |
|------|-------------|
| factual | Conflicting facts |
| date | Conflicting dates |
| amount | Conflicting amounts/numbers |
| claim | Conflicting claims |

### Key Code
```python
@dataclass
class Contradiction:
    id: str
    statement1: str
    source1: str
    statement2: str
    source2: str
    contradiction_type: str
    severity: str
    notes: str = ""

def get_contradictions_formatted(self) -> str:
    for severity in ["high", "medium", "low"]:
        contradictions = self.get_contradictions_by_severity(severity)
        if contradictions:
            lines.append(f"[{severity.upper()}]")
            ...
```

## Review Notes
- Contradictions flagged for legal review
- Severity helps prioritize investigation
- Formatted output grouped by severity
- Future: LLM-based contradiction detection

## Next Unit
Unit 22: Investigation Templates
