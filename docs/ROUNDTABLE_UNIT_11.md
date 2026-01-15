# Roundtable Unit 11: Lead Prioritization

## Unit Goal
Dynamically reprioritize leads based on investigation context.

## Success Criteria
1. [x] reprioritize_leads() method
2. [x] Entity-based priority boost
3. [x] Hypothesis-based priority boost
4. [x] get_lead_statistics() for monitoring
5. [x] Engine calls reprioritize during loop

## Changes Made

### state.py Methods
| Method | Description |
|--------|-------------|
| reprioritize_leads() | Adjust priorities based on context |
| get_lead_statistics() | Get lead counts and average priority |

### Priority Boost Factors
| Factor | Boost | Condition |
|--------|-------|-----------|
| Top entity | +0.15 | Lead mentions a top-5 entity |
| Hypothesis | +0.10 | Lead shares 2+ words with hypothesis |

### engine.py Changes
| Location | Change |
|----------|--------|
| _investigate_loop | Calls reprioritize_leads() every 2 iterations |

### Key Code
```python
def reprioritize_leads(self):
    top_entities = {e.name.lower() for e in self.get_top_entities(5)}
    hypothesis_words = set(self.hypothesis.lower().split()) if self.hypothesis else set()

    for lead in self.leads:
        if lead.investigated:
            continue

        boost = 0.0
        desc_lower = lead.description.lower()

        # Boost if mentions top entity
        for entity_name in top_entities:
            if entity_name in desc_lower:
                boost += 0.15
                break

        # Boost if related to hypothesis
        if len(set(desc_lower.split()) & hypothesis_words) >= 2:
            boost += 0.1

        lead.priority = min(1.0, lead.priority + boost)
```

## Review Notes
- Reprioritization adapts to investigation findings
- Top entities emerge from document analysis
- Hypothesis connection ensures focused investigation
- Priority capped at 1.0 to prevent runaway

## Next Unit
Unit 12: Investigation Summary
