# Roundtable Unit 17: Query Analysis & Classification

## Unit Goal
Analyze and classify queries to guide investigation strategy.

## Success Criteria
1. [x] QueryType enum for query categories
2. [x] classify_query() function
3. [x] Complexity estimation
4. [x] Entity extraction from query
5. [x] Integration into investigation flow

## Changes Made

### state.py - QueryType Enum
| Type | Description | Keywords |
|------|-------------|----------|
| FACTUAL | What happened? When? | "what happened", "when did" |
| ANALYTICAL | What does this mean? | "what does", "implications" |
| COMPARATIVE | How does X compare to Y? | "compare", "versus" |
| EVALUATIVE | Is this valid? | "valid", "assess", "evaluate" |
| PROCEDURAL | What steps? | "how to", "process" |
| UNKNOWN | Default | - |

### state.py - classify_query()
| Output Field | Description |
|--------------|-------------|
| type | Query type (string) |
| complexity | 1-5 complexity score |
| word_count | Number of words |
| potential_entities | Capitalized words as potential entities |

### Complexity Calculation
| Condition | Complexity |
|-----------|------------|
| <= 10 words | 1 |
| 11-20 words | 2 |
| 21-30 words | 3 |
| > 30 words | 4 |
| Contains connectors | +1 |

### engine.py Changes
- Query classification at investigation start
- Classification stored in state
- Logged as thinking step

### Key Code
```python
def classify_query(query: str) -> dict[str, Any]:
    query_type = QueryType.UNKNOWN

    if any(kw in query_lower for kw in factual_keywords):
        query_type = QueryType.FACTUAL
    ...

    return {
        "type": query_type.value,
        "complexity": complexity,
        "word_count": word_count,
        "potential_entities": potential_entities[:5],
    }
```

## Review Notes
- Query classification guides search strategy
- Complexity helps estimate investigation depth
- Potential entities seed initial searches
- Future: can adjust prompts based on query type

## Next Unit
Unit 18: Document Type Priority
