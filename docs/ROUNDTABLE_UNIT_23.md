# Roundtable Unit 23: Search Query Expansion

## Unit Goal
Expand search queries with legal synonyms and related terms.

## Success Criteria
1. [x] LEGAL_SYNONYMS dictionary
2. [x] expand_query() function
3. [x] generate_related_searches() function
4. [x] Integration with search system

## Changes Made

### search.py - LEGAL_SYNONYMS
| Term | Synonyms |
|------|----------|
| agreement | contract, covenant, arrangement, understanding |
| contract | agreement, covenant, arrangement |
| breach | violation, default, non-compliance, failure |
| terminate | cancel, end, rescind, void |
| liability | responsibility, obligation, duty |
| damages | compensation, remedy, recovery, losses |
| indemnify | hold harmless, compensate, reimburse |
| confidential | proprietary, secret, private |
| warranty | guarantee, representation, assurance |
| obligation | duty, requirement, responsibility |
| party | parties, signatory, counterparty |
| execute | sign, enter into, consummate |
| material | significant, substantial, important |
| consent | approval, permission, authorization |
| notice | notification, communication, written notice |

### search.py - Functions
| Function | Description |
|----------|-------------|
| expand_query(query, max_expansions) | Expand with synonyms |
| generate_related_searches(query) | Generate related queries |

### Key Code
```python
def expand_query(query: str, max_expansions: int = 3) -> list[str]:
    expanded = [query]
    query_lower = query.lower()

    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in query_lower:
            for synonym in synonyms[:max_expansions]:
                expanded_query = query_lower.replace(term, synonym)
                if expanded_query not in expanded:
                    expanded.append(expanded_query)

    return expanded[:max_expansions + 1]

def generate_related_searches(query: str) -> list[str]:
    related = []
    words = [w for w in query.lower().split() if len(w) > 3]

    # Generate phrase variations
    if len(words) >= 2:
        for i in range(len(words) - 1):
            related.append(f"{words[i]} {words[i+1]}")

    # Add common legal modifiers
    modifiers = ["material", "significant", "breach of", "failure to"]
    for word in words[:2]:
        for modifier in modifiers[:2]:
            if modifier not in query.lower():
                related.append(f"{modifier} {word}")

    return related[:5]
```

## Review Notes
- Synonyms improve recall without manual effort
- Max expansions prevents query explosion
- Related searches provide investigation directions
- Domain-specific synonyms for legal accuracy

## Next Unit
Unit 24: Evidence Strength Scoring
