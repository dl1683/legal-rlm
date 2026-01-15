# Roundtable Unit 25: Multi-Query Support

## Unit Goal
Handle compound queries and multiple simultaneous investigations.

## Success Criteria
1. [x] decompose_query() - break compound queries
2. [x] investigate_multi() - parallel/sequential investigations
3. [x] investigate_compound() - full compound query handling
4. [x] _merge_investigation_results() - combine results

## Changes Made

### engine.py - Methods
| Method | Description |
|--------|-------------|
| decompose_query() | LLM-based query decomposition |
| investigate_multi() | Run multiple queries parallel/sequential |
| investigate_compound() | Full compound query pipeline |
| _merge_investigation_results() | Combine findings |
| _create_failed_state() | Handle failed investigations |

### Query Decomposition
Uses LLM to analyze queries and determine:
- Is it compound? (multiple questions)
- Sub-queries with priorities
- Dependencies between sub-queries

### Merged Results
| Field | Description |
|-------|-------------|
| total_documents_read | Sum across all investigations |
| total_citations | Combined citation count |
| all_citations | All citations collected |
| all_entities | Merged entity map |
| all_facts | Deduplicated facts |
| all_hypotheses | Hypothesis from each sub-query |
| combined_confidence | Average confidence score |

### Key Code
```python
async def decompose_query(self, query: str) -> list[dict]:
    """Decompose compound query into sub-queries."""
    prompt = f"""Analyze this query and determine if compound.
    Query: {query}
    Return:
    {{
        "is_compound": true/false,
        "sub_queries": [
            {{"query": "...", "priority": 0-1, "depends_on": null/index}}
        ]
    }}"""
    response = await self.client.complete(prompt, tier=ModelTier.FLASH)
    return self._parse_json_safe(response, defaults)["sub_queries"]

async def investigate_multi(
    self,
    queries: list[str],
    repository_path: str | Path,
    parallel: bool = True,
) -> dict[str, InvestigationState]:
    """Run multiple investigations."""
    if parallel:
        tasks = [self.investigate(q, repository_path) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {q: r for q, r in zip(queries, results)}
    else:
        # Sequential execution
        ...

async def investigate_compound(
    self,
    query: str,
    repository_path: str | Path,
) -> dict[str, Any]:
    """Full compound query handling."""
    sub_queries = await self.decompose_query(query)
    independent = [sq for sq if sq["depends_on"] is None]
    dependent = [sq for sq if sq["depends_on"] is not None]

    # Parallel for independent
    results = await self.investigate_multi(independent_queries, ...)

    # Sequential for dependent (with context)
    for dep in dependent:
        parent = results[independent[dep["depends_on"]]]
        enriched = f"{dep['query']}\nContext: {parent.hypothesis}"
        results[dep["query"]] = await self.investigate(enriched, ...)

    return {"merged": self._merge_investigation_results(...)}
```

## Query Examples
| Type | Example |
|------|---------|
| Simple | "What are the contract terms?" |
| Compound | "Compare the 2020 and 2021 agreements and identify key differences" |
| Dependent | "What was the breach? Based on that, what are the potential damages?" |

## Review Notes
- Parallel execution significantly speeds up multi-query
- Dependencies handled sequentially with context enrichment
- Result merging preserves all findings with deduplication
- Failed sub-queries don't block others

## Next Unit
Unit 26: Document Clustering
