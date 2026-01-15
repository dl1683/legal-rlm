# Roundtable Unit 13: Synthesis Enhancement

## Unit Goal
Improve final synthesis with entity information and better structure.

## Success Criteria
1. [x] Enhanced SYNTHESIS_PROMPT with entities
2. [x] Include hypothesis in synthesis
3. [x] Include verification status
4. [x] More comprehensive analysis sections

## Changes Made

### engine.py - SYNTHESIS_PROMPT
| Section | Addition |
|---------|----------|
| Summary | Added citation_count, max_depth |
| Hypothesis | Added current hypothesis |
| Entities | Added Key Entities section |
| Instructions | Added entities, gaps, verification priority |

### engine.py - _synthesize()
| Change | Description |
|--------|-------------|
| entities_text | Added entity summary to prompt |
| hypothesis | Added hypothesis to prompt |
| citation_count | Added citation count metric |
| max_depth | Added investigation depth metric |

### Enhanced Output Sections
1. Direct answer to the query
2. Supporting evidence with citations
3. **Key entities and their roles** (NEW)
4. Issues with evidence
5. Risks and concerns
6. **Investigation gaps** (NEW)
7. Recommended next steps

### Key Code
```python
prompt = SYNTHESIS_PROMPT.format(
    query=state.query,
    docs_analyzed=state.documents_read,
    searches=state.searches_performed,
    citation_count=len(state.citations),
    max_depth=state.max_depth_reached,
    hypothesis=state.hypothesis or "No specific hypothesis formed",
    entities=entities_text or "No entities identified",
    findings=findings_text or "No specific findings accumulated",
    citations=citations_text or "No citations collected",
)
```

## Review Notes
- Synthesis now includes entity analysis
- Hypothesis provides investigation context
- Verification priority helps reliability
- Gaps section identifies areas for further investigation

## Next Unit
Unit 14: Progress Metrics
