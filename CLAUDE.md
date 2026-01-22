# Irys RLM - Project Instructions

## Overview
Irys RLM (Recursive Language Model) is a legal document analysis system that investigates queries against document repositories using multi-tier LLM reasoning.

## Evaluation & Improvement Workflow

### Test Repository
- **Location**: `C:\Users\devan\Downloads\CITIOM v Gulfstream\documents`
- **Case**: CITIOM v Gulfstream (aircraft 192-month inspection dispute)
- **Size**: 95 documents, multiple subfolders, large PDFs (500+ pages)

### Evaluation Set
- **File**: `legal_queries.json` (100 queries across 6 categories)
- **Categories**:
  1. `factual_extraction` (20 queries) - Simple fact lookups
  2. `multi_document_synthesis` (20 queries) - Cross-document analysis
  3. `timeline_construction` (15 queries) - Chronological analysis
  4. `contradiction_detection` (15 queries) - Finding inconsistencies
  5. `legal_analysis` (15 queries) - Legal reasoning
  6. `evidence_assessment` (15 queries) - Evidence evaluation

### Improvement Loop

```
┌─────────────────────────────────────────────────────────────┐
│  1. BASELINE: Run eval set, record metrics                  │
│     - python run_comparative_test.py [count]                │
│     - Track: time, citations, docs_read, accuracy           │
├─────────────────────────────────────────────────────────────┤
│  2. ANALYZE: Identify failure patterns                      │
│     - Which query types fail?                               │
│     - Search not finding docs? (0 docs_read)                │
│     - Synthesis weak? (low citations)                       │
│     - Too slow? (time per query)                            │
├─────────────────────────────────────────────────────────────┤
│  3. HYPOTHESIZE: Root cause analysis                        │
│     - Search layer issues (PDF indexing, folder nav)        │
│     - LLM decision issues (prompts, model selection)        │
│     - Caching/efficiency issues                             │
├─────────────────────────────────────────────────────────────┤
│  4. IMPLEMENT: Make targeted fix                            │
│     - Edit src/irys/rlm/engine.py                           │
│     - Edit src/irys/rlm/decisions.py                        │
│     - Edit src/irys/core/search.py                          │
├─────────────────────────────────────────────────────────────┤
│  5. VALIDATE: Re-run affected query subset                  │
│     - Quick test: python test_comprehensive.py --quick      │
│     - Targeted: test specific failing queries               │
├─────────────────────────────────────────────────────────────┤
│  6. COMPARE: Run full eval, compare to baseline             │
│     - Did metrics improve?                                  │
│     - Any regressions?                                      │
│     - Update baseline if successful                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        REPEAT LOOP
```

### Test Commands

```bash
# Quick single test (test_documents)
python test_comprehensive.py --quick

# Full test suite (test_documents)
python test_comprehensive.py

# CITIOM comparative test (limited queries)
python run_comparative_test.py 3

# Full CITIOM test
python run_comparative_test.py 12
```

### Key Metrics to Track

| Metric | Target | Current (2026-01-21) |
|--------|--------|----------------------|
| Simple query time | <30s | ~32s |
| Complex query time | <90s | ~120s |
| Avg citations | >5 | 5.33 |
| Docs found rate | >90% | **33.3%** (CITIOM) |

### Current Baseline (CITIOM repo)

```
Queries: 3/3 completed
Avg time per query: 65.74s
Avg citations: 5.33
Avg docs read: 2.0
Docs found rate: 33.3%  <-- CRITICAL ISSUE

By Category:
  timeline_construction  | 120.8s | 16 cites | SUCCESS
  multi_document_synth   | 44.31s | 0 cites  | FAILED
  factual_extraction     | 32.1s  | 0 cites  | FAILED
```

### Known Issues (as of 2026-01-21)

1. **Large PDF indexing**: 500+ page PDFs not being searched effectively
2. **Folder navigation**: Complex subfolder structures cause search misses
3. **Format support**: .mht files not being read
4. **Citation specificity**: Missing exhibit/page references

### Architecture

```
src/irys/
├── core/
│   ├── models.py      # GeminiClient (LITE/FLASH/PRO tiers)
│   ├── repository.py  # MatterRepository (file discovery)
│   ├── reader.py      # Document reading (PDF, DOCX, TXT)
│   └── search.py      # Text search across documents
├── rlm/
│   ├── engine.py      # Main investigation engine
│   ├── decisions.py   # LLM decision functions
│   ├── prompts.py     # Prompt templates
│   └── state.py       # InvestigationState, ThinkingStep
└── ui/
    └── app.py         # Streamlit UI
```

### Model Tiers

- **LITE** (gemini-2.5-flash-lite): Quick decisions, file picking, sufficiency checks
- **FLASH** (gemini-2.5-flash): Analysis, planning, fact extraction, simple synthesis
- **PRO** (gemini-2.5-pro): Complex synthesis only

### Recent Optimizations (2026-01-21)

1. Document extraction cache (avoid re-extracting same doc)
2. Search term deduplication (skip >50% word overlap)
3. FLASH model for simple query synthesis (vs PRO)
4. Early sufficiency exit (after 5+ facts)
5. Reduced iteration limits (max_depth=3, max_iterations=10)

**Result**: 30.8% faster overall (71s → 49s avg per query)

### Comparison: RLM vs Claude Code Subagent

For complex legal repos (CITIOM):
- **RLM**: Faster (~2 min) but lower accuracy on large repos
- **Claude Subagent**: Slower (~3 min) but better citations and specificity

Use Claude subagent for verification/comparison testing.

## Environment

```bash
# Required
GEMINI_API_KEY=xxx  # In .env file, never hardcode

# Install
pip install -r requirements.txt
```

## Next Priority Improvements

1. [ ] Fix PDF search for large documents (chunk-based indexing)
2. [ ] Add .mht file support
3. [ ] Improve subfolder traversal in search
4. [ ] Add exhibit/page citation extraction
5. [ ] Parallel document reading for faster processing
