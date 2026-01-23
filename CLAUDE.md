# Irys RLM - Project Instructions

## CRITICAL: Git Commit Rules

**COMMIT AFTER EVERY LOGICAL CHANGE. NO EXCEPTIONS.**

A "logical change" is ONE of the following:
- Fix a single bug
- Add a single function
- Update a single prompt
- Add a single test
- Fix imports in one file
- Any change that can be described in ONE sentence

**DO NOT:**
- Bundle multiple fixes into one commit
- Wait until "everything works" to commit
- Make commits with 500+ lines of changes
- Commit multiple unrelated changes together

**Commit message format:**
```
<short description of the ONE thing changed>

Committed by Devansh
```

**Example good commits:**
- "Fix citation text truncation - remove 200 char limit"
- "Add on_citation callback for external sources"
- "Filter out template-style queries with brackets"

**Example BAD commits:**
- "Various fixes and improvements" (too vague)
- "External search integration with multiple fixes" (too many things)

**WHY THIS MATTERS:**
- Easy to revert if something breaks
- Clear history of what changed when
- Proper code review possible
- Isolate bugs to specific changes

---

## Overview
Irys RLM (Recursive Language Model) is a legal document analysis system that investigates queries against document repositories using multi-tier LLM reasoning.

## Architecture

```
src/irys/
├── core/
│   ├── models.py      # GeminiClient (LITE/FLASH/PRO tiers)
│   ├── repository.py  # MatterRepository (file discovery)
│   ├── reader.py      # Document reading (PDF, DOCX, TXT, MHT)
│   └── search.py      # Text search across documents
├── rlm/
│   ├── engine.py      # Main investigation engine
│   ├── decisions.py   # LLM decision functions
│   ├── prompts.py     # Prompt templates
│   └── state.py       # InvestigationState, ThinkingStep
└── ui/
    └── app.py         # Streamlit UI
```

## Model Tiers

- **LITE** (gemini-2.5-flash-lite): Quick decisions, file picking, sufficiency checks
- **FLASH** (gemini-2.5-flash): Analysis, planning, fact extraction, simple synthesis
- **PRO** (gemini-2.5-pro): Complex synthesis only

## Test Repository
- **Location**: `C:\Users\devan\Downloads\CITIOM v Gulfstream\documents`
- **Case**: CITIOM v Gulfstream (aircraft 192-month inspection dispute)
- **Evaluation Set**: `legal_queries.json` (100 queries across 6 categories)

## Evaluation Categories
1. `factual_extraction` (20 queries) - Simple fact lookups
2. `multi_document_synthesis` (20 queries) - Cross-document analysis
3. `timeline_construction` (15 queries) - Chronological analysis
4. `contradiction_detection` (15 queries) - Finding inconsistencies
5. `legal_analysis` (15 queries) - Legal reasoning
6. `evidence_assessment` (15 queries) - Evidence evaluation

## Environment

```bash
# Required
GEMINI_API_KEY=xxx  # In .env file, never hardcode

# Install
pip install -r requirements.txt

# Run UI
python run_ui.py
```

## Improvement Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  1. BASELINE: Run eval set, record metrics                  │
├─────────────────────────────────────────────────────────────┤
│  2. ANALYZE: Identify failure patterns                      │
│     - Which query types fail?                               │
│     - Search not finding docs?                              │
│     - Synthesis weak?                                       │
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
│  5. VALIDATE: Test the fix                                  │
├─────────────────────────────────────────────────────────────┤
│  6. COMPARE: Run full eval, compare to baseline             │
└─────────────────────────────────────────────────────────────┘
```

## Key Files for Modifications

| Area | File | Purpose |
|------|------|---------|
| Search | `src/irys/core/search.py` | Document search logic |
| Engine | `src/irys/rlm/engine.py` | Investigation loop |
| Decisions | `src/irys/rlm/decisions.py` | LLM decision functions |
| Prompts | `src/irys/rlm/prompts.py` | Prompt templates |
| Reader | `src/irys/core/reader.py` | PDF/DOCX extraction |
