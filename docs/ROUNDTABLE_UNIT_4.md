# Roundtable Unit 4: Citation Verification

## Unit Goal
Verify citations by checking if quoted text actually exists in referenced documents.

## Success Criteria
1. [x] Citation dataclass has `verified` and `verification_note` fields
2. [x] `_verify_citations` method added to engine
3. [x] Verification runs before synthesis
4. [x] Verification stats tracked in state
5. [x] Formatted output shows verification status

## Changes Made

### state.py
| Change | Description |
|--------|-------------|
| Citation.verified | New field: None=unchecked, True=verified, False=not found |
| Citation.verification_note | New field for verification details |
| StepType.VERIFY | New step type for verification |
| get_unverified_citations() | Get citations needing verification |
| get_verification_stats() | Get verification statistics |
| get_citations_formatted() | Updated to show [VERIFIED]/[UNVERIFIED] |
| Display prefixes | Changed from emojis to ASCII for Windows compatibility |

### engine.py
| Change | Description |
|--------|-------------|
| `_verify_citations` | New method to verify all citations |
| investigate() | Added verification phase between loop and synthesis |

### Verification Algorithm
1. Normalize citation text (lowercase, collapse whitespace)
2. Load document and normalize full text
3. Check if first 50 chars of citation exist in document
4. If not found, try fuzzy match with 20-char chunks
5. Mark as verified/unverified with note

### Key Code
```python
async def _verify_citations(self, state: InvestigationState, repo: MatterRepository):
    unverified = state.get_unverified_citations()
    for citation in unverified:
        doc = repo.read(citation.document)
        citation_text = " ".join(citation.text.lower().split())
        doc_text = " ".join(doc.full_text.lower().split())

        if citation_text[:50] in doc_text:
            citation.verified = True
        else:
            # Fuzzy match with 20-char chunks
            ...
```

## Review Notes
- Verification uses document cache (no redundant reads)
- Fuzzy matching handles minor OCR/extraction differences
- Unverified citations still shown but flagged
- Windows-compatible ASCII prefixes for step display

## Next Unit
Unit 5: Rate Limiting
