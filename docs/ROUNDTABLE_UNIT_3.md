# Roundtable Unit 3: JSON Parsing & Fallbacks

## Unit Goal
Make JSON parsing robust with safe fallbacks to prevent investigation failures.

## Success Criteria
1. [x] `_parse_json_safe` method with default values
2. [x] All LLM response parsing uses safe fallbacks
3. [x] Type validation for nested structures (quotes, leads, concerns)
4. [x] Logging of parse failures for debugging

## Changes Made

### engine.py
| Method | Change |
|--------|--------|
| `_parse_json_safe` | New method - parse JSON with defaults |
| `_orient` | Uses `_parse_json_safe` with fallback hypothesis |
| `_analyze_search_results` | Uses `_parse_json_safe` with empty arrays |
| `_deep_read_document` | Uses `_parse_json_safe` with proper defaults |
| Lead validation | Check `isinstance(lead_data, dict)` before access |
| Quote validation | Check `isinstance(quote, dict)` before access |
| Concern validation | Check `isinstance(concern, str)` before use |

### Key Code
```python
def _parse_json_safe(self, text: str, defaults: dict) -> dict:
    """Parse JSON from LLM response with safe fallback to defaults."""
    try:
        json_str = self._extract_json(text)
        result = json.loads(json_str)
        # Merge with defaults for any missing keys
        for key, value in defaults.items():
            if key not in result:
                result[key] = value
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse failed: {e}, using defaults")
        return defaults
```

### Default Values Used
| Method | Defaults |
|--------|----------|
| `_orient` | issues=[], relevant_folders=[], initial_searches=[], hypothesis="Investigating..." |
| `_analyze_search_results` | key_facts=[], new_leads=[], hypothesis_update=None, next_searches=[] |
| `_deep_read_document` | key_facts=[], quotes=[], entities={...}, connections=[], concerns=[] |

## Review Notes
- Parse failures now logged with warning level
- Investigation continues even with malformed LLM responses
- Type checking prevents crashes from unexpected data structures

## Next Unit
Unit 4: Citation Verification
