# Roundtable Unit 22: Investigation Templates

## Unit Goal
Provide predefined templates for common legal investigation types.

## Success Criteria
1. [x] InvestigationTemplate dataclass
2. [x] 5 predefined templates
3. [x] suggest_template() auto-suggestion
4. [x] Template accessors

## Changes Made

### templates.py - InvestigationTemplate Class
| Field | Description |
|-------|-------------|
| name | Template display name |
| description | What the template does |
| initial_searches | Seed search terms |
| key_document_types | Priority document types |
| entity_focus | Entity types to prioritize |
| key_questions | Questions to answer |
| red_flags | Patterns to watch for |
| synthesis_focus | What to emphasize in output |

### Predefined Templates
| Template | Use Case |
|----------|----------|
| contract_review | Analyze contract terms and risks |
| litigation_analysis | Analyze pleadings and evidence |
| due_diligence | Transaction review |
| regulatory_compliance | Compliance review |
| ip_review | Intellectual property analysis |

### Functions
| Function | Description |
|----------|-------------|
| get_template(name) | Get template by name |
| get_template_names() | List all templates |
| suggest_template(query) | Auto-suggest based on query |

### Key Code
```python
@dataclass
class InvestigationTemplate:
    name: str
    description: str
    initial_searches: list[str]
    key_document_types: list[str]
    entity_focus: list[str]
    key_questions: list[str]
    red_flags: list[str]
    synthesis_focus: list[str]

TEMPLATES = {
    "contract_review": InvestigationTemplate(...),
    "litigation_analysis": InvestigationTemplate(...),
    ...
}

def suggest_template(query: str) -> Optional[str]:
    if "contract" in query_lower:
        return "contract_review"
    ...
```

## Review Notes
- Templates guide investigation strategy
- Auto-suggestion based on query keywords
- Red flags help identify issues
- Synthesis focus ensures relevant output

## Next Unit
Unit 23: Search Query Expansion
