# Roundtable Unit 10: Entity Extraction

## Unit Goal
Track and organize entities (people, companies, dates) found during investigation.

## Success Criteria
1. [x] Entity dataclass with type, sources, mentions
2. [x] Entity tracking in InvestigationState
3. [x] add_entity() with deduplication by name
4. [x] add_entities_from_analysis() for batch processing
5. [x] Serialization support for checkpoints
6. [x] Engine integration for deep reads

## Changes Made

### state.py - Entity Class
| Field | Description |
|-------|-------------|
| name | Entity name (display version) |
| entity_type | "person", "company", "date", "amount", "location", "other" |
| sources | List of documents where found |
| mentions | Total mention count |
| context | Optional context snippet |

### state.py - InvestigationState Methods
| Method | Description |
|--------|-------------|
| add_entity() | Add/update entity by normalized name |
| add_entities_from_analysis() | Process LLM analysis output |
| get_entities_by_type() | Filter by entity type |
| get_top_entities() | Get most mentioned entities |
| get_entities_formatted() | Human-readable summary |

### engine.py Changes
| Location | Change |
|----------|--------|
| _deep_read_document | Calls add_entities_from_analysis() |

### Key Code
```python
def add_entity(self, name, entity_type, source, context=None):
    key = name.lower().strip()
    if key in self.entities:
        self.entities[key].add_mention(source)
        return self.entities[key]
    entity = Entity(name=name.strip(), entity_type=entity_type, ...)
    self.entities[key] = entity
    return entity

def add_entities_from_analysis(self, entities_dict, source):
    for entity_type, names in entities_dict.items():
        if isinstance(names, list):
            for name in names:
                self.add_entity(name, entity_type, source)
```

## Review Notes
- Entities deduplicated by normalized name
- Mention counts help identify key players
- Source tracking enables provenance
- Serialization preserves entity data across checkpoints

## Next Unit
Unit 11: Lead Prioritization
