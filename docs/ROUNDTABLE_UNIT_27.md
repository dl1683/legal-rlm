# Roundtable Unit 27: Relevance Feedback

## Unit Goal
Enable learning from user feedback to improve search and investigation quality.

## Success Criteria
1. [x] FeedbackType enum
2. [x] RelevanceFeedback dataclass
3. [x] add_feedback() method
4. [x] Term boosting/demotion based on feedback
5. [x] apply_feedback_to_leads() method
6. [x] Serialization support

## Changes Made

### state.py - FeedbackType Enum
| Value | Description |
|-------|-------------|
| RELEVANT | Item is relevant to query |
| NOT_RELEVANT | Item is not relevant |
| PARTIALLY_RELEVANT | Somewhat relevant |
| HELPFUL | Item was helpful |
| NOT_HELPFUL | Item was not helpful |

### state.py - RelevanceFeedback Class
| Field | Description |
|-------|-------------|
| id | Unique identifier |
| item_type | citation/lead/fact/evidence |
| item_id | ID of rated item |
| feedback | FeedbackType value |
| query | Query context |
| timestamp | When feedback given |
| notes | Optional notes |
| terms_to_boost | Terms from positive feedback |
| terms_to_demote | Terms from negative feedback |

### InvestigationState Methods
| Method | Description |
|--------|-------------|
| add_feedback() | Add user feedback |
| get_feedback_by_type() | Filter feedback |
| get_boosted_terms() | Terms from positive feedback |
| get_demoted_terms() | Terms from negative feedback |
| get_feedback_summary() | Statistics |
| apply_feedback_to_leads() | Adjust lead priorities |

### Key Code
```python
class FeedbackType(Enum):
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"

@dataclass
class RelevanceFeedback:
    id: str
    item_type: str
    item_id: str
    feedback: FeedbackType
    query: str
    terms_to_boost: list[str]
    terms_to_demote: list[str]

def add_feedback(self, item_type, item_id, feedback_type, notes=""):
    fb = RelevanceFeedback.create(...)

    # Extract terms for boosting/demotion
    item_text = self._get_item_text(item_type, item_id)
    terms = self._extract_key_terms(item_text)

    if feedback_type in (RELEVANT, HELPFUL):
        fb.terms_to_boost = terms
    elif feedback_type in (NOT_RELEVANT, NOT_HELPFUL):
        fb.terms_to_demote = terms

    self.feedback.append(fb)
    return fb

def apply_feedback_to_leads(self):
    boosted = set(self.get_boosted_terms())
    demoted = set(self.get_demoted_terms())

    for lead in self.leads:
        desc_words = set(lead.description.lower().split())
        boost_count = len(desc_words & boosted)
        demote_count = len(desc_words & demoted)

        lead.priority += boost_count * 0.1
        lead.priority -= demote_count * 0.1
```

## Usage Example
```python
# Add positive feedback on a citation
state.add_feedback(
    item_type="citation",
    item_id="abc123",
    feedback_type=FeedbackType.RELEVANT,
    notes="This contract clause is exactly what we needed"
)

# Add negative feedback on a lead
state.add_feedback(
    item_type="lead",
    item_id="xyz789",
    feedback_type=FeedbackType.NOT_RELEVANT,
)

# Apply feedback to reprioritize leads
state.apply_feedback_to_leads()

# Get feedback summary
summary = state.get_feedback_summary()
print(f"Positive: {summary['positive']}, Boosted terms: {summary['boosted_terms']}")
```

## How It Works
1. User provides feedback on items (citations, leads, facts)
2. System extracts key terms from rated items
3. Positive feedback → terms added to boost list
4. Negative feedback → terms added to demote list
5. Lead priorities adjusted based on term overlap
6. Future searches can use boosted/demoted terms

## Review Notes
- Feedback persists across checkpoint saves
- Term extraction uses simple frequency-based approach
- Lead priorities bounded to [0.0, 1.0]
- Can be integrated with search ranking for query expansion

## Next Unit
Unit 28: Answer Quality Assessment
