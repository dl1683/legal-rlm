# Roundtable Unit 24: Evidence Strength Scoring

## Unit Goal
Score the strength of evidence found during investigations.

## Success Criteria
1. [x] EVIDENCE_SOURCE_WEIGHTS dictionary
2. [x] EvidenceItem dataclass
3. [x] calculate_strength() method
4. [x] Evidence tracking in InvestigationState
5. [x] Serialization support

## Changes Made

### state.py - EVIDENCE_SOURCE_WEIGHTS
| Document Type | Weight |
|--------------|--------|
| contract | 1.0 |
| agreement | 1.0 |
| judgment | 1.0 |
| order | 0.95 |
| declaration | 0.9 |
| affidavit | 0.9 |
| complaint | 0.85 |
| motion | 0.8 |
| report | 0.7 |
| memo | 0.65 |
| letter | 0.5 |
| email | 0.4 |
| draft | 0.3 |

### state.py - EvidenceItem Class
| Field | Description |
|-------|-------------|
| id | Unique identifier |
| claim | The claim being supported |
| source_doc | Source document |
| page | Page number |
| quote | Supporting quote |
| strength_score | 0-100 score |
| strength_level | strong/moderate/weak/insufficient |
| factors | Score breakdown |
| corroborating_sources | Supporting documents |
| contradicting_sources | Contradicting documents |

### Strength Calculation Factors
| Factor | Weight | Description |
|--------|--------|-------------|
| source_type | 0-30 | Document type reliability |
| verification | 0-25 | Whether citation verified |
| corroboration | 0-25 | Support from other sources |
| contradictions | -15-0 | Penalty for contradictions |
| specificity | 0-20 | How specific the quote is |

### state.py - Methods
| Method | Description |
|--------|-------------|
| add_evidence() | Add evidence with strength calculation |
| get_evidence_by_strength() | Filter by strength level |
| get_strong_evidence() | Get strong/moderate items |
| recalculate_evidence_strength() | Update with corroboration |
| get_evidence_summary() | Summary statistics |
| get_evidence_formatted() | Human-readable output |

### Key Code
```python
@dataclass
class EvidenceItem:
    id: str
    claim: str
    source_doc: str
    page: Optional[int]
    quote: str
    strength_score: float = 0.0
    strength_level: str = "unknown"
    factors: dict = field(default_factory=dict)
    corroborating_sources: list[str] = field(default_factory=list)

    def calculate_strength(
        self,
        verified: bool = False,
        corroboration_count: int = 0,
        contradiction_count: int = 0,
        specificity: float = 0.5,
    ):
        factors = {}
        # Factor 1: Source type (0-30)
        factors["source_type"] = get_source_weight(self.source_doc) * 30
        # Factor 2: Verification (0-25)
        factors["verification"] = 25 if verified else 5
        # Factor 3: Corroboration (0-25)
        factors["corroboration"] = min(corroboration_count * 8, 25)
        # Factor 4: Contradiction penalty (-15 to 0)
        factors["contradictions"] = -min(contradiction_count * 5, 15)
        # Factor 5: Specificity (0-20)
        factors["specificity"] = specificity * 20

        total_score = max(0, min(100, sum(factors.values())))
        self.strength_score = total_score
        self.strength_level = self._determine_level(total_score)
```

## Strength Levels
| Level | Score Range | Meaning |
|-------|-------------|---------|
| strong | 75-100 | High reliability |
| moderate | 50-74 | Reasonable support |
| weak | 25-49 | Limited support |
| insufficient | 0-24 | Not reliable |

## Review Notes
- Multi-factor scoring provides nuanced assessment
- Corroboration from multiple sources increases strength
- Contradictions reduce evidence reliability
- Document type weights reflect legal practice
- Recalculation allows dynamic updates as more evidence found

## Next Unit
Unit 25: Multi-Query Support
