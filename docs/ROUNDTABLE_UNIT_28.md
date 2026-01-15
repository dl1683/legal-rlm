# Roundtable Unit 28: Answer Quality Assessment

## Unit Goal
Automatically assess quality of generated answers to identify gaps and improvements.

## Success Criteria
1. [x] AnswerQualityAssessment dataclass
2. [x] Multi-factor quality scoring
3. [x] Issue identification
4. [x] Recommendation generation
5. [x] Integration with InvestigationState

## Changes Made

### state.py - AnswerQualityAssessment Class
| Field | Description |
|-------|-------------|
| overall_score | 0-100 quality score |
| quality_level | excellent/good/adequate/poor |
| factors | Score breakdown by factor |
| issues | Identified problems |
| recommendations | Suggested improvements |

### Quality Factors (100 points total)
| Factor | Max | Description |
|--------|-----|-------------|
| length | 15 | Answer length appropriateness |
| citations | 25 | Citation coverage |
| verification | 20 | Citation verification rate |
| query_coverage | 15 | Query term coverage |
| entities | 10 | Entity identification |
| facts | 10 | Fact density |
| document_coverage | 5 | Document coverage |

### Quality Levels
| Level | Score Range |
|-------|-------------|
| excellent | 80-100 |
| good | 60-79 |
| adequate | 40-59 |
| poor | 0-39 |

### Methods
| Method | Description |
|--------|-------------|
| AnswerQualityAssessment.assess() | Static assessment method |
| InvestigationState.assess_answer_quality() | Assess investigation's answer |
| get_formatted() | Human-readable report |
| to_dict() / from_dict() | Serialization |

### Key Code
```python
@dataclass
class AnswerQualityAssessment:
    overall_score: float = 0.0
    quality_level: str = "unknown"
    factors: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    @classmethod
    def assess(
        cls,
        answer: str,
        query: str,
        citations: list,
        verified_count: int,
        entities_found: int,
        facts_count: int,
        documents_read: int,
    ) -> "AnswerQualityAssessment":
        factors = {}
        issues = []
        recommendations = []

        # Factor 1: Answer length (0-15)
        if len(answer) < 200:
            factors["length"] = 5
            issues.append("Answer may be too brief")
        elif len(answer) < 2000:
            factors["length"] = 15
        ...

        # Factor 2: Citations (0-25)
        if len(citations) >= 10:
            factors["citations"] = 25
        else:
            issues.append("Limited citation support")

        # Calculate total and determine level
        total_score = sum(factors.values())
        quality_level = "excellent" if total_score >= 80 else ...

        return cls(
            overall_score=total_score,
            quality_level=quality_level,
            factors=factors,
            issues=issues,
            recommendations=recommendations,
        )
```

## Usage Example
```python
# Assess after investigation completes
assessment = state.assess_answer_quality()

print(f"Quality: {assessment.quality_level} ({assessment.overall_score}/100)")

if assessment.issues:
    print("Issues found:")
    for issue in assessment.issues:
        print(f"  - {issue}")

if assessment.quality_level == "poor":
    print("Recommendations:")
    for rec in assessment.recommendations:
        print(f"  - {rec}")

# Get formatted report
print(assessment.get_formatted())
```

## Sample Output
```
Answer Quality Assessment
========================================
Overall Score: 72.5/100 (GOOD)

Factor Breakdown:
  citations: 20.0
  verification: 15.0
  query_coverage: 12.0
  length: 15.0
  entities: 7.0
  facts: 3.0

Issues Identified:
  - Many citations unverified

Recommendations:
  - Verify more citations against source documents
```

## Review Notes
- Provides automated quality gate for answers
- Issues help identify specific problems
- Recommendations guide improvement
- Can be used to trigger additional investigation
- Integrates naturally with investigation workflow

## Next Unit
Unit 29: Document Summarization
