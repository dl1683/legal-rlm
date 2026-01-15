"""Investigation templates for common legal query types.

Provides predefined search strategies and focus areas for different
types of legal investigations.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InvestigationTemplate:
    """Template for a type of legal investigation."""
    name: str
    description: str
    initial_searches: list[str]
    key_document_types: list[str]
    entity_focus: list[str]  # Types of entities to prioritize
    key_questions: list[str]
    red_flags: list[str]  # Patterns to watch for
    synthesis_focus: list[str]  # What to emphasize in final output


# Predefined investigation templates
TEMPLATES: dict[str, InvestigationTemplate] = {
    "contract_review": InvestigationTemplate(
        name="Contract Review",
        description="Analyze contract terms, obligations, and risks",
        initial_searches=[
            "obligation", "liability", "indemnification", "termination",
            "warranty", "limitation", "confidential", "breach"
        ],
        key_document_types=["contract", "agreement", "amendment", "exhibit"],
        entity_focus=["companies", "people", "dates", "amounts"],
        key_questions=[
            "What are the key obligations of each party?",
            "What are the termination conditions?",
            "What limitations of liability exist?",
            "What indemnification provisions apply?",
        ],
        red_flags=[
            "unlimited liability", "perpetual", "irrevocable",
            "sole discretion", "without cause", "waive"
        ],
        synthesis_focus=[
            "Key obligations", "Risk areas", "Termination rights",
            "Financial exposure", "Recommended actions"
        ],
    ),

    "litigation_analysis": InvestigationTemplate(
        name="Litigation Analysis",
        description="Analyze pleadings, evidence, and case strategy",
        initial_searches=[
            "claim", "allegation", "damages", "evidence",
            "witness", "exhibit", "motion", "order"
        ],
        key_document_types=["complaint", "motion", "order", "declaration", "exhibit"],
        entity_focus=["people", "companies", "dates", "amounts"],
        key_questions=[
            "What are the main claims/allegations?",
            "What evidence supports each claim?",
            "What is the timeline of events?",
            "What are the potential damages?",
        ],
        red_flags=[
            "fraud", "willful", "intentional", "punitive",
            "bad faith", "misrepresentation"
        ],
        synthesis_focus=[
            "Claims summary", "Evidence strength", "Timeline",
            "Exposure analysis", "Strategy recommendations"
        ],
    ),

    "due_diligence": InvestigationTemplate(
        name="Due Diligence",
        description="Comprehensive review for transactions",
        initial_searches=[
            "material", "disclosure", "representation", "warranty",
            "encumbrance", "litigation", "compliance", "consent"
        ],
        key_document_types=["agreement", "disclosure", "report", "certificate"],
        entity_focus=["companies", "people", "amounts", "dates"],
        key_questions=[
            "What material contracts exist?",
            "Are there any pending litigations?",
            "What representations were made?",
            "What consents are required?",
        ],
        red_flags=[
            "undisclosed", "material adverse", "default",
            "pending litigation", "regulatory investigation"
        ],
        synthesis_focus=[
            "Material findings", "Risk areas", "Disclosure gaps",
            "Required consents", "Recommended conditions"
        ],
    ),

    "regulatory_compliance": InvestigationTemplate(
        name="Regulatory Compliance",
        description="Review regulatory requirements and compliance",
        initial_searches=[
            "compliance", "regulation", "requirement", "violation",
            "penalty", "filing", "report", "audit"
        ],
        key_document_types=["report", "filing", "correspondence", "memo"],
        entity_focus=["dates", "amounts", "companies"],
        key_questions=[
            "What regulations apply?",
            "What are the compliance requirements?",
            "Are there any violations or penalties?",
            "What filings are required?",
        ],
        red_flags=[
            "violation", "penalty", "non-compliance", "audit finding",
            "enforcement action", "investigation"
        ],
        synthesis_focus=[
            "Applicable regulations", "Compliance status",
            "Violations/penalties", "Remediation needs"
        ],
    ),

    "ip_review": InvestigationTemplate(
        name="IP Review",
        description="Intellectual property analysis",
        initial_searches=[
            "patent", "trademark", "copyright", "license",
            "infringement", "royalty", "assignment", "ownership"
        ],
        key_document_types=["agreement", "license", "assignment", "certificate"],
        entity_focus=["companies", "dates", "amounts"],
        key_questions=[
            "What IP assets exist?",
            "What licenses are in place?",
            "Are there any infringement issues?",
            "What ownership rights exist?",
        ],
        red_flags=[
            "infringement", "invalidity", "prior art",
            "unlicensed use", "expired"
        ],
        synthesis_focus=[
            "IP inventory", "License terms", "Ownership chain",
            "Risk areas", "Protection recommendations"
        ],
    ),
}


def get_template(template_name: str) -> Optional[InvestigationTemplate]:
    """Get an investigation template by name."""
    return TEMPLATES.get(template_name)


def get_template_names() -> list[str]:
    """Get list of available template names."""
    return list(TEMPLATES.keys())


def suggest_template(query: str) -> Optional[str]:
    """Suggest a template based on query content."""
    query_lower = query.lower()

    # Simple keyword matching for template suggestion
    if any(kw in query_lower for kw in ["contract", "agreement", "terms"]):
        return "contract_review"
    if any(kw in query_lower for kw in ["lawsuit", "litigation", "claim", "complaint"]):
        return "litigation_analysis"
    if any(kw in query_lower for kw in ["due diligence", "acquisition", "merger", "transaction"]):
        return "due_diligence"
    if any(kw in query_lower for kw in ["compliance", "regulatory", "regulation"]):
        return "regulatory_compliance"
    if any(kw in query_lower for kw in ["patent", "trademark", "copyright", "ip", "intellectual property"]):
        return "ip_review"

    return None
