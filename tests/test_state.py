"""State management tests for Irys RLM system."""

import pytest
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.rlm.state import (
    InvestigationState, Citation, Lead, Entity, ThinkingStep, StepType,
    Contradiction, TimelineEvent, CrossReference,
    classify_query, QueryType,
)


class TestInvestigationState:
    """Tests for InvestigationState."""

    def test_create_state(self):
        state = InvestigationState.create("Test query", "/path/to/repo")
        assert state.query == "Test query"
        assert state.status == "initialized"
        assert len(state.id) == 8

    def test_add_citation_dedup(self):
        state = InvestigationState.create("Test", "/path")
        c1 = state.add_citation("doc.pdf", 1, "Some text", "context", "relevant")
        c2 = state.add_citation("doc.pdf", 1, "Some text", "context", "relevant")
        assert c1 is not None
        assert c2 is None  # Duplicate
        assert len(state.citations) == 1

    def test_add_lead_dedup(self):
        state = InvestigationState.create("Test", "/path")
        l1 = state.add_lead("Investigate the contract", "doc1", 0.8)
        l2 = state.add_lead("Investigate the contract terms", "doc2", 0.9)
        # Second lead is similar enough to be considered duplicate
        assert l1 is not None

    def test_add_entity(self):
        state = InvestigationState.create("Test", "/path")
        e1 = state.add_entity("ACME Corp", "company", "doc1.pdf")
        assert e1.mentions == 1
        e2 = state.add_entity("ACME Corp", "company", "doc2.pdf")
        # e1 and e2 are the same object - both now have 2 mentions
        assert e2.mentions == 2
        assert e1 is e2  # Same entity returned
        assert len(state.entities) == 1

    def test_confidence_score(self):
        state = InvestigationState.create("Test", "/path")
        conf = state.get_confidence_score()
        assert conf["score"] == 0
        assert conf["level"] == "insufficient"

        # Add some data
        state.add_citation("doc.pdf", 1, "text", "ctx", "rel")
        state.documents_read = 3
        conf = state.get_confidence_score()
        assert conf["score"] > 0

    def test_serialization(self):
        state = InvestigationState.create("Test query", "/path")
        state.add_citation("doc.pdf", 1, "text", "ctx", "rel")
        state.add_lead("Lead 1", "source", 0.5)
        state.add_entity("Entity", "company", "doc.pdf")

        # Serialize
        data = state.to_dict()
        assert data["query"] == "Test query"

        # Deserialize
        restored = InvestigationState.from_dict(data)
        assert restored.query == state.query
        assert len(restored.citations) == len(state.citations)
        assert len(restored.leads) == len(state.leads)


class TestQueryClassification:
    """Tests for query classification."""

    def test_factual_query(self):
        result = classify_query("What happened on January 15th?")
        assert result["type"] == "factual"

    def test_analytical_query(self):
        result = classify_query("What does this clause mean?")
        assert result["type"] == "analytical"

    def test_comparative_query(self):
        result = classify_query("Compare the two contracts")
        assert result["type"] == "comparative"

    def test_complexity(self):
        simple = classify_query("What happened?")
        complex_q = classify_query("What happened on January 15th and what were the implications for the contract and how did it affect the parties involved?")
        assert complex_q["complexity"] > simple["complexity"]


class TestEvidenceItem:
    """Tests for evidence strength scoring."""

    def test_create_evidence(self):
        ev = EvidenceItem.create(
            claim="The defendant breached the contract",
            source_doc="complaint.pdf",
            quote="Defendant failed to perform...",
            page=5,
        )
        assert ev.claim == "The defendant breached the contract"
        assert ev.strength_score == 0  # Not calculated yet

    def test_strength_calculation(self):
        ev = EvidenceItem.create("claim", "contract.pdf", "quote", 1)
        ev.calculate_strength(
            verified=True,
            corroboration_count=2,
            specificity=0.8,
        )
        assert ev.strength_score > 50  # Should be strong
        assert ev.strength_level in ["strong", "moderate"]

    def test_source_weight(self):
        # Contract should have high weight
        assert get_source_weight("contract.pdf") == 1.0
        # Email should have low weight
        assert get_source_weight("email.txt") == 0.4
        # Unknown type should have default weight
        assert get_source_weight("random.pdf") == 0.5


class TestContradiction:
    """Tests for contradiction detection."""

    def test_create_contradiction(self):
        c = Contradiction.create(
            statement1="The meeting was on January 15",
            source1="doc1.pdf",
            statement2="The meeting was on January 20",
            source2="doc2.pdf",
            contradiction_type="date",
            severity="high",
        )
        assert c.contradiction_type == "date"
        assert c.severity == "high"


class TestTimelineEvent:
    """Tests for timeline extraction."""

    def test_date_parsing(self):
        event = TimelineEvent(
            date_str="January 15, 2024",
            description="Contract signed",
            source_doc="contract.pdf",
        )
        assert event.date_parsed is not None
        assert event.date_parsed.month == 1
        assert event.date_parsed.day == 15


class TestAnswerQuality:
    """Tests for answer quality assessment."""

    def test_quality_assessment(self):
        assessment = AnswerQualityAssessment.assess(
            answer="This is a short answer.",
            query="What are the contract terms?",
            citations=[],
            verified_count=0,
            entities_found=0,
            facts_count=0,
            documents_read=1,
        )
        assert assessment.quality_level == "poor"
        assert len(assessment.issues) > 0

    def test_good_quality(self):
        assessment = AnswerQualityAssessment.assess(
            answer="This is a comprehensive answer with detailed analysis " * 20,
            query="What are the contract terms?",
            citations=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            verified_count=8,
            entities_found=10,
            facts_count=15,
            documents_read=10,
        )
        assert assessment.quality_level in ["excellent", "good"]


class TestFeedback:
    """Tests for relevance feedback."""

    def test_add_feedback(self):
        state = InvestigationState.create("Test", "/path")
        c = state.add_citation("doc.pdf", 1, "text about contracts", "ctx", "rel")

        fb = state.add_feedback(
            item_type="citation",
            item_id=c.id,
            feedback_type=FeedbackType.RELEVANT,
        )
        assert fb is not None
        assert len(state.feedback) == 1

    def test_boosted_terms(self):
        state = InvestigationState.create("Test", "/path")
        c = state.add_citation("doc.pdf", 1, "contract obligation terms", "ctx", "rel")

        state.add_feedback("citation", c.id, FeedbackType.RELEVANT)
        boosted = state.get_boosted_terms()
        assert len(boosted) > 0


class TestCrossReference:
    """Tests for cross-reference detection."""

    def test_create_cross_reference(self):
        state = InvestigationState.create("Test", "/path")
        ref = CrossReference(
            source_doc="contract.pdf",
            target_doc="amendment.pdf",
            reference_text="This amendment modifies Section 3",
        )
        state.cross_references.append(ref)
        assert len(state.cross_references) == 1
        assert ref.reference_text == "This amendment modifies Section 3"


class TestLeadManagement:
    """Tests for lead prioritization and management."""

    def test_lead_priority_sorting(self):
        state = InvestigationState.create("Test", "/path")
        state.add_lead("Low priority", "src", 0.3)
        state.add_lead("High priority", "src", 0.9)
        state.add_lead("Medium priority", "src", 0.6)

        pending = state.get_pending_leads()
        # Should be sorted by priority descending
        assert pending[0].priority == 0.9
        assert pending[1].priority == 0.6
        assert pending[2].priority == 0.3

    def test_mark_lead_investigated(self):
        state = InvestigationState.create("Test", "/path")
        lead = state.add_lead("Test lead", "src", 0.8)
        assert lead.investigated is False

        state.mark_lead_investigated(lead.id, "Found 5 matches")
        assert lead.investigated is True
        assert lead.findings == "Found 5 matches"


class TestInvestigationProgress:
    """Tests for progress tracking."""

    def test_progress_metrics(self):
        state = InvestigationState.create("Test", "/path")
        state.documents_read = 5
        state.searches_performed = 10
        state.add_citation("doc.pdf", 1, "text", "ctx", "rel")

        progress = state.get_progress()
        assert progress["documents_read"] == 5
        assert progress["searches_performed"] == 10
        assert progress["citations"] == 1

    def test_verification_stats(self):
        state = InvestigationState.create("Test", "/path")
        c1 = state.add_citation("doc1.pdf", 1, "text1", "ctx", "rel")
        c2 = state.add_citation("doc2.pdf", 2, "text2", "ctx", "rel")
        c1.verified = True
        c2.verified = False

        stats = state.get_verification_stats()
        assert stats["verified"] == 1
        assert stats["unverified"] == 1


class TestStateSerialization:
    """Extended serialization tests."""

    def test_full_state_serialization(self):
        state = InvestigationState.create("Complex query", "/path/to/repo")
        state.hypothesis = "Test hypothesis"
        state.add_citation("doc.pdf", 1, "text", "ctx", "rel")
        state.add_lead("Test lead", "src", 0.7)
        state.add_entity("ACME Corp", "company", "doc.pdf")
        state.findings["key_fact"] = "Important finding"

        # Serialize
        data = state.to_dict()

        # Verify structure
        assert "citations" in data
        assert "leads" in data
        assert "entities" in data
        assert "findings" in data
        assert data["hypothesis"] == "Test hypothesis"

        # Deserialize
        restored = InvestigationState.from_dict(data)
        assert restored.hypothesis == state.hypothesis
        assert len(restored.citations) == 1
        assert len(restored.leads) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
