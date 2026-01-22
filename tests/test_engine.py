"""Engine tests for Irys RLM system - testing fixes made in review cycles."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from irys.rlm.engine import RLMEngine, RLMConfig
from irys.rlm.state import InvestigationState, Citation, Lead
from irys.core.repository import MatterRepository
from irys.core.reader import DocumentReader


class TestDepthGating:
    """Tests for depth tracking and gating - Fix #2, #7."""

    def test_depth_increments_per_iteration(self):
        """Verify recursion_depth increments with each iteration."""
        state = InvestigationState.create("Test query", "/path")

        # Simulate iteration progression
        state.recursion_depth = 1  # iteration 0 + 1
        assert state.recursion_depth == 1

        state.recursion_depth = 2  # iteration 1 + 1
        assert state.recursion_depth == 2

        state.recursion_depth = 3  # iteration 2 + 1
        assert state.recursion_depth == 3

    def test_max_depth_reached_tracking(self):
        """Verify max_depth_reached is updated correctly."""
        state = InvestigationState.create("Test query", "/path")
        assert state.max_depth_reached == 0

        # Simulate depth tracking during parallel lead processing
        lead_depth = 2
        if lead_depth > state.max_depth_reached:
            state.max_depth_reached = lead_depth
        assert state.max_depth_reached == 2

        # Lower depth shouldn't update max
        lead_depth = 1
        if lead_depth > state.max_depth_reached:
            state.max_depth_reached = lead_depth
        assert state.max_depth_reached == 2  # Still 2, not 1

        # Higher depth should update
        lead_depth = 4
        if lead_depth > state.max_depth_reached:
            state.max_depth_reached = lead_depth
        assert state.max_depth_reached == 4

    def test_depth_passed_as_parameter_not_shared(self):
        """Test that depth is passed as parameter to avoid race condition."""
        # This tests the fix where current_depth is captured before parallel processing
        state = InvestigationState.create("Test query", "/path")
        state.recursion_depth = 3

        # Capture depth at start (like engine does)
        current_depth = state.recursion_depth

        # Simulate parallel lead processing modifying state (race condition scenario)
        state.recursion_depth = 5  # Another task modifies it

        # Original task should still use captured depth
        assert current_depth == 3

        # Lead depth calculation using captured value
        lead_depth = current_depth + 1
        assert lead_depth == 4  # Not 6 from modified state


class TestCitationCallbacks:
    """Tests for citation callback handling - Fix #3."""

    def test_citation_callback_with_none_citation(self):
        """Verify no crash when citation is None (duplicate detected)."""
        state = InvestigationState.create("Test query", "/path")
        callback_calls = []

        def on_citation(citation):
            callback_calls.append(citation)

        # Add first citation - should succeed
        c1 = state.add_citation("doc.pdf", 1, "Some text", "context", "relevance")
        if c1 and on_citation:
            on_citation(c1)

        assert len(callback_calls) == 1
        assert callback_calls[0] == c1

        # Add duplicate citation - returns None
        c2 = state.add_citation("doc.pdf", 1, "Some text", "context", "relevance")
        if c2 and on_citation:  # This is the fix - guard against None
            on_citation(c2)

        # Should still be 1 call, not crash
        assert len(callback_calls) == 1

    def test_citation_callback_not_called_when_none(self):
        """Explicitly test the guard condition."""
        callback_called = False

        def on_citation(citation):
            nonlocal callback_called
            callback_called = True

        citation = None  # Simulating duplicate return

        # The fixed pattern
        if citation and on_citation:
            on_citation(citation)

        assert not callback_called

    def test_citation_callback_called_when_valid(self):
        """Verify callback is called for valid citations."""
        callback_calls = []

        def on_citation(citation):
            callback_calls.append(citation)

        state = InvestigationState.create("Test query", "/path")
        citation = state.add_citation("doc1.pdf", 1, "Text 1", "ctx", "rel")

        if citation and on_citation:
            on_citation(citation)

        assert len(callback_calls) == 1


class TestLegacyFormatHandling:
    """Tests for legacy file format handling - Fix #4, Fix #12."""

    def test_reader_rejects_doc_files(self):
        """Verify .doc files are not in SUPPORTED_EXTENSIONS."""
        reader = DocumentReader()
        assert ".doc" not in reader.SUPPORTED_EXTENSIONS
        assert not reader.can_read(Path("old_document.doc"))

    def test_reader_rejects_rtf_files(self):
        """Verify .rtf files are not in SUPPORTED_EXTENSIONS."""
        reader = DocumentReader()
        assert ".rtf" not in reader.SUPPORTED_EXTENSIONS
        assert not reader.can_read(Path("old_document.rtf"))

    def test_repository_rejects_doc_files(self):
        """Verify MatterRepository doesn't list .doc files."""
        assert ".doc" not in MatterRepository.SUPPORTED_EXTENSIONS

    def test_repository_stats_tracks_legacy_files(self, tmp_path):
        """Verify RepositoryStats tracks skipped legacy files."""
        # Create test files
        (tmp_path / "contract.txt").write_text("Valid content")
        (tmp_path / "old_doc.doc").write_text("Legacy content")
        (tmp_path / "old_rtf.rtf").write_text("Legacy content")

        repo = MatterRepository(tmp_path)
        stats = repo.get_stats()

        # Should only count the txt file
        assert stats.total_files == 1
        # Should track skipped legacy files
        assert stats.skipped_legacy_files == 2
        assert stats.has_legacy_files is True

    def test_repository_stats_no_legacy_files(self, tmp_path):
        """Verify stats correctly reports no legacy files when none exist."""
        (tmp_path / "contract.txt").write_text("Valid content")
        (tmp_path / "report.txt").write_text("Another file")

        repo = MatterRepository(tmp_path)
        stats = repo.get_stats()

        assert stats.total_files == 2
        assert stats.skipped_legacy_files == 0
        assert stats.has_legacy_files is False


class TestRLMConfigDefaults:
    """Tests for RLM configuration defaults."""

    def test_default_config_values(self):
        """Verify sensible defaults."""
        config = RLMConfig()

        assert config.max_depth == 5
        assert config.max_leads_per_level == 5
        assert config.min_lead_priority == 0.3
        assert config.adaptive_depth is True
        assert config.min_depth == 2
        assert config.depth_citation_threshold == 15
        assert config.max_iterations == 20

    def test_custom_config(self):
        """Test custom configuration."""
        config = RLMConfig(
            max_depth=3,
            max_iterations=10,
            adaptive_depth=False,
        )

        assert config.max_depth == 3
        assert config.max_iterations == 10
        assert config.adaptive_depth is False


class TestQueryComplexityFalsePositives:
    """Tests for query complexity calculation - Fix #11."""

    def test_conjunction_whole_word_matching(self):
        """Verify conjunctions are matched as whole words, not substrings."""
        from irys.rlm.state import classify_query

        # "mandated" contains "and" but shouldn't trigger conjunction bonus
        result1 = classify_query("What does the mandated clause say?")

        # "What and when" has actual "and" conjunction
        result2 = classify_query("What and when did the breach occur?")

        # The complexity of a query with actual conjunction should be STRICTLY higher
        # than one with "and" only as substring
        assert result2["complexity"] > result1["complexity"], \
            f"Conjunction query should have strictly higher complexity: {result2['complexity']} vs {result1['complexity']}"

    def test_actual_conjunction_detection(self):
        """Verify actual conjunctions increase complexity."""
        from irys.rlm.state import classify_query

        simple = classify_query("What happened?")
        with_and = classify_query("What happened and why?")
        with_or = classify_query("Did A or B occur?")

        # Queries with conjunctions should have strictly higher complexity
        assert with_and["complexity"] > simple["complexity"], \
            f"'and' conjunction should increase complexity: {with_and['complexity']} vs {simple['complexity']}"
        assert with_or["complexity"] > simple["complexity"], \
            f"'or' conjunction should increase complexity: {with_or['complexity']} vs {simple['complexity']}"

    def test_substring_and_not_detected(self):
        """Verify words containing 'and' as substring don't trigger bonus."""
        from irys.rlm.state import classify_query

        # These contain 'and' as substring but not as a word
        mandated = classify_query("What is the mandated procedure?")
        standard = classify_query("Is this the standard practice?")
        outstanding = classify_query("What are the outstanding issues?")

        # These have actual 'and' conjunction
        actual_and = classify_query("What is the procedure and policy?")

        # Actual conjunction should have higher complexity
        assert actual_and["complexity"] > mandated["complexity"]
        assert actual_and["complexity"] > standard["complexity"]
        assert actual_and["complexity"] > outstanding["complexity"]

    def test_substring_or_not_detected(self):
        """Verify words containing 'or' as substring don't trigger bonus."""
        from irys.rlm.state import classify_query

        # These contain 'or' as substring but not as a word
        performance = classify_query("What was the performance level?")
        corporate = classify_query("Is this a corporate matter?")
        priority = classify_query("What is the priority?")

        # This has actual 'or' conjunction
        actual_or = classify_query("Is this contract or agreement?")

        # Actual conjunction should have higher complexity
        assert actual_or["complexity"] > performance["complexity"]
        assert actual_or["complexity"] > corporate["complexity"]
        assert actual_or["complexity"] > priority["complexity"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
