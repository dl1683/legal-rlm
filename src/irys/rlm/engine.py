"""RLM Engine - Recursive Language Model investigation engine.

OPTIMIZED VERSION:
- Document extraction cache (don't re-extract same doc)
- Search term deduplication (skip similar searches)
- Smart model selection (FLASH for simple queries)
- Early sufficiency checks (after first iteration)
"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from pathlib import Path
import asyncio
import logging

from ..core.models import GeminiClient, ModelTier
from ..core.repository import MatterRepository
from ..core.search import SearchResults
from .state import InvestigationState, StepType, ThinkingStep, Citation, Lead, classify_query
from . import decisions

logger = logging.getLogger(__name__)


@dataclass
class RLMConfig:
    """Configuration for RLM engine."""
    max_depth: int = 3  # Reduced from 5
    max_leads_per_level: int = 3  # Reduced from 5
    max_documents_per_search: int = 5  # Reduced from 10
    # Dynamic excerpt limits based on query complexity
    excerpt_chars_simple: int = 8000  # Fast path for simple queries
    excerpt_chars_complex: int = 40000  # Full coverage for complex queries
    parallel_reads: int = 3  # Reduced from 5
    checkpoint_dir: Optional[str] = None
    checkpoint_interval: int = 5
    max_iterations: int = 10  # Reduced from 20
    # New optimization settings
    early_exit_facts: int = 5  # Exit early if we have this many facts
    skip_similar_searches: bool = True
    use_flash_for_simple: bool = True


@dataclass
class InvestigationCache:
    """Cache to avoid redundant work."""
    extracted_docs: set = field(default_factory=set)  # Docs we've extracted facts from
    searched_terms: set = field(default_factory=set)  # Search terms we've used

    def has_extracted(self, filepath: str) -> bool:
        """Check if we've already extracted facts from this doc."""
        return filepath in self.extracted_docs

    def mark_extracted(self, filepath: str):
        """Mark a doc as extracted."""
        self.extracted_docs.add(filepath)

    def is_similar_search(self, term: str) -> bool:
        """Check if we've done a similar search."""
        term_lower = term.lower()
        term_words = set(term_lower.split())

        for existing in self.searched_terms:
            existing_words = set(existing.lower().split())
            # If >50% word overlap, consider it similar
            if term_words and existing_words:
                overlap = len(term_words & existing_words) / min(len(term_words), len(existing_words))
                if overlap > 0.5:
                    return True
        return False

    def add_search(self, term: str):
        """Record a search term."""
        self.searched_terms.add(term.lower())


class RLMEngine:
    """
    Recursive Language Model investigation engine.

    OPTIMIZED loop:
    1. Plan - create investigation strategy
    2. Execute - search and read documents (with caching)
    3. Check - LLM decides if sufficient (check early!)
    4. Synthesize - use appropriate model based on query type
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        config: Optional[RLMConfig] = None,
        on_step: Optional[Callable[[ThinkingStep], None]] = None,
        on_citation: Optional[Callable[[Citation], None]] = None,
        on_progress: Optional[Callable[[dict], None]] = None,
    ):
        self.client = gemini_client
        self.config = config or RLMConfig()
        self.on_step = on_step
        self.on_citation = on_citation
        self.on_progress = on_progress

    async def investigate(
        self,
        query: str,
        repository_path: str | Path,
    ) -> InvestigationState:
        """Run full recursive investigation."""
        repo = MatterRepository(repository_path)
        state = InvestigationState.create(query, str(repository_path))
        cache = InvestigationCache()

        # Classify the query
        state.query_classification = classify_query(query)
        is_simple = state.query_classification['type'] == 'factual' and state.query_classification['complexity'] <= 2
        self._is_simple_query = is_simple  # Store for dynamic limits

        self._emit_step(
            state,
            StepType.THINKING,
            f"Query classified as {state.query_classification['type']} (complexity: {state.query_classification['complexity']}/5)" +
            (" [SIMPLE - will use fast path]" if is_simple else ""),
        )

        try:
            # Phase 1: Create plan
            await self._create_plan(state, repo)

            # Phase 2: Investigation loop (optimized)
            await self._investigate_loop(state, repo, cache)

            # Phase 3: Final synthesis (model based on query type)
            await self._synthesize(state, is_simple)

            state.complete()

        except Exception as e:
            state.fail(str(e))
            raise

        return state

    async def _create_plan(self, state: InvestigationState, repo: MatterRepository):
        """Phase 1: Create investigation plan using LLM."""
        self._emit_step(state, StepType.THINKING, "Analyzing repository structure...")

        stats = repo.get_stats()
        structure = repo.get_structure()
        structure_str = "\n".join(f"  {folder}: {count} files" for folder, count in structure.items())

        # Use decisions layer for planning
        plan = await decisions.create_plan(
            query=state.query,
            repo_structure=structure_str,
            total_files=stats.total_files,
            client=self.client,
        )

        state.hypothesis = plan.get("success_criteria", "Investigating query")
        state.findings["issues"] = plan.get("key_issues", [])
        state.findings["initial_plan"] = plan

        # Create leads from search terms - LIMIT TO 3
        for term in plan.get("search_terms", [])[:3]:
            if isinstance(term, str):
                state.add_lead(f"Search for: {term}", source="initial_plan")

        # Fallback if no leads
        if not state.leads:
            terms = await decisions.extract_search_terms(state.query, self.client)
            for term in terms[:2]:
                state.add_lead(f"Search for: {term}", source="fallback")

        self._emit_step(
            state,
            StepType.THINKING,
            f"Plan created with {len(state.leads)} initial leads",
            details=plan,
        )

    async def _investigate_loop(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        cache: InvestigationCache,
    ):
        """Phase 2: Iterative investigation with optimizations."""
        iteration = 0

        while iteration < self.config.max_iterations:
            state.recursion_depth = iteration + 1

            pending_leads = state.get_pending_leads()
            if not pending_leads:
                self._emit_step(state, StepType.THINKING, "No more leads to investigate")
                break

            # Take fewer leads
            leads_to_process = pending_leads[:self.config.max_leads_per_level]

            self._emit_step(
                state,
                StepType.THINKING,
                f"Processing {len(leads_to_process)} leads (iteration {iteration + 1})",
            )

            # Process leads
            for lead in leads_to_process:
                await self._investigate_lead(state, repo, lead, cache)

            iteration += 1

            # OPTIMIZATION: Check sufficiency EARLY (after first iteration)
            facts_count = len(state.findings.get("accumulated_facts", []))

            # Early exit if we have enough facts for simple queries
            if facts_count >= self.config.early_exit_facts:
                findings_summary = self._format_findings(state)
                is_sufficient = await decisions.is_sufficient(
                    query=state.query,
                    findings=findings_summary,
                    client=self.client,
                )

                if is_sufficient:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Early exit: {facts_count} facts sufficient",
                    )
                    break

            # Save checkpoint periodically
            if self.config.checkpoint_dir and iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(state, iteration)

    async def _investigate_lead(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        lead: Lead,
        cache: InvestigationCache,
    ):
        """Investigate a single lead with caching."""
        if state.recursion_depth > self.config.max_depth:
            state.mark_lead_investigated(lead.id, "Max depth reached")
            return

        if state.recursion_depth > state.max_depth_reached:
            state.max_depth_reached = state.recursion_depth

        # Determine if this is a search or read lead
        if lead.description.startswith("Read document:"):
            filepath = lead.description.replace("Read document:", "").strip()

            # OPTIMIZATION: Skip if already extracted
            if cache.has_extracted(filepath):
                state.mark_lead_investigated(lead.id, "Already extracted")
                return

            await self._read_document(state, repo, filepath, cache)
            state.mark_lead_investigated(lead.id, "Document read")
        else:
            # Extract search term
            search_term = self._extract_search_term(lead.description)

            # OPTIMIZATION: Skip similar searches
            if self.config.skip_similar_searches and cache.is_similar_search(search_term):
                state.mark_lead_investigated(lead.id, f"Similar search already done")
                logger.info(f"Skipping similar search: {search_term}")
                return

            cache.add_search(search_term)
            self._emit_step(state, StepType.SEARCH, f"Searching: {search_term}")

            # Perform search (using smart_search for OR fallback)
            results = repo.smart_search(search_term, context_lines=2)
            state.searches_performed += 1

            if not results.hits:
                state.mark_lead_investigated(lead.id, "No results found")
                return

            self._emit_step(
                state,
                StepType.FINDING,
                f"Found {len(results.hits)} matches",
            )

            # Let LLM pick the most relevant hits
            relevant_hits = await decisions.pick_relevant_hits(
                query=state.query,
                results=results,
                client=self.client,
            )

            # Analyze results and read docs (with caching)
            await self._analyze_results(state, repo, results, relevant_hits, cache)

            state.mark_lead_investigated(lead.id, f"Found {len(results.hits)} matches")

    async def _analyze_results(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        results: SearchResults,
        relevant_hits: list,
        cache: InvestigationCache,
    ):
        """Analyze search results using LLM."""
        # Use decisions layer to analyze
        analysis = await decisions.analyze_results(
            query=state.query,
            results=results,
            client=self.client,
        )

        # Store facts
        facts = analysis.get("facts", [])
        state.add_facts(facts)

        # Add citations from relevant hits (limit to 3)
        for hit in relevant_hits[:3]:
            citation = state.add_citation(
                document=hit.file_path,
                page=hit.page_num,
                text=hit.match_text[:200],
                context=hit.context[:300],
                relevance=f"Found via search: {results.query}",
            )
            if citation and self.on_citation:
                self.on_citation(citation)

        # OPTIMIZATION: Only add leads for docs we haven't extracted
        for filepath in analysis.get("read_deeper", [])[:2]:
            if isinstance(filepath, str) and not cache.has_extracted(filepath):
                state.add_lead(f"Read document: {filepath}", source="analysis")

        # OPTIMIZATION: Limit additional searches
        for term in analysis.get("additional_searches", [])[:1]:
            if isinstance(term, str) and not cache.is_similar_search(term):
                state.add_lead(f"Search for: {term}", source="analysis")

        # Get candidate files from search results
        candidate_files = list(results.by_file().keys())

        # OPTIMIZATION: Dynamic document prioritization using LLM
        if candidate_files:
            key_issues = state.findings.get("issues", [])
            already_read = list(cache.extracted_docs)

            prioritized_files = await decisions.prioritize_documents(
                query=state.query,
                candidate_files=candidate_files,
                already_read=already_read,
                key_issues=key_issues,
                client=self.client,
            )

            # Filter to unread files only
            top_files = [f for f in prioritized_files if not cache.has_extracted(f)]
        else:
            top_files = []

        await self._batch_read(state, repo, top_files[:self.config.parallel_reads], cache)

    async def _batch_read(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_paths: list[str],
        cache: InvestigationCache,
    ):
        """Read multiple documents in parallel."""
        # Filter out already extracted docs
        to_read = [fp for fp in file_paths if not cache.has_extracted(fp)]

        if not to_read:
            return

        self._emit_step(
            state,
            StepType.READING,
            f"Reading {len(to_read)} documents",
        )

        tasks = [self._read_document(state, repo, fp, cache) for fp in to_read]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _read_document(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        file_path: str,
        cache: InvestigationCache,
    ):
        """Read and extract facts from a document."""
        # OPTIMIZATION: Skip if already extracted
        if cache.has_extracted(file_path):
            logger.debug(f"Skipping already extracted: {file_path}")
            return

        self._emit_step(state, StepType.READING, f"Reading: {Path(file_path).name}")

        try:
            doc = repo.read(file_path)
            state.documents_read += 1
            cache.mark_extracted(file_path)  # Mark as extracted

            # Dynamic excerpt limit based on query complexity
            excerpt_limit = (
                self.config.excerpt_chars_simple
                if getattr(self, '_is_simple_query', False)
                else self.config.excerpt_chars_complex
            )
            content = doc.get_excerpt(excerpt_limit)

            # Dynamic extraction limit (matching excerpt)
            extraction_limit = 10000 if getattr(self, '_is_simple_query', False) else 35000

            # Use decisions layer to extract facts
            extraction = await decisions.extract_facts(
                query=state.query,
                filename=doc.filename,
                content=content,
                client=self.client,
                max_content_chars=extraction_limit,
            )

            # Store facts
            facts = extraction.get("facts", [])
            state.add_facts(facts)

            # Add citations from quotes (limit to 2)
            for quote in extraction.get("quotes", [])[:2]:
                if isinstance(quote, dict) and "text" in quote:
                    citation = state.add_citation(
                        document=doc.path,
                        page=quote.get("page"),
                        text=quote["text"][:200],
                        context="",
                        relevance=quote.get("relevance", "Direct quote"),
                    )
                    if citation and self.on_citation:
                        self.on_citation(citation)

            # Skip adding reference leads - reduces iteration depth

        except Exception as e:
            self._emit_step(state, StepType.ERROR, f"Failed to read {file_path}: {e}")

    async def _synthesize(self, state: InvestigationState, is_simple: bool = False):
        """Phase 3: Final synthesis - use appropriate model."""
        self._emit_step(state, StepType.SYNTHESIS, "Synthesizing final analysis...")

        # Compile evidence
        facts = state.findings.get("accumulated_facts", [])
        evidence = "\n".join(f"- {fact}" for fact in facts[:20])  # Limit facts

        citations = state.get_citations_formatted()

        # OPTIMIZATION: Use FLASH for simple queries
        if is_simple and self.config.use_flash_for_simple:
            logger.info("Using FLASH model for simple query synthesis")
            response = await decisions.synthesize_simple(
                query=state.query,
                evidence=evidence or "No specific findings accumulated",
                citations=citations or "No citations collected",
                client=self.client,
            )
        else:
            response = await decisions.synthesize(
                query=state.query,
                evidence=evidence or "No specific findings accumulated",
                citations=citations or "No citations collected",
                client=self.client,
            )

        state.findings["final_output"] = response
        self._emit_step(state, StepType.SYNTHESIS, "Analysis complete")

    def _emit_step(
        self,
        state: InvestigationState,
        step_type: StepType,
        content: str,
        details: Optional[dict] = None,
    ):
        """Emit a thinking step and call callback."""
        step = state.add_step(step_type, content, details)
        if self.on_step:
            self.on_step(step)
        self._emit_progress(state)

    def _emit_progress(self, state: InvestigationState):
        """Emit progress update."""
        if self.on_progress:
            self.on_progress(state.get_progress())

    def _extract_search_term(self, lead_description: str) -> str:
        """Extract search term from lead description."""
        import re

        # Remove common prefixes
        prefixes = ["Search for:", "Investigate:", "Find:", "Look for:"]
        term = lead_description
        for prefix in prefixes:
            if term.startswith(prefix):
                term = term[len(prefix):].strip()
                break

        # Clean up
        term = re.sub(r'\bAND\b|\bOR\b|\bNOT\b', ' ', term, flags=re.IGNORECASE)
        term = re.sub(r'[\'\"()]', ' ', term)
        term = re.sub(r'\s+', ' ', term).strip()

        return term[:50] if term else "contract"

    def _format_findings(self, state: InvestigationState) -> str:
        """Format current findings for LLM."""
        facts = state.findings.get("accumulated_facts", [])
        citations_count = len(state.citations)
        docs_read = state.documents_read

        lines = [
            f"Documents read: {docs_read}",
            f"Citations found: {citations_count}",
            "",
            "Key facts found:",
        ]

        for fact in facts[:15]:  # Limit facts in summary
            lines.append(f"- {fact}")

        if not facts:
            lines.append("- No facts extracted yet")

        return "\n".join(lines)

    def _save_checkpoint(self, state: InvestigationState, iteration: int):
        """Save investigation checkpoint."""
        if not self.config.checkpoint_dir:
            return

        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{state.id}_iter{iteration}.json"
        state.save_checkpoint(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    async def resume_investigation(
        self,
        checkpoint_path: str | Path,
    ) -> InvestigationState:
        """Resume investigation from checkpoint."""
        state = InvestigationState.load_checkpoint(checkpoint_path)
        repo = MatterRepository(state.repository_path)
        cache = InvestigationCache()

        self._emit_step(state, StepType.THINKING, "Resuming investigation from checkpoint")

        try:
            if state.status not in ("completed", "failed"):
                await self._investigate_loop(state, repo, cache)
                await self._synthesize(state)
                state.complete()

        except Exception as e:
            state.fail(str(e))
            raise

        return state

    async def summarize_documents(
        self,
        file_paths: list[Path],
        repository: Optional[MatterRepository] = None,
    ) -> dict[str, Any]:
        """Create summaries for multiple documents."""
        if not file_paths:
            return {"individual_summaries": [], "collection_summary": None}

        summaries = []
        for fp in file_paths:
            try:
                if repository:
                    doc = repository.read(str(fp))
                else:
                    temp_repo = MatterRepository(fp.parent)
                    doc = temp_repo.read(str(fp))

                content = doc.get_excerpt(self.config.excerpt_chars_complex)

                extraction = await decisions.extract_facts(
                    query="Summarize this document",
                    filename=doc.filename,
                    content=content,
                    client=self.client,
                )

                summaries.append({
                    "filename": doc.filename,
                    "facts": extraction.get("facts", []),
                    "quotes": extraction.get("quotes", []),
                    "references": extraction.get("references", []),
                })

            except Exception as e:
                logger.error(f"Failed to summarize {fp}: {e}")

        return {
            "individual_summaries": summaries,
            "document_count": len(summaries),
        }
