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
import re

from ..core.models import GeminiClient, ModelTier
from ..core.repository import MatterRepository
from ..core.search import SearchResults
from ..core.external_search import ExternalSearchManager
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
    # External search settings
    enable_external_search: bool = True
    max_case_law_results: int = 5
    max_web_results: int = 5


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
        # Initialize external search manager (enabled by default)
        self.external_search = ExternalSearchManager() if self.config.enable_external_search else None
        self._external_research: dict = {}  # Store external research results

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
            # Check if small repository - can skip complex RLM search
            if repo.is_small_repo:
                total_chars = repo.metadata.total_chars if repo.metadata else 0
                self._emit_step(
                    state,
                    StepType.THINKING,
                    f"Small repository detected ({total_chars:,} chars) - loading all content directly",
                )
                await self._direct_answer(state, repo)
            else:
                # Full RLM investigation for large repositories
                # Mimics how a lawyer works:

                # Phase 1: Create investigation plan based on the query and repo structure
                await self._create_plan(state, repo)

                # Phase 2: Investigation loop with continuous recalibration
                # - Reads documents, extracts facts, accumulates research triggers
                # - Continuously checks: sufficient? need to replan? need external research?
                # - External search is triggered MID-LOOP after triggers are accumulated
                #   (not upfront with generic terms - that produced irrelevant results)
                await self._investigate_loop(state, repo, cache)

                # Phase 3: Final synthesis
                await self._synthesize(state, is_simple)

                # Save learnings from this query for future reference
                if repo.metadata and state.findings.get("accumulated_facts"):
                    key_facts = state.findings["accumulated_facts"][:3]
                    if key_facts:
                        repo.add_learning(query, "; ".join(key_facts))

            state.complete()

        except Exception as e:
            state.fail(str(e))
            raise
        finally:
            # Clean up external search sessions
            if self.external_search:
                try:
                    await self.external_search.close()
                except Exception:
                    pass

        return state

    async def _direct_answer(self, state: InvestigationState, repo: MatterRepository):
        """
        Direct answer mode for small repositories.

        Skips: Complex RLM loop (leads, iterations, replanning, context management)
        Keeps: External search tools (case law, web) when query needs them

        Flow:
        1. Load all documents directly (no search needed - small enough)
        2. Create plan to determine if external research is needed
        3. Execute external searches if needed
        4. Synthesize answer from all sources
        """
        self._emit_step(state, StepType.READING, "Loading all documents...")

        # Load all content directly - no complex search needed for small repos
        all_content = repo.get_all_content()
        state.documents_read = len(repo.list_files())

        self._emit_step(
            state,
            StepType.FINDING,
            f"Loaded {len(all_content):,} chars from {state.documents_read} documents",
        )

        # For small repos, do a quick trigger extraction before deciding on external search
        if self.config.enable_external_search and self.external_search:
            self._emit_step(state, StepType.THINKING, "Scanning content for research triggers...")

            # Quick LITE call to extract triggers from loaded content
            triggers_result = await decisions.extract_triggers_from_content(
                content=all_content,
                client=self.client,
            )

            # Accumulate triggers into state
            state.add_triggers(triggers_result)

            # Check if meaningful triggers were found
            if state.has_external_triggers(min_triggers=2):
                triggers_summary = state.get_trigger_summary()
                self._emit_step(
                    state,
                    StepType.THINKING,
                    f"Research triggers found: {triggers_summary[:100]}...",
                )

                # Extract entity names from content for context
                content_sample = all_content[:5000]
                potential_entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content_sample)
                quick_entities = list(set(potential_entities))[:10]

                # Generate specific queries using triggers
                result = await decisions.generate_external_queries(
                    query=state.query,
                    facts=[],
                    entities=quick_entities,
                    client=self.client,
                    triggers=triggers_summary,
                )

                case_law_queries = result.get("case_law_queries", [])
                web_queries = result.get("web_queries", [])

                if case_law_queries or web_queries:
                    state.findings["initial_plan"] = {
                        "case_law_searches": case_law_queries,
                        "web_searches": web_queries,
                    }
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Generated {len(case_law_queries)} case law + {len(web_queries)} web queries from triggers",
                    )
                    await self._execute_external_searches(state)

        # Compile external research if any and add citations
        external_text = ""
        if self._external_research:
            formatted = self._format_external_research()
            if formatted.get("case_law"):
                external_text += f"\n\n=== CASE LAW ===\n{formatted['case_law']}"

                # Add citations for case law
                for case in self._external_research.get("case_law", [])[:5]:
                    citation = state.add_citation(
                        document=f"[Case Law] {case.get('case_name', 'Unknown Case')}",
                        page=None,
                        text=case.get('snippet', '') or '',
                        context=f"Citation: {case.get('citation', 'N/A')} | Court: {case.get('court', 'N/A')}",
                        relevance="External case law research",
                    )
                    if citation and self.on_citation:
                        self.on_citation(citation)

            if formatted.get("web"):
                external_text += f"\n\n=== REGULATIONS/STANDARDS ===\n{formatted['web']}"

                # Add citations for web results
                for result in self._external_research.get("web", [])[:5]:
                    citation = state.add_citation(
                        document=f"[Web] {result.get('title', 'Unknown Source')}",
                        page=None,
                        text=result.get('content', '') or '',
                        context=f"URL: {result.get('url', 'N/A')}",
                        relevance="External regulatory research",
                    )
                    if citation and self.on_citation:
                        self.on_citation(citation)

        # Include any prior learnings from this repo
        learnings = repo.get_learnings(5)
        learnings_context = ""
        if learnings:
            learnings_context = "\n\nPrior learnings from this repository:\n" + "\n".join(
                f"- Q: {l['query'][:50]}... -> {l['finding'][:100]}..." for l in learnings
            )

        # Synthesize answer from all sources
        self._emit_step(state, StepType.SYNTHESIS, "Analyzing all sources...")

        # Adjust instructions based on whether query explicitly asks for external research
        query_lower = state.query.lower()
        asks_for_external = any(term in query_lower for term in [
            'case law', 'precedent', 'regulation', 'regulatory', 'guidance', 'legal standard'
        ])

        if asks_for_external and external_text:
            instructions = """INSTRUCTIONS:
1. The query asks for EXTERNAL LEGAL RESEARCH - prioritize the CASE LAW and REGULATIONS sections
2. Cite specific cases with their citations (e.g., "Horror Inc. v. Miller, 335 F. Supp. 3d 273")
3. Reference specific regulations and their requirements
4. Connect external research to the case documents where relevant
5. If case law or regulations are limited, acknowledge that"""
        else:
            instructions = """INSTRUCTIONS:
1. Answer based primarily on the CASE DOCUMENTS
2. Support with case law and regulations if provided
3. Cite specific sources for each fact
4. Be precise with names, dates, amounts, and legal terms"""

        response = await self.client.complete(
            f"""You are a senior legal analyst. Answer this query using all provided sources.

Query: {state.query}

=== CASE DOCUMENTS ===
{all_content}
{external_text}
{learnings_context}

{instructions}""",
            tier=ModelTier.FLASH,
        )

        state.findings["final_output"] = response
        state.findings["mode"] = "direct_answer"
        self._emit_step(state, StepType.SYNTHESIS, "Analysis complete")

    def _format_external_research(self) -> dict[str, str]:
        """Format external research results for synthesis prompts.

        Returns:
            Dict with 'case_law' and 'web' keys containing formatted text
        """
        result = {"case_law": "", "web": ""}

        if not self._external_research:
            return result

        # Format case law results
        case_law = self._external_research.get("case_law", [])
        if case_law:
            case_lines = []
            for c in case_law[:5]:
                snippet = c.get('snippet') or c.get('opinion_text') or 'No snippet available'
                case_lines.append(
                    f"- **{c.get('case_name', 'Unknown')}** ({c.get('citation') or 'No citation'})\n"
                    f"  Court: {c.get('court', 'Unknown')} | Date: {c.get('date_filed', 'Unknown')}\n"
                    f"  Snippet: {snippet[:300]}..."
                )
            result["case_law"] = "\n\n".join(case_lines)

            # Include analysis if available
            analysis = self._external_research.get("analysis", {}).get("case_law", {})
            if analysis:
                summary = analysis.get("summary", "")
                if summary:
                    result["case_law"] += f"\n\n**Legal Standards Identified:** {summary}"

        # Format web results
        web = self._external_research.get("web", [])
        if web:
            web_lines = []
            for r in web[:5]:
                web_lines.append(
                    f"- **{r.get('title', 'Untitled')}**\n"
                    f"  URL: {r.get('url', '')}\n"
                    f"  Content: {r.get('content', '')[:300]}..."
                )
            result["web"] = "\n\n".join(web_lines)

            # Include Tavily's AI answer if available
            if self._external_research.get("web_answer"):
                result["web"] = f"**Summary:** {self._external_research['web_answer']}\n\n" + result["web"]

            # Include analysis if available
            analysis = self._external_research.get("analysis", {}).get("web", {})
            if analysis:
                summary = analysis.get("summary", "")
                if summary:
                    result["web"] += f"\n\n**Regulatory Context:** {summary}"

        return result

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

        # Emit the reasoning (show LLM's thinking process)
        reasoning = plan.get("reasoning", "")
        challenges = plan.get("potential_challenges", "")
        key_issues = plan.get("key_issues", [])

        if reasoning:
            self._emit_step(
                state,
                StepType.THINKING,
                f"STRATEGY: {reasoning}",
            )
        if key_issues:
            issues_str = ", ".join(key_issues[:3])
            self._emit_step(
                state,
                StepType.THINKING,
                f"KEY ISSUES: {issues_str}",
            )
        if challenges:
            self._emit_step(
                state,
                StepType.THINKING,
                f"CHALLENGES: {challenges}",
            )

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

    async def _execute_external_searches(self, state: InvestigationState):
        """Execute external searches (case law, web) ONLY when the LLM plan requests them.

        This mimics how a real lawyer works:
        1. First, read the repository documents
        2. Based on what's found, identify if external research is needed
        3. Only then search case law or web for specific legal questions

        External searches are TOOLS, not mandatory steps:
        - Case law: Use when legal precedent questions arise from the documents
        - Web search: Use when regulations/standards need verification (e.g., state law, industry norms)

        The LLM plan decides what external searches are needed based on:
        - The nature of the query
        - What's likely in the repository
        - What gaps might need external verification
        """
        plan = state.findings.get("initial_plan", {})
        case_law_queries = plan.get("case_law_searches", [])
        web_queries = plan.get("web_searches", [])

        # Only proceed if the LLM identified external searches as needed
        if not case_law_queries and not web_queries:
            self._emit_step(
                state,
                StepType.THINKING,
                "No external research needed - focusing on repository documents",
            )
            return

        self._external_research = {"case_law": [], "web": [], "analysis": {}}

        # Execute case law searches
        if case_law_queries and self.external_search:
            self._emit_step(
                state,
                StepType.THINKING,
                f"Searching case law for: {', '.join(case_law_queries[:2])}...",
            )

            for query in case_law_queries[:2]:  # Limit to 2 case law searches
                try:
                    cases = await self.external_search.search_case_law(
                        query,
                        max_results=self.config.max_case_law_results
                    )
                    if cases:
                        self._external_research["case_law"].extend(cases)
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"Found {len(cases)} relevant cases for '{query[:30]}...'",
                        )
                except Exception as e:
                    logger.warning(f"Case law search failed for '{query}': {e}")

            # Analyze case law results (results are dicts from to_dict())
            if self._external_research["case_law"]:
                case_law_text = "\n\n".join([
                    f"**{c.get('case_name', 'Unknown')}** ({c.get('citation') or 'No citation'})\n"
                    f"Court: {c.get('court', 'Unknown')}\nDate: {c.get('date_filed', 'Unknown')}\n"
                    f"Snippet: {(c.get('snippet') or c.get('opinion_text', ''))[:500] if c.get('snippet') or c.get('opinion_text') else 'No summary'}"
                    for c in self._external_research["case_law"][:5]
                ])

                analysis = await decisions.analyze_case_law_results(
                    query=state.query,
                    case_law_results=case_law_text,
                    client=self.client,
                )
                self._external_research["analysis"]["case_law"] = analysis

        # Execute web searches
        if web_queries and self.external_search:
            self._emit_step(
                state,
                StepType.THINKING,
                f"Searching web for regulations/standards: {', '.join(web_queries[:2])}...",
            )

            for query in web_queries[:2]:  # Limit to 2 web searches
                try:
                    result_data = await self.external_search.search_web(
                        query,
                        max_results=self.config.max_web_results
                    )
                    # search_web returns dict with 'results' key
                    web_results = result_data.get("results", [])
                    if web_results:
                        self._external_research["web"].extend(web_results)
                        self._emit_step(
                            state,
                            StepType.FINDING,
                            f"Found {len(web_results)} web results for '{query[:30]}...'",
                        )
                    # Also capture the AI-generated answer if available
                    if result_data.get("answer"):
                        self._external_research["web_answer"] = result_data["answer"]
                except Exception as e:
                    logger.warning(f"Web search failed for '{query}': {e}")

            # Analyze web results (results are dicts)
            if self._external_research["web"]:
                web_text = "\n\n".join([
                    f"**{r.get('title', 'Untitled')}**\nURL: {r.get('url', '')}\n"
                    f"Content: {r.get('content', 'No content')[:500]}"
                    for r in self._external_research["web"][:5]
                ])

                analysis = await decisions.analyze_web_results(
                    query=state.query,
                    web_results=web_text,
                    client=self.client,
                )
                self._external_research["analysis"]["web"] = analysis

        # Store in state findings for reference
        state.findings["external_research"] = self._external_research

        # Add citations for external sources (so they appear in Citations tab)
        if self._external_research.get("case_law"):
            for case in self._external_research["case_law"][:5]:
                citation = state.add_citation(
                    document=f"[Case Law] {case.get('case_name', 'Unknown Case')}",
                    page=None,
                    text=case.get('snippet', '') or case.get('opinion_text', '') or '',
                    context=f"Citation: {case.get('citation', 'N/A')} | Court: {case.get('court', 'N/A')}",
                    relevance="External case law research",
                )
                if citation and self.on_citation:
                    self.on_citation(citation)

        if self._external_research.get("web"):
            for result in self._external_research["web"][:5]:
                citation = state.add_citation(
                    document=f"[Web] {result.get('title', 'Unknown Source')}",
                    page=None,
                    text=result.get('content', '') or '',
                    context=f"URL: {result.get('url', 'N/A')}",
                    relevance="External regulatory research",
                )
                if citation and self.on_citation:
                    self.on_citation(citation)

    async def _investigate_loop(
        self,
        state: InvestigationState,
        repo: MatterRepository,
        cache: InvestigationCache,
    ):
        """Phase 2: Iterative investigation with continuous recalibration.

        Mimics how a lawyer works:
        1. Follow leads, read documents, extract facts
        2. Continuously assess: Do we have enough? Should we change approach?
        3. Dynamically decide if external research is needed based on what we find
        4. Stop when we have sufficient evidence
        """
        iteration = 0
        external_search_triggered = False  # Track if we've done external search

        while iteration < self.config.max_iterations:
            state.recursion_depth = iteration + 1

            pending_leads = state.get_pending_leads()
            if not pending_leads:
                self._emit_step(state, StepType.THINKING, "No more leads to investigate")
                break

            # Take leads to process
            leads_to_process = pending_leads[:self.config.max_leads_per_level]

            self._emit_step(
                state,
                StepType.THINKING,
                f"Processing {len(leads_to_process)} leads (iteration {iteration + 1})",
            )

            # Process leads in parallel
            tasks = [self._investigate_lead(state, repo, lead, cache) for lead in leads_to_process]
            await asyncio.gather(*tasks, return_exceptions=True)

            iteration += 1
            facts_count = len(state.findings.get("accumulated_facts", []))
            findings_summary = self._format_findings(state)

            # === RECALIBRATION POINT ===
            # After each iteration, reassess our approach like a lawyer would

            # 1. Check if we have enough to answer
            if facts_count >= self.config.early_exit_facts:
                is_sufficient = await decisions.is_sufficient(
                    query=state.query,
                    findings=findings_summary,
                    client=self.client,
                )

                if is_sufficient:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Sufficient evidence gathered ({facts_count} facts)",
                    )
                    break

            # 2. Continuous replanning - check if we should change approach
            # For complex queries, replan frequently based on what we're finding
            prev_facts_count = state.findings.get("_prev_facts_count", 0)
            facts_delta = facts_count - prev_facts_count
            state.findings["_prev_facts_count"] = facts_count

            # Replan if:
            # - We're not finding new facts (stalled)
            # - Every iteration for complex queries
            # - Every 2 iterations for simpler queries
            is_complex = state.query_classification.get('complexity', 3) >= 3
            is_stalled = facts_delta < 2 and iteration > 1
            should_check_replan = is_stalled or (is_complex and iteration > 0) or (iteration > 1 and iteration % 2 == 0)

            if should_check_replan and pending_leads:  # Only replan if we still have work to do
                plan_summary = state.findings.get("initial_plan", {}).get("reasoning", "")
                should_change = await decisions.should_replan(
                    query=state.query,
                    plan=plan_summary,
                    results=findings_summary,
                    client=self.client,
                )

                if should_change:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Recalibrating... (stalled={is_stalled}, complex={is_complex})",
                    )
                    new_plan = await decisions.replan(
                        query=state.query,
                        previous_approach=plan_summary,
                        findings=findings_summary,
                        client=self.client,
                    )

                    # Add new leads from replanning
                    new_leads_added = 0
                    for term in new_plan.get("search_terms", [])[:3]:
                        if not cache.is_similar_search(term):
                            state.add_lead(f"Search for: {term}", source="replan")
                            new_leads_added += 1
                    for filepath in new_plan.get("files_to_check", [])[:2]:
                        if not cache.has_extracted(filepath):
                            state.add_lead(f"Read document: {filepath}", source="replan")
                            new_leads_added += 1

                    if new_leads_added > 0:
                        self._emit_step(
                            state,
                            StepType.THINKING,
                            f"Added {new_leads_added} new leads from replanning",
                        )

                    # Check if replanning suggests external research
                    if new_plan.get("needs_external_research") and not external_search_triggered:
                        if "initial_plan" not in state.findings:
                            state.findings["initial_plan"] = {}
                        state.findings["initial_plan"]["case_law_searches"] = new_plan.get("case_law_searches", [])
                        state.findings["initial_plan"]["web_searches"] = new_plan.get("web_searches", [])

            # 3. Dynamic external search - trigger if we discover we need it
            # (e.g., found references to case law, regulations, state-specific rules)
            if not external_search_triggered and self.config.enable_external_search and self.external_search:
                # Check if findings suggest we need external research
                needs_external = await self._check_if_external_needed(state)
                if needs_external:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        "Findings suggest external legal research would help...",
                    )
                    await self._execute_external_searches(state)
                    external_search_triggered = True

            # Save checkpoint periodically
            if self.config.checkpoint_dir and iteration % self.config.checkpoint_interval == 0:
                self._save_checkpoint(state, iteration)

    async def _check_if_external_needed(self, state: InvestigationState) -> bool:
        """Check if accumulated triggers suggest we need external legal research.

        Uses trigger-based approach:
        1. Check if meaningful triggers have been accumulated from documents
        2. If yes, use LITE to generate specific queries using those triggers
        3. If no triggers, skip external search entirely (zero LLM cost)

        Returns True if queries were generated, False otherwise.
        """
        # Must have read some documents first
        if state.documents_read < 2:
            return False

        # Check if we have meaningful triggers from document analysis
        if not state.has_external_triggers(min_triggers=2):
            # No triggers accumulated - skip external search entirely
            return False

        facts = state.findings.get("accumulated_facts", [])
        entities = [e.name for e in state.entities.values()][:10] if state.entities else []
        triggers = state.get_trigger_summary()

        self._emit_step(
            state,
            StepType.THINKING,
            f"Research triggers found: {triggers[:100]}...",
        )

        # Generate specific queries using accumulated triggers
        result = await decisions.generate_external_queries(
            query=state.query,
            facts=facts,
            entities=entities,
            client=self.client,
            triggers=triggers,
        )

        case_law_queries = result.get("case_law_queries", [])
        web_queries = result.get("web_queries", [])

        # If any queries were generated, store them for execution
        if case_law_queries or web_queries:
            if "initial_plan" not in state.findings:
                state.findings["initial_plan"] = {}
            state.findings["initial_plan"]["case_law_searches"] = case_law_queries
            state.findings["initial_plan"]["web_searches"] = web_queries

            self._emit_step(
                state,
                StepType.THINKING,
                f"Generated {len(case_law_queries)} case law + {len(web_queries)} web queries from triggers",
            )
            return True

        return False

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

            # OPTIMIZATION: Skip LLM call if <= 15 results (long-context LLMs handle this fine)
            if len(results.hits) <= 15:
                relevant_hits = results.hits
                logger.info(f"Skipping pick_relevant_hits: only {len(results.hits)} results")
            else:
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
                text=hit.match_text,
                context=hit.context,
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
            if facts:
                self._emit_step(
                    state,
                    StepType.FINDING,
                    f"Found {len(facts)} facts in {doc.filename}",
                )
            state.add_facts(facts)

            # Accumulate external research triggers
            triggers = extraction.get("external_triggers", {})
            if triggers:
                added = state.add_triggers(triggers)
                if added > 0:
                    self._emit_step(
                        state,
                        StepType.THINKING,
                        f"Noted {added} research trigger(s) from {doc.filename}",
                    )

            # Emit insights from the extraction (show LLM's thinking)
            insights = extraction.get("insights", "")
            gaps = extraction.get("gaps", "")
            next_steps = extraction.get("next_steps", "")

            if insights or gaps:
                insight_msg = f"From {doc.filename}:\n"
                if insights:
                    insight_msg += f"  LEARNED: {insights}\n"
                if gaps:
                    insight_msg += f"  GAPS: {gaps}\n"
                if next_steps:
                    insight_msg += f"  NEXT: {next_steps}"
                self._emit_step(state, StepType.THINKING, insight_msg.strip())

            # Add citations from quotes (limit to 2)
            for quote in extraction.get("quotes", [])[:2]:
                if isinstance(quote, dict) and "text" in quote:
                    citation = state.add_citation(
                        document=doc.path,
                        page=quote.get("page"),
                        text=quote["text"],
                        context="",
                        relevance=quote.get("relevance", "Direct quote"),
                    )
                    if citation and self.on_citation:
                        self.on_citation(citation)

            # Skip adding reference leads - reduces iteration depth

        except Exception as e:
            self._emit_step(state, StepType.ERROR, f"Failed to read {file_path}: {e}")

    async def _synthesize(self, state: InvestigationState, is_simple: bool = False):
        """Phase 3: Final synthesis using ALL THREE sources.

        Sources:
        1. Local document search (facts and citations from repo)
        2. Case law search (CourtListener)
        3. Web search (Tavily - regulations/standards)
        """
        self._emit_step(state, StepType.SYNTHESIS, "Synthesizing from all sources...")

        # SOURCE 1: Local document evidence
        facts = state.findings.get("accumulated_facts", [])
        evidence = "\n".join(f"- {fact}" for fact in facts[:20])
        citations = state.get_citations_formatted()

        # SOURCES 2 & 3: External research (case law + web)
        external_formatted = self._format_external_research()
        case_law_text = external_formatted.get("case_law", "No case law found")
        web_text = external_formatted.get("web", "No web results found")

        # Track sources used
        state.findings["sources_used"] = ["local_documents", "case_law", "web_search"]

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
            # Full synthesis with all three sources clearly separated
            response = await decisions.synthesize(
                query=state.query,
                evidence=evidence or "No specific findings accumulated",
                citations=citations or "No citations collected",
                external_research=f"=== CASE LAW (CourtListener) ===\n{case_law_text}\n\n=== REGULATIONS/STANDARDS (Web) ===\n{web_text}",
                client=self.client,
            )

        state.findings["final_output"] = response
        self._emit_step(state, StepType.SYNTHESIS, "Analysis complete (3 sources: docs, case law, web)")

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
