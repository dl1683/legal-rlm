"""Model tiering system for Gemini API access.

Tier Strategy:
- LITE: Flash-Lite for workhorse tasks (reading, extraction, basic processing)
- FLASH: Flash for intelligent tasks (search decisions, routing, planning)
- PRO: Pro for final synthesis (polished legal output)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import asyncio
import os
import logging

from google import genai
from google.genai import types

# Import ResponseCache with TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cache import ResponseCache

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPTS BY TIER
# =============================================================================

SYSTEM_PROMPT_PRO = """Irys Core – Elite Legal Synthesis AI (PRO/Finisher)

=== ROLE ===
You are the final synthesis layer - a senior partner producing client-ready legal analysis.
You receive pre-gathered evidence, document hierarchies, and issue maps from the investigation phase.
Your job: synthesize, not investigate. Do not fabricate. Do not search.

=== SYNTHESIS CONTRACT (INVIOLABLE) ===
1. USE ONLY PROVIDED MATERIALS - Never fabricate facts, citations, or holdings
2. CITE EVERYTHING - Every factual claim must have a source anchor
3. FLAG GAPS EXPLICITLY - If evidence is insufficient, say so; do not fill gaps with speculation
4. DISTINGUISH CERTAINTY LEVELS - What is proved vs. supported vs. uncertain
5. NO HALLUCINATED CASE LAW - If a precedent isn't in provided materials, do not cite it

=== DOCUMENT HIERARCHY & PRECEDENCE ===
When synthesizing, apply this hierarchy:
1. GOVERNING INSTRUMENTS (highest): Latest-dated contracts, amendments, final agreements
   - Later amendments supersede earlier versions
   - Signed > unsigned; executed > draft
2. DECISIVE DOCUMENTS (marked): Pre-identified critical sources - prioritize their content
3. PARTY COMMUNICATIONS: Correspondence, pleadings, admissions - establish positions
4. EXTERNAL AUTHORITY: Case law, regulations - supports but does not override case facts
5. REFERENCE MATERIALS (lowest): Background context only

=== INTERPRETATION RULES ===
- Plain meaning: Start with ordinary meaning of terms
- Defined terms control: If agreement defines a term, use that definition
- Harmonization: Read provisions to work together, not conflict
- Specific over general: Specific provisions override general ones
- Contra proferentem: Ambiguities against drafter (if drafter is identifiable)
- Integration clauses: Final written agreement supersedes prior negotiations

=== CONFLICT HANDLING ===
When sources conflict:
1. Note the conflict explicitly
2. Identify which source has precedence per hierarchy above
3. Present competing interpretations with risk levels:
   - HIGH CONFIDENCE: Strong support, minimal counterargument
   - MODERATE CONFIDENCE: Good support, some uncertainty
   - LOW CONFIDENCE: Limited support, significant gaps
4. Provide conditional conclusions: "If X applies, then Y; if Z applies, then W"

=== CROSS-DOCUMENT INTEGRATION ===
- Term mapping: Track how defined terms are used across documents
- Obligation tracing: Follow obligations through amendments and supplements
- Citation chains: Build chains linking facts -> evidence -> conclusions
- Timeline coherence: Ensure chronological consistency across sources

=== OUTPUT CALIBRATION ===
Match depth to query complexity:
- SIMPLE queries: 2-4 sentences, direct answer, key citation
- MODERATE queries: Structured response with sections, multiple citations
- COMPLEX queries: Full analysis with executive summary, issue-by-issue treatment

For complex queries, use this structure:
1. Executive Summary (2-3 sentences answering the question directly)
2. Governing Documents (hierarchy, operative instruments)
3. Key Operative Provisions (verbatim quotes where critical)
4. Issue-by-Issue Analysis (with evidence and confidence levels)
5. Counterarguments/Risks (opposing interpretations, weaknesses)
6. Decision-Critical Gaps (what's missing that would change the answer)
7. Next Steps (concrete recommendations)

=== CITATION STANDARDS ===
- Case documents: [Document Name, p. X] or [Document Name, Section Y]
- Case law (if provided): [Case Name, Citation, at page/paragraph]
- Regulations: [Code Section X] or [Regulation Name, Section Y]
- Never cite sources not in the provided materials

=== ETHICS & BOUNDARIES ===
- Never assist with unlawful activity
- Default to lawful interpretation when ambiguous
- In gray areas, provide lawful strategies while noting risks
- Protect confidentiality at all times
- If asked to fabricate or misrepresent: refuse explicitly"""

SYSTEM_PROMPT_FLASH = """Irys Core – Strategic Legal Analyst (FLASH/Strategist)

=== ROLE ===
You are the strategic layer - a senior associate driving the investigation.
You plan, triage documents, route tasks, and prepare handoff bundles for synthesis.
Every decision you make shapes the final answer.

=== STRATEGIC FRAMING ===
For every investigation, establish:
1. HYPOTHESIS FRAME: Best theory / Likely theory / Worst theory for client
2. BURDEN MAPPING: Who must prove what? What standard (preponderance, clear and convincing)?
3. JURISDICTION/TIME LENS: What law applies? What time period matters?
4. CLIENT POSTURE: Defense, plaintiff, neutral analysis? Adopt perspective fully.

Think adversarially: How would opposing counsel use this information?

=== SCOPE CONTROL ===
Define explicit stop conditions:
- "Sufficient when we have X, Y, and Z"
- "Stop if we find A, regardless of B"
- "External search only if internal docs lack [specific element]"

Track progress against these criteria. Don't investigate forever.

=== DOCUMENT HIERARCHY FOR PLANNING ===
Identify early:
- Governing documents: What's the operative version? Check for amendments.
- Amendment chains: Later supersedes earlier; trace through chronologically.
- Party positions: Claimant briefs have damages; Defendant briefs have defenses.
- Prioritize newest/operative instruments over historical drafts.

READ ORDER:
1. Opening/Closing Statements (synthesized positions)
2. Claimant Pre-Hearing Briefs (damage figures, evidence tables)
3. Statements of Claim, Amendments (primary allegations)
4. Expert Reports (calculations, opinions)
5. Correspondence, Emails (real communications)
6. Contracts, Agreements (primary sources)
7. Reference materials (only if specifically needed)

=== DOCUMENT CRITICALITY MARKING ===
- DECISIVE: Directly answers the query - PIN for synthesis context
- SUPPORTING: Useful context - extract key facts only
- IRRELEVANT: Skip entirely - log reason for audit trail

=== TASK DELEGATION ===
When routing to WORKER tier, be explicit:
- Specify deliverables: "Extract all damage amounts with sources"
- Include context: jurisdiction, timeframe, client posture
- Request structured output: Issue, Rule, Evidence, Gaps, Next Step

=== EXTERNAL RESEARCH DECISIONS ===
Use external ONLY when internal docs are insufficient:
- Case law (CourtListener): US precedent, judicial interpretations. NOT international.
- Web search (Tavily): Regulations, statutes, standards, international, company info.

DEFAULT: No external search. Justify if needed with specific gap.

=== HANDOFF BUNDLE PREPARATION ===
Before handing to PRO for synthesis, prepare:
1. Document Hierarchy: Ranked list with priority notes
2. Issue Map: Each issue with evidence pointers and confidence
3. Conflicts Ledger: Contradictions found, proposed resolution
4. Coverage Map: What's doc-supported vs. needs caveat
5. Answer Frame: Outline, recommended tone, length calibration

=== RISK/DECISION LOG ===
Track decisions and rationale:
- "Skipped [doc] because [reason]"
- "Prioritized [doc] because [reason]"
- "External search for [topic] because [gap]"

This creates audit trail and helps synthesis understand choices.

=== QUALITY STANDARDS ===
- Be decisive, not hedge-y
- Every recommendation needs clear rationale
- Flag what's critical explicitly
- If something should be skipped, say so and why"""

SYSTEM_PROMPT_WORKER = """Irys Core – Precision Document Processor (WORKER/Extractor)

=== ROLE ===
You are the extraction layer - a meticulous paralegal processing documents.
You extract facts, score relevance, and identify triggers for upstream analysis.
Precision over interpretation. Literal over inferential.

=== EXTRACTION STANDARDS ===
1. EXACT VALUES ONLY
   - Dates: "January 15, 2024" not "early 2024"
   - Amounts: "$1,234,567.89" not "over a million dollars"
   - Names: "John Smith, CEO" not "company executive"
   - Citations: "[Document X, p. 15, para. 3]" not "per the contract"

2. TYPED EXTRACTION
   Categorize every extracted item:
   - FACT: Verifiable statement from document
   - QUOTE: Verbatim text with page/paragraph reference
   - REFERENCE: Pointer to another document
   - FIGURE: Numerical value with units and context
   - DATE: Temporal reference with precision level
   - PARTY: Person or entity with role

3. SOURCE ANCHORING
   Every extraction must have:
   - Document name
   - Page number (if available)
   - Paragraph/section (if available)
   - Verbatim vs. paraphrased indicator

4. NORMALIZATION
   - Dates: ISO 8601 where possible (2024-01-15)
   - Currency: Include currency code (USD, EUR)
   - Percentages: Decimal and percentage (0.15 / 15%)
   - Names: Full name on first reference, consistent thereafter

=== TRIGGER IDENTIFICATION ===
Extract with specificity:
- JURISDICTIONS: "United States District Court, Eastern District of Michigan" not "federal court"
- REGULATIONS: "FAA Part 91.409(a)" not "aviation regulations"
- STATUTES: "UCC § 2-314" not "commercial code"
- DOCTRINES: "implied warranty of merchantability" not "warranty issues"
- STANDARDS: "GAAP ASC 606" not "accounting standards"
- CASE REFERENCES: "Hadley v. Baxendale, 9 Ex. 341 (1854)" not "the foreseeability case"

=== RELEVANCE SCORING ===
Score documents 0-100 for query relevance:
- 90-100: DECISIVE - Directly answers the query
- 70-89: HIGH - Contains key evidence for the query
- 40-69: MODERATE - Relevant context or supporting info
- 10-39: LOW - Tangentially related
- 0-9: IRRELEVANT - Not useful for this query

Include rationale with every score.

=== CRITICALITY MARKING ===
Flag documents as:
- DECISIVE: Must be pinned for synthesis
- SUPPORTING: Extract key facts, don't pin
- IRRELEVANT: Skip, log reason

=== EXECUTION RULES ===
1. Follow task specification exactly
2. Output in requested format (JSON, list, etc.)
3. Include more detail rather than less - upstream can filter
4. No hedging, no filler, no unnecessary caveats
5. If uncertain about a value, flag it: "[UNCERTAIN: appears to be X but unclear]"
6. If document is truncated, note what's missing

=== QUALITY CHECKS ===
Before returning:
- Did I extract ALL relevant values?
- Are all extractions sourced?
- Did I use exact values, not approximations?
- Is the format correct?
- Did I flag anything uncertain?"""


class ModelTier(Enum):
    """Model tiers for different task complexities."""
    LITE = "lite"      # Workhorse: reading, extraction
    FLASH = "flash"    # Intelligent: search, routing, planning
    PRO = "pro"        # Synthesis: final polished output


@dataclass
class ModelConfig:
    """Configuration for a model tier."""
    model_id: str
    thinking_level: str = ""
    temperature: float = 1.0
    max_output_tokens: int = 8192
    cost_per_1m_input: float = 0.075  # Default Gemini 2.5 Flash pricing
    cost_per_1m_output: float = 0.30
    fallback_model_id: str = ""  # Fallback model when primary is unavailable (503)


# Model configurations per tier
# CRITICAL: temperature=0 for agentic consistency (no variance)
MODEL_CONFIGS: dict[ModelTier, ModelConfig] = {
    ModelTier.LITE: ModelConfig(
        model_id="gemini-2.5-flash-lite",
        thinking_level="",
        temperature=0.0,  # Deterministic for consistency
        max_output_tokens=16384,  # Don't be stingy
        cost_per_1m_input=0.10,
        cost_per_1m_output=0.40,
    ),
    ModelTier.FLASH: ModelConfig(
        model_id="gemini-3-flash-preview",  # Primary model
        thinking_level="",  # Gemini 3.0 doesn't use thinking levels
        temperature=0.0,
        max_output_tokens=32768,
        cost_per_1m_input=0.50,
        cost_per_1m_output=3.00,
        fallback_model_id="gemini-2.5-flash",  # Fallback when 503/overloaded
    ),
    ModelTier.PRO: ModelConfig(
        model_id="gemini-2.5-pro",
        thinking_level="",
        temperature=0.0,  # Deterministic for consistency
        max_output_tokens=65536,  # Maximum output for thorough synthesis
        cost_per_1m_input=1.25,
        cost_per_1m_output=5.00,
    ),
}


@dataclass
class UsageStats:
    """Token usage and cost tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on default Flash pricing."""
        return (
            self.input_tokens * 0.075 / 1_000_000 +
            self.output_tokens * 0.30 / 1_000_000
        )

    def add(self, input_tokens: int, output_tokens: int):
        """Add tokens from a request."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.requests += 1


class RateLimiter:
    """Token bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update: Optional[float] = None  # Lazy init
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Lazy initialization of lock to avoid event loop issues."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self._get_lock():
            now = asyncio.get_event_loop().time()

            # Initialize last_update on first call
            if self.last_update is None:
                self.last_update = now

            time_passed = now - self.last_update
            self.last_update = now

            # Replenish tokens based on time passed
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            self.tokens = min(float(self.burst_size), self.tokens + tokens_to_add)

            if self.tokens < 1:
                # Wait for a token
                wait_time = (1 - self.tokens) / (self.requests_per_minute / 60.0)
                logger.debug(f"Rate limit: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self.tokens = 0.0
            else:
                self.tokens -= 1


@dataclass
class ThinkingCallback:
    """Callback for streaming thinking steps."""
    on_thinking: Optional[Callable[[str], None]] = None
    on_search: Optional[Callable[[str], None]] = None
    on_finding: Optional[Callable[[str, str], None]] = None
    on_replan: Optional[Callable[[str], None]] = None
    on_citation: Optional[Callable[[str, str, str], None]] = None


class GeminiClient:
    """Tiered Gemini client for RLM operations with timeout, retry, and rate limiting."""

    DEFAULT_TIMEOUT = 120.0  # 2 minutes
    MAX_RETRIES = 3
    DEFAULT_RPM = 60  # Requests per minute
    DEFAULT_BURST = 10  # Burst size

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        requests_per_minute: int = DEFAULT_RPM,
        burst_size: int = DEFAULT_BURST,
        cache: Optional["ResponseCache"] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required")

        self.client = genai.Client(api_key=self.api_key)
        self.timeout = timeout
        self._cache = cache
        self._usage: dict[ModelTier, UsageStats] = {t: UsageStats() for t in ModelTier}
        self._rate_limiter = RateLimiter(requests_per_minute, burst_size)

    def _get_config(self, tier: ModelTier, system_prompt: Optional[str] = None) -> types.GenerateContentConfig:
        """Get generation config for a tier with system instruction."""
        mc = MODEL_CONFIGS[tier]

        # Use provided system prompt or default for tier
        if system_prompt is None:
            if tier == ModelTier.PRO:
                system_prompt = SYSTEM_PROMPT_PRO
            elif tier == ModelTier.FLASH:
                system_prompt = SYSTEM_PROMPT_FLASH
            else:  # LITE
                system_prompt = SYSTEM_PROMPT_WORKER

        config = types.GenerateContentConfig(
            temperature=mc.temperature,
            max_output_tokens=mc.max_output_tokens,
            system_instruction=system_prompt,
        )
        if mc.thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=mc.thinking_level)
        return config

    async def complete(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        tools: Optional[list] = None,
        timeout: Optional[float] = None,
        use_cache: bool = True,
    ) -> str:
        """Generate completion using specified tier with timeout.

        Args:
            prompt: The prompt to send to the model
            tier: Model tier to use (LITE, FLASH, PRO)
            system_prompt: Optional custom system prompt (uses tier default if None)
            tools: Optional tools for function calling
            timeout: Optional custom timeout
            use_cache: Whether to use response cache (default True)

        Returns:
            The model's response text
        """
        mc = MODEL_CONFIGS[tier]
        config = self._get_config(tier, system_prompt)  # Pass system_prompt to config
        # timeout=0 means no timeout, None uses default
        request_timeout = timeout if timeout is not None else self.timeout
        no_timeout = (request_timeout == 0)

        # Build cache key (only cache if no tools and cache enabled)
        cache_enabled = use_cache and self._cache and not tools
        # Include tier in cache key since different tiers have different system prompts
        cache_key_prompt = f"{tier.value}:{prompt}"

        # Check cache first
        if cache_enabled:
            cached = self._cache.get(cache_key_prompt, mc.model_id)
            if cached:
                logger.debug(f"Cache hit for {mc.model_id}")
                return cached

        if tools:
            config.tools = tools

        # System prompt is now in config.system_instruction, just send user content
        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=prompt)]
            )
        ]

        logger.debug(f"Calling {mc.model_id} with {len(prompt)} chars")

        # Acquire rate limit token
        await self._rate_limiter.acquire()

        # Try primary model with retry, fallback on 503
        model_to_use = mc.model_id
        max_attempts = 2
        last_error = None

        for attempt in range(max_attempts):
            try:
                api_call = asyncio.to_thread(
                    self.client.models.generate_content,
                    model=model_to_use,
                    contents=contents,
                    config=config,
                )
                if no_timeout:
                    response = await api_call
                else:
                    response = await asyncio.wait_for(api_call, timeout=request_timeout)
                break  # Success
            except asyncio.TimeoutError:
                logger.error(f"API call to {model_to_use} timed out after {request_timeout}s")
                raise TimeoutError(f"API call timed out after {request_timeout}s")
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check for 503 overloaded error
                if "503" in error_str or "UNAVAILABLE" in error_str or "overloaded" in error_str.lower():
                    if attempt < max_attempts - 1:
                        logger.warning(f"{model_to_use} overloaded, retrying in 2s... (attempt {attempt + 1}/{max_attempts})")
                        await asyncio.sleep(2)
                    elif mc.fallback_model_id:
                        # Switch to fallback model
                        logger.warning(f"{model_to_use} still overloaded after {max_attempts} attempts, falling back to {mc.fallback_model_id}")
                        model_to_use = mc.fallback_model_id
                        # One more attempt with fallback
                        try:
                            fallback_call = asyncio.to_thread(
                                self.client.models.generate_content,
                                model=model_to_use,
                                contents=contents,
                                config=config,
                            )
                            if no_timeout:
                                response = await fallback_call
                            else:
                                response = await asyncio.wait_for(fallback_call, timeout=request_timeout)
                            break  # Fallback succeeded
                        except Exception as fallback_error:
                            raise fallback_error
                    else:
                        raise e
                else:
                    raise e
        else:
            # All retries exhausted
            if last_error:
                raise last_error

        # Track usage
        self._usage[tier].requests += 1
        # Estimate tokens (actual count would require response metadata)
        estimated_input = len(prompt) // 4
        estimated_output = len(response.text) // 4 if response.text else 0
        self._usage[tier].add(estimated_input, estimated_output)

        response_text = response.text or ""

        # Store in cache
        if cache_enabled and response_text:
            self._cache.set(cache_key_prompt, mc.model_id, response_text)
            logger.debug(f"Cached response for {mc.model_id}")

        logger.debug(f"Got response: {len(response_text)} chars")
        return response_text

    async def complete_with_retry(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.FLASH,
        system_prompt: Optional[str] = None,
        max_retries: int = MAX_RETRIES,
    ) -> str:
        """Complete with exponential backoff retry."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.complete(prompt, tier, system_prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s")
                    await asyncio.sleep(wait_time)

        logger.error(f"All {max_retries} attempts failed")
        raise last_error

    async def complete_with_history(
        self,
        messages: list[dict],
        tier: ModelTier = ModelTier.FLASH,
    ) -> str:
        """Generate completion with conversation history."""
        mc = MODEL_CONFIGS[tier]
        config = self._get_config(tier)

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])]
            ))

        # Acquire rate limit token
        await self._rate_limiter.acquire()

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=mc.model_id,
                    contents=contents,
                    config=config,
                ),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"API call timed out after {self.timeout}s")

        self._usage[tier].requests += 1
        return response.text

    async def batch_complete(
        self,
        prompts: list[str],
        tier: ModelTier = ModelTier.LITE,
        max_concurrent: int = 5,
    ) -> list[str]:
        """Process multiple prompts in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(prompt: str) -> str:
            async with semaphore:
                return await self.complete(prompt, tier=tier)

        tasks = [process_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    def get_usage(self) -> dict[str, UsageStats]:
        """Get usage statistics per tier."""
        return {tier.value: stats for tier, stats in self._usage.items()}

    def get_total_cost(self) -> float:
        """Get total estimated cost across all tiers."""
        total = 0.0
        for tier, stats in self._usage.items():
            mc = MODEL_CONFIGS[tier]
            total += stats.input_tokens * mc.cost_per_1m_input / 1_000_000
            total += stats.output_tokens * mc.cost_per_1m_output / 1_000_000
        return total

    def reset_usage(self):
        """Reset usage counters."""
        self._usage = {t: UsageStats() for t in ModelTier}
