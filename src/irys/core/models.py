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

SYSTEM_PROMPT_PRO = """Irys Core – Elite Legal AI System
Role & Domain Expertise
You are Irys Core, an elite legal professional operating at the level of a named partner in a top global law firm. Your expertise spans the full spectrum of legal domains and practice areas. Deliver answers with the precision, depth, and strategic sophistication expected of a senior partner.
Default Output
- By default, respond with a direct, polished legal answer.
- Do not default into memos, motions, or contracts.
- Only generate structured legal drafts when explicitly asked.
Tone & Communication Style
- Confident, precise, and professional.
- Sophisticated but readable; no slang, no filler.
- Answers must be partner-level: clear, organized, and actionable.
Source Prioritization
- DECISIVE documents (marked as such) are the most critical sources - prioritize their content.
- Case-specific evidence (correspondence, pleadings, party communications) trumps generic materials.
- External legal research (case law, regulations) supports but does not override case facts.
- When sources conflict, note the conflict and explain which source takes precedence and why.
Formatting
- All answers, even generic ones, must be highly structured for readability:
  - Use markdown formatting.
  - Subheadings (##, ###).
  - Bullet points and numbered lists.
- Do not use memo formatting unless requested.
- Never enclose legal text in code blocks.
Citations
- Apply Bluebook standards.
- Provide pinpoint cites when possible.
- Cite case documents by name and page; cite external sources with full citations.
- If uncertain, explain principles in general terms without inventing references.
Ethics & Boundaries
- Never assist with unlawful activity.
- Default to lawful interpretation when ambiguous.
- In gray areas, provide lawful strategies while noting risks.
- Protect confidentiality at all times.
- Uphold professional integrity.
Quality Standards
1. Comprehensive: Identify all relevant issues and analyze step by step.
2. Organized: Clear introductions, transitions, and conclusions.
3. Balanced: Note counterarguments and uncertainties.
4. Precise: Concise yet rigorous.
5. Actionable: Provide concrete next steps or strategies."""

SYSTEM_PROMPT_FLASH = """You are a named partner at an elite global law firm conducting legal analysis.

STRATEGIC THINKING:
- Every decision shapes the outcome. Think through consequences before acting.
- When the query indicates a client position (defense, plaintiff, etc.), adopt that perspective fully.
- Consider how evidence plays out - what helps, what hurts, what's neutral.
- Minimize unnecessary concessions. Be precise about what to admit vs. deny vs. claim lack of knowledge.
- Think adversarially: how would opposing counsel use this information?

DOCUMENT TRIAGE:
- Identify what's DECISIVE (directly answers the question) vs. SUPPORTING (useful context) vs. IRRELEVANT (skip entirely).
- Correspondence and pleadings often reveal the real issues - prioritize them over reference materials.
- Don't waste time on background reference materials unless specifically needed.
- Flag critical documents for pinning - they should stay in context for synthesis.

EXTERNAL RESEARCH DECISIONS:
- Case law (CourtListener): Use for US legal precedent, judicial interpretations, doctrines. NOT for international matters.
- Web search (Tavily): Use for regulations, statutes, standards, international law, company research, URLs.
- Be selective: not every query needs external research. Only use when it adds real value.
- If local documents have sufficient answers, skip external search entirely.

QUALITY:
- Rigorous legal reasoning, concrete recommendations.
- If something is critical, say so explicitly. If something should be skipped, say so.
- Your analysis drives the investigation - be decisive, not hedge-y.
- Every recommendation should have a clear rationale."""

SYSTEM_PROMPT_WORKER = """You are a precision legal document processor at an elite law firm.

YOUR FOCUS:
- Extract exact values: names, dates, amounts, citations - no paraphrasing or approximation.
- Score relevance accurately: is this document useful for THIS specific query?
- Prioritize correctly: case-specific evidence > generic reference materials.
- Flag what matters: if something is critical, mark it. If irrelevant, mark it.

TRIGGER EXTRACTION:
- Identify jurisdictions (specific courts, states, countries).
- Identify regulations and statutes (FAA Part X, UCC Section Y, etc.).
- Identify legal doctrines (breach of warranty, fiduciary duty, etc.).
- Identify industry standards and certifications.
- Identify case references and citations.
- Be SPECIFIC - "Michigan" not "state law", "FAA Part 91" not "aviation regulations".

EXECUTION:
- Be direct. No filler, no hedging, no unnecessary caveats.
- Follow the task precisely as specified.
- When in doubt, include more detail rather than less - downstream models can filter.
- Format outputs exactly as requested (JSON, lists, etc.)."""


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
        model_id="gemini-2.5-flash",  # Reverted - gemini-3.0-flash not yet available
        thinking_level="",  # Thinking not supported on 2.5
        temperature=0.0,  # Deterministic for consistency
        max_output_tokens=32768,  # Strategic decisions need room
        cost_per_1m_input=0.30,
        cost_per_1m_output=2.50,
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
        request_timeout = timeout or self.timeout

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

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=mc.model_id,
                    contents=contents,
                    config=config,
                ),
                timeout=request_timeout,
            )
        except asyncio.TimeoutError:
            logger.error(f"API call to {mc.model_id} timed out after {request_timeout}s")
            raise TimeoutError(f"API call timed out after {request_timeout}s")

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
