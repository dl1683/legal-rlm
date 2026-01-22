"""Prompt templates for LLM decisions.

All prompts used by the decisions layer are defined here.
Organized by model tier: WORKER (cheap/fast), MID (balanced), HIGH (expensive/thorough).
"""

# =============================================================================
# WORKER TIER PROMPTS (LITE model - quick decisions)
# =============================================================================

P_PICK_FILES = """Given this query and list of files, which files are most likely to contain relevant information?

Query: {query}

Files in repository:
{file_list}

Pick the 3-5 most relevant files. Consider:
- Filename matches query terms
- Document type (contracts, agreements, reports are usually important)
- Likely to contain the answer

Reply with just the filenames, one per line. No explanations."""


P_PICK_HITS = """Given this query and search results, which results are most relevant?

Query: {query}

Search Results:
{hits}

Pick the 5-10 most relevant results by their number (e.g., [1], [5], [8]).
Consider which ones directly answer or relate to the query.

Reply with just the numbers, comma-separated. Example: 1, 3, 5, 8"""


P_IS_SUFFICIENT = """Query: {query}

Evidence gathered so far:
{findings}

Is this sufficient to answer the query? Consider:
- Do we have direct evidence addressing the question?
- Are there citations from source documents?
- Is there enough detail to give a useful answer?

Reply with only YES or NO."""


P_SHOULD_REPLAN = """Query: {query}

Current approach: {plan}

Results so far: {results}

Is the current approach working? Consider:
- Are we finding relevant information?
- Should we try different search terms?
- Should we look at different files?

Reply with only CONTINUE or REPLAN."""


P_EXTRACT_SEARCH_TERMS = """Query: {query}

Extract 3-5 specific search terms that would help find relevant information in documents.
Include:
- Key nouns and phrases from the query
- Legal/domain-specific terms
- Names, dates, or specific identifiers mentioned

Reply with just the search terms, one per line. No explanations."""


P_CLASSIFY_QUERY = """Classify this query:

Query: {query}

Reply with one word:
- SIMPLE: Basic factual question (who, what, when, where)
- ANALYTICAL: Requires understanding and explanation
- COMPLEX: Requires deep analysis across multiple sources"""


P_PRIORITIZE_DOCUMENTS = """You are ranking candidate documents for relevance to a legal query.

Query: {query}

Key issues identified:
{key_issues}

Candidate files (from search results):
{candidate_files}

Already read documents:
{already_read}

Score each candidate from 0-100 for likely relevance to the query and key issues.

CRITICAL PRIORITY RULES:
1. For actual damages/costs/figures: CLAIMANT documents have the real numbers (Claimant briefs, Statements of Claim)
2. DEFENDANT/RESPONDENT briefs contain arguments but often NOT the actual figures
3. When BOTH party briefs exist, READ BOTH - they contain different information

PRIORITY ORDER (score higher):
1. Opening Statements, Closing Arguments (95-100) - BEST SOURCE: synthesized key facts, final figures
2. Claimant/Plaintiff Pre-Hearing Briefs (95-100) - have actual damage figures and evidence tables
3. Statements of Claim, Amended Claims (90-95) - primary allegations with amounts
4. Expert Reports from CLAIMANT side (85-90) - damage calculations
5. Defendant/Respondent briefs (70-80) - arguments but not actual figures

Consider:
- Document type as above
- "Claimant" or "CITIOM" in filename = higher priority for damage queries
- "Gulfstream" or "Respondent" in filename = lower priority for actual figures
- Prefer files NOT already read unless clearly essential

Reply with JSON only:
{{
    "ranked_files": [
        {{"file": "path/to/file.pdf", "score": 95, "reason": "brief reason"}},
        {{"file": "path/to/file2.docx", "score": 72, "reason": "brief reason"}}
    ]
}}"""


# =============================================================================
# MID TIER PROMPTS (FLASH model - analysis and planning)
# =============================================================================

P_CREATE_PLAN = """You are investigating a legal query against a document repository.

Query: {query}

Repository structure:
{repo_structure}

Total files: {total_files}

Create a brief investigation plan:
1. What are the key issues to investigate?
2. Which files/folders should we search first?
   - For damages/costs: prioritize CLAIMANT briefs and Statements of Claim
   - Always check BOTH party briefs if they exist
3. What search terms should we use?
   - Provide individual search terms (1-2 words each) - these work best for grep-style search
   - Provide OR groups for synonyms/variants (e.g., ["cost", "price", "amount"])
   - Include document reference patterns (e.g., "GAC", "CIT", "Exhibit") - these often cite actual figures
   - AVOID long phrases - they often fail to match
4. What would constitute a sufficient answer?

Reply in JSON:
{{
    "key_issues": ["issue1", "issue2"],
    "priority_files": ["file1.pdf", "file2.docx"],
    "search_terms": ["term1", "term2", "term3"],
    "or_groups": [["term1", "synonym1"], ["term2", "variant2", "variant3"]],
    "success_criteria": "What we need to find to answer the query"
}}"""


P_ANALYZE_RESULTS = """You are analyzing search results for a legal investigation.

Query: {query}

Search Results:
{results}

Extract the key information:
1. What facts are directly relevant to the query?
2. What documents should we read more deeply?
3. What additional searches might help?

Reply in JSON:
{{
    "facts": ["fact1", "fact2"],
    "citations": [{{"text": "quote", "source": "filename", "page": 1}}],
    "read_deeper": ["file1.pdf", "file2.docx"],
    "additional_searches": ["term1", "term2"]
}}"""


P_EXTRACT_FACTS = """You are extracting facts from a document for a legal investigation.

Query: {query}

Document: {filename}
Content:
{content}

CRITICAL: Legal precision is paramount. Extract ALL relevant facts with EXACT values.

Extract:
1. ALL facts directly relevant to the query - do NOT summarize or paraphrase
   - Include EXACT dollar amounts, dates, percentages, durations
   - Include EXACT names, titles, document references
   - If a table or comparison exists, extract each row's data
2. Key quotes with page numbers - use VERBATIM text
3. References to other documents or exhibits

Be thorough - missing a single fact or figure could affect the legal outcome.

Reply in JSON:
{{
    "facts": ["fact1 with exact values", "fact2 with exact values"],
    "quotes": [{{"text": "exact verbatim quote", "page": 1, "relevance": "why important"}}],
    "references": ["mentioned_doc1.pdf", "mentioned_doc2.pdf"]
}}"""


P_REPLAN = """The current investigation approach is not working well.

Query: {query}

What we tried: {previous_approach}

What we found: {findings}

Create a new approach:
1. What went wrong with the previous approach?
2. What should we try instead?
3. What search terms should we use now?

Reply in JSON:
{{
    "diagnosis": "What went wrong",
    "new_approach": "What to try now",
    "search_terms": ["term1", "term2"],
    "files_to_check": ["file1.pdf"]
}}"""


# =============================================================================
# HIGH TIER PROMPTS (PRO model - final synthesis)
# =============================================================================

P_SYNTHESIZE = """You are a senior legal analyst preparing a response to a query.

Query: {query}

Evidence gathered:
{evidence}

Citations:
{citations}

Write a clear, well-organized response that:
1. Directly answers the query
2. Cites specific sources for each claim
3. Notes any gaps or uncertainties
4. Is appropriate in length for the complexity of the question

Format your response with clear sections if the answer is complex."""


P_SYNTHESIZE_SIMPLE = """Answer this factual question concisely based on the evidence.

Query: {query}

Evidence:
{evidence}

Sources:
{citations}

Provide a brief, direct answer (2-4 sentences). Include the key facts and cite sources.
Do not elaborate unnecessarily - just answer the question directly."""


P_RESOLVE_CONTRADICTIONS = """You are analyzing potentially contradictory evidence.

Query: {query}

Contradictory findings:
{contradictions}

Analyze:
1. Are these truly contradictory or just different aspects?
2. Which sources are more authoritative?
3. How should we reconcile or present these differences?

Reply in JSON:
{{
    "is_true_contradiction": true/false,
    "analysis": "Explanation of the contradiction",
    "resolution": "How to handle this in the final answer",
    "preferred_interpretation": "Which view is better supported"
}}"""


# =============================================================================
# TOOL CALLING PROMPTS
# =============================================================================

P_DECIDE_ACTION = """You are investigating a query. Decide the next action.

Query: {query}

Current state:
- Files searched: {files_searched}
- Documents read: {docs_read}
- Facts found: {num_facts}
- Key findings: {findings_summary}

Available actions:
- search: Search for text in files (params: query, files)
- read: Read a document fully (params: filepath)
- done: Finish investigation (params: reason)

What's the best next action? Be efficient - if we have enough info, finish.

Reply in JSON:
{{
    "action": "search" | "read" | "done",
    "params": {{}},
    "reason": "Brief explanation"
}}"""
