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


P_CLASSIFY_QUERY = """Classify this legal query for synthesis complexity.

Query: {query}

You will have ALL relevant document content, facts, and citations when synthesizing.
The question is: does this query need sophisticated legal reasoning, or is it straightforward?

SIMPLE (use faster model):
- Direct factual lookups ("What is the contract date?", "Who signed the agreement?")
- Single-document answers ("What does clause 5 say?")
- Basic summaries ("List the parties involved")

COMPLEX (use advanced model):
- Multi-document synthesis ("What are the key issues in this dispute?")
- Legal analysis ("Draft a responsive pleading", "Analyze liability")
- Strategic reasoning ("What are our strongest arguments?")
- Timeline construction across sources
- Contradiction analysis
- Anything requiring legal judgment or persuasive writing

Reply with only: SIMPLE or COMPLEX"""


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

INCLUDE:
- Names of people, companies, places
- Domain-specific terms (legal, technical, industry terms)
- Specific identifiers (dates, numbers, document names)
- Key nouns that are the SUBJECT of the query

DO NOT INCLUDE:
- Instruction verbs (analyze, explain, describe, summarize, find, identify, etc.)
- Common words (the, what, how, why, when, where, which)
- Generic terms (information, document, data, work, thing)

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

=== DOCUMENTS IN REPOSITORY ({total_files} files) ===
{file_list}

=== DOCUMENT PRIORITIZATION ===
Based on the filenames above, categorize and prioritize:

READ FIRST (case-specific, high value):
- Emails, correspondence, letters (contain actual communications)
- Pleadings, complaints, answers, motions (define the dispute)
- Contracts, agreements (primary source documents)
- Witness statements, declarations, affidavits

READ LATER (supporting):
- Expert reports, analyses
- Invoices, receipts, financial records

SKIP OR DEPRIORITIZE (generic reference):
- Generic legal acts, statutes, codes (e.g., "Business Corporations Act")
- Manuals, handbooks, guidelines (unless specifically relevant)
- Template documents, forms

=== YOUR TASK ===
1. Look at the FILENAMES above and identify which documents are likely case-specific vs generic reference
2. Select 2-3 PRIORITY FILES to read first (emails, correspondence, pleadings - NOT generic acts)
3. Generate search terms only AFTER identifying priority files
4. Plan external searches (case law, web) based on the query type

=== EXTERNAL SEARCH CAPABILITIES ===
- CASE LAW (CourtListener): U.S. court opinions, precedents
- WEB SEARCH (Tavily): Regulations, statutes, standards, company info

Reply in JSON:
{{
    "reasoning": "Brief analysis of what files look case-specific vs generic, and your strategy",
    "key_issues": ["issue1", "issue2"],
    "priority_files": ["exact_filename_from_list.pdf", "another_file.pdf"],
    "skip_files": ["generic_act.pdf", "reference_manual.pdf"],
    "search_terms": ["term1", "term2"],
    "case_law_searches": [],
    "web_searches": [],
    "success_criteria": "What we need to find",
    "potential_challenges": "What might be difficult"
}}

CRITICAL:
- priority_files must be EXACT filenames from the list above
- DO NOT prioritize generic legal acts or reference documents
- Emails and correspondence almost always contain the most relevant case-specific information"""


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
4. External research triggers - things that suggest we need to look up external sources:
   - Jurisdictions mentioned (e.g., "Michigan law", "Federal court", "UK jurisdiction")
   - Regulations/statutes cited (e.g., "FAA Part 91", "UCC § 2-314", "GDPR")
   - Legal doctrines referenced (e.g., "breach of warranty", "negligent misrepresentation")
   - Industry standards mentioned (e.g., "192-month inspection", "GAAP", "ISO 9001")
   - Case law or precedents cited (e.g., specific case names)

Also provide your analysis:
- What did you learn from this document that helps answer the query?
- What gaps remain - what information is still missing?
- What other documents should we look at based on references here?

Be thorough - missing a single fact or figure could affect the legal outcome.

Reply in JSON:
{{
    "facts": ["fact1 with exact values", "fact2 with exact values"],
    "quotes": [{{"text": "exact verbatim quote", "page": 1, "relevance": "why important"}}],
    "references": ["mentioned_doc1.pdf", "mentioned_doc2.pdf"],
    "insights": "What I learned from this document and how it helps answer the query",
    "gaps": "What information is still missing or unclear",
    "next_steps": "What we should look for next based on what we found",
    "external_triggers": {{
        "jurisdictions": ["any jurisdictions mentioned"],
        "regulations_statutes": ["any regulations, statutes, or legal codes cited"],
        "legal_doctrines": ["any legal theories or doctrines referenced"],
        "industry_standards": ["any industry standards or practices mentioned"],
        "case_references": ["any case law or precedents cited"]
    }}
}}"""


P_REPLAN = """The current investigation approach needs adjustment.

Query: {query}

What we tried: {previous_approach}

What we found: {findings}

Reassess and create a new approach:
1. What's working? What's not working?
2. What should we try differently?
3. What new search terms might help?
4. Are there specific documents we should read?
5. Do the findings suggest we need EXTERNAL research?
   - Case law: if we need legal precedents or judicial interpretations
   - Web search: if we need regulations, statutes, or industry standards

Reply in JSON:
{{
    "diagnosis": "What's working and what's not",
    "new_approach": "What to try now",
    "search_terms": ["term1", "term2"],
    "files_to_check": ["file1.pdf"],
    "needs_external_research": true/false,
    "case_law_searches": ["legal issue to search"],
    "web_searches": ["regulation or standard to look up"]
}}"""


# =============================================================================
# HIGH TIER PROMPTS (PRO model - final synthesis)
# =============================================================================

P_SYNTHESIZE = """You are a senior legal analyst preparing a response to a query.

Query: {query}

=== DECISIVE DOCUMENTS (Full Content) ===
{pinned_content}

=== CASE-SPECIFIC EVIDENCE ===
{evidence}

=== EXTERNAL LEGAL RESEARCH ===
{external_research}

=== CITATIONS ===
{citations}

IMPORTANT: PRIORITIZE information from the DECISIVE DOCUMENTS section - these are the most critical sources for answering this query. They have been identified as directly relevant and their full content is provided above.

Write a clear, well-organized response that:
1. Directly answers the query, prioritizing DECISIVE document content
2. Supports legal conclusions with relevant case law and regulations when available
3. Cites specific sources for each claim (both case documents AND external sources)
4. Distinguishes between case facts and general legal principles
5. Notes any gaps or uncertainties
6. Is appropriate in length for the complexity of the question

When citing:
- Case documents: [Document Name, p. X]
- Case law: [Case Name, Citation]
- Regulations/Web: [Source Name]

Format your response with clear sections if the answer is complex."""


P_SYNTHESIZE_SIMPLE = """Answer this factual question concisely based on the evidence.

Query: {query}
{pinned_content}
Evidence:
{evidence}

Sources:
{citations}

Provide a brief, direct answer (2-4 sentences). Include the key facts and cite sources.
If DECISIVE document content is provided above, prioritize that information.
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


# =============================================================================
# EXTERNAL SEARCH PROMPTS
# =============================================================================

P_ANALYZE_CASE_LAW = """You found these case law results from CourtListener.

Query context: {query}

Case Law Results:
{case_law_results}

Extract the key legal principles and precedents:
1. What legal standards or tests do these cases establish?
2. How might they apply to the current query?
3. Are there any directly applicable holdings?

Reply in JSON:
{{
    "key_precedents": [
        {{"case": "Case Name", "citation": "citation", "holding": "relevant holding", "applicability": "how it applies"}}
    ],
    "legal_standards": ["standard 1", "standard 2"],
    "summary": "Brief summary of how this case law informs the query"
}}"""


P_ANALYZE_WEB_RESULTS = """You found these web search results about legal regulations/standards.

Query context: {query}

Web Results:
{web_results}

Extract the key regulatory information:
1. What regulations or standards are relevant?
2. What are the key requirements or thresholds?
3. How do they apply to the current situation?

Reply in JSON:
{{
    "regulations": [
        {{"name": "Regulation Name", "source": "source URL", "key_requirements": "relevant requirements"}}
    ],
    "standards": ["industry standard 1", "legal standard 2"],
    "summary": "Brief summary of the regulatory context"
}}"""


P_SHOULD_SEARCH_EXTERNAL = """Based on the investigation so far, should we search external sources?

Query: {query}

Facts found so far:
{facts_found}

Key issues:
{key_issues}

Consider:
1. Are there legal issues that would benefit from case law precedents?
2. Are there regulatory or compliance questions that need external verification?
3. Would external sources help prevent hallucination about legal standards?

Reply in JSON:
{{
    "search_case_law": true/false,
    "case_law_queries": ["query 1", "query 2"],
    "search_web": true/false,
    "web_queries": ["query 1", "query 2"],
    "reasoning": "Why or why not to search external sources"
}}"""


P_GENERATE_EXTERNAL_QUERIES = """You are analyzing case facts to determine if external legal research is needed.

Query: {query}

Facts found from documents:
{facts}

Entities identified:
{entities}

Research triggers identified from documents:
{triggers}

Based on the facts AND the research triggers, generate SPECIFIC external search queries.

=== WHEN TO USE EACH SOURCE ===

CASE LAW (CourtListener) - Use when:
- Query involves US legal precedent, judicial interpretations, or standards of proof
- US jurisdictions are mentioned (e.g., "Michigan", "Delaware", federal courts)
- Legal doctrines need authority (breach of warranty, negligence, fiduciary duty, etc.)
- Need to support legal arguments with US case citations
- LIMITATION: CourtListener is US-focused. DO NOT use for international matters.

WEB SEARCH (Tavily) - Use when:
- Query asks about regulations, statutes, or industry standards
- User provides a URL/link to look up
- Need to verify regulatory compliance requirements
- Triggers include specific regulations (FAA, SEC, OSHA, state codes)
- International matters - use web for non-US jurisdictions
- Company/entity research (background, public records, news)

RETURN EMPTY ARRAYS when:
- That source type is not relevant to this specific query
- Local documents already have sufficient information
- No meaningful triggers for that source type
- The query asks only about case-specific facts (no external authority needed)

=== HOW TO USE TRIGGERS ===
- If US jurisdictions found (e.g., "Michigan") → case_law_queries
- If international jurisdictions found → web_queries (NOT case law)
- If regulations/statutes found (e.g., "FAA Part 91") → web_queries
- If US legal doctrines found → case_law_queries
- If industry standards found → web_queries
- If specific case references found → case_law_queries (to find those cases)

=== EXAMPLES ===

Query about US case law:
  case_law_queries: ["Michigan breach of warranty aircraft maintenance"]
  web_queries: []

Query about regulations:
  case_law_queries: []
  web_queries: ["FAA Part 91 inspection requirements"]

Query about international matter (e.g., UK, EU):
  case_law_queries: []
  web_queries: ["UK aviation maintenance regulations", "EASA inspection standards"]

Query needing both (US legal + regulatory):
  case_law_queries: ["Delaware fiduciary duty directors"]
  web_queries: ["SEC disclosure requirements public companies"]

Query about case facts only:
  case_law_queries: []
  web_queries: []
  reasoning: "Query asks only about facts in documents, no external authority needed"

Reply in JSON only:
{{
    "case_law_queries": ["specific query 1", "specific query 2"],
    "web_queries": ["specific regulation/standard query"],
    "reasoning": "Brief explanation of source selection and which triggers informed searches"
}}

If no external research is needed, reply:
{{
    "case_law_queries": [],
    "web_queries": [],
    "reasoning": "Reason why external research not needed"
}}"""


P_EXTRACT_TRIGGERS = """Scan this legal document content and identify any external research triggers.

Content (excerpt):
{content}

Identify mentions of:
1. Jurisdictions - specific courts, states, countries, or legal systems mentioned
2. Regulations/Statutes - specific laws, codes, regulations, or statutory references
3. Legal doctrines - legal theories, causes of action, or legal principles
4. Industry standards - technical standards, professional practices, certifications
5. Case references - any cited cases or legal precedents

Only include SPECIFIC items actually mentioned in the text. Do NOT infer or guess.
Return empty lists for categories with no mentions.

Reply in JSON only:
{{
    "jurisdictions": ["specific jurisdictions mentioned"],
    "regulations_statutes": ["specific regulations or statutes cited"],
    "legal_doctrines": ["specific legal doctrines referenced"],
    "industry_standards": ["specific standards mentioned"],
    "case_references": ["specific cases cited"]
}}"""


# =============================================================================
# CONSOLIDATED PROMPTS (Reducing LLM calls)
# =============================================================================

P_CHECKPOINT = """Query: {query}

Evidence gathered so far:
{findings}

Current approach: {plan}

Evaluate the investigation status:

1. SUFFICIENCY: Do we have enough evidence to answer the query?
   - Is there direct evidence addressing the question?
   - Are there citations from source documents?
   - Is there enough detail for a useful answer?

2. PROGRESS: Is the current approach working?
   - Are we finding relevant information?
   - Are we stalled or making progress?

3. NEXT STEPS: If not sufficient, what should we do?
   - Different search terms?
   - Specific documents to read?
   - Change strategy entirely?

Reply in JSON:
{{
    "sufficient": true/false,
    "should_replan": true/false,
    "progress_assessment": "brief assessment of what's working/not working",
    "next_steps": ["specific action 1", "specific action 2"],
    "new_search_terms": ["term1", "term2"],
    "files_to_check": ["file1.pdf", "file2.pdf"]
}}"""


P_ANALYZE_SEARCH = """You are analyzing search results for a legal investigation.

Query: {query}

Key issues identified:
{key_issues}

Search Results:
{results}

Already read documents:
{already_read}

Perform a COMPLETE analysis in ONE pass:

1. RELEVANT HITS: Which search results are most relevant? (by number)
2. KEY FACTS: What facts are directly relevant to the query?
3. DOCUMENT PRIORITY: Rank candidate documents for deeper reading.
   - Score 0-100 for relevance
   - CRITICAL: Pleadings, correspondence, party briefs > reference materials
   - Prioritize unread documents
4. ADDITIONAL NEEDS: What else might help?

DOCUMENT CRITICALITY:
- Mark any document as "DECISIVE" if it appears to directly answer the query
- Mark as "IRRELEVANT" if clearly not useful for this specific query
- Mark as "SUPPORTING" for useful context

Reply in JSON:
{{
    "relevant_hit_numbers": [1, 3, 5],
    "facts": ["fact1 with exact values", "fact2"],
    "citations": [{{"text": "quote", "source": "filename", "page": 1}}],
    "ranked_documents": [
        {{"file": "path/file.pdf", "score": 95, "criticality": "DECISIVE", "reason": "brief reason"}},
        {{"file": "path/file2.pdf", "score": 70, "criticality": "SUPPORTING", "reason": "brief reason"}},
        {{"file": "path/file3.pdf", "score": 10, "criticality": "IRRELEVANT", "reason": "brief reason"}}
    ],
    "additional_searches": ["term1"],
    "read_deeper": ["file1.pdf"]
}}"""


P_ANALYZE_EXTERNAL = """You are analyzing external legal research results.

Query context: {query}

=== CASE LAW RESULTS ===
{case_law_results}

=== WEB/REGULATORY RESULTS ===
{web_results}

Analyze ALL external research in ONE pass:

1. CASE LAW ANALYSIS:
   - What legal standards or tests do these cases establish?
   - Are there directly applicable holdings?
   - How do they apply to our situation?

2. REGULATORY ANALYSIS:
   - What regulations or standards are relevant?
   - What are the key requirements or thresholds?
   - How do they apply to the current situation?

3. SYNTHESIS:
   - How do case law and regulations interact?
   - What's the combined legal framework?

Reply in JSON:
{{
    "key_precedents": [
        {{"case": "Case Name", "citation": "citation", "holding": "relevant holding", "applicability": "how it applies"}}
    ],
    "legal_standards": ["standard 1", "standard 2"],
    "regulations": [
        {{"name": "Regulation Name", "source": "source", "key_requirements": "requirements"}}
    ],
    "regulatory_standards": ["standard 1", "standard 2"],
    "combined_framework": "How case law and regulations together inform this situation",
    "summary": "Brief unified summary of external legal context"
}}"""
