"""Prompt templates for LLM decisions.

All prompts used by the decisions layer are defined here.
Organized by model tier: WORKER (cheap/fast), MID (balanced), HIGH (expensive/thorough).
"""

# =============================================================================
# WORKER TIER PROMPTS (LITE model - quick decisions)
# =============================================================================

P_PICK_FILES = """Select files most likely to answer this query.

Query: {query}

Files:
{file_list}

Select 3-5 files. Prioritize:
- Pleadings, briefs, statements (define disputes, contain positions)
- Contracts, agreements (primary source documents)
- Correspondence, emails (actual party communications)
- Expert reports (specialized analysis)
- Documents with names/terms matching the query

Deprioritize:
- Generic reference materials (statutes, manuals, guidelines)
- Template documents

Reply with filenames only, one per line."""


P_PICK_HITS = """Select the most relevant search hits for this query.

Query: {query}

Search Results:
{hits}

Select 5-10 hits by number. Prioritize:
- Direct answers to the query
- Specific facts, figures, dates relevant to the issue
- Key contractual provisions or legal conclusions
- Party admissions or positions

Deprioritize:
- Boilerplate language
- Generic definitions
- Tangential references

Reply with numbers only, comma-separated. Example: 1, 3, 5, 8"""


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


P_PRIORITIZE_DOCUMENTS = """Rank these documents for relevance to the query.

Query: {query}

Key issues: {key_issues}

Candidates:
{candidate_files}

Already read: {already_read}

SCORING (0-100):
- 90-100: Directly answers the query (pleadings, key contracts, party briefs on point)
- 70-89: Contains key supporting evidence
- 40-69: Relevant context
- 10-39: Tangentially related
- 0-9: Not useful for this query

DOCUMENT VALUE HIERARCHY:
- Opening/Closing statements: Synthesized positions, final figures
- Party briefs: Claimant briefs have damages; Defendant briefs have defenses
- Pleadings (complaints, answers): Define the dispute
- Contracts/agreements: Primary source terms
- Correspondence: Actual party communications
- Expert reports: Specialized analysis
- Reference materials: Low priority unless specifically needed

RULES:
- Unread documents > already read (unless essential)
- When BOTH party briefs exist, prioritize both - they contain different information
- For damages/figures: look for claimant/plaintiff sources
- Match document type to query type

Reply JSON only:
{{
    "ranked_files": [
        {{"file": "exact_filename.pdf", "score": 95, "reason": "brief reason"}}
    ]
}}"""


# =============================================================================
# MID TIER PROMPTS (FLASH model - analysis and planning)
# =============================================================================

P_ASSESS_SMALL_REPO = """You have ALL documents from this legal matter. Assess the query and determine what's needed.

Query: {query}

=== ALL DOCUMENTS ===
{content}

=== ASSESSMENT TASK ===
1. COMPLEXITY: Is the synthesis straightforward or does it require sophisticated legal reasoning?
2. SUFFICIENCY: Can you answer this query using ONLY these documents?

Apply your external research judgment here. Most queries about case facts, drafting, or
document-based analysis need no external search. Only recommend external search when the
query explicitly requires legal authority not present in the documents.

ANSWER FROM DOCS (no external search) when:
- Query asks about facts, dates, amounts, parties, events in the case
- Query asks for drafting help using case materials
- Query asks for summaries or analysis of the documents themselves
- The documents contain sufficient information to answer

SEARCH EXTERNALLY only when:
- Query EXPLICITLY asks for case law, precedents, or legal standards
- Query asks about regulations/statutes NOT referenced in documents
- You cannot provide a legally supportable answer without external authority
- The gap is specific and articulable (not just "more context would help")

=== OUTPUT (JSON only) ===
{{
  "complexity": "simple" | "complex",
  "can_answer_from_docs": true | false,
  "reasoning": "Brief explanation of your assessment",
  "gap": "Only if can_answer_from_docs=false: what specific external authority is needed and why",
  "case_law_searches": [],
  "web_searches": []
}}

IMPORTANT: If can_answer_from_docs is true, search arrays MUST be empty.
Only populate searches if there is a genuine, specific gap that external research would fill."""


P_CHECK_SEARCH_SUFFICIENCY = """You searched for external legal authority and found these results.

Query: {query}

=== ORIGINAL GAP ===
{original_gap}

=== SEARCH RESULTS ===
{results_summary}

=== QUESTION ===
Do these results adequately fill the gap, or is there still a CRITICAL missing piece?

CRITICAL means: Your answer would be WRONG or MISLEADING without this information.
NOT critical: "More would be nice" or "additional context could help"

Default: The results are sufficient. Proceed to synthesis.

=== OUTPUT (JSON only) ===
{{
  "sufficient": true | false,
  "reasoning": "Brief explanation",
  "if_not_sufficient_what_missing": "Only if sufficient=false: specific critical gap remaining",
  "additional_search": ""
}}

IMPORTANT: Only set sufficient=false if you have a specific, critical piece of information
that is absolutely required. If you got relevant results, proceed with what you have."""


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


P_ANALYZE_RESULTS = """Analyze these search results strategically.

Query: {query}

Search Results:
{results}

EXTRACT:
1. Facts directly answering or advancing the query
2. Key quotes worth preserving (with source and page)
3. Documents requiring deeper read (potential goldmines)
4. Gaps - what's missing that we still need?

STRATEGIC ASSESSMENT:
- Are we finding what we need?
- Should we pivot search strategy?
- What document types are yielding results vs. dead ends?

Reply JSON:
{{
    "facts": ["specific fact with exact values"],
    "citations": [{{"text": "verbatim quote", "source": "filename", "page": 1}}],
    "read_deeper": ["file1.pdf"],
    "additional_searches": ["more specific term"],
    "assessment": "Brief strategic assessment of progress"
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

P_SYNTHESIZE = """Query: {query}

=== DECISIVE DOCUMENTS ===
{pinned_content}

=== EVIDENCE GATHERED ===
{evidence}

=== EXTERNAL RESEARCH ===
{external_research}
"""


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


P_GENERATE_EXTERNAL_QUERIES = """Determine if this query requires external legal research.

Query: {query}

Facts found: {facts}

Entities: {entities}

Triggers found in documents: {triggers}

═══════════════════════════════════════════════════════════════════════════════
FIRST: Does this query actually NEED external research?
═══════════════════════════════════════════════════════════════════════════════

DEFAULT: NO EXTERNAL SEARCH. Most queries can be answered from documents alone.

DO NOT SEARCH when query asks about:
- "What is the main issue/dispute?" → Answer from documents
- "What happened?" / "Summarize the facts" → Answer from documents
- "What does the contract say?" → Answer from documents
- "Who are the parties?" → Answer from documents
- "What are the claims/allegations?" → Answer from documents
- Any question about WHAT IS IN THE DOCUMENTS

Just because a document MENTIONS a jurisdiction or regulation does NOT mean
you need to search for it. Triggers are informational, not commands.

ONLY SEARCH when query EXPLICITLY requires:
- "What does [jurisdiction] law say about X?" → Search needed
- "Find case law supporting X" → Search needed
- "What are the legal standards for X?" → Search needed
- "Is this compliant with [specific regulation]?" → Search needed

═══════════════════════════════════════════════════════════════════════════════
IF search IS needed, use appropriate source:
═══════════════════════════════════════════════════════════════════════════════

CASE LAW (CourtListener) - US only:
- US legal precedent, judicial interpretations
- US jurisdictions (Michigan, Delaware, federal)
- Legal doctrines needing authority

WEB SEARCH (Tavily):
- Regulations, statutes, standards
- International jurisdictions (NOT CourtListener)
- Company/entity background research

═══════════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════════

Reply JSON:
{{
    "case_law_queries": [],
    "web_queries": [],
    "reasoning": "Why search is or is not needed for THIS SPECIFIC QUERY"
}}

If the query is about document facts/issues/parties/claims, return empty arrays."""


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


P_ANALYZE_SEARCH = """Analyze search results for this investigation.

Query: {query}

Key issues: {key_issues}

Search Results:
{results}

Already read: {already_read}

═══════════════════════════════════════════════════════════════════════════════
DOCUMENT CRITICALITY - BE STRATEGIC WITH MEMORY
═══════════════════════════════════════════════════════════════════════════════

DECISIVE (gets loaded in full for synthesis):
- ONLY case-specific documents that DIRECTLY answer the query
- Contracts, agreements specific to this matter
- Party correspondence, emails about the dispute
- Pleadings, briefs, statements specific to this case
- Expert reports specific to this matter

NEVER DECISIVE (always SUPPORTING or IRRELEVANT):
- Statutes, acts, codes (e.g., "Business Corporations Act")
- Generic regulations or legal references
- Manuals, handbooks, guidelines
- Template documents
- Background/reference materials

SUPPORTING: Useful context but don't need full content in synthesis
IRRELEVANT: Not useful for this query - skip

═══════════════════════════════════════════════════════════════════════════════

Reply JSON:
{{
    "relevant_hit_numbers": [1, 3, 5],
    "facts": ["fact with exact values"],
    "citations": [{{"text": "quote", "source": "filename", "page": 1}}],
    "ranked_documents": [
        {{"file": "path/file.pdf", "score": 95, "criticality": "DECISIVE", "reason": "case-specific, answers query"}},
        {{"file": "statute.pdf", "score": 40, "criticality": "SUPPORTING", "reason": "reference material only"}}
    ],
    "additional_searches": ["term"],
    "read_deeper": ["file.pdf"],
    "assessment": "Brief strategic assessment"
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
