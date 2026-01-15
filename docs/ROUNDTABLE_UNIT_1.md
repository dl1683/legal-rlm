# Roundtable Unit 1: Foundation & Security Hardening

## Unit Goal
Establish solid foundation for RLM legal document analysis system with proper security, error handling, and architectural patterns.

## Success Criteria
1. [x] No hardcoded API keys in codebase
2. [x] All API calls have timeout handling
3. [x] Structured logging throughout
4. [x] Token/cost tracking implemented
5. [x] Global document caching
6. [ ] Parallel lead processing (deferred to Unit 2)

## Constraints
- Must maintain backward compatibility with existing tests
- No breaking changes to public API
- Keep simple UI for now (focus on backend)

---

## Cycle 1: Discovery (COMPLETE)

### Key Findings
- 3 CRITICAL security issues (hardcoded API keys)
- No timeout handling on API calls
- Silent JSON parsing failures
- Sequential lead processing bottleneck
- Per-search document caching (not global)

---

## Cycle 2: Design (COMPLETE)

### Implementation Plan
See detailed plan in previous version of this document.

---

## Cycle 3: Implementation (COMPLETE)

### Changes Made

#### Security Fixes
| File | Change |
|------|--------|
| run.py | Removed hardcoded key, added env validation |
| test_rlm.py | Removed hardcoded key, added env check |
| test_basic.py | Made Gemini test conditional on env var |

#### Timeout & Retry
| File | Change |
|------|--------|
| models.py | Added 120s default timeout with asyncio.wait_for |
| models.py | Added complete_with_retry() with exponential backoff |
| models.py | Added UsageStats class for token tracking |
| models.py | Added get_usage(), get_total_cost() methods |

#### Logging
| File | Change |
|------|--------|
| models.py | Added logger, debug/error logging |
| search.py | Replaced print() with logger.warning() |
| repository.py | Added logger, info/debug logging |
| engine.py | Added logger import |
| run.py | Configured logging with format string |

#### Caching
| File | Change |
|------|--------|
| repository.py | Added global _doc_cache dict |
| repository.py | Cache shared with search_engine |
| repository.py | read() checks cache before parsing |

---

## Cycle 4: Review

### What Was Accomplished
1. **Security**: All hardcoded API keys removed
2. **Reliability**: 120s timeout on all API calls
3. **Observability**: Structured logging throughout
4. **Cost Control**: Token tracking with estimated costs
5. **Performance**: Global document caching

### What Was Deferred to Unit 2
1. Parallel lead processing (asyncio.gather)
2. JSON parsing fallback defaults
3. Citation verification loop
4. Rate limiting

### Code Quality Assessment
- Security: GOOD (no exposed credentials)
- Error handling: IMPROVED (timeouts, logging)
- Performance: IMPROVED (caching)
- Testing: NEEDS WORK (add more unit tests)

---

## Next Unit Proposal

### Unit 2: Performance & Parallel Processing

**Goal**: Optimize investigation speed through parallel lead processing and improve error recovery.

**Success Criteria**:
1. Leads processed in parallel with asyncio.gather
2. JSON parsing has safe fallback defaults
3. Investigation time reduced by 30%+
4. Add rate limiting to prevent API overload

**Key Files**:
- engine.py (parallel leads)
- models.py (rate limiting)
- state.py (thread-safe updates)

---

## Cycle Handoff Pack

**Unit Goal:** Foundation & Security Hardening

**Cycle Number:** 4 (Review)

**Subagents Run:**
- Repo Cartographer (Discovery)
- Research Scout (Discovery)
- Implementation (Cycles 2-3)
- Review (Cycle 4)

**Key Decisions Made:**
1. Use environment variables for all API keys
2. 120 second default timeout for API calls
3. Global document caching at repository level
4. Defer parallel lead processing to Unit 2

**Artifacts Produced:**
- Updated models.py with timeout/retry/tracking
- Updated search.py with logging
- Updated repository.py with global cache
- Updated run.py/test files without hardcoded keys

**Open Questions:**
- Should rate limiting be per-tier or global?
- What's the optimal parallel lead count?
- Should we add checkpointing for long investigations?

**Next Cycle Must Start By:**
Implementing parallel lead processing in engine.py

**Acceptance Criteria for Next Cycle:**
- Leads processed in parallel
- Investigation time reduced by 30%+
- No race conditions in state updates
