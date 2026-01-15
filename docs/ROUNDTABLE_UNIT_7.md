# Roundtable Unit 7: Batch Document Processing

## Unit Goal
Optimize investigation speed by processing multiple documents in parallel.

## Success Criteria
1. [x] _batch_deep_read() method for parallel document analysis
2. [x] batch_read_parallel() in repository for threaded reading
3. [x] Configurable parallel_reads limit
4. [x] Error handling for individual document failures

## Changes Made

### engine.py
| Method | Description |
|--------|-------------|
| _batch_deep_read() | Process multiple documents with asyncio.gather |
| _analyze_search_results | Now uses batch reading for top files |

### repository.py
| Method | Description |
|--------|-------------|
| batch_read_parallel() | Thread-pool based parallel reading |

### Key Code
```python
async def _batch_deep_read(self, state, repo, file_paths):
    """Process multiple documents in parallel."""
    tasks = [
        self._deep_read_document(state, repo, fp)
        for fp in file_paths
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log any errors without stopping
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Document read failed: {file_paths[i]}")
```

### Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| parallel_reads | 5 | Max documents to read in parallel |

## Review Notes
- Parallel reading significantly speeds up deep analysis phase
- Individual failures don't block other documents
- Rate limiting still applies to LLM calls
- Document cache prevents redundant parsing

## Next Unit
Unit 8: Search Result Ranking
