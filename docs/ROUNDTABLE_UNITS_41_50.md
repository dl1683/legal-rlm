# Roundtable Units 41-50: API & Integration

## Overview
Units 41-50 implement the high-level API, convenience functions, and final polish.

## Unit 41: High-Level API (Irys Class)
Main entry point for the system:
- `Irys(config)` - Initialize with configuration
- `investigate()` - Run full investigation
- `summarize()` - Summarize documents
- `search()` - Search repository
- Callback support for progress/steps

## Unit 42: Investigation Result
Structured result container:
- `state` - Full investigation state
- `output` - Formatted output
- `confidence` - Confidence metrics
- `quality` - Quality assessment
- `to_format()` - Convert to other formats

## Unit 43: Batch Investigation
`batch_investigate()` - Run multiple queries:
- Parallel or sequential execution
- Returns list of results
- Error handling per query

## Unit 44: Quick Functions
Convenience functions for common tasks:
- `quick_search()` - Simple search
- `quick_summarize()` - Simple summarization

## Unit 45: Repository Analysis
`analyze_repository()` - Get repository overview:
- File statistics
- Document categorization
- Structure analysis

## Unit 46: Default Instance
`get_irys()` - Get/create default instance

## Unit 47: Sync Wrappers
Synchronous versions for non-async code:
- `investigate_sync()`
- `search_sync()`

## Unit 48: Version Info
- `__version__` - Version string
- `get_version()` - Get version

## Unit 49: Health Check
`health_check()` - System status check

## Unit 50: Public API Exports
Complete `__all__` with all public APIs

## Usage Example
```python
from irys.api import Irys, IrysConfig

# Initialize
config = IrysConfig(
    api_key="your-key",
    max_depth=5,
    output_format="markdown",
)
irys = Irys(config)

# Set callbacks
irys.on_progress(lambda p: print(f"Progress: {p['progress_percent']}%"))

# Run investigation
result = await irys.investigate(
    query="What are the contract obligations?",
    repository="./documents",
)

print(f"Status: {result.status}")
print(f"Confidence: {result.confidence['score']}/100")
print(result.output)

# Quick functions
results = await quick_search("termination", "./documents")
summary = await quick_summarize(["doc1.pdf", "doc2.pdf"])
```

## All Files
- src/irys/api.py
