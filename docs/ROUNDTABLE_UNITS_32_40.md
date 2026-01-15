# Roundtable Units 32-40: Core Utilities

## Overview
Units 32-40 implement core utility functions across error handling, logging, text processing, batch operations, and configuration.

## Unit 32: Error Recovery
- RetryableError / FatalError exceptions
- RetryConfig dataclass
- retry_async() decorator with exponential backoff
- retry_sync() for synchronous functions

## Unit 33: Logging & Telemetry
- TelemetryEvent dataclass
- TelemetryCollector for tracking operations
- setup_logging() configuration
- Duration tracking with start_operation/end_operation

## Unit 34: Text Utilities
- clean_text() - normalize text
- truncate_text() - safe truncation
- extract_sentences() - sentence splitting
- extract_numbers() - find numbers/currency
- normalize_whitespace() - whitespace handling

## Unit 35: Batch Processing
- batch_process() - async batch execution
- chunk_list() - split lists
- parallel_map() - concurrent mapping with limit

## Unit 36: Progress Tracking
- ProgressTracker dataclass
- advance() / set() methods
- percent, elapsed_seconds, eta_seconds properties
- Callback support for UI updates

## Unit 37: Configuration Management
- SystemConfig dataclass
- Model, investigation, performance settings
- from_dict() / to_dict() serialization

## Unit 38: Validation Utilities
- validate_query() - query validation
- validate_file_path() - path validation

## Unit 39: Date/Time Utilities
- parse_date_flexible() - multi-format parsing
- format_duration() - human-readable durations

## Unit 40: String Similarity
- levenshtein_distance() - edit distance
- similarity_ratio() - string similarity 0-1
- jaccard_similarity() - word set similarity

## Key Code Examples

### Retry Decorator
```python
@retry_async(RetryConfig(max_retries=3, initial_delay=1.0))
async def make_api_call():
    return await client.complete(prompt)
```

### Batch Processing
```python
results = await batch_process(
    items=documents,
    processor=summarize_document,
    batch_size=10,
)
```

### Progress Tracking
```python
tracker = ProgressTracker(total=100)
tracker.add_callback(lambda t: print(f"{t.percent}%"))
tracker.advance(10, "Processing documents...")
```

## All Files
- src/irys/core/utils.py
