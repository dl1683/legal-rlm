# Roundtable Unit 29: Document Summarization

## Unit Goal
Generate structured summaries of documents and document collections.

## Success Criteria
1. [x] SUMMARIZE_DOCUMENT_PROMPT
2. [x] SUMMARIZE_COLLECTION_PROMPT
3. [x] summarize_document() method
4. [x] summarize_documents() method
5. [x] Parallel summarization

## Changes Made

### engine.py - Prompts
| Prompt | Purpose |
|--------|---------|
| SUMMARIZE_DOCUMENT_PROMPT | Single document summary |
| SUMMARIZE_COLLECTION_PROMPT | Collection summary |

### engine.py - Methods
| Method | Description |
|--------|-------------|
| summarize_document() | Summarize one document |
| summarize_documents() | Summarize multiple + collection |

### Document Summary Output
| Field | Description |
|-------|-------------|
| summary | 2-3 sentence overview |
| document_type | contract/pleading/correspondence/other |
| parties | Key parties mentioned |
| key_dates | Important dates with descriptions |
| key_terms | Key provisions or claims |
| amounts | Dollar amounts with context |
| concerns | Notable issues |

### Collection Summary Output
| Field | Description |
|-------|-------------|
| collection_summary | 3-5 sentence overview |
| parties | All parties across documents |
| timeline | Chronological events |
| themes | Main themes identified |
| document_relationships | How documents relate |
| gaps | Missing information |

### Key Code
```python
SUMMARIZE_DOCUMENT_PROMPT = """
Document: {filename}
Document Type: {doc_type}
Content: {content}

Create a structured summary including:
1. Document Type and Purpose
2. Key Parties/Entities
3. Important Dates
4. Key Terms/Provisions
5. Critical Numbers
6. Notable Issues

Respond in JSON format:
{{
    "summary": "...",
    "document_type": "...",
    "parties": [...],
    "key_dates": [...],
    "key_terms": [...],
    "amounts": [...],
    "concerns": [...]
}}
"""

async def summarize_document(self, file_path: Path) -> dict:
    doc = repository.read(str(file_path))
    content = doc.full_text[:self.config.excerpt_chars]

    prompt = SUMMARIZE_DOCUMENT_PROMPT.format(...)
    response = await self.client.complete(prompt, tier=ModelTier.FLASH)
    return self._parse_json_safe(response, defaults)

async def summarize_documents(self, file_paths: list[Path]) -> dict:
    # Parallel individual summaries
    tasks = [self.summarize_document(fp) for fp in file_paths]
    individual = await asyncio.gather(*tasks)

    # Generate collection summary
    prompt = SUMMARIZE_COLLECTION_PROMPT.format(
        document_list=...,
        summaries=...,
    )
    collection = await self.client.complete(prompt, tier=ModelTier.FLASH)

    return {
        "individual_summaries": individual,
        "collection_summary": collection,
    }
```

## Usage Example
```python
# Summarize single document
summary = await engine.summarize_document(Path("contract.pdf"))
print(f"Summary: {summary['summary']}")
print(f"Parties: {summary['parties']}")

# Summarize multiple documents
result = await engine.summarize_documents([
    Path("contract.pdf"),
    Path("amendment.pdf"),
    Path("correspondence.pdf"),
])

print(f"Collection: {result['collection_summary']['collection_summary']}")
print(f"Themes: {result['collection_summary']['themes']}")

for doc in result['individual_summaries']:
    print(f"- {doc['filename']}: {doc['summary']}")
```

## Review Notes
- Parallel summarization for efficiency
- Structured JSON output for programmatic use
- Document type auto-detection from filename
- Collection summary identifies relationships
- Useful for repository orientation

## Next Unit
Unit 30: Caching & Performance
