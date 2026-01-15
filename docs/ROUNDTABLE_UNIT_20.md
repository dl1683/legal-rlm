# Roundtable Unit 20: Timeline Extraction

## Unit Goal
Extract and track dates/events to build investigation timeline.

## Success Criteria
1. [x] TimelineEvent dataclass with date parsing
2. [x] add_timeline_event() with deduplication
3. [x] get_timeline_sorted() for chronological view
4. [x] get_timeline_formatted() for display

## Changes Made

### state.py - TimelineEvent Class
| Field | Description |
|-------|-------------|
| date_str | Original date string |
| date_parsed | Parsed datetime (if possible) |
| description | Event description |
| source_doc | Source document |
| page | Page number |
| event_type | filing, agreement, correspondence, deadline, general |

### Date Formats Parsed
| Format | Example |
|--------|---------|
| MM/DD/YYYY | 01/15/2024 |
| MM/DD/YY | 01/15/24 |
| YYYY-MM-DD | 2024-01-15 |
| Month DD, YYYY | January 15, 2024 |

### state.py - Methods
| Method | Description |
|--------|-------------|
| add_timeline_event() | Add event with deduplication |
| get_timeline_sorted() | Get events in chronological order |
| get_timeline_formatted() | Get human-readable timeline |

### Key Code
```python
@dataclass
class TimelineEvent:
    date_str: str
    date_parsed: Optional[datetime] = None
    ...

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        patterns = [
            (r"(\d{1,2})/(\d{1,2})/(\d{4})", "%m/%d/%Y"),
            (r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", None),  # Month name
        ]
        for pattern, fmt in patterns:
            match = re.search(pattern, date_str)
            if match:
                return datetime.strptime(...)

def get_timeline_sorted(self) -> list[TimelineEvent]:
    def sort_key(e):
        if e.date_parsed:
            return (0, e.date_parsed)
        return (1, datetime.max)
    return sorted(self.timeline, key=sort_key)
```

## Review Notes
- Timeline provides chronological view of events
- Date parsing handles multiple formats
- Unparseable dates sorted to end
- Deduplication prevents duplicate events

## Next Unit
Unit 21: Contradiction Detection
