# Roundtable Unit 31: Output Formatting

## Unit Goal
Provide multiple output formats for investigation results.

## Success Criteria
1. [x] OutputFormatter protocol
2. [x] MarkdownFormatter
3. [x] HTMLFormatter
4. [x] JSONFormatter
5. [x] PlainTextFormatter
6. [x] get_formatter() factory

## Changes Made

### output/formatters.py (NEW)
| Formatter | Description |
|-----------|-------------|
| MarkdownFormatter | GitHub-flavored Markdown |
| HTMLFormatter | Standalone HTML document |
| JSONFormatter | Structured JSON |
| PlainTextFormatter | Simple plain text |

### Key Code
```python
class MarkdownFormatter:
    def format(self, state: Any) -> str:
        lines = [
            f"# Investigation Report",
            f"**Query:** {state.query}",
            "## Key Findings",
            ...
        ]
        return "\n".join(lines)

def get_formatter(format_type: str) -> OutputFormatter:
    formatters = {
        "markdown": MarkdownFormatter(),
        "html": HTMLFormatter(),
        "json": JSONFormatter(),
        "text": PlainTextFormatter(),
    }
    return formatters.get(format_type.lower(), PlainTextFormatter())
```

## Usage
```python
from irys.output import get_formatter

formatter = get_formatter("markdown")
output = formatter.format(state)
print(output)
```

## Next Unit
Unit 32: Error Recovery
