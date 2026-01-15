# Roundtable Unit 6: Checkpointing & Resume

## Unit Goal
Allow saving investigation state to disk and resuming later.

## Success Criteria
1. [x] InvestigationState.to_dict() serialization
2. [x] InvestigationState.from_dict() deserialization
3. [x] save_checkpoint() and load_checkpoint() methods
4. [x] Periodic checkpoints during investigation
5. [x] resume_investigation() method in engine

## Changes Made

### state.py
| Method | Description |
|--------|-------------|
| to_dict() | Serialize entire state to dictionary |
| from_dict() | Deserialize state from dictionary |
| save_checkpoint() | Save state to JSON file |
| load_checkpoint() | Load state from JSON file |

### engine.py
| Change | Description |
|--------|-------------|
| RLMConfig.checkpoint_dir | Optional directory for checkpoints |
| RLMConfig.checkpoint_interval | Save every N iterations (default: 5) |
| _save_checkpoint() | Save checkpoint with iteration number |
| resume_investigation() | Resume from checkpoint file |

### Serialized Fields
- All thinking steps with timestamps
- All citations with verification status
- All leads with investigation status
- Findings dictionary
- Hypothesis and metrics
- Start/completion timestamps

### Key Code
```python
def to_dict(self) -> dict:
    return {
        "id": self.id,
        "query": self.query,
        "thinking_steps": [...],
        "citations": [...],
        "leads": [...],
        "findings": self.findings,
        ...
    }

async def resume_investigation(self, checkpoint_path):
    state = InvestigationState.load_checkpoint(checkpoint_path)
    if state.status not in ("completed", "failed"):
        await self._investigate_loop(state, repo)
        await self._verify_citations(state, repo)
        await self._synthesize(state)
    return state
```

## Review Notes
- Checkpoints include full state for complete resume
- Both iteration-specific and "latest" checkpoints saved
- Resume skips already-completed phases
- JSON format is human-readable for debugging

## Next Unit
Unit 7: Batch Document Processing
