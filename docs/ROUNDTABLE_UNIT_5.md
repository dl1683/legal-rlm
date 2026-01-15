# Roundtable Unit 5: Rate Limiting

## Unit Goal
Prevent API overload with token bucket rate limiting.

## Success Criteria
1. [x] RateLimiter class with token bucket algorithm
2. [x] Configurable requests per minute and burst size
3. [x] Integrated into GeminiClient.complete()
4. [x] Integrated into GeminiClient.complete_with_history()
5. [x] Lazy initialization to avoid event loop issues

## Changes Made

### models.py
| Change | Description |
|--------|-------------|
| RateLimiter class | Token bucket rate limiter |
| GeminiClient.__init__ | Added requests_per_minute, burst_size params |
| GeminiClient.complete | Added rate limiter acquire before API call |
| GeminiClient.complete_with_history | Added rate limiter acquire |

### Rate Limiter Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| requests_per_minute | 60 | Max sustained request rate |
| burst_size | 10 | Max burst before throttling |

### Key Code
```python
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_update: Optional[float] = None
        self._lock: Optional[asyncio.Lock] = None

    async def acquire(self):
        async with self._get_lock():
            # Replenish tokens based on time passed
            tokens_to_add = time_passed * (self.requests_per_minute / 60.0)
            self.tokens = min(float(self.burst_size), self.tokens + tokens_to_add)

            if self.tokens < 1:
                wait_time = (1 - self.tokens) / (self.requests_per_minute / 60.0)
                await asyncio.sleep(wait_time)
```

## Review Notes
- Token bucket allows bursts while enforcing average rate
- Lazy lock initialization prevents event loop issues
- Rate limiting is transparent to callers
- Logs wait times at debug level

## Next Unit
Unit 6: Checkpointing & Resume
