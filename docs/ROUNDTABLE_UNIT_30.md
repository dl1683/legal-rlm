# Roundtable Unit 30: Caching & Performance

## Unit Goal
Add caching mechanisms to improve performance and reduce redundant operations.

## Success Criteria
1. [x] LRUCache class with TTL support
2. [x] DiskCache for persistent caching
3. [x] ResponseCache for LLM responses
4. [x] Cache statistics
5. [x] cached() decorator

## Changes Made

### cache.py (NEW)
| Component | Description |
|-----------|-------------|
| CacheEntry | Single cache entry with metadata |
| LRUCache | In-memory LRU cache with TTL |
| DiskCache | Persistent disk-based cache |
| ResponseCache | Specialized cache for LLM responses |
| cached() | Decorator for caching functions |

### LRUCache Features
| Feature | Description |
|---------|-------------|
| max_size | Maximum number of entries |
| ttl_seconds | Time-to-live for entries |
| get/set | Standard cache operations |
| touch() | Record cache hit |
| evict_oldest() | Remove least recently used |

### DiskCache Features
| Feature | Description |
|---------|-------------|
| max_size_mb | Maximum cache size in MB |
| Persistent | Survives program restart |
| JSON storage | Human-readable cache files |
| Automatic eviction | Enforces size limits |

### ResponseCache
| Feature | Description |
|---------|-------------|
| Prompt+model key | Unique key per request |
| 1 hour default TTL | Reasonable for most use cases |
| Size tracking | Track response sizes |

### Key Code
```python
class LRUCache(Generic[T]):
    def __init__(self, max_size: int = 100, ttl_seconds: Optional[int] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry[T]] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> Optional[T]:
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if entry.is_expired:
            self._remove(key)
            return None
        entry.touch()
        self._move_to_front(key)
        return entry.value

    def set(self, key: str, value: T, size_bytes: int = 0):
        while len(self._cache) >= self.max_size:
            self._evict_oldest()
        self._cache[key] = CacheEntry(key=key, value=value, ...)

class ResponseCache:
    def _make_key(self, prompt: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()

    def get(self, prompt: str, model: str) -> Optional[str]:
        return self._cache.get(self._make_key(prompt, model))

    def set(self, prompt: str, model: str, response: str):
        self._cache.set(self._make_key(prompt, model), response)
```

## Usage Example
```python
from irys.core.cache import LRUCache, ResponseCache, DiskCache

# In-memory cache
cache = LRUCache[dict](max_size=100, ttl_seconds=3600)
cache.set("key1", {"data": "value"})
result = cache.get("key1")

# Response cache
response_cache = ResponseCache(max_size=500, ttl_seconds=3600)
response_cache.set(prompt, "gemini-flash", response)
cached_response = response_cache.get(prompt, "gemini-flash")

# Disk cache
disk_cache = DiskCache(cache_dir="./cache", max_size_mb=100)
disk_cache.set("document_summary", summary_data)

# Get statistics
print(cache.get_stats())
```

## Performance Impact
| Operation | Without Cache | With Cache |
|-----------|--------------|------------|
| Same prompt | API call | ~0ms |
| Document read | Disk read | ~0ms |
| Search result | Full search | ~0ms |

## Review Notes
- LRU ensures frequently used items stay cached
- TTL prevents stale data
- Disk cache survives restarts
- Response cache reduces API costs
- Statistics help monitor cache effectiveness

## Next Unit
Unit 31: Output Formatting
