# Irys RLM Client Libraries

Client libraries for integrating with the Irys RLM document investigation API.

## Python Client

### Installation

```bash
cd clients/python
pip install -r requirements.txt
```

Or just install httpx:
```bash
pip install httpx
```

### Quick Start

```python
import asyncio
from irys_client import IrysClient

async def main():
    # Create client
    client = IrysClient("http://your-irys-server:8000")
    
    # Check health
    health = await client.health()
    print(f"Status: {health['status']}")
    
    # Run investigation
    result = await client.investigate(
        query="What are the payment terms?",
        s3_prefix="matters/case-001",
    )
    
    print(result["analysis"])
    
    # Clean up
    await client.close()

asyncio.run(main())
```

### Using Context Manager

```python
async with IrysClient("http://your-server:8000") as client:
    result = await client.investigate(
        query="What are the key obligations?",
        s3_prefix="matters/case-001",
    )
```

### Available Methods

| Method | Description |
|--------|-------------|
| `health()` | Check service status |
| `investigate(query, s3_prefix, ...)` | Run full investigation |
| `get_job(job_id)` | Get job status/results |
| `list_jobs(status, limit)` | List recent jobs |
| `search(query, s3_prefix, max_results)` | Quick keyword search |

### Using Webhooks (Recommended for Production)

Instead of polling, use webhooks:

```python
# Start investigation with callback
result = await client.investigate(
    query="What are the payment terms?",
    s3_prefix="matters/case-001",
    callback_url="https://your-service.com/webhook/irys",
    wait=False,  # Don't poll, use webhook instead
)

job_id = result["job_id"]
# Results will be POSTed to your callback URL when ready
```

## Node.js/TypeScript Client

See `API_DOCS.md` for TypeScript client example.

## Integration Tips

1. **Use webhooks** for production to avoid polling overhead
2. **Check health** before starting investigations
3. **Handle errors** gracefully with retries
4. **Set reasonable timeouts** (investigations can take 30-120 seconds)
5. **Cache results** for repeated queries

