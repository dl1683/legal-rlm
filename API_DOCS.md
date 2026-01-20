# Irys RLM API Documentation

REST API for document investigation and analysis using recursive language models.

**Base URL:** `http://your-server:8000`  
**OpenAPI Docs:** `http://your-server:8000/docs`

---

## Configuration

### Required Environment Variables

Create a `.env` file in the project root:

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` with your values:

```env
# ============================================
# REQUIRED
# ============================================

# Google Gemini API Key
# Get from: https://aistudio.google.com/app/apikey
GEMINI_API_KEY=AIzaSy...your_key_here

# S3 Bucket for document storage
S3_BUCKET=your-company-documents

# AWS Region
S3_REGION=us-east-1

# ============================================
# AWS CREDENTIALS (choose one method)
# ============================================

# Option 1: Environment variables (for local/testing)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...

# Option 2: IAM Role (recommended for EC2)
# Leave AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY empty
# Attach an IAM role with S3 read permissions to your EC2 instance

# ============================================
# OPTIONAL - Resource Limits
# ============================================

# Max concurrent investigations (default: 3)
IRYS_MAX_CONCURRENT_JOBS=3

# Max documents per investigation (default: 50)
IRYS_MAX_DOCS_PER_JOB=50

# Temp storage limit in MB (default: 500)
IRYS_MAX_TEMP_SIZE_MB=500
```

### S3 Bucket Structure

Organize documents by matter/case:

```
your-bucket/
├── matters/
│   ├── case-001/
│   │   ├── contract.pdf
│   │   ├── amendment.pdf
│   │   └── correspondence/
│   │       └── email_chain.pdf
│   ├── case-002/
│   │   └── ...
```

When calling the API, reference the S3 prefix:
```json
{"s3_prefix": "matters/case-001"}
```

---

## API Endpoints

### Health Check

Check service status and connectivity.

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "gemini_connected": true,
  "s3_connected": true,
  "active_jobs": 2,
  "temp_storage_mb": 125.5,
  "uptime_seconds": 3600
}
```

**Use this for:**
- Load balancer health checks
- Monitoring/alerting
- Debugging connectivity issues

---

### Start Investigation

Start an async document investigation. Returns immediately with a job ID.

```
POST /investigate
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What are the payment terms and key obligations?",
  "s3_prefix": "matters/case-001",
  "callback_url": "https://your-service.com/webhook/irys",
  "options": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Investigation question |
| `s3_prefix` | string | ✅ | S3 folder containing documents |
| `callback_url` | string | ❌ | URL to POST results when complete |
| `options` | object | ❌ | Additional options (reserved) |

**Response:**
```json
{
  "job_id": "inv_abc123def456",
  "status": "pending",
  "message": "Investigation started",
  "estimated_seconds": 60
}
```

**Status Codes:**
- `200` - Job started successfully
- `429` - Too many concurrent jobs (wait and retry)
- `400` - Invalid request

---

### Get Investigation Results

Poll for investigation status and results.

```
GET /investigate/{job_id}
```

**Response (pending/processing):**
```json
{
  "job_id": "inv_abc123def456",
  "status": "processing",
  "query": "What are the payment terms?",
  "s3_prefix": "matters/case-001",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": null,
  "documents_processed": 0
}
```

**Response (completed):**
```json
{
  "job_id": "inv_abc123def456",
  "status": "completed",
  "query": "What are the payment terms?",
  "s3_prefix": "matters/case-001",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:31:05Z",
  "duration_seconds": 65.2,
  "documents_processed": 5,
  "analysis": "Based on the contract documents, the payment terms are...",
  "citations": [
    {
      "file": "contract.pdf",
      "page": 12,
      "text": "Payment shall be due within 30 days..."
    }
  ],
  "entities": {
    "companies": ["ACME Corp", "TechServices LLC"],
    "people": ["John Smith", "Jane Doe"],
    "dates": ["January 1, 2024", "December 31, 2024"],
    "amounts": ["$50,000", "$5,000/month"]
  }
}
```

**Response (failed):**
```json
{
  "job_id": "inv_abc123def456",
  "status": "failed",
  "error": "S3 access denied: bucket 'your-bucket' prefix 'invalid/path'",
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:02Z"
}
```

**Job Status Values:**
| Status | Description |
|--------|-------------|
| `pending` | Job queued, waiting to start |
| `processing` | Investigation in progress |
| `completed` | Results ready |
| `failed` | Error occurred |

---

### List Jobs

List recent investigation jobs.

```
GET /jobs?status=completed&limit=10
```

**Query Parameters:**
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `status` | string | all | Filter by status |
| `limit` | int | 20 | Max results (1-100) |

**Response:**
```json
[
  {"job_id": "inv_abc123", "status": "completed", ...},
  {"job_id": "inv_def456", "status": "processing", ...}
]
```

---

### Quick Search

Keyword search across documents (synchronous, faster).

```
POST /search
Content-Type: application/json
```

**Request:**
```json
{
  "query": "payment terms",
  "s3_prefix": "matters/case-001",
  "max_results": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "file": "contract.pdf",
      "page": 12,
      "text": "Payment terms: Net 30 days from invoice date",
      "context": "...surrounding text for context..."
    }
  ],
  "total_matches": 15,
  "documents_searched": 5
}
```

---

## Chat Service Integration Guide

### Python Client Example

```python
import asyncio
import httpx
from typing import Optional

class IrysClient:
    """Client for Irys RLM API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def health(self) -> dict:
        """Check service health."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def investigate(
        self,
        query: str,
        s3_prefix: str,
        callback_url: Optional[str] = None,
        wait: bool = True,
        poll_interval: float = 5.0,
        timeout: float = 300.0,
    ) -> dict:
        """
        Run a document investigation.

        Args:
            query: Investigation question
            s3_prefix: S3 folder with documents
            callback_url: Optional webhook URL
            wait: If True, poll until complete
            poll_interval: Seconds between polls
            timeout: Max wait time in seconds

        Returns:
            Investigation results
        """
        # Start the job
        response = await self.client.post(
            f"{self.base_url}/investigate",
            json={
                "query": query,
                "s3_prefix": s3_prefix,
                "callback_url": callback_url,
            }
        )
        response.raise_for_status()
        job = response.json()

        if not wait:
            return job

        # Poll for results
        job_id = job["job_id"]
        elapsed = 0.0

        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            result = await self.get_job(job_id)

            if result["status"] == "completed":
                return result
            elif result["status"] == "failed":
                raise Exception(f"Investigation failed: {result.get('error')}")

        raise TimeoutError(f"Investigation timed out after {timeout}s")

    async def get_job(self, job_id: str) -> dict:
        """Get job status and results."""
        response = await self.client.get(
            f"{self.base_url}/investigate/{job_id}"
        )
        response.raise_for_status()
        return response.json()

    async def search(
        self,
        query: str,
        s3_prefix: str,
        max_results: int = 20,
    ) -> dict:
        """Quick keyword search."""
        response = await self.client.post(
            f"{self.base_url}/search",
            json={
                "query": query,
                "s3_prefix": s3_prefix,
                "max_results": max_results,
            }
        )
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close the client."""
        await self.client.aclose()


# Usage in your chat service:
async def handle_user_question(user_query: str, matter_id: str):
    """Handle a user question about documents."""

    client = IrysClient("http://your-irys-server:8000")

    try:
        # Check if service is healthy
        health = await client.health()
        if health["status"] != "healthy":
            return "Document service is currently unavailable."

        # Run investigation
        result = await client.investigate(
            query=user_query,
            s3_prefix=f"matters/{matter_id}",
            timeout=120.0,
        )

        # Format response for user
        response = result["analysis"]

        # Add citations
        if result.get("citations"):
            response += "\n\n**Sources:**\n"
            for cite in result["citations"][:3]:
                response += f"- {cite['file']}, page {cite['page']}\n"

        return response

    except Exception as e:
        return f"Error analyzing documents: {e}"

    finally:
        await client.close()
```

### Node.js/TypeScript Client Example

```typescript
import axios, { AxiosInstance } from 'axios';

interface InvestigateRequest {
  query: string;
  s3_prefix: string;
  callback_url?: string;
}

interface JobResult {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  query: string;
  analysis?: string;
  citations?: Array<{ file: string; page: number; text: string }>;
  entities?: Record<string, string[]>;
  error?: string;
}

class IrysClient {
  private client: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      timeout: 30000,
    });
  }

  async health(): Promise<any> {
    const response = await this.client.get('/health');
    return response.data;
  }

  async investigate(
    query: string,
    s3Prefix: string,
    options: {
      callbackUrl?: string;
      wait?: boolean;
      pollInterval?: number;
      timeout?: number;
    } = {}
  ): Promise<JobResult> {
    const { wait = true, pollInterval = 5000, timeout = 300000 } = options;

    // Start job
    const response = await this.client.post('/investigate', {
      query,
      s3_prefix: s3Prefix,
      callback_url: options.callbackUrl,
    });

    if (!wait) {
      return response.data;
    }

    // Poll for results
    const jobId = response.data.job_id;
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      await this.sleep(pollInterval);

      const result = await this.getJob(jobId);

      if (result.status === 'completed') {
        return result;
      } else if (result.status === 'failed') {
        throw new Error(`Investigation failed: ${result.error}`);
      }
    }

    throw new Error(`Investigation timed out after ${timeout}ms`);
  }

  async getJob(jobId: string): Promise<JobResult> {
    const response = await this.client.get(`/investigate/${jobId}`);
    return response.data;
  }

  async search(
    query: string,
    s3Prefix: string,
    maxResults: number = 20
  ): Promise<any> {
    const response = await this.client.post('/search', {
      query,
      s3_prefix: s3Prefix,
      max_results: maxResults,
    });
    return response.data;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// Usage:
const irys = new IrysClient('http://your-irys-server:8000');

async function handleQuestion(query: string, matterId: string) {
  const result = await irys.investigate(query, `matters/${matterId}`);
  return result.analysis;
}
```

### Webhook Integration

Instead of polling, receive results via webhook:

**1. Start investigation with callback:**
```json
POST /investigate
{
  "query": "What are the payment terms?",
  "s3_prefix": "matters/case-001",
  "callback_url": "https://your-chat-service.com/webhooks/irys"
}
```

**2. Irys will POST results to your callback URL:**
```json
POST https://your-chat-service.com/webhooks/irys
Content-Type: application/json

{
  "job_id": "inv_abc123def456",
  "status": "completed",
  "query": "What are the payment terms?",
  "analysis": "Based on the documents...",
  "citations": [...],
  "entities": {...},
  "documents_processed": 5,
  "duration_seconds": 65.2
}
```

**3. Handle webhook in your service:**
```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/webhooks/irys")
async def handle_irys_webhook(request: Request):
    result = await request.json()

    job_id = result["job_id"]
    status = result["status"]

    if status == "completed":
        # Send analysis to user
        analysis = result["analysis"]
        await send_to_user(job_id, analysis)
    else:
        # Handle failure
        error = result.get("error", "Unknown error")
        await notify_failure(job_id, error)

    return {"received": True}
```

---

## cURL Examples

### Health Check
```bash
curl http://localhost:8000/health
```

### Start Investigation
```bash
curl -X POST http://localhost:8000/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the payment terms and obligations?",
    "s3_prefix": "matters/case-001"
  }'
```

### Get Results
```bash
curl http://localhost:8000/investigate/inv_abc123def456
```

### Quick Search
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "payment",
    "s3_prefix": "matters/case-001",
    "max_results": 10
  }'
```

### List Jobs
```bash
curl "http://localhost:8000/jobs?status=completed&limit=5"
```

---

## File Upload Endpoints

Upload documents directly without S3. Useful for testing via Postman.

### Upload & Investigate

Start an investigation with uploaded files.

```
POST /upload/investigate
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Investigation question |
| `files` | file[] | ✅ | Document files (PDF, DOCX, TXT) |
| `callback_url` | string | ❌ | Webhook URL for results |

**Response:**
```json
{
  "job_id": "upload_abc123def456",
  "status": "pending",
  "message": "Investigation started",
  "files_received": 3,
  "estimated_seconds": 60
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/upload/investigate \
  -F "query=What are the payment terms?" \
  -F "files=@contract.pdf" \
  -F "files=@amendment.pdf"
```

### Upload & Search

Quick search across uploaded files (synchronous).

```
POST /upload/search
Content-Type: multipart/form-data
```

**Form Fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | ✅ | Search query |
| `files` | file[] | ✅ | Document files to search |
| `max_results` | int | ❌ | Maximum results (default: 20) |

**Response:**
```json
{
  "results": [
    {"file": "contract.pdf", "page": 5, "text": "...matching text..."}
  ],
  "total_matches": 12,
  "files_searched": 2
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/upload/search \
  -F "query=payment" \
  -F "files=@contract.pdf" \
  -F "max_results=10"
```

---

## S3 URL Endpoints

Download and analyze documents by specific S3 URLs instead of prefixes.

### Investigate by URLs

```
POST /investigate/urls
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "What are the key terms?",
  "s3_urls": [
    "s3://my-bucket/contracts/main.pdf",
    "s3://my-bucket/contracts/amendment.pdf"
  ],
  "callback_url": "https://your-service.com/webhook"
}
```

**Supported URL formats:**
- `s3://bucket/key`
- `https://bucket.s3.region.amazonaws.com/key`
- `https://s3.region.amazonaws.com/bucket/key`

**Response:**
```json
{
  "job_id": "urls_abc123def456",
  "status": "pending",
  "message": "Investigation started for 2 files",
  "estimated_seconds": 60
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/investigate/urls \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the payment obligations?",
    "s3_urls": [
      "s3://my-bucket/docs/contract.pdf",
      "s3://my-bucket/docs/addendum.pdf"
    ]
  }'
```

### Search by URLs

Quick search across documents from S3 URLs (synchronous).

```
POST /search/urls
Content-Type: application/json
```

**Request Body:**
```json
{
  "query": "payment",
  "s3_urls": [
    "s3://my-bucket/contracts/main.pdf"
  ],
  "max_results": 20
}
```

**Response:**
```json
{
  "results": [...],
  "total_matches": 8,
  "documents_searched": 1
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/search/urls \
  -H "Content-Type: application/json" \
  -d '{
    "query": "liability",
    "s3_urls": ["s3://my-bucket/contracts/main.pdf"],
    "max_results": 10
  }'
```

---

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message here"
}
```

### Common Errors

| Status | Error | Solution |
|--------|-------|----------|
| `400` | Invalid request | Check request body format |
| `404` | Job not found | Verify job_id is correct |
| `429` | Too many jobs | Wait for jobs to complete |
| `500` | Internal error | Check server logs |

### S3 Errors
| Error | Cause | Solution |
|-------|-------|----------|
| Access Denied | IAM permissions | Add S3 read permissions |
| Bucket Not Found | Wrong bucket name | Check S3_BUCKET env var |
| No Documents | Empty prefix | Verify s3_prefix path |

---

## Rate Limits & Best Practices

### Limits
- Max concurrent jobs: 3 (configurable)
- Max documents per job: 50
- Max document size: 10 MB
- Request timeout: 5 minutes

### Best Practices

1. **Use webhooks** instead of polling for production
2. **Check health** before starting investigations
3. **Handle failures** gracefully with retries
4. **Cache results** for repeated queries
5. **Monitor job counts** to avoid hitting limits

### Retry Strategy
```python
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
async def investigate_with_retry(client, query, s3_prefix):
    return await client.investigate(query, s3_prefix)
```

---

## Quick Start Checklist

- [ ] Set `GEMINI_API_KEY` in `.env`
- [ ] Set `S3_BUCKET` in `.env`
- [ ] Configure AWS credentials (env vars or IAM role)
- [ ] Upload documents to S3 bucket
- [ ] Start server: `python run_server.py`
- [ ] Test health: `curl http://localhost:8000/health`
- [ ] Run first investigation
- [ ] Set up webhook endpoint (optional)
- [ ] Deploy to EC2 (see DEPLOY.md)
```

