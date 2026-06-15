# RAG System Evaluation

Automated daily evaluation pipeline for the RAG chatbot. Fetches traces from Langfuse, scores them using GPT-4o-mini, and sends a summary report to Google Sheet.

## How It Works

1. Fetches the last 24 hours of traces from Langfuse
2. Filters for MSU student assistant conversations
3. Skips fallback responses
4. Evaluates each trace on 3 criteria using GPT-4o-mini
5. Sends daily average scores to n8n webhook

## Metrics

| Metric | Description |
|---|---|
| Faithfulness | Are all claims grounded in the retrieved context? |
| Answer Relevance | Does the response actually answer the student's question? |
| Correctness | Is the information factually correct based on context? |
| Overall | Average of the three scores above |

## Setup

1. Clone the repo
2. Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
N8N_WEBHOOK_URL=your_n8n_webhook_url
LOOKBACK_DAYS=1
TRACE_NAME=your_trace_name
```

3. Start the services:

```bash
docker compose up -d --build
```

## Schedule

Runs every day at **midnight Malaysian time (MYT / UTC+8)**.

## Stack

- Python 3.12
- Langfuse — trace fetching
- OpenAI GPT-4o-mini — evaluation
- Ofelia — job scheduling
- n8n & Google Sheets — report delivery

## Project Structure

```
.
├── evaluator.py        # Main evaluation script
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env               # Environment variables
```