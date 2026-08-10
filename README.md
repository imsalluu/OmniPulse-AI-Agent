# InsureFlow AI Agent

An advanced, real-time AI Agent & Co-pilot for Insurance Sales & Technical Support calls.

## Overview
InsureFlow AI Agent processes real-time audio transcripts via WebSockets, executes a parallel NLU pipeline (Contextual Correction, Sentiment Analysis, NER, Pinecone RAG Vector Search), triggers automated tools (Function Calling), and generates real-time advice and coaching for agents while tracking sales conversation phases in Redis.

## Features
- **Real-Time Streaming & Buffer:** Ingests live audio transcript chunks over WebSocket with silence/word-count thresholding.
- **Parallel NLU Pipeline:** Runs OpenAI GPT-4o-mini and Pinecone vector search simultaneously to reduce latency.
- **Phonetic & Contextual Correction:** Automatically fixes domain-specific misheard terms (e.g., `psp` -> `premium` or `PHP`).
- **Sentiment & Entity Extraction:** Detects customer mood (`Interested`, `Frustrated`, `Neutral`) and extracts entities (`PERSON`, `AMOUNT`, `DATE`, `PRODUCT`, `SPOUSE_NAME`).
- **Multi-turn Hybrid RAG Search:** Combines multi-turn conversation context, sales phase state, and keyword relevance scoring to retrieve precise playbook tactics from Pinecone.
- **Agent Action Triggers (Function Calling):** Automatically triggers insurance tools mid-call (Live Quote Calculation, SMS Document Dispatch, CRM Profile Lookup) via OpenAI Function Calling.
- **Cognitive Response Generator:** Incorporates executed tool results into structured coaching advice (< 40 words) tailored to Sales (4-Block) or Technical (3-Step) domains.
- **Sales State Machine & Memory:** Tracks sales phase (`Hook`, `Discovery`, `Objection`, `Closing`) and persists session history in Redis Cloud with local in-memory fallback.
- **Live Dashboard Broadcast:** Streams insights and executed tool results in real-time to front-end agent dashboards via `/transcript-stream` WebSocket.

## Architecture
```
[WebSocket Client / Orchestrator] 
           │
           ▼
     [ChunkBuffer] 
           │
           ▼
[Parallel NLU Pipeline (asyncio.gather)]
├── 1. Contextual Correction (OpenAI GPT-4o-mini)
├── 2. Sentiment Analyzer (OpenAI)
├── 3. Entity Extractor (NER - OpenAI)
└── 4. Multi-turn Hybrid RAG (Pinecone + Keyword Rerank)
           │
           ▼
[Agent Action Engine (Function Calling)]
├── calculate_insurance_quote
├── send_policy_document_sms
└── fetch_customer_crm_profile
           │
           ▼
 [Cognitive Response Engine] ──► [Sales State Machine] ──► [Redis Memory]
           │
           ▼
 [FastAPI WebSocket Broadcast] ──► [Front-End Agent Dashboard]
```

## Tech Stack
- **Framework:** FastAPI, Uvicorn, Python Asyncio
- **WebSockets:** Python `websockets` library
- **LLM, Function Calling & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
- **Vector Database:** Pinecone
- **In-Memory Store:** Redis
- **Configuration:** Python-dotenv

## Setup & Running

1. **Clone the repository:**
   ```bash
   git clone https://github.com/imsalluu/ai_agent.git
   cd ai_agent
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file with your credentials:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=sales-playbook
   PINECONE_NAMESPACE=sales-playbook-v1
   REDIS_HOST=your_redis_host
   REDIS_PORT=6379
   REDIS_PASSWORD=your_redis_password
   ORCHESTRATOR_WS_URL=ws://localhost:8000/stream
   AGENT_PORT=5051
   CHUNK_BUFFER_SECONDS=1.5
   ```

4. **Start the Agent Server:**
   ```bash
   python main.py
   ```
