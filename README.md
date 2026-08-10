# InsureFlow AI Agent

An advanced, real-time AI Agent & Co-pilot for Insurance Sales & Technical Support calls.

## Overview
InsureFlow AI Agent processes real-time audio transcripts via WebSockets, executes a parallel NLU pipeline (Contextual Correction, Sentiment Analysis, NER, Pinecone RAG Vector Search), and generates real-time advice and coaching for agents while tracking sales conversation phases in Redis.

## Features
- **Real-Time Streaming & Buffer:** Ingests live audio transcript chunks over WebSocket with silence/word-count thresholding.
- **Parallel NLU Pipeline:** Runs OpenAI GPT-4o-mini and Pinecone vector search simultaneously to reduce latency.
- **Phonetic & Contextual Correction:** Automatically fixes domain-specific misheard terms (e.g., `psp` -> `premium` or `PHP`).
- **Sentiment & Entity Extraction:** Detects customer mood (`Interested`, `Frustrated`, `Neutral`) and extracts entities (`PERSON`, `AMOUNT`, `DATE`, `PRODUCT`, `SPOUSE_NAME`).
- **Cognitive Response Generator:** Generates structured coaching advice (< 40 words) tailored to Sales (4-Block) or Technical (3-Step) domains.
- **Sales State Machine & Memory:** Tracks sales phase (`Hook`, `Discovery`, `Objection`, `Closing`) and persists session history in Redis Cloud.
- **Live Dashboard Broadcast:** Streams insights in real-time to front-end agent dashboards via `/transcript-stream` WebSocket.

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
└── 4. Knowledge Retriever (Pinecone Vector RAG)
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
- **LLM & Embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`)
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
