import asyncio
import os
import json
import uvicorn
from fastapi import FastAPI, WebSocket
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# --- Modular Imports ---
from core.buffer import ChunkBuffer
from core.orchestrator_listener import OrchestratorListener
from nlu.intent_detector import NLUManager
from nlu.ner_extractor import EntityExtractor
from nlu.sentiment_analyzer import SentimentAnalyzer
from engine.memory import CallMemory
from engine.response_generator import ResponseEngine
from engine.state_machine import SalesStateMachine

# Load environment variables
load_dotenv()

# --- Component Initialization ---
memory = CallMemory()
nlu = NLUManager()
ner = EntityExtractor()
sentiment = SentimentAnalyzer()
response_gen = ResponseEngine()
fsm = SalesStateMachine()
buffer = ChunkBuffer(window_seconds=float(os.getenv("CHUNK_BUFFER_SECONDS", 1.5)))

# Dashboard Subscriber List
ai_subscribers = []

async def process_intelligence(raw_text):
    """
    THE PARALLEL NLU PIPELINE:
    Executes multiple AI tasks at once to minimize latency.
    """
    call_id = "test_call_123" 
    print(f"\n[AI BRAIN] Analyzing Flow: '{raw_text}'")

    try:
        # Step 1: Run NLU tasks in Parallel using asyncio.gather
        # This reduces round-trip time to OpenAI/Pinecone significantly
        tasks = [
            nlu.contextual_correction(raw_text),
            sentiment.analyze_sentiment(raw_text),
            ner.extract_entities(raw_text),
            nlu.get_relevant_tactic(raw_text) # Pinecone Search
        ]
        
        # Execute all tasks simultaneously
        cleaned_text, mood_result, entities, knowledge_context = await asyncio.gather(*tasks)

        # Step 2: Generate response using the 4-Block Template
        advice = await response_gen.generate_advice(cleaned_text, knowledge_context)
        
        if advice:
            # Step 3: Determine Phase and Update Memory
            current_state = memory.get_state(call_id)
            new_phase = fsm.determine_phase(cleaned_text, current_state)
            memory.set_state(call_id, new_phase)
            memory.update_transcript(call_id, cleaned_text)

            # Step 4: Prepare JSON payload for the Dashboard/UI
            payload = {
                "type": "ai_insight",
                "phase": new_phase,
                "mood": mood_result.split('|')[0].strip(),
                "entities": entities,
                "advice": advice,
                "raw_text": cleaned_text
            }
            
            # Step 5: Broadcast to all connected Dashboards
            for ws in ai_subscribers[:]:
                try:
                    await ws.send_json(payload)
                except Exception:
                    ai_subscribers.remove(ws)
            
            print(f"--- AI ADVICE BROADCASTED SUCCESSFULLY ---")

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")

# --- LIFESPAN MANAGEMENT ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to Orchestrator via WebSocket
    orchestrator_url = os.getenv("ORCHESTRATOR_WS_URL")
    print(f"[SYSTEM] Establishing link with Orchestrator: {orchestrator_url}")
    
    listener = OrchestratorListener(buffer, process_intelligence)
    listener_task = asyncio.create_task(listener.start())
    
    yield
    
    # Shutdown: Cleanup
    listener_task.cancel()
    print("[SYSTEM] AI Agent shutting down...")

# Initialize FastAPI App
app = FastAPI(lifespan=lifespan)

# --- DASHBOARD WEBSOCKET ENDPOINT ---
@app.websocket("/transcript-stream")
async def transcript_stream_endpoint(websocket: WebSocket):
    """
    WebSocket server for the Front-end Dashboard.
    Usage: ws://localhost:5051/transcript-stream
    """
    await websocket.accept()
    ai_subscribers.append(websocket)
    print(f"[SYSTEM] Dashboard connected. Active Subscribers: {len(ai_subscribers)}")
    try:
        while True:
            # Keep the connection open to receive broadcasted advice
            await asyncio.sleep(3600)
    except Exception:
        if websocket in ai_subscribers:
            ai_subscribers.remove(websocket)
        print("[SYSTEM] Dashboard disconnected.")

if __name__ == "__main__":
    agent_port = int(os.getenv("AGENT_PORT", 5051))
    uvicorn.run(app, host="0.0.0.0", port=agent_port)