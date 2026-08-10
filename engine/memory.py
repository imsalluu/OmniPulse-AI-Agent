import redis
import os
import json
from dotenv import load_dotenv

load_dotenv()

class CallMemory:
    def __init__(self):
        # In-memory fallbacks when Redis is not available
        self._local_transcripts = {}
        self._local_states = {}
        self.use_redis = False
        
        redis_host = os.getenv("REDIS_HOST")
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_password = os.getenv("REDIS_PASSWORD")

        if redis_host:
            try:
                port = int(redis_port) if str(redis_port).isdigit() else 6379
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=port,
                    password=redis_password,
                    decode_responses=True,
                    socket_connect_timeout=2
                )
                # Test connection ping
                self.redis_client.ping()
                self.use_redis = True
            except Exception as e:
                print(f"[MEMORY WARNING] Redis connection unavailable ({e}). Using local in-memory store.")
                self.use_redis = False

    def update_transcript(self, call_id: str, new_text: str):
        """Append new text to the conversation history in Redis or local memory."""
        if self.use_redis:
            try:
                key = f"transcript:{call_id}"
                self.redis_client.append(key, f" {new_text}")
                self.redis_client.expire(key, 3600)
                return
            except Exception as e:
                print(f"[MEMORY ERROR] Redis update failed: {e}. Falling back to local memory.")
                self.use_redis = False

        self._local_transcripts[call_id] = self._local_transcripts.get(call_id, "") + f" {new_text}"

    def get_full_history(self, call_id: str) -> str:
        """Retrieve the entire conversation context."""
        if self.use_redis:
            try:
                return self.redis_client.get(f"transcript:{call_id}") or ""
            except Exception:
                self.use_redis = False

        return self._local_transcripts.get(call_id, "")

    def set_state(self, call_id: str, state: str):
        """Track the current phase (Hook, Discovery, Objection, Closing)."""
        if self.use_redis:
            try:
                self.redis_client.set(f"state:{call_id}", state)
                return
            except Exception:
                self.use_redis = False

        self._local_states[call_id] = state

    def get_state(self, call_id: str) -> str:
        if self.use_redis:
            try:
                return self.redis_client.get(f"state:{call_id}") or "Discovery"
            except Exception:
                self.use_redis = False

        return self._local_states.get(call_id, "Discovery")