import redis
import os
import json
from dotenv import load_dotenv

load_dotenv()

class CallMemory:
    def __init__(self):
        # Connect to Redis Cloud
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            password=os.getenv("REDIS_PASSWORD"),
            decode_responses=True
        )

    def update_transcript(self, call_id, new_text):
        """Append new text to the conversation history in Redis."""
        key = f"transcript:{call_id}"
        self.redis_client.append(key, f" {new_text}")
        # Set expiry to 1 hour to clean up old calls
        self.redis_client.expire(key, 3600)

    def get_full_history(self, call_id):
        """Retrieve the entire conversation context."""
        return self.redis_client.get(f"transcript:{call_id}") or ""

    def set_state(self, call_id, state):
        """Track the current phase (Hook, Discovery, Objection, Closing)."""
        self.redis_client.set(f"state:{call_id}", state)

    def get_state(self, call_id):
        return self.redis_client.get(f"state:{call_id}") or "Discovery"