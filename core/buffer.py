import time

class ChunkBuffer:
    def __init__(self, window_seconds=1.5):
        self.buffer = ""
        self.window_seconds = window_seconds
        self.last_update = time.time()
        # Trigger processing if more than 12 words accumulate, even without silence
        self.word_threshold = 12 

    def add_chunk(self, text):
        """Adds new words to the buffer and updates the timestamp."""
        if text.strip():
            self.buffer += " " + text.strip()
            self.last_update = time.time()

    def is_ready(self):
        """
        Determines if the buffer is ready to be processed by the AI.
        Triggers on either a silence timeout or a word count limit.
        """
        words = self.buffer.strip().split()
        word_count = len(words)
        silence_duration = time.time() - self.last_update
        
        # Condition 1: Sufficient silence after speaking
        # Condition 2: High word count (prevents buffer overflow during long speech)
        if word_count > 0:
            if silence_duration >= self.window_seconds or word_count >= self.word_threshold:
                return True
        return False

    def get_and_clear(self):
        """Returns the accumulated string and resets the buffer."""
        content = self.buffer.strip()
        self.buffer = ""
        return content