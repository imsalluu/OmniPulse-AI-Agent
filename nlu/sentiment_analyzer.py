import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class SentimentAnalyzer:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def analyze_sentiment(self, text):
        """
        Determines the customer's mood: Interested, Frustrated, or Neutral.
        """
        prompt = (
            f"Analyze the sentiment of this customer statement: '{text}'. "
            f"Return a single line: [LABEL] | [SCORE 0.0 to 1.0]. "
            f"Example: Interested | 0.8"
        )
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"[SENTIMENT ERROR] {e}")
            return "Neutral | 0.5"