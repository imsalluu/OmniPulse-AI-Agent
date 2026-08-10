import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

class EntityExtractor:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=openai_key) if openai_key else None

    async def extract_entities(self, text):
        """
        Extracts Names, Dates, Amounts, and Products from sales transcript.
        """
        if not self.client:
            return {}

        prompt = (
            f"Analyze this insurance sales transcript: '{text}'. "
            f"Extract key entities: [PERSON, AMOUNT, DATE, PRODUCT, SPOUSE_NAME]. "
            f"Return only a JSON object."
        )
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            import json
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[NER ERROR] {e}")
            return {}