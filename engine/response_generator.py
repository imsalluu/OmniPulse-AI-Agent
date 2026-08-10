from openai import AsyncOpenAI
import os

class ResponseEngine:
    def __init__(self):
        openai_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=openai_key) if openai_key else None

    async def generate_advice(self, customer_input, knowledge_context=None, executed_actions=None):
        """
        Cognitive Response Engine:
        1. Detects Domain (Sales vs Tech) using retrieved Pinecone Rules.
        2. Incorporates results from Action Engine tool executions (if any).
        3. Applies phonetic mapping (psp/php) based on that Domain.
        4. Formats Sales advice in 4-Blocks or Technical advice in 3-Steps.
        """
        if not self.client:
            return "[Validation] Understood. [Pivot] Let us discuss your insurance needs. [Knowledge] We offer tailored wealth and protection plans. [Soft Close] Would you like a quote?"

        # If no knowledge was found in Pinecone, we use a default fallback instruction
        if not knowledge_context:
            knowledge_context = (
                "No direct matches found in database. "
                "Default Behavior: Identify if the topic is about wealth/protection. "
                "If not, acknowledge and pivot back to insurance discovery."
            )

        actions_context = ""
        if executed_actions:
            import json
            actions_context = f"\nExecuted Tool Results:\n{json.dumps(executed_actions, indent=2)}\n(IMPORTANT: Use these exact figures/status in your advice if applicable!)\n"

        # The core logic prompt for the AI Agent
        prompt = (
            f"You are a Senior Wealth Advisor and Lead Systems Engineer.\n"
            f"Customer Speech: '{customer_input}'.\n\n"
            f"Retrieved Intelligence Rules & Tactics:\n{knowledge_context}\n"
            f"{actions_context}\n"
            f"STRICT INSTRUCTIONS:\n"
            f"1. Detect Domain: Use the 'Dynamic Domain Detection' rule to set context.\n"
            f"2. Correct Phonetics: If 'psp' or 'php' is used, map it based on the detected domain.\n"
            f"3. Format Response:\n"
            f"   - IF [SALES]: Use the 4-Block Universal Template ([Validation], [Pivot], [Knowledge], [Soft Close]).\n"
            f"   - IF [TECHNICAL]: Provide [Analysis], [Technical Insight], [Verification Question].\n"
            f"4. Constraint: Keep the advice for the human agent under 40 words total. "
            f"Be authoritative yet deeply caring."
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a highly intelligent, context-aware sales and tech coach."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3 # Low temperature for consistent rule-following
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating advice: {str(e)}"