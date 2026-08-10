import os
from openai import AsyncOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class NLUManager:
    def __init__(self):
        # Initialize OpenAI and Pinecone clients
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        self.namespace = os.getenv("PINECONE_NAMESPACE", "sales-playbook-v1")

    async def contextual_correction(self, raw_text):
        """
        Initial NLP pass to clean the transcript based on high-level patterns.
        """
        prompt = (
            f"Context: Insurance Sales & Tech support. Raw Transcript: '{raw_text}'. "
            f"Task: Correct technical misheard words. "
            f"Example: 'psp' -> 'premium' (if sales) or 'PHP' (if tech). "
            f"Return ONLY the corrected sentence."
        )
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    async def get_relevant_tactic(self, cleaned_text):
        """
        Retrieves multiple context chunks from Pinecone to provide the AI 
        with both specific tactics and global intelligence rules.
        """
        embedding_res = await self.client.embeddings.create(
            input=cleaned_text,
            model="text-embedding-3-small"
        )
        vector = embedding_res.data[0].embedding

        # Query top 3 results to capture both Tactic and Intelligence Rules
        results = self.index.query(
            vector=vector,
            top_k=3,
            include_metadata=True,
            namespace=self.namespace
        )
        
        if results['matches']:
            # Combine all retrieved text pieces into a single context block
            combined_context = ""
            for match in results['matches']:
                if match['score'] > 0.60:
                    combined_context += f"\n--- Knowledge Chunk ---\n{match['metadata']['text']}\n"
            
            return combined_context if combined_context else None
        
        return None