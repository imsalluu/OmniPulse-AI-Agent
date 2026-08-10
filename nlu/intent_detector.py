import os
from openai import AsyncOpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class NLUManager:
    def __init__(self):
        # Initialize OpenAI and Pinecone clients safely
        openai_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index = os.getenv("PINECONE_INDEX_NAME", "sales-playbook")

        self.client = AsyncOpenAI(api_key=openai_key) if openai_key else None
        
        self.index = None
        self.namespace = os.getenv("PINECONE_NAMESPACE", "sales-playbook-v1")
        if pinecone_key and pinecone_index:
            try:
                self.pc = Pinecone(api_key=pinecone_key)
                self.index = self.pc.Index(pinecone_index)
            except Exception as e:
                print(f"[PINECONE WARNING] Could not initialize Pinecone index ({e}). Vector RAG disabled.")

    async def contextual_correction(self, raw_text: str) -> str:
        """
        Initial NLP pass to clean the transcript based on high-level patterns.
        """
        if not self.client:
            return raw_text.strip()

        prompt = (
            f"Context: Insurance Sales & Tech support. Raw Transcript: '{raw_text}'. "
            f"Task: Correct technical misheard words. "
            f"Example: 'psp' -> 'premium' (if sales) or 'PHP' (if tech). "
            f"Return ONLY the corrected sentence."
        )
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[NLU CORRECTION ERROR] {e}")
            return raw_text.strip()

    async def get_relevant_tactic(self, cleaned_text: str):
        """
        Retrieves multiple context chunks from Pinecone to provide the AI 
        with both specific tactics and global intelligence rules.
        """
        if not self.client or not self.index:
            return None

        try:
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
            
            if results and results.get('matches'):
                combined_context = ""
                for match in results['matches']:
                    if match.get('score', 0) > 0.60 and 'metadata' in match:
                        combined_context += f"\n--- Knowledge Chunk ---\n{match['metadata'].get('text', '')}\n"
                
                return combined_context if combined_context else None
        except Exception as e:
            print(f"[PINECONE RAG ERROR] {e}")

        return None