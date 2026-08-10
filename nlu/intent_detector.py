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

    async def get_relevant_tactic(self, cleaned_text: str, conversation_history: str = "", current_phase: str = "Discovery"):
        """
        Multi-turn Hybrid RAG Engine:
        1. Assembles multi-turn query context merging sales stage, recent turns, and current speech.
        2. Embeds the multi-turn context for dense vector similarity.
        3. Performs hybrid keyword relevance scoring (boosting matches containing specific product names or objection triggers).
        """
        if not self.client or not self.index:
            return None

        # Step 1: Assemble Multi-Turn Context Query
        short_history = " ".join(conversation_history.strip().split()[-40:]) if conversation_history else ""
        multi_turn_query = f"Sales Phase: {current_phase}. Call History: {short_history}. Current Customer Utterance: {cleaned_text}"

        try:
            embedding_res = await self.client.embeddings.create(
                input=multi_turn_query,
                model="text-embedding-3-small"
            )
            vector = embedding_res.data[0].embedding

            # Step 2: Query top 5 candidates for hybrid reranking
            results = self.index.query(
                vector=vector,
                top_k=5,
                include_metadata=True,
                namespace=self.namespace
            )
            
            if results and results.get('matches'):
                # Step 3: Hybrid Keyword & Score Reranking
                keywords = [w.lower() for w in cleaned_text.split() if len(w) > 3]
                scored_matches = []

                for match in results['matches']:
                    base_score = match.get('score', 0)
                    text = match.get('metadata', {}).get('text', '')
                    text_lower = text.lower()

                    # Keyword boosting: +0.05 for each key domain word match
                    keyword_boost = sum(0.05 for kw in keywords if kw in text_lower)
                    final_score = base_score + keyword_boost

                    if final_score >= 0.55 and text:
                        scored_matches.append((final_score, text))

                # Sort by hybrid final score descending
                scored_matches.sort(key=lambda x: x[0], reverse=True)

                combined_context = ""
                for score, chunk_text in scored_matches[:3]:
                    combined_context += f"\n--- Knowledge Chunk (Relevance: {score:.2f}) ---\n{chunk_text}\n"

                return combined_context if combined_context else None

        except Exception as e:
            print(f"[HYBRID RAG ERROR] {e}")

        return None
