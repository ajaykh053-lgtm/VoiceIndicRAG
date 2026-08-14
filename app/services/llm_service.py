"""LLM Service using Groq API (Llama 3.1 8B) for ultra-low-latency generation with Indic support."""

import logging
import os
import time
from typing import List, Tuple
from app.core.config import settings
from app.models.schemas import SearchResultItem

logger = logging.getLogger("llm_service")


class LLMService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self._groq_client = None
        self._init_client()

    def _init_client(self):
        key = (settings.GROQ_API_KEY or "").strip("\"' ")
        if key and key != "your_groq_key_here" and len(key) > 5:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=key)
                logger.info("Groq client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self._groq_client = None

    def build_prompt(self, query: str, contexts: List[SearchResultItem], language: str = "hi") -> Tuple[str, str]:
        """Constructs system prompt and context-grounded user prompt."""
        system_prompt = (
            "You are an expert multilingual assistant specializing in Indian languages and English. "
            "Your task is to answer the user's question accurately, concisely, and ONLY based on the provided context passages. "
            "If the context does not contain the answer, politely state that you do not have enough information based on the documents. "
            "Always respond in the same language as the user query (defaulting to Hindi/English as appropriate). "
            "Do NOT make up facts."
        )

        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts.append(f"[संदर्भ {i} | ID: {ctx.chunk_id}]:\n{ctx.text.strip()}")

        context_block = "\n\n".join(formatted_contexts) if formatted_contexts else "कोई संदर्भ उपलब्ध नहीं है।"

        user_content = (
            f"नीचे दिए गए संदर्भों (Contexts) के आधार पर प्रश्न का उत्तर दें:\n\n"
            f"--- संदर्भ शुरू ---\n{context_block}\n--- संदर्भ समाप्त ---\n\n"
            f"प्रश्न (Question): {query}\n\n"
            f"सटीक उत्तर (Answer):"
        )
        return system_prompt, user_content

    def generate_answer(
        self,
        query: str,
        contexts: List[SearchResultItem],
        language: str = "hi"
    ) -> Tuple[str, str, float]:
        """
        Calls Groq Llama 3.1 to generate grounded response.
        Returns: (answer_text, model_name, latency_ms)
        """
        start_t = time.time()
        system_prompt, user_prompt = self.build_prompt(query, contexts, language)

        if self._groq_client is not None:
            try:
                response = self._groq_client.chat.completions.create(
                    model=settings.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=settings.GROQ_MAX_TOKENS,
                    temperature=settings.GROQ_TEMPERATURE,
                )
                answer = response.choices[0].message.content.strip()
                latency_ms = (time.time() - start_t) * 1000.0
                return answer, settings.GROQ_MODEL, latency_ms
            except Exception as e:
                logger.error(f"Groq generation failed: {e}. Using deterministic local synthesis fallback.")

        # Local deterministic synthesis fallback
        latency_ms = (time.time() - start_t) * 1000.0
        if contexts:
            top_ctx = contexts[0].text
            clean_top = top_ctx.split("\n")[-1].strip()
            answer = f"प्राप्त संदर्भ के आधार पर: {clean_top}"
        else:
            answer = "दिए गए संदर्भ में इस प्रश्न की जानकारी उपलब्ध नहीं है।"

        return answer, "local-fallback-synthesizer", max(10.0, latency_ms)


llm_service = LLMService()
