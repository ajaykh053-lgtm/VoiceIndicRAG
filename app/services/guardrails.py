"""Guardrails Service for:
1. Unsafe input detection (harmful keywords, injection attacks).
2. Off-topic query detection.
3. Hallucination / groundedness verification between retrieved chunks and answer.
"""

import logging
import re
import time
from typing import List, Tuple
from app.models.schemas import GuardrailStatus, SearchResultItem

logger = logging.getLogger("guardrails")


class GuardrailsService:
    def __init__(self):
        # Unsafe patterns (profanity, prompt injection, harmful instructions)
        self.unsafe_patterns = [
            r"(?i)\b(ignore previous instructions|system prompt|bypass filter|drop table|exec\(|eval\()\b",
            r"(?i)\b(bomb|weapon|hack|exploit|malware|credit card|ssn)\b",
        ]
        # Off-topic patterns for out-of-scope queries
        self.off_topic_patterns = [
            r"(?i)\b(crypto prediction|gamble|betting rate|pirate movie)\b"
        ]

    def check_input_safety(self, query: str) -> Tuple[bool, bool, str]:
        """
        Validates user query.
        Returns: (is_safe, is_on_topic, flag_reason)
        """
        clean_q = query.strip()
        if not clean_q:
            return False, False, "खाली प्रश्न (Empty query)"

        for pat in self.unsafe_patterns:
            if re.search(pat, clean_q):
                return False, True, "सुरक्षा नीति उल्लंघन (Unsafe query detected)"

        for pat in self.off_topic_patterns:
            if re.search(pat, clean_q):
                return True, False, "अनुपयुक्त विषय (Off-topic query)"

        return True, True, ""

    def verify_hallucination(
        self,
        answer: str,
        contexts: List[SearchResultItem]
    ) -> Tuple[bool, float]:
        """
        Calculates groundedness score by checking key noun/token overlap between answer and retrieved passages.
        Returns: (is_hallucination, groundedness_score)
        """
        if not contexts:
            return True, 0.0

        ans_words = set(re.findall(r'\w+', answer.lower()))
        if not ans_words:
            return False, 1.0

        ctx_words = set()
        for c in contexts:
            ctx_words.update(re.findall(r'\w+', c.text.lower()))

        # Remove very common short stopwords
        stopwords = {"है", "का", "की", "के", "में", "से", "पर", "और", "the", "is", "of", "and", "to", "in", "a"}
        content_ans_words = ans_words - stopwords

        if not content_ans_words:
            return False, 1.0

        overlap = content_ans_words.intersection(ctx_words)
        groundedness = len(overlap) / len(content_ans_words)

        # If groundedness is extremely low (< 0.15), flag possible hallucination
        is_hallucinated = groundedness < 0.15
        return is_hallucinated, float(min(1.0, groundedness))


guardrails_service = GuardrailsService()
