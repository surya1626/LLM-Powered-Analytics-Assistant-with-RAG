"""
routes/synthesizer.py
──────────────────
Merges SQL-derived structured insights with RAG-based review analysis
into a single coherent answer for HYBRID queries.

Exports
-------
    Synthesizer.merge(query, sql_summary, rag_summary) -> str
"""

from __future__ import annotations

import os
import logging
from config import LLM, OPENAI_API_KEY

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior e-commerce business analyst.
You will be given:
  (A) A structured data insight derived from a database query.
  (B) A qualitative insight derived from customer review analysis.

Synthesise both into ONE cohesive 3-5 sentence business insight that directly
answers the user's question. Use specific numbers from (A) and customer voice
from (B). Write in clear, professional prose. Do not use bullet points."""


def _call_openai(messages: list) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=LLM,
        messages=messages,
        temperature=0.4,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


class Synthesizer:
    """Merges structured + unstructured insights for HYBRID queries."""

    def merge(self, query: str, sql_summary: str, rag_summary: str) -> str:
        """
        Synthesise SQL and RAG outputs into a single insight.

        Parameters
        ----------
        query       : str   Original user question.
        sql_summary : str   Plain-English summary of SQL result.
        rag_summary : str   Review sentiment / theme analysis.

        Returns
        -------
        str   Synthesised business insight.
        """
        if not sql_summary and not rag_summary:
            return "No data available to synthesise."

        user_content = (
            f"User Question: {query}\n\n"
            f"(A) Structured Data Insight:\n{sql_summary or 'No structured data available.'}\n\n"
            f"(B) Customer Review Insight:\n{rag_summary or 'No review data available.'}\n\n"
            "Please synthesise these into one coherent business insight."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        try:
            return _call_openai(messages)
        except Exception as e:
            logger.error(f"Synthesizer error: {e}")
            return f"{sql_summary}\n\n{rag_summary}"
