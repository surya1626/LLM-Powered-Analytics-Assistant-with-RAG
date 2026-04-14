"""
routes/sentiment.py
────────────────
Performs RAG-based sentiment analysis and theme extraction on
retrieved review chunks.

Exports
-------
    SentimentAnalyser.analyse(query, chunks) -> dict
"""

from __future__ import annotations

import os
import json
import logging
import re
from typing import List, Dict
from config import LLM, OPENAI_API_KEY

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior customer-experience analyst for an e-commerce company.
You will be given a user question and a set of real customer review excerpts.

Your task:
1. Answer the user's question using ONLY the provided review excerpts.
2. Classify the overall sentiment as one of: positive, negative, mixed.
3. List the top 3 complaint or praise themes as short labels (2-4 words each).
4. Keep your answer concise and grounded — do not invent facts not in the excerpts.

Respond in this exact JSON format (no markdown fences):
{
  "answer": "<your 3-5 sentence answer>",
  "sentiment": "<positive|negative|mixed>",
  "themes": ["<theme1>", "<theme2>", "<theme3>"]
}
"""


def _call_openai(messages: list, model: str = LLM) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=LLM,
        messages=messages,
        temperature=0.3,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


def _format_chunks(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[Review {i}] (order {c['order_id']}):\n{c['text']}")
    return "\n\n".join(parts)


class SentimentAnalyser:
    """RAG-based sentiment and theme extractor."""

    def analyse(self, query: str, chunks: List[Dict]) -> dict:
        """
        Analyse review chunks in the context of the user's question.

        Parameters
        ----------
        query  : str          The user's original question.
        chunks : list[dict]   Retrieved review chunks (text, order_id, score).

        Returns
        -------
        dict with keys: answer, sentiment, themes
        """
        if not chunks:
            return {
                "answer"   : "No relevant reviews were found for this query.",
                "sentiment": "mixed",
                "themes"   : [],
            }

        context = _format_chunks(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"User Question: {query}\n\nReview Excerpts:\n{context}"},
        ]

        raw = _call_openai(messages)
        logger.info(f"Sentiment LLM raw: {raw[:200]}")

        try:
            # Strip accidental markdown fences
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            result = json.loads(clean)
            # Normalise keys
            return {
                "answer"   : result.get("answer", raw),
                "sentiment": result.get("sentiment", "mixed").lower(),
                "themes"   : result.get("themes", []),
            }
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON from sentiment LLM. Returning raw text.")
            return {
                "answer"   : raw,
                "sentiment": "mixed",
                "themes"   : [],
            }
