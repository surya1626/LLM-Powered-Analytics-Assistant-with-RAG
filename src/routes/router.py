from __future__ import annotations

import os
import logging
import re
from config import LLM, OPENAI_API_KEY

logger = logging.getLogger(__name__)

VALID_ROUTES = {"SQL", "RAG", "HYBRID"}

SYSTEM_PROMPT = """You are a query router for an e-commerce analytics assistant.

You must classify each user query into EXACTLY ONE of three routing classes:

SQL    — The query asks for aggregated or filtered structured data that can be answered
          by running a SQL query on the orders, products, payments, sellers, or customers tables.
          Examples: totals, counts, averages, rankings, comparisons of numeric values.

RAG    — The query asks for qualitative insights, opinions, sentiment, or themes that come
          from reading unstructured customer review text.
          Examples: complaints, sentiments, what customers say, review themes.

HYBRID — The query requires BOTH structured data (SQL) AND qualitative review analysis (RAG).
          Examples: "Which category has worst reviews and what do customers say?"

Reply with ONLY one word: SQL, RAG, or HYBRID. No explanation, no punctuation.
"""


def _call_openai(messages: list) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=LLM,
        messages=messages,
        temperature=0,
        max_tokens=10,
    )
    return resp.choices[0].message.content.strip()


class QueryRouter:
    """LLM-based query router."""

    def classify(self, query: str) -> str:
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": query},
            ]
            route = _call_openai(messages).upper()
            route = re.sub(r"[^A-Z]", "", route)   # strip stray punctuation
            if route not in VALID_ROUTES:
                route = self._heuristic_route(query)
            logger.info(f"Route: {route}  Query: {query[:60]}…")
            return route
        except Exception as e:
            logger.warning(f"Router LLM call failed ({e}), using heuristics.")
            return self._heuristic_route(query)

    @staticmethod
    def _heuristic_route(query: str) -> str:
        """Simple keyword-based fallback router."""
        q = query.lower()
        rag_keywords  = {"review", "complain", "sentiment", "opinion", "customer say",
                         "feedback", "complaint", "experience", "feel", "mention",
                         "theme", "issue", "problem", "suggest"}
        sql_keywords  = {"top", "total", "average", "count", "revenue", "sale",
                         "order", "product", "seller", "payment", "delivery",
                         "how many", "list", "show", "rank", "compare", "month",
                         "year", "state", "city", "highest", "lowest"}

        has_rag = any(k in q for k in rag_keywords)
        has_sql = any(k in q for k in sql_keywords)

        if has_rag and has_sql:
            return "HYBRID"
        if has_rag:
            return "RAG"
        return "SQL"
