"""
routes/nl_to_sql.py
────────────────
Converts a natural-language question into SQLite SQL using an LLM,
executes it against the Olist database, summarises the result in
plain English, and optionally renders a Plotly chart.

Pipeline
--------
    User Question
        │
        ▼
    LLM (schema-aware system prompt)  →  SQL string
        │
        ▼
    SQLite execute (pandas.read_sql)  →  DataFrame
        │
        ▼
    LLM (summarise DataFrame)         →  plain-English insight
        │
        ▼
    LLM (pick chart type)             →  Plotly figure  (optional)

Exports
-------
    NLtoSQL(db_path)
    NLtoSQL.execute(question) -> dict
        keys: sql, df, summary, chart, error
"""

from __future__ import annotations

import os
import re
import logging
import sqlite3
from pathlib import Path
from config import LLM, OPENAI_API_KEY
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Schema helper
# ══════════════════════════════════════════════════════════════════════════════

def _get_schema(db_path: Path) -> str:
    """
    Read every table + column name from SQLite and return a
    compact schema string to inject into the LLM system prompt.
    """
    conn = sqlite3.connect(db_path)
    lines = []
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    for (tname,) in tables:
        cols = conn.execute(f"PRAGMA table_info({tname})").fetchall()
        col_str = ", ".join(c[1] for c in cols)
        lines.append(f"TABLE {tname}({col_str})")

    conn.close()

    lines.append(
        "\nVIEW analytics_view -- pre-joined view combining: "
        "orders, order_items, products, category_translation, "
        "sellers, customers, payments, order_reviews. "
        "Key columns: order_id, order_status, order_year, order_month, "
        "order_weekday, delivery_days, estimated_days, is_late_delivery, "
        "price, freight_value, item_total, category_english, "
        "seller_state, customer_state, payment_type, payment_value, "
        "review_score, sentiment_label, has_review_text."
    )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# LLM call
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(messages: list, temperature: float = 0) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    resp = client.chat.completions.create(
        model=LLM,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Generate SQL
# ══════════════════════════════════════════════════════════════════════════════

SQL_SYSTEM_PROMPT = """You are an expert SQLite query writer for the Olist Brazilian e-commerce dataset.

DATABASE SCHEMA:
{schema}

STRICT RULES:
1. Output ONLY the raw SQL query -- no markdown, no explanation, no backticks, no comments.
2. Always prefer analytics_view for queries needing data from multiple tables.
3. Use individual tables only when the query is about one specific table.
4. Always add LIMIT 50 unless the user asks for all rows.
5. Use LOWER() for all string comparisons.
6. Date columns are stored as text 'YYYY-MM-DD HH:MM:SS' -- use SUBSTR() or strftime().
7. For revenue queries use SUM(price) or SUM(item_total).
8. NEVER use DROP, DELETE, INSERT, UPDATE, or ALTER.
9. If the question cannot be answered with SQL, output exactly: CANNOT_ANSWER
"""


def _generate_sql(question: str, schema: str) -> str:
    system = SQL_SYSTEM_PROMPT.format(schema=schema)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": f"Write a SQL query to answer: {question}"},
    ]
    raw = _call_llm(messages, temperature=0)
    sql = re.sub(r"```(?:sql)?|```", "", raw).strip()
    return sql


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Execute SQL
# ══════════════════════════════════════════════════════════════════════════════

def _execute_sql(sql: str, db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Summarise result
# ══════════════════════════════════════════════════════════════════════════════

def _summarise(question: str, df: pd.DataFrame) -> str:
    if df.empty:
        return "The query returned no results."

    preview = df.head(10).to_markdown(index=False)
    messages = [
        {"role": "system", "content": "You are a concise e-commerce data analyst."},
        {"role": "user", "content": (
            f"User question: {question}\n\n"
            f"Query result (first {min(10, len(df))} of {len(df)} rows):\n{preview}\n\n"
            "Write a 2-3 sentence plain-English business insight from this data. "
            "Mention specific numbers. Be concise."
        )},
    ]
    return _call_llm(messages, temperature=0.3)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Auto chart
# ══════════════════════════════════════════════════════════════════════════════

def _pick_chart_type(question: str, df: pd.DataFrame) -> str:
    if df.empty or len(df.columns) < 2:
        return "none"

    col_info = {col: str(df[col].dtype) for col in df.columns}
    messages = [
        {"role": "system", "content": "You pick the best chart type for data."},
        {"role": "user", "content": (
            f"Question: {question}\n"
            f"DataFrame columns and dtypes: {col_info}\n"
            f"Number of rows: {len(df)}\n\n"
            "Pick the single best chart from: bar, line, pie, scatter, none.\n"
            "Reply with ONE word only."
        )},
    ]
    choice = _call_llm(messages, temperature=0).lower().strip()
    return choice if choice in ("bar", "line", "pie", "scatter") else "none"


def _render_chart(chart_type: str, question: str, df: pd.DataFrame):
    if chart_type == "none" or df.empty:
        return None

    import plotly.express as px

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    try:
        if chart_type == "bar" and cat_cols and num_cols:
            return px.bar(
                df, x=cat_cols[0], y=num_cols[0], title=question,
                color=num_cols[0], color_continuous_scale="Blues",
            )
        elif chart_type == "line":
            x = cat_cols[0] if cat_cols else df.columns[0]
            y = num_cols[0] if num_cols else df.columns[1]
            return px.line(df, x=x, y=y, title=question, markers=True,
                           color_discrete_sequence=["#0f3460"])
        elif chart_type == "pie" and cat_cols and num_cols:
            return px.pie(df, names=cat_cols[0], values=num_cols[0], title=question)
        elif chart_type == "scatter" and len(num_cols) >= 2:
            return px.scatter(df, x=num_cols[0], y=num_cols[1], title=question,
                              color_discrete_sequence=["#e94560"])
    except Exception as e:
        logger.warning(f"Chart render failed ({chart_type}): {e}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class NLtoSQL:
    """
    Natural-language to SQL engine.

    Usage
    -----
        nl2sql = NLtoSQL(db_path)
        result = nl2sql.execute("Top 5 categories by revenue?")

        result["sql"]      # generated SQL string
        result["df"]       # pandas DataFrame
        result["summary"]  # plain-English insight
        result["chart"]    # Plotly figure (or None)
        result["error"]    # error message string (or None)
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._schema: str | None = None

    def _get_schema(self) -> str:
        if self._schema is None:
            self._schema = _get_schema(self.db_path)
        return self._schema

    def execute(self, question: str) -> dict:
        """
        Run the full NL -> SQL -> execute -> summarise -> chart pipeline.

        Returns dict with keys: sql, df, summary, chart, error
        """
        result = {
            "sql"    : None,
            "df"     : None,
            "summary": "",
            "chart"  : None,
            "error"  : None,
        }

        try:
            # Step 1: Generate SQL
            logger.info(f"Generating SQL for: {question}")
            sql = _generate_sql(question, self._get_schema())
            result["sql"] = sql
            logger.info(f"Generated SQL:\n{sql}")

            if sql.strip().upper() == "CANNOT_ANSWER":
                result["error"] = "This question cannot be answered with structured data."
                return result

            # Step 2: Execute
            df = _execute_sql(sql, self.db_path)
            result["df"] = df
            logger.info(f"Query returned {len(df)} rows, {len(df.columns)} columns.")

            # Step 3: Summarise
            result["summary"] = _summarise(question, df)

            # Step 4: Auto chart
            chart_type = _pick_chart_type(question, df)
            result["chart"] = _render_chart(chart_type, question, df)
            logger.info(f"Chart type chosen: {chart_type}")

        except Exception as e:
            logger.error(f"NLtoSQL error: {e}")
            result["error"] = str(e)

        return result