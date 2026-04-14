"""
LLM-Powered Analytics Assistant with RAG
Streamlit UI — app.py
"""

import os, traceback
import streamlit as st
import pandas as pd
from config import DATA_DIR, OPENAI_API_KEY,DB_DIR
from ingest import OlistDataLoader

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="Olist Analytics Assistant", page_icon="🛒",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1a1a2e,#0f3460);padding:2rem 2.5rem;
  border-radius:14px;margin-bottom:1.5rem;text-align:center;color:white;}
.main-header h1{font-size:2.2rem;margin:0 0 .4rem;}
.main-header p{opacity:.85;font-size:1.05rem;margin:0;}
.route-badge{display:inline-block;padding:3px 14px;border-radius:20px;
  font-size:12px;font-weight:700;letter-spacing:.05em;margin-bottom:8px;}
.badge-sql{background:#dbeafe;color:#1e40af;}
.badge-rag{background:#d1fae5;color:#065f46;}
.badge-hybrid{background:#ede9fe;color:#5b21b6;}
</style>
""", unsafe_allow_html=True)


def setup_database():
    loader = OlistDataLoader(
        data_dir=DATA_DIR,
        db_path=DB_DIR
    )
    return loader.run()

@st.cache_resource(show_spinner="🔧 Loading database and FAISS index…")
def load_system():
    from rag.embedder import build_faiss_index
    from rag.retriever import Retriever
    from routes.nl_to_sql import NLtoSQL
    from routes.router import QueryRouter
    from routes.synthesizer import Synthesizer
    from routes.sentiment import SentimentAnalyser
    db_path = setup_database()
    index, chunks = build_faiss_index(db_path)

    return {
            "retriever": Retriever(index, chunks), 
            "nl2sql": NLtoSQL(db_path),
            "router": QueryRouter(), 
            "synth": Synthesizer(), 
            "sentiment": SentimentAnalyser()
            }


with st.sidebar:
    st.markdown("## 🛒 Olist Assistant")
    st.divider()
    st.subheader("📋 Example Queries")
    examples = {
        "📊 SQL": ["Top 5 product categories by revenue?","Monthly order counts for 2017",
                   "Sellers with highest average review score","Average delivery time by state"],
        "💬 RAG": ["What are customers saying about delivery delays?",
                   "Main complaints in 1-star reviews","What do customers love?"],
        "🔀 Hybrid": ["Which categories have worst reviews and what do customers say?",
                      "Compare bed & bath sales with customer sentiment"],
    }
    for group, qs in examples.items():
        with st.expander(group, expanded=False):
            for q in qs:
                if st.button(q, key=f"ex_{q}", use_container_width=True):
                    st.session_state["pending_query"] = q
    st.divider()
    st.caption("GPT-4O-MINI · FAISS · sentence-transformers · Plotly · Streamlit")

st.markdown("""
<div class="main-header">
    <h1>🛒 Olist Analytics Assistant</h1>
    <p>Ask anything about orders, products, sellers, or customer reviews — in plain English.</p>
</div>""", unsafe_allow_html=True)

try:
    system = load_system()
except Exception as e:
    st.error(f"❌ System initialisation failed: {e}")
    with st.expander("Full traceback"): st.code(traceback.format_exc())
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []


def run_query(query):
    route = system["router"].classify(query)
    result = {"route": route, "query": query}
    if route in ("SQL", "HYBRID"):
        sr = system["nl2sql"].execute(query)
        result.update({"sql": sr.get("sql"), "df": sr.get("df"),
                        "sql_text": sr.get("summary",""), "chart": sr.get("chart"),
                        "sql_error": sr.get("error")})
    if route in ("RAG", "HYBRID"):
        chunks = system["retriever"].retrieve(query, k=5)
        rr = system["sentiment"].analyse(query, chunks)
        result.update({"rag_text": rr.get("answer",""), "sentiment": rr.get("sentiment"),
                        "themes": rr.get("themes",[]), "chunks": chunks})
    if route == "HYBRID":
        result["synthesis"] = system["synth"].merge(
            query, result.get("sql_text",""), result.get("rag_text",""))
    return result


def render_result(result):
    route = result["route"]
    bmap = {"SQL":("badge-sql","🗄️ SQL Path"),"RAG":("badge-rag","💬 Review Analysis"),
            "HYBRID":("badge-hybrid","🔀 Hybrid")}
    cls, label = bmap.get(route,("badge-sql",route))
    st.markdown(f'<span class="route-badge {cls}">{label}</span>', unsafe_allow_html=True)

    if result.get("sql_error"): st.error(f"SQL Error: {result['sql_error']}")
    if result.get("sql"):
        with st.expander("🔍 Generated SQL", expanded=False):
            st.code(result["sql"], language="sql")
    df = result.get("df")
    if df is not None and not df.empty:
        st.dataframe(df, use_container_width=True, height=min(300,50+35*len(df)))
    if result.get("sql_text"): st.info(f"📊 **Insight:** {result['sql_text']}")
    if result.get("chart"): st.plotly_chart(result["chart"], use_container_width=True)

    if result.get("rag_text"):
        if result.get("sql_text") or df is not None: st.divider()
        smap = {"positive":"😊 Positive","negative":"😞 Negative","mixed":"😐 Mixed"}
        sent = result.get("sentiment","")
        if sent: st.markdown(f"**Sentiment:** `{smap.get(sent,sent)}`")
        st.markdown(f"**📝 Review Analysis:**\n\n{result['rag_text']}")
        themes = result.get("themes",[])
        if themes: st.markdown("**🔑 Top Themes:** "+"  ·  ".join(f"`{t}`" for t in themes))
        with st.expander(f"📄 Source Excerpts ({len(result.get('chunks',[]))})", expanded=False):
            for i,c in enumerate(result.get("chunks",[]),1):
                st.markdown(f"**Excerpt {i}** — score `{c['score']:.3f}` | order `{c['order_id']}`")
                st.caption(c["text"][:400]); st.divider()

    if result.get("synthesis"):
        st.divider()
        st.success(f"**🔀 Combined Insight:**\n\n{result['synthesis']}")


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pending = st.session_state.pop("pending_query", None)
user_query = st.chat_input("Ask a question about Olist data…") or pending

if user_query:
    st.session_state.history.append({"role":"user","content":user_query})
    with st.chat_message("user"): st.markdown(user_query)
    with st.chat_message("assistant"):
        with st.spinner("Analysing…"):
            try:
                result = run_query(user_query)
                render_result(result)
                summary = (result.get("synthesis") or result.get("sql_text")
                           or result.get("rag_text") or "Query completed.")
                st.session_state.history.append({"role":"assistant","content":summary})
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                with st.expander("Full traceback"): st.code(traceback.format_exc())
                st.session_state.history.append({"role":"assistant","content":err})
