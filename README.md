# 🛒 Olist Analytics Assistant

An LLM-powered analytics chatbot built on the [Brazilian E-Commerce (Olist) dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Ask questions in plain English — the system routes them to SQL queries, RAG-based review analysis, or a hybrid of both.

---

## 🚀 Features

- **Natural Language to SQL** — converts plain English into SQLite queries and returns DataFrames + Plotly charts
- **RAG (Retrieval-Augmented Generation)** — semantic search over customer reviews using FAISS + sentence-transformers
- **Hybrid Mode** — combines structured data insights with unstructured review sentiment
- **Smart Query Routing** — automatically classifies each question as SQL, RAG, or Hybrid
- **Streamlit UI** — conversational chat interface with sidebar example queries

---

## 🗂️ Project Structure

```
├── app.py                  # Main Streamlit UI
├── config.py               # Paths and API keys
├── ingest.py               # OlistDataLoader — CSV → SQLite
├── routes/
│   ├── router.py           # QueryRouter — classifies questions
    ├── synthesizer.py      # Merges SQL + RAG answers (Hybrid)
    ├── sentiment.py        # SentimentAnalyser — themes + tone
│   └── nl_to_sql.py        # NLtoSQL — GPT-powered SQL generation
├── rag/
│   ├── embedder.py         # Builds FAISS index from review text
│   └── retriever.py        # Retriever — semantic search over chunks
└── data/                   # Olist CSV files go here
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/surya1626/LLM-Powered-Analytics-Assistant-with-RAG
cd LLM-Powered-Analytics-Assistant-with-RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your OpenAI API key

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

### 4. Download the Olist dataset

Download from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files in the `data/` directory (or wherever `DATA_DIR` points in `config.py`).

### 5. Run the app

```bash
streamlit run app.py
```

The app will automatically ingest the CSVs into SQLite and build the FAISS index on first launch.

---

## 💬 Example Queries

| Type | Example |
|------|---------|
| 📊 SQL | `Top 5 product categories by revenue?` |
| 📊 SQL | `Monthly order counts for 2017` |
| 📊 SQL | `Sellers with highest average review score` |
| 💬 RAG | `What are customers saying about delivery delays?` |
| 💬 RAG | `Main complaints in 1-star reviews` |
| 🔀 Hybrid | `Which categories have worst reviews and what do customers say?` |

---

## 🧠 How It Works

1. **User asks a question** in the chat input
2. **QueryRouter** classifies it as `SQL`, `RAG`, or `HYBRID`
3. For **SQL**: `NLtoSQL` sends the question + schema to GPT-3.5, executes the returned SQL on SQLite, and renders a DataFrame + optional chart
4. For **RAG**: `Retriever` finds the top-5 most relevant review chunks via FAISS cosine similarity, then `SentimentAnalyser` synthesizes an answer with sentiment and themes
5. For **HYBRID**: both pipelines run, and `Synthesizer` merges them into a unified insight
6. Results are rendered in the Streamlit chat UI

---

## 🔧 Configuration (`config.py`)

| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Path to Olist CSV files |
| `DB_DIR` | Path to the SQLite database file |
| `OPENAI_API_KEY` | Your OpenAI key (also loadable from `.env`) |

---

## ⚠️ Known Issues

- `SentimentAnalyser` and `Synthesizer` are currently commented out in `load_system()` — RAG and Hybrid routes will raise a `KeyError` until they are re-enabled
- First launch may take 1–3 minutes to ingest CSVs and build the FAISS index (cached after first run)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM | gpt-4o-mini |
| Vector Search | FAISS |
| Embeddings | sentence-transformers |
| Database | SQLite |
| Charts | Plotly |
| Data | pandas |

---

## 📄 License

MIT
