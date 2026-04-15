import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")


BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data/raw"
DB_DIR = BASE_DIR / "data/DB/olist.db"


VECTORSTORE_DIR  = BASE_DIR / "vectorstore"
INDEX_PATH = VECTORSTORE_DIR / "faiss.index"
FAISS_INDEX_PATH = INDEX_PATH
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.pkl"
METADATA_PATH = VECTORSTORE_DIR / "metadata.pkl"

CHUNK_WORDS   = 200
OVERLAP_WORDS = 30
MODEL_NAME    = "sentence-transformers/all-MiniLM-L6-v2"

LLM = "gpt-4o-mini"


