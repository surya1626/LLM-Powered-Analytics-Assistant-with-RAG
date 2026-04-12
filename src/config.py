import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = "data/olist.db"
FAISS_INDEX_PATH = "data/faiss.index"


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR,"data/raw")
DB_DIR = os.path.join(BASE_DIR,"data/DB/olist.db")





