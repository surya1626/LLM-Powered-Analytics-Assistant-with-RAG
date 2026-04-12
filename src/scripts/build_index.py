from pathlib import Path
import logging

from src.rag.embedder import build_faiss_index
from src.config import DB_DIR

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    db_path = Path(DB_DIR)

    index, chunks = build_faiss_index(db_path)

    print(f"\n✅ Index ready: {index.ntotal} vectors")
    print(f"✅ Total chunks: {len(chunks)}")