from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSIST_DIR = BASE_DIR / "chroma_db"
CONVERSATIONS_DIR = BASE_DIR / "conversations"

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
RETRIEVER_K = 4
