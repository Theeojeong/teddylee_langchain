import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

RAG_DIR = BASE_DIR / "rag"
RAW_DATA_DIR = RAG_DIR / "data" / "raw"
VECTOR_DIR = RAG_DIR / "vectorstore"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# KMMLU
KMMLU_DATASET_NAME = os.getenv("KMMLU_DATASET_NAME", "HAERAEHUB/KMMLU")
KMMLU_CATEGORY = os.getenv("KMMLU_CATEGORY", "Criminal-Law")
KMMLU_SPLIT = os.getenv("KMMLU_SPLIT", "test")

# Batch
BATCH_COMPLETION_WINDOW = os.getenv("BATCH_COMPLETION_WINDOW", "24h")
BATCH_ENDPOINT = os.getenv("BATCH_ENDPOINT", "/v1/chat/completions")


def timestamp_dir(prefix: str) -> Path:
    import datetime as _dt

    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    d = OUTPUTS_DIR / f"{prefix}_{ts}"
    d.mkdir(parents=True, exist_ok=True)
    return d

