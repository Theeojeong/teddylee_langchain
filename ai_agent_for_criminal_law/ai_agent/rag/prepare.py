from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any
import json
import os
import faiss
import numpy as np
from tqdm import tqdm

from openai import OpenAI

from ..config import RAW_DATA_DIR, VECTOR_DIR, EMBEDDING_MODEL


INDEX_FILE = VECTOR_DIR / "index.faiss"
META_FILE = VECTOR_DIR / "meta.json"


def _ensure_sample_raw():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = list(RAW_DATA_DIR.glob("*.txt"))
    if not files:
        sample = RAW_DATA_DIR / "criminal_law_ko_sample.txt"
        sample.write_text(
            (
                "형법 개론\n"
                "형법은 범죄와 형벌을 규정하는 법률로서, 구성요건해당성, 위법성, 책임의 원칙을 기반으로 한다.\n"
                "구성요건은 범죄의 성립 요건을 의미하며, 고의와 과실 등 주관적 요소와 행위의 객관적 요소가 포함된다.\n"
                "정당방위, 긴급피난 등 위법성조각사유가 존재할 수 있으며, 책임 조각사유로는 심신상실, 책임무능력 등이 있다.\n"
                "공동정범과 간접정범, 교사범과 방조범의 구별, 미수범 처벌 요건 등도 중요하다.\n"
            ),
            encoding="utf-8",
        )


def _load_raw_texts() -> List[str]:
    _ensure_sample_raw()
    texts: List[str] = []
    for p in RAW_DATA_DIR.glob("*.txt"):
        try:
            texts.append(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return texts


def _clean_text(t: str) -> str:
    return " ".join(t.replace("\r", " ").replace("\n", " ").split())


def _split_chunks(text: str, max_chars: int = 700, overlap: int = 100) -> List[str]:
    text = _clean_text(text)
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _embed_texts(client: OpenAI, texts: List[str], batch_size: int = 128) -> np.ndarray:
    vecs: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
    arr = np.array(vecs, dtype="float32")
    # Normalize for inner product = cosine
    faiss.normalize_L2(arr)
    return arr


def build_index() -> Tuple[faiss.Index, Dict[str, Any]]:
    client = OpenAI()
    raw_texts = _load_raw_texts()
    chunks: List[str] = []
    src_ids: List[int] = []
    for i, t in enumerate(raw_texts):
        cs = _split_chunks(t)
        chunks.extend(cs)
        src_ids.extend([i] * len(cs))

    if not chunks:
        raise RuntimeError("No RAG chunks prepared")

    X = _embed_texts(client, chunks)
    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)

    meta = {
        "embedding_model": EMBEDDING_MODEL,
        "chunks": chunks,
        "src_ids": src_ids,
        "dim": dim,
    }

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return index, meta


def load_index() -> Tuple[faiss.Index, Dict[str, Any]]:
    if not INDEX_FILE.exists() or not META_FILE.exists():
        return build_index()
    index = faiss.read_index(str(INDEX_FILE))
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    return index, meta


def retrieve(query: str, k: int = 3) -> List[str]:
    client = OpenAI()
    index, meta = load_index()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    v = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(v)
    D, I = index.search(v, k)
    chunks: List[str] = meta["chunks"]
    result = [chunks[int(i)] for i in I[0] if int(i) < len(chunks) and int(i) >= 0]
    return result


if __name__ == "__main__":
    build_index()
    print("RAG index built at:", INDEX_FILE)

