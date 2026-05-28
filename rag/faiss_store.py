import json
import faiss
import numpy as np
from pathlib import Path
from rag.embedder import embed_text, embed_texts

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "faiss"
INDEX_PATH = INDEX_DIR / "law.index"
METADATA_PATH = INDEX_DIR / "chunks_metadata.json"

# =========================
# GLOBAL CACHE (FIX)
# =========================
_faiss_index = None
_faiss_metadata = None

def ensure_dirs():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

def build_faiss_index(dataset_path: Path):
    ensure_dirs()
    with open(dataset_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = []
    metadata = []
    for c in chunks:
        text = c.get("retrieval_text") or c.get("text")
        if not text:
            continue
        texts.append(text)
        metadata.append({
            "chunk_id": c["chunk_id"],
            "article_id": c["article_id"],
            "article_title": c.get("article_title", ""),
            "text": c.get("text", "")
        })

    embeddings = np.array(embed_texts(texts)).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_faiss():
    global _faiss_index, _faiss_metadata
    if _faiss_index is not None and _faiss_metadata is not None:
        return _faiss_index, _faiss_metadata
    if not INDEX_PATH.exists():
        raise FileNotFoundError("FAISS index not built. Run build_faiss_index() first.")
    _faiss_index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _faiss_metadata = json.load(f)
    return _faiss_index, _faiss_metadata

def search_faiss(query: str, top_k: int = 5):
    index, metadata = load_faiss()
    q = np.array([embed_text(query)]).astype("float32")
    scores, idxs = index.search(q, top_k)
    results = []
    for i, idx in enumerate(idxs[0]):
        if idx == -1:
            continue
        doc = metadata[idx]
        results.append({
            "chunk_id": doc["chunk_id"],
            "article_id": doc["article_id"],
            "article_title": doc.get("article_title", ""),
            "text": doc.get("text", ""),
            "score": float(scores[0][i]),
            "source": "faiss"
        })
    return results