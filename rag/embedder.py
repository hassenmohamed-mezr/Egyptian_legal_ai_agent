import os

from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

hf_token = os.environ.get("HF_TOKEN")

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
hf_token = os.environ.get("HF_TOKEN")

# =========================
# LOAD MODEL
# =========================

model = SentenceTransformer(
    "BAAI/bge-m3",
    token=hf_token
)

# =========================
# SINGLE TEXT EMBEDDING
# =========================

def embed_text(text: str):

    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding

# =========================
# MULTIPLE TEXT EMBEDDINGS
# =========================

def embed_texts(texts: list):

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )

    return embeddings