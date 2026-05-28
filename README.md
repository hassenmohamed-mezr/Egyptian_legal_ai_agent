# ⚖️ Lex AI — Egyptian Legal AI Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-green?style=for-the-badge&logo=fastapi"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/FAISS-GPU-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Ollama-Local_LLM-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

> **A production-grade RAG (Retrieval-Augmented Generation) system specialized in Egyptian Labor Law.**
> Understands both direct legal questions and real-world narratives, and responds in plain language grounded strictly in legal texts.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problems This System Solves](#problems-this-system-solves)
- [System Architecture](#system-architecture)
- [Detailed Pipeline](#detailed-pipeline)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [API Reference](#api-reference)
- [Real-World Examples](#real-world-examples)
- [Models Used](#models-used)
- [Performance Optimizations](#performance-optimizations)

---

## Overview

**Lex AI** is not a simple legal chatbot — it is a full-stack intelligent pipeline that understands context, distinguishes between a direct question and a human narrative, and rewrites queries to extract the maximum amount of precise legal information from a structured corpus.

The system is built on three core pillars:

**1. Hybrid Retrieval** — Combines dense semantic search (FAISS) with sparse lexical search (BM25) using dynamically computed fusion weights.

**2. Intelligent Query Rewriting** — Detects when a user is narrating a story rather than asking a question, and automatically decomposes the narrative into targeted legal sub-questions before retrieval.

**3. Full-Article Context Delivery** — Instead of passing a random chunk to the LLM, the system reconstructs and delivers the complete legal article, preserving every clause and condition.

---

## Problems This System Solves

Standard RAG systems fail on legal texts in three well-known ways. This project addresses all three:

| Problem | Impact | Solution Applied |
|---------|--------|-----------------|
| Retrieving a single chunk from a multi-part article | LLM sees incomplete legal text and produces incorrect or partial answers | **Article-level merging** — all chunks belonging to an article are merged in order before being passed to the LLM |
| Searching with the user's raw colloquial text | User narratives contain no legal terminology; retrieval returns irrelevant results | **Query Rewriter** — an LLM call decomposes the narrative into precise legal questions used for retrieval |
| Relying solely on FAISS semantic search | Dense search weakens on domain-specific legal terms and article numbers | **Dynamic Hybrid Fusion** — FAISS + BM25 scores are normalized and fused with query-length-aware weights |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   User  (Streamlit UI)                   │
│          Direct Question  ──or──  Story / Scenario       │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                    FastAPI  /chat                        │
│                      routes.py                           │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│                     rag_service.py                       │
│                                                          │
│   1. query_rewriter   →  detect type + extract questions │
│   2. retrieve_chunks  →  multi-query search + dedup      │
│   3. generate_answer  →  prompt-aware generation         │
└───────────┬──────────────────────────┬───────────────────┘
            │                          │
            ▼                          ▼
┌───────────────────┐     ┌────────────────────────────────┐
│   query_rewriter  │     │        hybrid_retriever        │
│                   │     │                                │
│  Step 1: detect   │     │  FAISS  (semantic / dense)     │
│  narrative vs     │     │    +                           │
│  direct question  │     │  BM25   (lexical / sparse)     │
│                   │     │    ↓                           │
│  Step 2: extract  │     │  Log-squash + Normalization    │
│  2–4 legal        │     │    ↓                           │
│  sub-questions    │     │  Dynamic Fusion Weights        │
│  (LLM call × 1-2) │     │    ↓                           │
└───────────────────┘     │  Article-Level Merging         │
                          │    ↓                           │
                          │  Pre-Rerank Filtering          │
                          │    ↓                           │
                          │  Cross-Encoder Reranker        │
                          └────────────────────────────────┘
                                         │
                                         ▼
                           ┌────────────────────────┐
                           │      ollama_client     │
                           │   Qwen 2.5:7B  (local) │
                           │  narrative / direct    │
                           │  prompt template       │
                           └────────────────────────┘
```

---

## Detailed Pipeline

### Stage 1 — Query Intelligence

```python
# query_rewriter.py
rewrite_result = rewrite_query(text=query)
# → {
#     "is_narrative": True,
#     "questions": [
#         "Is temporary suspension from work without notice legal?",
#         "What are the conditions for suspending an employee?",
#         ...
#     ]
#   }
```

The rewriter sends a lightweight LLM call first:
*"Is this text a direct legal question or a narrative/scenario?"*

If a narrative is detected, a second LLM call extracts 2–4 explicit legal questions from it. This means a user writing:

> *"My salary was deducted without any reason or written notice"*

automatically produces retrieval queries like:
- What are the legal conditions for wage deductions?
- What is the maximum deduction allowed per month?
- What are the procedures for appealing a deduction decision?

This bridges the vocabulary gap between colloquial user language and formal legal text — a problem that kills retrieval quality in naive RAG implementations.

---

### Stage 2 — Hybrid Retrieval

#### Step 1 — Dual Retrieval

```python
faiss_results = search_faiss(query, top_k=candidate_k)    # semantic
bm25_results  = bm25_store.search(query, top_k=candidate_k)  # lexical
```

**FAISS (IndexFlatIP):** Searches by meaning — understands that "wrongful termination" and "unjust contract ending" are semantically close even when phrased differently.

**BM25Okapi:** Searches by exact token overlap — excels on precise legal terminology such as article numbers, specific legal phrases, and domain-specific vocabulary that dense embeddings tend to smooth over.

#### Step 2 — Normalization + Dynamic Fusion

```python
# BM25 scores undergo log-squashing first to suppress outlier dominance
bm25_scores = np.log1p(bm25_scores)
bm25_scores = normalize_scores(bm25_scores)
faiss_scores = normalize_scores(faiss_scores)

# Weights shift dynamically based on query length
if query_len <= 3:
    faiss_weight, bm25_weight = 0.55, 0.45   # short query → BM25 matters more
else:
    faiss_weight, bm25_weight = 0.72, 0.28   # long query  → semantics matters more

final_score = faiss_weight * faiss_norm + bm25_weight * bm25_norm
```

This decision is grounded in an empirical observation about legal queries: short queries (2–3 tokens) are typically precise legal terms that benefit from exact matching, while longer queries carry rich semantic context that FAISS handles better. Static weights would sacrifice one at the expense of the other.

#### Step 3 — Article-Level Merging

This is the single most important architectural decision separating this system from off-the-shelf RAG:

```python
def _merge_article_text(article_id: str) -> str:
    """
    Reconstructs the full text of a legal article from all its
    ordered chunks. The LLM always sees the complete article,
    never an arbitrary fragment.
    """
    chunks = _ARTICLE_INDEX.get(article_id, [])
    chunks.sort(key=lambda c: c.get("chunk_order", 0))
    return "\n".join(c.get("text", "") for c in chunks)
```

Egyptian labor law articles are split across multiple chunks during indexing. Retrieving only the top-scoring chunk — the standard RAG approach — means the LLM might see a list of punishable acts but miss the procedural conditions that govern them, producing legally inaccurate responses.

The `_ARTICLE_INDEX` is built once at startup from `bm25_store.chunks`, which is already loaded into memory. There is zero additional I/O cost. Every article that reaches the LLM is complete and correctly ordered.

#### Step 4 — Pre-Rerank Filtering

```python
filtered = [
    r for r in results
    if r["faiss_score"] >= 0.55 or r["bm25_score"] >= 1.5
]

# Fallback: if nothing passes the threshold, take the top 12
if not filtered:
    filtered = results[:12]
```

Cross-Encoder reranking is computationally expensive — it reads the full query and full article text jointly for every candidate. This filter removes weak results before they reach the reranker, cutting reranking time significantly without sacrificing quality, since genuinely relevant articles will comfortably clear at least one of these thresholds.

#### Step 5 — Cross-Encoder Reranking

```python
# BAAI/bge-reranker-v2-m3 — multilingual, Arabic-capable
scores = reranker.predict([(query, article_text) for ...])

# Hybrid bonus rewards articles that both FAISS and BM25 agreed on
r["rerank_score"] = rerank_score + 0.15 * r["faiss_score"]
```

A Cross-Encoder reads the question and the candidate article in the same attention context, producing a relevance score far more accurate than cosine similarity on independent embeddings. It is applied last — after filtering — because its quadratic attention cost makes it impractical as a first-pass retriever.

The hybrid bonus (`+ 0.15 * faiss_score`) gives a slight edge to articles that both retrieval signals agreed on, rewarding inter-system consensus.

---

### Stage 3 — Multi-Query Deduplication

When multiple questions are extracted from a narrative, each runs through the full hybrid retrieval pipeline independently. Results are then merged with article-level deduplication:

```python
seen: Dict[str, Dict] = {}   # article_id → best-scoring chunk

for question in questions:
    results = hybrid_search(query=question, top_k=top_k)
    for chunk in results:
        aid = chunk["article_id"]
        # Keep only the highest score for each article across all questions
        if aid not in seen or chunk["score"] > seen[aid]["score"]:
            seen[aid] = chunk

# Sort by best score and cap at top_k
merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
```

This guarantees no article appears twice in the final context window, and that each article is represented by its strongest retrieval signal regardless of which sub-question surfaced it.

---

### Stage 4 — Prompt-Aware Generation

The system selects a different prompt template based on the detected query type:

**Direct question** → `LEGAL_USER_PROMPT_TEMPLATE`
Instructs the LLM to give a direct answer, cite the article number, and explain its practical implication — without quoting the legal text verbatim.

**Narrative / scenario** → `NARRATIVE_USER_PROMPT_TEMPLATE`
Instructs the LLM to analyze the user's legal situation, identify all relevant articles, and lay out concrete actionable steps — framed around the user's specific story.

Both templates explicitly prohibit verbatim reproduction of legal text. The model is required to explain in its own words, making responses accessible to non-lawyers.

---

## Key Engineering Decisions

### `hybrid_retriever.py`
- **Dynamic fusion weights** that adapt to query length rather than using fixed ratios
- **Log-squash on BM25 scores** to prevent high-frequency terms from dominating the fusion
- **Lazy-loaded, cached article index** — built once from already-resident memory, never rebuilt
- **Pre-rerank filter** that shields the Cross-Encoder from unnecessary computation

### `query_rewriter.py`
- **Two-step LLM pipeline**: detection first (cheap), extraction second (only when needed)
- **Robust JSON parsing** with regex fallback — handles markdown fences and malformed LLM output gracefully
- **Graceful degradation**: if extraction fails for any reason, the original query is returned unchanged, ensuring the system never silently returns empty results
- **Consistent retry logic** mirroring `ollama_client` — unified error handling across all LLM calls

### `bm25_store.py`
- **Arabic-aware tokenizer** with diacritic removal and hamza normalization (`أ`, `إ`, `آ` → `ا`)
- **Domain-tuned stopword list** — removes Arabic function words that add noise without legal meaning
- **Enriched corpus**: each document is indexed as `retrieval_text + article_title`, improving recall on title-level queries

### `reranker.py`
- `BAAI/bge-reranker-v2-m3` — a state-of-the-art multilingual cross-encoder with strong Arabic performance
- **FP16 inference** on GPU — halves memory usage with negligible quality impact
- **`torch.inference_mode()`** instead of `no_grad()` — faster and lower overhead for pure inference workloads

---

## Project Structure

```
egyptian_legal_ai_agent/
│
├── api/
│   └── routes.py                  # FastAPI endpoint definitions + Pydantic I/O models
│
├── app/
│   ├── main.py                    # Application entry point and startup
│   ├── config.py                  # Environment configuration
│   └── settings.py                # Pydantic settings management
│
├── llm/
│   ├── ollama_client.py           # Ollama HTTP client with retry logic + prompt dispatch
│   ├── prompts.py                 # System prompt + direct/narrative user templates
│   ├── query_rewriter.py          # Narrative detection + legal question extraction
│   ├── gemini_client.py           # Gemini API client (alternative LLM backend)
│   └── parser.py                  # LLM output parsing utilities
│
├── rag/
│   ├── hybrid_retriever.py        # ← Core of the system
│   ├── faiss_store.py             # FAISS index builder, loader, and search
│   ├── bm25_store.py              # BM25 index with Arabic-aware tokenization
│   ├── reranker.py                # Cross-Encoder reranking with hybrid bonus
│   ├── embedder.py                # BAAI/bge-m3 embedding (single + batch)
│   └── preprocess.py             # Dataset cleaning and chunk preparation
│
├── services/
│   ├── rag_service.py             # End-to-end pipeline orchestration
│   ├── chat_service.py            # Conversation session management
│   └── retrieval_service.py       # Unified retrieval interface
│
├── agent/
│   ├── agent_core.py              # Agent reasoning loop
│   ├── tool_registry.py           # Tool registration and metadata
│   └── tool_router.py             # Query-to-tool routing logic
│
├── tools/
│   ├── legal_search_tool.py       # Hybrid search tool (agent-callable)
│   ├── article_lookup_tool.py     # Direct article lookup by ID
│   └── summarize_tool.py          # Legal text summarization tool
│
├── data/
│   ├── processed/
│   │   └── law_ar_rag_optimized.json    # Preprocessed and chunked legal corpus
│   └── faiss/
│       ├── law.index                    # Persisted FAISS index
│       └── chunks_metadata.json         # Chunk metadata aligned with FAISS index
│
├── ui/
│   └── streamlit_app.py           # Full Streamlit UI with progress steps + narrative display
│
└── tests/
    ├── test_hybrid.py
    ├── test_search.py
    └── test_ollama_rag.py
```

---

## Installation & Setup

### Prerequisites

```
Python 3.10+
CUDA-capable GPU (recommended — CPU mode works but is significantly slower)
Ollama installed locally
```

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull the LLM via Ollama

```bash
ollama pull qwen2.5:7b
```

### 3. Configure environment variables

```env
# .env
OLLAMA_HOST=http://localhost:11434
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Build the FAISS index

```bash
python -c "
from rag.faiss_store import build_faiss_index
from pathlib import Path
build_faiss_index(Path('data/processed/law_ar_rag_optimized.json'))
"
```

### 5. Start the FastAPI server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300
```

### 6. Launch the Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

---

## API Reference

### `POST /chat`

**Request**

```json
{
  "query": "I was suspended from work without explanation. Is this legal?",
  "top_k": 6
}
```

**Response**

```json
{
  "answer": "Based on Article 145, your employer has the right to suspend you temporarily, but only under specific conditions...",
  "sources": [
    {
      "article_id": "145",
      "article_title": "Article 145",
      "chunk_id": "145_1",
      "score": 0.8921,
      "text": "..."
    }
  ],
  "query": "I was suspended from work without explanation. Is this legal?",
  "is_narrative": true,
  "questions_used": [
    "What are the legal conditions for temporary work suspension?",
    "Must the employer notify the employee of a suspension decision?",
    "What can an employee do if suspended without cause?"
  ]
}
```

**Fields**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | LLM-generated legal explanation in plain Arabic |
| `sources` | `array` | Retrieved articles with scores (full merged text) |
| `is_narrative` | `boolean` | `true` if the input was detected as a story/scenario |
| `questions_used` | `array` | Legal sub-questions used for retrieval (visible in UI) |

---

## Real-World Examples

### Direct Question

> **User:** What is the maximum allowed wage deduction?

The system identifies this as a direct question, runs a single hybrid search, retrieves Article 143 in full, and answers immediately.

### Narrative / Real-World Scenario

> **User:** I've been working at a private company for 3 years. Last week I had a conflict with my manager over a project mistake. Two days later I was suddenly suspended and told there's an internal investigation. I don't know if this is legal, and they said I might be fired.

**Automatically extracted legal questions:**
- Is suspending an employee during an internal investigation legal?
- What are the conditions and limits on temporary work suspension?
- What are the employee's rights in case of termination?
- How can an employee appeal a suspension decision?

**Articles retrieved (complete text):** 145, 146, 147, 148

**Result:** The LLM receives the full text of all four articles and the original user narrative, producing a structured legal analysis of the user's situation with actionable steps.

---

## Models Used

| Model | Role | Source |
|-------|------|--------|
| `BAAI/bge-m3` | Dense embeddings — 1024-dim, multilingual | HuggingFace |
| `BAAI/bge-reranker-v2-m3` | Cross-Encoder reranking — multilingual | HuggingFace |
| `qwen2.5:7b` | Answer generation + query rewriting | Ollama (local) |

---

## Performance Optimizations

| Optimization | Technique | Impact |
|-------------|-----------|--------|
| Reduced GPU memory usage | FP16 inference on all neural models | ~50% memory reduction |
| Faster inference | `torch.inference_mode()` over `no_grad()` | Lower overhead per call |
| Avoid repeated index loading | Global FAISS cache with lazy initialization | Index loaded once per process lifetime |
| Avoid repeated BM25 corpus builds | BM25 index built at startup, reused for all queries | Zero per-query corpus overhead |
| Free article reconstruction | Article index built from already-loaded BM25 data | Zero additional I/O |
| Protect Cross-Encoder from overload | Pre-rerank filter before expensive reranking | Fewer candidates = faster reranking |

---

<p align="center">
  Built with ❤️ for Egyptian workers who deserve to know their rights
</p>