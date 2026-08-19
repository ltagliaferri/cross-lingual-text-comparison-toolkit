"""
Embedding utilities: model loading, chunk generation, caching.
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_by_paragraph(text: str, min_chars: int = 300, max_chars: int = 1600) -> list[str]:
    """Split text into paragraph-based chunks within a target size range."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks, buffer = [], ""
    for p in paragraphs:
        if len(buffer) + len(p) < max_chars:
            buffer = (buffer + "\n\n" + p).strip()
        else:
            if len(buffer) >= min_chars:
                chunks.append(buffer)
                buffer = p
            else:
                buffer = (buffer + "\n\n" + p).strip()
    if buffer and len(buffer) >= min_chars:
        chunks.append(buffer)
    return chunks


# ---------------------------------------------------------------------------
# Corpus JSON helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_corpus(chunks: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_embedded_corpora(embed_dir, corpus_ids: list[str]) -> dict:
    """
    Load chunks + cached embeddings for each corpus id (as produced by
    embed_corpus.py). Returns {corpus_id: {'chunks': [...], 'embeddings': ndarray}}.
    """
    embed_dir = Path(embed_dir)
    result = {}
    for corpus_id in corpus_ids:
        result[corpus_id] = {
            'chunks':     load_corpus(embed_dir / f'{corpus_id}.json'),
            'embeddings': np.load(embed_dir / f'{corpus_id}.npy'),
        }
    return result


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_model(name: str = "sentence-transformers/LaBSE"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            'sentence-transformers is required to embed a corpus. Install it '
            'with: pip install "cross-lingual-toolkit[embeddings]"'
        )
    print(f"Loading model {name}…")
    return SentenceTransformer(name)


def embed_chunks(chunks: list[dict], model,
                 cache_path: Optional[Path] = None,
                 batch_size: int = 16) -> np.ndarray:
    """Embed a list of chunk dicts and optionally cache to disk."""
    if cache_path and Path(cache_path).exists():
        print(f"  Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks…")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"  Cached to {cache_path}")

    return embeddings


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def mean_pairwise_sim(emb_a: np.ndarray, emb_b: np.ndarray,
                      same: bool = False) -> float:
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(emb_a, emb_b)
    if same:
        n = sims.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return float(sims[mask].mean())
    return float(sims.mean())


def permutation_test(emb_a: np.ndarray, emb_b: np.ndarray,
                     n_perm: int = 1000, seed: int = 42) -> float:
    """
    One-sided permutation test: is within-A similarity higher than A-to-B?
    Returns p-value (fraction of permutations where shuffled gap >= observed gap).
    """
    observed = mean_pairwise_sim(emb_a, emb_a, same=True) - mean_pairwise_sim(emb_a, emb_b)
    combined = np.vstack([emb_a, emb_b])
    n_a = len(emb_a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(combined))
        perm_a = combined[idx[:n_a]]
        perm_b = combined[idx[n_a:]]
        diff = mean_pairwise_sim(perm_a, perm_a, same=True) - mean_pairwise_sim(perm_a, perm_b)
        if diff >= observed:
            count += 1
    return count / n_perm
