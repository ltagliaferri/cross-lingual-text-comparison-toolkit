"""
Embedding utilities: model loading, chunk generation, caching.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _hard_wrap(text: str, max_chars: int) -> List[str]:
    """Cut text at the last whitespace before max_chars (mid-run only if none)."""
    if max_chars <= 0:
        raise ValueError(f'max_chars must be positive, got {max_chars}')
    out = []
    while len(text) > max_chars:
        cut = max((m.start() for m in re.finditer(r'\s', text[:max_chars + 1])), default=0)
        if cut <= 0:
            cut = max_chars
        out.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        out.append(text)
    return out


def _pack(pieces: List[str], max_chars: int) -> List[str]:
    """Greedily rejoin pieces with spaces into runs of at most max_chars."""
    out, buf = [], ""
    for piece in pieces:
        if buf and len(buf) + 1 + len(piece) > max_chars:
            out.append(buf)
            buf = piece
        else:
            buf = f"{buf} {piece}" if buf else piece
    if buf:
        out.append(buf)
    return out


def _split_paragraph(p: str, max_chars: int) -> List[str]:
    """
    Break a paragraph longer than max_chars at sentence boundaries, then at
    single newlines, then at whitespace, so no piece exceeds max_chars.
    """
    if len(p) <= max_chars:
        return [p]
    for pattern in (r'(?<=[.!?])\s+', r'\n'):
        parts = [x.strip() for x in re.split(pattern, p) if x.strip()]
        if len(parts) > 1:
            pieces = []
            for part in parts:
                pieces.extend(_split_paragraph(part, max_chars))
            return _pack(pieces, max_chars)
    return _hard_wrap(p, max_chars)


def chunk_by_paragraph(text: str, min_chars: int = 300, max_chars: int = 800) -> List[str]:
    """
    Split text into paragraph-based chunks of at most max_chars characters.

    Paragraphs (blank-line separated) are merged until the next one would
    push a chunk past max_chars. A paragraph that is itself longer than
    max_chars is first split at sentence ends, then single newlines, then
    whitespace. min_chars only drops a text whose whole length is shorter
    than it; any chunk of a longer text, including the last, is kept even
    when it is shorter than min_chars.

    Raises ValueError if max_chars is not positive or min_chars is negative.
    """
    if max_chars <= 0:
        raise ValueError(f'max_chars must be positive, got {max_chars}')
    if min_chars < 0:
        raise ValueError(f'min_chars must not be negative, got {min_chars}')
    paragraphs = []
    for p in re.split(r'\n\s*\n', text):
        if p.strip():
            paragraphs.extend(_split_paragraph(p.strip(), max_chars))
    chunks, buffer = [], ""
    for p in paragraphs:
        candidate = f"{buffer}\n\n{p}" if buffer else p
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            chunks.append(buffer)
            buffer = p
    if buffer and (chunks or len(buffer) >= min_chars):
        chunks.append(buffer)
    return chunks


# ---------------------------------------------------------------------------
# Corpus JSON helpers
# ---------------------------------------------------------------------------

def load_corpus(path: str) -> List[dict]:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_corpus(chunks: List[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load_embedded_corpora(embed_dir, corpus_ids: List[str]) -> dict:
    """
    Load chunks + cached embeddings for each corpus id (as produced by
    embed_corpus.py). Returns {corpus_id: {'chunks': [...], 'embeddings': ndarray}}.
    """
    embed_dir = Path(embed_dir)
    missing = [str(embed_dir / f'{cid}{ext}') for cid in corpus_ids
               for ext in ('.json', '.npy') if not (embed_dir / f'{cid}{ext}').exists()]
    if missing:
        raise FileNotFoundError(
            'Embeddings not found:\n  ' + '\n  '.join(missing) + '\n'
            'Run xlt-embed-corpus with the same --config / --output-root first.'
        )
    result = {}
    for corpus_id in corpus_ids:
        chunks = load_corpus(embed_dir / f'{corpus_id}.json')
        embeddings = np.load(embed_dir / f'{corpus_id}.npy')
        rerun = 'rerun xlt-embed-corpus with the same --config / --output-root.'
        if embeddings.ndim != 2:
            raise ValueError(
                f'{embed_dir / corpus_id}.npy is not a 2-D array of embeddings '
                f'(shape {embeddings.shape}); {rerun}'
            )
        if len(chunks) != len(embeddings):
            raise ValueError(
                f'{embed_dir / corpus_id}.json has {len(chunks)} chunks but '
                f'{corpus_id}.npy has {len(embeddings)} embeddings; {rerun}'
            )
        result[corpus_id] = {'chunks': chunks, 'embeddings': embeddings}
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


def truncation_report(texts: List[str], model, limit: Optional[int] = None) -> Optional[dict]:
    """
    How many of `texts` are longer than the model's input window.

    Sentence-transformer models expose the tokenizer they were trained with
    and `max_seq_length`, the number of word-pieces they read before silently
    truncating. Returns {'over', 'total', 'limit', 'suggested_max_chars'},
    where the suggestion is the chunk length (in characters) that fits the
    window at this corpus's characters-per-word-piece ratio, with headroom.
    Returns None when the model exposes no tokenizer or limit, so a stub or
    an unusual model never blocks embedding.
    """
    tokenizer = getattr(model, 'tokenizer', None)
    limit = limit or getattr(model, 'max_seq_length', None)
    if tokenizer is None or not limit or not texts:
        return None
    try:
        lengths = [len(tokenizer(t, add_special_tokens=True, truncation=False)['input_ids'])
                   for t in texts]
    except Exception:          # a tokenizer with another calling convention
        return None
    total_pieces = sum(lengths)
    if not total_pieces:
        return None
    chars_per_piece = sum(len(t) for t in texts) / total_pieces
    return {
        'over': sum(1 for n in lengths if n > limit),
        'total': len(texts),
        'limit': int(limit),
        'suggested_max_chars': int(chars_per_piece * (limit - 2) * 0.85),
    }


def _cache_key(texts: List[str], model_name: Optional[str]) -> str:
    """SHA-256 over the model name and every chunk text, for embed_chunks."""
    h = hashlib.sha256((model_name or '').encode('utf-8'))
    for t in texts:
        h.update(b'\0' + t.encode('utf-8'))
    return h.hexdigest()


def embed_chunks(chunks: List[dict], model,
                 cache_path: Optional[Path] = None,
                 batch_size: int = 16,
                 model_name: Optional[str] = None) -> np.ndarray:
    """
    Embed a list of chunk dicts and optionally cache to disk. Next to the
    .npy cache a `.key` file stores a hash of `model_name` and the chunk
    texts; the cache is reused only if that hash matches, so changing the
    texts, the chunking settings or the model recomputes the embeddings.
    """
    texts = [c["text"] for c in chunks]
    key = _cache_key(texts, model_name)
    key_path = Path(f'{cache_path}.key') if cache_path else None
    if cache_path and Path(cache_path).exists():
        stored = key_path.read_text().strip() if key_path.exists() else None
        if stored == key:
            cached = np.load(cache_path)
            if cached.ndim == 2 and len(cached) == len(chunks):
                print(f"  Loading cached embeddings from {cache_path}")
                return cached
        print(f"  Cached embeddings in {cache_path} do not match the current "
              f"chunks or model; recomputing")

    report = truncation_report(texts, model)
    if report and report['over']:
        print(f"  WARNING: {report['over']} of {report['total']} chunks are longer than "
              f"the model's input window of {report['limit']} word-pieces, so the text "
              f"past the window is not embedded. Lower chunking.max_chars (about "
              f"{report['suggested_max_chars']} characters fits this corpus) and rerun "
              f"xlt-embed-corpus.")
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
        key_path.write_text(key + '\n')
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


def similarity_gap(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    The permutation test's statistic: mean within-group similarity
    averaged over both groups, minus the mean between-group similarity,
    (within(A) + within(B)) / 2 - between(A, B). Symmetric in A and B.
    """
    return ((mean_pairwise_sim(emb_a, emb_a, same=True)
             + mean_pairwise_sim(emb_b, emb_b, same=True)) / 2
            - mean_pairwise_sim(emb_a, emb_b))


def _gap_from_sims(sims: np.ndarray, g1: np.ndarray, g2: np.ndarray) -> float:
    def within(g):
        block = sims[np.ix_(g, g)]
        return (block.sum() - np.trace(block)) / (len(g) * (len(g) - 1))
    return (within(g1) + within(g2)) / 2 - sims[np.ix_(g1, g2)].mean()


def permutation_test(emb_a: np.ndarray, emb_b: np.ndarray,
                     n_perm: int = 1000, seed: int = 42) -> float:
    """
    One-sided permutation test: are the two groups more similar within
    themselves than to each other?

    Statistic: similarity_gap() — the mean of the two within-group
    similarities minus the between-group similarity — so the result does
    not depend on which corpus is passed first. Group labels are shuffled
    `n_perm` times; the p-value is (count + 1) / (n_perm + 1), where count
    is the number of shuffles whose gap is at least the observed one, so
    it is never exactly 0.

    Caveats: chunks of one text are not independent (neighbours share
    vocabulary and topic), which overstates significance; and in a
    cross-lingual study the groups always differ in language (and often
    genre), so a small p alone does not show an authorial difference.

    Raises ValueError if either group has fewer than 2 chunks, since
    within-group similarity is then undefined; if the inputs are not 2-D
    arrays with the same number of columns; or if n_perm < 1.
    """
    emb_a, emb_b = np.asarray(emb_a), np.asarray(emb_b)
    if emb_a.ndim != 2 or emb_b.ndim != 2:
        raise ValueError(
            f'permutation test needs 2-D embedding arrays (one row per chunk), '
            f'got shapes {emb_a.shape} and {emb_b.shape}'
        )
    if emb_a.shape[1] != emb_b.shape[1]:
        raise ValueError(
            f'embedding dimensions differ ({emb_a.shape[1]} vs. {emb_b.shape[1]}); '
            'embed both corpora with the same model'
        )
    if n_perm != int(n_perm) or n_perm < 1:
        raise ValueError(f'n_perm must be a whole number of at least 1, got {n_perm}')
    n_perm = int(n_perm)
    if len(emb_a) < 2 or len(emb_b) < 2:
        raise ValueError(
            f'permutation test needs at least 2 chunks per corpus '
            f'(got {len(emb_a)} and {len(emb_b)})'
        )
    from sklearn.metrics.pairwise import cosine_similarity
    combined = np.vstack([emb_a, emb_b])
    # Put rows in a canonical order so the shuffles drawn for a given seed
    # are the same whichever corpus was passed first.
    order = np.lexsort(combined.T[::-1])
    sims = cosine_similarity(combined[order])
    position = np.empty(len(order), dtype=int)
    position[order] = np.arange(len(order))
    n_a = len(emb_a)
    observed = _gap_from_sims(sims, position[:n_a], position[n_a:])

    n_small = min(n_a, len(emb_b))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(combined))
        if _gap_from_sims(sims, idx[:n_small], idx[n_small:]) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)
