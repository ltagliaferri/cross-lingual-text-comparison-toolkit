import numpy as np
from pytest import approx

from cross_lingual_toolkit import embed


# ---------------------------------------------------------------------------
# chunk_by_paragraph()
# ---------------------------------------------------------------------------

def test_chunk_by_paragraph_merges_short_paragraphs():
    text = "Short one.\n\nShort two.\n\nShort three."
    chunks = embed.chunk_by_paragraph(text, min_chars=10, max_chars=1000)
    # all three paragraphs fit comfortably under max_chars, so they merge
    # into a single chunk rather than three tiny ones.
    assert len(chunks) == 1
    assert "Short one." in chunks[0]
    assert "Short three." in chunks[0]


def test_chunk_by_paragraph_splits_when_max_exceeded():
    para_a = "A" * 50
    para_b = "B" * 50
    chunks = embed.chunk_by_paragraph(f"{para_a}\n\n{para_b}", min_chars=10, max_chars=60)
    assert len(chunks) == 2
    assert chunks[0] == para_a
    assert chunks[1] == para_b


def test_chunk_by_paragraph_empty_text_returns_empty_list():
    assert embed.chunk_by_paragraph("", min_chars=10, max_chars=100) == []


def test_chunk_by_paragraph_drops_trailing_buffer_under_min_chars():
    chunks = embed.chunk_by_paragraph("hi", min_chars=100, max_chars=1000)
    assert chunks == []


# ---------------------------------------------------------------------------
# save_corpus() / load_corpus()
# ---------------------------------------------------------------------------

def test_save_and_load_corpus_roundtrip(tmp_path):
    chunks = [{"chunk_id": "a_0000", "text": "hello", "author": "x"}]
    path = tmp_path / "sub" / "a.json"
    embed.save_corpus(chunks, str(path))
    assert path.exists()
    assert embed.load_corpus(str(path)) == chunks


# ---------------------------------------------------------------------------
# cosine_sim() / mean_pairwise_sim()
# ---------------------------------------------------------------------------

def test_cosine_sim_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert embed.cosine_sim(v, v) == approx(1.0)


def test_cosine_sim_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert embed.cosine_sim(a, b) == approx(0.0)


def test_mean_pairwise_sim_same_excludes_diagonal():
    # three identical vectors: every off-diagonal pair has similarity 1.0,
    # so the mean (excluding self-pairs) should be exactly 1.0.
    emb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert embed.mean_pairwise_sim(emb, emb, same=True) == approx(1.0)


def test_mean_pairwise_sim_orthogonal_groups_is_zero():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert embed.mean_pairwise_sim(a, b) == approx(0.0)
