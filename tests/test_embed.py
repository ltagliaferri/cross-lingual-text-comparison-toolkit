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


def test_chunk_by_paragraph_text_shorter_than_min_chars_gives_nothing():
    assert embed.chunk_by_paragraph("hi", min_chars=100, max_chars=1000) == []


def _words(s):
    return s.split()


def test_chunk_by_paragraph_caps_single_newline_text_at_max_chars():
    # regression: a text with single-newline paragraphs used to become one
    # chunk of any length (5,999 characters here)
    line = "This sentence has some words in it and ends with a full stop."
    text = "\n".join([line] * 96)
    assert len(text) > 5900
    chunks = embed.chunk_by_paragraph(text, min_chars=300, max_chars=1600)
    assert len(chunks) > 1
    assert all(len(c) <= 1600 for c in chunks)
    assert _words(" ".join(chunks)) == _words(text)


def test_chunk_by_paragraph_caps_one_unbroken_paragraph():
    text = " ".join(["word"] * 2000)       # no sentence ends, no newlines
    chunks = embed.chunk_by_paragraph(text, min_chars=300, max_chars=500)
    assert all(len(c) <= 500 for c in chunks)
    assert _words(" ".join(chunks)) == _words(text)


def test_chunk_by_paragraph_short_buffer_never_overflows_max_chars():
    # regression: a short buffer used to absorb the next paragraph
    # regardless of max_chars
    text = "Heading\n\n" + "x" * 590
    chunks = embed.chunk_by_paragraph(text, min_chars=300, max_chars=595)
    assert all(len(c) <= 595 for c in chunks)
    assert "Heading" in chunks[0]


def test_chunk_by_paragraph_keeps_the_closing_paragraph():
    # regression: a final paragraph shorter than min_chars was dropped
    body = "\n\n".join(["x" * 790, "y" * 790])         # one full 1,582-char chunk
    tail = "The closing paragraph. " * 4                 # ~90 chars: cannot join it
    chunks = embed.chunk_by_paragraph(body + "\n\n" + tail, min_chars=300, max_chars=1600)
    assert "The closing paragraph." in chunks[-1]
    assert all(len(c) <= 1600 for c in chunks)


def test_chunk_by_paragraph_keeps_a_short_tail_as_its_own_chunk():
    # the previous chunk was flushed because the tail would not fit, so a
    # tail shorter than min_chars stays on its own rather than being merged
    text = "x" * 1000 + "\n\n" + "y" * 1590 + "\n\n" + "z" * 50
    chunks = embed.chunk_by_paragraph(text, min_chars=300, max_chars=1600)
    assert [len(c) for c in chunks] == [1000, 1590, 50]


# ---------------------------------------------------------------------------
# permutation_test()
# ---------------------------------------------------------------------------

import pytest  # noqa: E402


def _clusters(seed=0, n_a=8, n_b=10, noise_a=0.05, noise_b=0.05, dim=16):
    rng = np.random.default_rng(seed)
    centre_a = np.zeros(dim); centre_a[0] = 1.0
    centre_b = np.zeros(dim); centre_b[1] = 1.0
    a = centre_a + noise_a * rng.standard_normal((n_a, dim))
    b = centre_b + noise_b * rng.standard_normal((n_b, dim))
    return a, b


def test_permutation_test_well_separated_groups_give_the_minimum_p_not_zero():
    a, b = _clusters()
    p = embed.permutation_test(a, b, n_perm=199)
    assert p == approx(1 / 200)
    assert p > 0


def test_permutation_test_identical_distributions_give_a_large_p():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((15, 16))
    b = rng.standard_normal((15, 16))
    assert embed.permutation_test(a, b, n_perm=199) > 0.2


def test_permutation_test_is_symmetric_in_corpus_order():
    # tight A, diffuse B: the old within(A) - between statistic gave
    # p ~ 0 one way round and ~ 1 the other
    a, b = _clusters(noise_a=0.02, noise_b=1.5, n_a=10, n_b=14)
    assert embed.permutation_test(a, b, n_perm=199) == embed.permutation_test(b, a, n_perm=199)


def test_permutation_test_rejects_a_one_chunk_corpus():
    a, b = _clusters()
    with pytest.raises(ValueError, match="at least 2 chunks"):
        embed.permutation_test(a[:1], b)
    with pytest.raises(ValueError, match="at least 2 chunks"):
        embed.permutation_test(a, b[:1])


def test_permutation_test_bounds_and_reproducibility():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((6, 8))
    b = rng.standard_normal((7, 8))
    p1 = embed.permutation_test(a, b, n_perm=99, seed=7)
    p2 = embed.permutation_test(a, b, n_perm=99, seed=7)
    assert p1 == p2
    assert 1 / 100 <= p1 <= 1.0


def test_similarity_gap_is_symmetric():
    a, b = _clusters(noise_b=0.8)
    assert embed.similarity_gap(a, b) == approx(embed.similarity_gap(b, a))


# ---------------------------------------------------------------------------
# load_embedded_corpora()
# ---------------------------------------------------------------------------

def test_load_embedded_corpora_missing_files_point_at_embed_corpus(tmp_path):
    with pytest.raises(FileNotFoundError, match="xlt-embed-corpus"):
        embed.load_embedded_corpora(tmp_path, ["x"])


# ---------------------------------------------------------------------------
# Hardening review 2: stale caches, chunking guards, permutation_test input
# ---------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, dim=4):
        self.dim, self.calls = dim, 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.ones((len(texts), self.dim))


def test_embed_chunks_recomputes_a_cache_with_the_wrong_row_count(tmp_path):
    cache = tmp_path / "x.npy"
    np.save(cache, np.zeros((8, 4)))
    chunks = [{"text": f"t{i}"} for i in range(3)]
    model = _FakeModel()
    out = embed.embed_chunks(chunks, model, cache)
    assert model.calls == 1
    assert out.shape == (3, 4)
    assert np.load(cache).shape == (3, 4)


def test_embed_chunks_reuses_a_cache_that_matches(tmp_path):
    cache = tmp_path / "x.npy"
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    model = _FakeModel()
    embed.embed_chunks(chunks, model, cache, model_name="m")
    out = embed.embed_chunks(chunks, model, cache, model_name="m")
    assert model.calls == 1
    assert out.shape == (3, 4)


@pytest.mark.parametrize("texts, name", [(["a", "b", "x"], "m"), (["a", "b", "c"], "other")])
def test_embed_chunks_recomputes_when_texts_or_model_change(tmp_path, texts, name):
    cache = tmp_path / "x.npy"
    model = _FakeModel()
    embed.embed_chunks([{"text": t} for t in "abc"], model, cache, model_name="m")
    embed.embed_chunks([{"text": t} for t in texts], model, cache, model_name=name)
    assert model.calls == 2


def test_embed_chunks_recomputes_a_cache_without_a_key(tmp_path):
    cache = tmp_path / "x.npy"
    np.save(cache, np.zeros((3, 4)))
    model = _FakeModel()
    embed.embed_chunks([{"text": t} for t in "abc"], model, cache)
    assert model.calls == 1


def test_load_embedded_corpora_rejects_a_0d_array(tmp_path):
    embed.save_corpus([{"text": "a"}], str(tmp_path / "x.json"))
    np.save(tmp_path / "x.npy", np.float64(1.0))
    with pytest.raises(ValueError, match="not a 2-D array.*xlt-embed-corpus"):
        embed.load_embedded_corpora(tmp_path, ["x"])


def test_load_embedded_corpora_rejects_a_1d_array_without_contradicting_itself(tmp_path):
    embed.save_corpus([{"text": "a"}, {"text": "b"}, {"text": "c"}], str(tmp_path / "x.json"))
    np.save(tmp_path / "x.npy", np.zeros(3))
    with pytest.raises(ValueError) as exc:
        embed.load_embedded_corpora(tmp_path, ["x"])
    assert "not a 2-D array" in str(exc.value)
    assert "3 embeddings" not in str(exc.value)


def test_permutation_test_rejects_fractional_n_perm():
    a = np.eye(4)[:2]
    with pytest.raises(ValueError, match="whole number"):
        embed.permutation_test(a, np.eye(4)[2:], n_perm=2.5)


def test_load_embedded_corpora_rejects_mismatched_chunks_and_embeddings(tmp_path):
    embed.save_corpus([{"text": "a"}, {"text": "b"}], str(tmp_path / "x.json"))
    np.save(tmp_path / "x.npy", np.zeros((5, 4)))
    with pytest.raises(ValueError, match="2 chunks .* 5 embeddings.*xlt-embed-corpus"):
        embed.load_embedded_corpora(tmp_path, ["x"])


@pytest.mark.parametrize("kwargs", [{"max_chars": 0}, {"max_chars": -5},
                                    {"min_chars": -1}])
def test_chunk_by_paragraph_rejects_bad_limits(kwargs):
    with pytest.raises(ValueError):
        embed.chunk_by_paragraph("some text " * 20, **kwargs)


def test_permutation_test_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        embed.permutation_test(np.ones(5), np.ones(5))


def test_permutation_test_rejects_mismatched_dimensions():
    a, _ = _clusters(dim=16)
    _, b = _clusters(dim=8)
    with pytest.raises(ValueError, match="dimensions differ"):
        embed.permutation_test(a, b)


@pytest.mark.parametrize("n_perm", [0, -3])
def test_permutation_test_rejects_non_positive_n_perm(n_perm):
    a, b = _clusters()
    with pytest.raises(ValueError, match="n_perm"):
        embed.permutation_test(a, b, n_perm=n_perm)


def test_permutation_report_shows_the_actual_error():
    from cross_lingual_toolkit.analysis_a_umap import permutation_report
    a, _ = _clusters(dim=16)
    _, b = _clusters(dim=8)
    report = permutation_report(a, b, "A", "B", 10)
    assert "dimensions differ" in report
    assert "not enough chunks" not in report and "min_chars" not in report


def test_permutation_report_keeps_the_chunking_hint_for_too_few_chunks():
    from cross_lingual_toolkit.analysis_a_umap import permutation_report
    a, b = _clusters()
    report = permutation_report(a[:1], b, "A", "B", 10)
    assert "at least 2 chunks" in report and "min_chars" in report
