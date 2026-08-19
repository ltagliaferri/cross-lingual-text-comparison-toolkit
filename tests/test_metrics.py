"""
Tests for the corpus-independent measures. These import only
cross_lingual_toolkit.metrics, so they run without matplotlib, spaCy or
any downloaded model.
"""

from pytest import approx

from cross_lingual_toolkit.metrics import (
    adjectives_from_window_heuristic, collect_windows, count_cluster_forms,
    count_cluster_terms, dep_variants, dunning_g2, mattr, normalize, ttr,
)


# ---------------------------------------------------------------------------
# ttr() / mattr()
# ---------------------------------------------------------------------------

def test_ttr_all_unique_tokens_is_one():
    assert ttr(["a", "b", "c"]) == approx(1.0)


def test_ttr_repeated_tokens():
    # 2 unique / 4 total
    assert ttr(["a", "a", "b", "b"]) == approx(0.5)


def test_ttr_empty_list_returns_zero():
    assert ttr([]) == 0.0


def test_mattr_returns_one_value_per_window_start():
    tokens = ["a", "b", "c", "a", "b"]
    result = mattr(tokens, window=3)
    # 5 tokens, window 3 -> 5 - 3 + 1 = 3 window positions
    assert len(result) == 3
    assert result[0] == approx(ttr(tokens[0:3]))
    assert result[1] == approx(ttr(tokens[1:4]))
    assert result[2] == approx(ttr(tokens[2:5]))


def test_mattr_short_text_returns_a_single_element_list():
    # regression: mattr used to return a bare float here, which made the
    # caller's `enumerate(...)` and np.convolve blow up on any text
    # shorter than the window.
    tokens = ["a", "b", "c"]
    result = mattr(tokens, window=10)
    assert result == [approx(ttr(tokens))]


def test_mattr_empty_text_is_still_iterable():
    assert mattr([], window=10) == [0.0]


def test_mattr_rejects_a_non_positive_window():
    import pytest
    with pytest.raises(ValueError):
        mattr(["a"], window=0)


# ---------------------------------------------------------------------------
# normalize() / count_cluster_terms() / count_cluster_forms()
# ---------------------------------------------------------------------------

def test_normalize_scales_per_10k_by_default():
    assert normalize({"a": 5}, total=100) == {"a": 500.0}


def test_normalize_respects_custom_per():
    assert normalize({"a": 1}, total=2, per=10) == {"a": 5.0}


def test_normalize_empty_corpus_returns_zeros_not_a_zero_division():
    assert normalize({"a": 0, "b": 0}, total=0) == {"a": 0.0, "b": 0.0}


def test_count_cluster_terms_sums_every_configured_term():
    clusters = {
        "cluster_a": {"label": "A", "terms": ["apple", "pear"]},
        "cluster_b": {"label": "B", "terms": ["car"]},
    }
    tokens = ["apple", "pear", "pear", "car", "other"]
    assert count_cluster_terms(tokens, clusters) == {"cluster_a": 3, "cluster_b": 1}


def test_count_cluster_forms_counts_configured_forms():
    clusters = {
        "cluster_a": {"label": "A", "en": ["apple", "pear"]},
        "cluster_b": {"label": "B", "en": ["car", "train"]},
    }
    # "pear" appears twice, so it contributes twice to cluster_a's count —
    # this counts every token occurrence, not distinct types.
    tokens = ["apple", "pear", "pear", "car", "other"]
    counts = count_cluster_forms(tokens, clusters, "en")
    assert counts == {"cluster_a": 3, "cluster_b": 1}


def test_count_cluster_forms_only_counts_first_matching_cluster():
    # a token matching more than one cluster's forms is only counted once
    clusters = {"a": {"en": ["x"]}, "b": {"en": ["x"]}}
    counts = count_cluster_forms(["x"], clusters, "en")
    assert sum(counts.values()) == 1


# ---------------------------------------------------------------------------
# dunning_g2()
# ---------------------------------------------------------------------------

def test_dunning_g2_both_zero_is_zero():
    assert dunning_g2(0, 0, 1000, 1000) == 0.0


def test_dunning_g2_equal_proportions_is_near_zero():
    # same rate in both corpora -> no over/under-representation
    assert dunning_g2(10, 100, 1000, 10000) == approx(0.0, abs=1e-9)


def test_dunning_g2_positive_when_a_overrepresented():
    assert dunning_g2(100, 10, 1000, 1000) > 0


def test_dunning_g2_negative_when_b_overrepresented():
    assert dunning_g2(10, 100, 1000, 1000) < 0


def test_dunning_g2_empty_corpus_is_zero_not_a_zero_division():
    assert dunning_g2(0, 5, 0, 1000) == 0.0
    assert dunning_g2(5, 0, 1000, 0) == 0.0


# ---------------------------------------------------------------------------
# dep_variants()
# ---------------------------------------------------------------------------

def test_dep_variants_covers_both_passive_conventions():
    # spaCy's English models emit "nsubjpass"; UD-trained models and
    # Stanza emit "nsubj:pass". A config saying "nsubj" must match both.
    variants = dep_variants("nsubj")
    assert {"nsubj", "nsubjpass", "nsubj:pass"} <= variants


def test_dep_variants_treats_obj_and_dobj_as_the_same_role():
    assert "dobj" in dep_variants("obj")
    assert "obj" in dep_variants("dobj")


def test_dep_variants_passes_through_an_unknown_role():
    assert "iobj" in dep_variants("iobj")


# ---------------------------------------------------------------------------
# collect_windows() / adjectives_from_window_heuristic()
# ---------------------------------------------------------------------------

def test_collect_windows_excludes_anchor_and_respects_size():
    tokens = ["a", "b", "ANCHOR", "c", "d", "e"]
    windows = collect_windows(tokens, {"ANCHOR"}, window=2)
    assert len(windows) == 1
    assert windows[0] == ["a", "b", "c", "d"]


def test_collect_windows_clips_at_start_of_list():
    tokens = ["ANCHOR", "a", "b", "c"]
    windows = collect_windows(tokens, {"ANCHOR"}, window=5)
    assert windows[0] == ["a", "b", "c"]


def test_collect_windows_clips_at_end_of_list():
    tokens = ["a", "b", "c", "ANCHOR"]
    windows = collect_windows(tokens, {"ANCHOR"}, window=5)
    assert windows[0] == ["a", "b", "c"]


def test_collect_windows_finds_every_occurrence():
    tokens = ["ANCHOR", "x", "ANCHOR"]
    windows = collect_windows(tokens, {"ANCHOR"}, window=1)
    assert len(windows) == 2


def test_heuristic_returns_empty_when_no_suffixes_configured():
    # "wordzzq" would match a configured suffix, but with no suffixes
    # passed in, the heuristic must not silently assume any language's
    # morphology — it should find nothing rather than guess.
    counter = adjectives_from_window_heuristic(["wordzzq", "other"], stopwords=set(), suffixes=())
    assert counter == {}


def test_heuristic_matches_configured_suffixes():
    counter = adjectives_from_window_heuristic(
        ["wordzzq", "short", "alsozzq"], stopwords=set(), suffixes=("zzq",)
    )
    assert counter == {"wordzzq": 1, "alsozzq": 1}


def test_heuristic_skips_stopwords_and_short_tokens():
    counter = adjectives_from_window_heuristic(
        ["zzq", "wordzzq"], stopwords={"wordzzq"}, suffixes=("zzq",)
    )
    # "zzq" is too short (<=4 chars) to count; "wordzzq" is stopword-excluded
    assert counter == {}
