from pytest import approx

from cross_lingual_toolkit.cross_corpus_frequency import (
    dunning_g2, tokenize_generic, count_clusters, normalize,
)


# ---------------------------------------------------------------------------
# dunning_g2()
# ---------------------------------------------------------------------------

def test_dunning_g2_both_zero_is_zero():
    assert dunning_g2(0, 0, 1000, 1000) == 0.0


def test_dunning_g2_equal_proportions_is_near_zero():
    # same rate in both corpora -> no over/under-representation
    assert dunning_g2(10, 100, 1000, 10000) == approx(0.0, abs=1e-9)


def test_dunning_g2_positive_when_a_overrepresented():
    g2 = dunning_g2(100, 10, 1000, 1000)
    assert g2 > 0


def test_dunning_g2_negative_when_b_overrepresented():
    g2 = dunning_g2(10, 100, 1000, 1000)
    assert g2 < 0


# ---------------------------------------------------------------------------
# tokenize_generic() — regression test for the Unicode-letter genericity fix
# ---------------------------------------------------------------------------

def test_tokenize_generic_handles_multiple_languages_diacritics():
    text = "café, à la carte, größer naïve wörld"
    assert tokenize_generic(text) == [
        "café", "la", "carte", "größer", "naïve", "wörld",
    ]


def test_tokenize_generic_drops_single_char_and_digit_tokens():
    assert tokenize_generic("a bb 123 cc4") == ["bb", "cc"]


# ---------------------------------------------------------------------------
# count_clusters() / normalize()
# ---------------------------------------------------------------------------

def test_count_clusters_counts_configured_forms():
    clusters = {
        "cluster_a": {"label": "A", "en": ["apple", "pear"]},
        "cluster_b": {"label": "B", "en": ["car", "train"]},
    }
    # "pear" appears twice, so it contributes twice to cluster_a's count —
    # this counts every token occurrence, not distinct types.
    tokens = ["apple", "pear", "pear", "car", "other"]
    counts = count_clusters(tokens, clusters, "en")
    assert counts == {"cluster_a": 3, "cluster_b": 1}


def test_count_clusters_only_counts_first_matching_cluster():
    # a token matching more than one cluster's forms is only counted once
    clusters = {
        "a": {"en": ["x"]},
        "b": {"en": ["x"]},
    }
    counts = count_clusters(["x"], clusters, "en")
    assert sum(counts.values()) == 1


def test_normalize_scales_per_10k_by_default():
    assert normalize({"a": 5}, total=100) == {"a": 500.0}


def test_normalize_respects_custom_per():
    assert normalize({"a": 1}, total=2, per=10) == {"a": 5.0}
