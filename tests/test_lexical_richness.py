from pytest import approx

from cross_lingual_toolkit.lexical_richness import ttr, mattr


# ---------------------------------------------------------------------------
# ttr()
# ---------------------------------------------------------------------------

def test_ttr_all_unique_tokens_is_one():
    assert ttr(["a", "b", "c"]) == approx(1.0)


def test_ttr_repeated_tokens():
    # 2 unique / 4 total
    assert ttr(["a", "a", "b", "b"]) == approx(0.5)


def test_ttr_empty_list_returns_zero():
    assert ttr([]) == 0.0


# ---------------------------------------------------------------------------
# mattr()
# ---------------------------------------------------------------------------

def test_mattr_falls_back_to_scalar_ttr_for_short_text():
    tokens = ["a", "b", "c"]
    assert mattr(tokens, window=10) == approx(ttr(tokens))


def test_mattr_returns_one_value_per_window_start():
    tokens = ["a", "b", "c", "a", "b"]
    result = mattr(tokens, window=3)
    # 5 tokens, window 3 -> 5 - 3 + 1 = 3 window positions
    assert len(result) == 3
    assert result[0] == approx(ttr(tokens[0:3]))
    assert result[1] == approx(ttr(tokens[1:4]))
    assert result[2] == approx(ttr(tokens[2:5]))
