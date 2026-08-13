from cross_lingual_toolkit.adjective_window import (
    collect_windows, adjectives_from_window_heuristic,
)


# ---------------------------------------------------------------------------
# collect_windows()
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


# ---------------------------------------------------------------------------
# adjectives_from_window_heuristic() — regression test for the genericity
# fix: no suffixes configured means no guessing, not a hardcoded fallback.
# ---------------------------------------------------------------------------

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
