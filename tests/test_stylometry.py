"""
Same-language stylometry (stylometry.py): Delta measures, sampling,
features, selectors, validation, attribution, General Imposters, rolling
Delta, and the CLI — over synthetic "authors" who differ only in how often
they use a handful of function words.
"""

import csv
import json
import os
import random

import numpy as np
import pytest

from cross_lingual_toolkit import corpus, stylometry as st


FUNCTION = ["the", "of", "and", "to", "in", "that", "it", "with", "as", "for"]
CONTENT = ["garden", "stone", "light", "river", "house", "road", "tree", "field"]
PROFILES = {   # extra weight on three function words per author
    "A": {"the": 8, "of": 6, "in": 5},
    "B": {"and": 8, "to": 6, "that": 5},
    "C": {"it": 8, "with": 6, "as": 5},
    "D": {"for": 8, "the": 1, "to": 5},
    "E": {"with": 7, "of": 1, "that": 6},
}


def author_text(author, n_words, seed):
    """Deterministic text whose function-word mix identifies `author`."""
    vocab = FUNCTION + CONTENT
    weights = [PROFILES[author].get(w, 2) for w in FUNCTION] + [1.5] * len(CONTENT)
    return " ".join(random.Random(seed).choices(vocab, weights=weights, k=n_words))


def _z(groups, sample_words=200, mfw=50):
    """z-scored samples for {name: text}; returns (z, labels)."""
    samples, labels = [], []
    for name, text in groups.items():
        group_samples, _ = st.make_samples(text.split(), sample_words)
        samples += group_samples
        labels += [name] * len(group_samples)
    words, _ = st.select_features(samples, mfw)
    z, _, _, _ = st.zscore(st.relative_frequencies(samples, words))
    return z, np.array(labels)


# ---------------------------------------------------------------------------
# Delta measures
# ---------------------------------------------------------------------------

def test_burrows_delta_is_the_mean_absolute_difference():
    assert st.burrows_delta([1.0, -1.0, 0.0], [0.0, 1.0, 2.0]) == pytest.approx(5 / 3)


def test_cosine_delta_is_one_minus_cosine_similarity():
    assert st.cosine_delta([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert st.cosine_delta([1.0, 1.0], [2.0, 2.0]) == pytest.approx(0.0)
    assert st.cosine_delta([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)
    assert st.cosine_delta([3.0, 4.0], [4.0, 3.0]) == pytest.approx(1 - 24 / 25)


# ---------------------------------------------------------------------------
# Features and samples
# ---------------------------------------------------------------------------

def test_zscore_drops_a_zero_sd_feature():
    matrix = np.array([[0.1, 0.5, 0.0], [0.3, 0.5, 0.2]])
    z, mean, sd, keep = st.zscore(matrix)
    assert keep.tolist() == [True, False, True]
    assert int((~keep).sum()) == 1
    assert z.tolist() == [[pytest.approx(-1.0), pytest.approx(-1.0)],
                         [pytest.approx(1.0), pytest.approx(1.0)]]
    assert mean.tolist() == pytest.approx([0.2, 0.1])
    assert sd.tolist() == pytest.approx([0.1, 0.1])     # population SD


def test_mfw_ties_break_alphabetically_and_excluded_words_never_count():
    samples = [["pear", "apple", "fig", "fig"], ["Rome", "apple", "pear", "fig"]]
    samples = [[t.lower() for t in s] for s in samples]
    words, counts = st.select_features(samples, 3, exclude_words=["rome"])
    assert words == ["fig", "apple", "pear"]            # 3, then 2-2 alphabetical
    assert counts == [3, 2, 2]
    assert "rome" not in st.select_features(samples, 10, exclude_words=["rome"])[0]
    # the core compares exclude_words as given; normalizing is the caller's job
    assert "rome" in st.select_features(samples, 10, exclude_words=["Rome"])[0]


def test_the_script_normalizes_exclude_words(study, tmp_path):
    study["stylometry"]["exclude_words"] = ["THE", " Of"]
    study["stylometry"]["verification"] = []
    del study["stylometry"]["rolling"]
    st.analyze(study)
    words = [r["word"] for r in _read_csv(tmp_path / "results" / "stylometry_features.csv")]
    assert "the" not in words and "of" not in words and "and" in words


def test_stylo_core_imports_without_config_corpus_or_plotting(monkeypatch):
    import importlib
    import sys

    import cross_lingual_toolkit as package
    monkeypatch.setattr(package, "stylo_core", package.stylo_core)   # restored afterwards
    for name in list(sys.modules):
        if name.startswith("cross_lingual_toolkit.stylo"):
            monkeypatch.delitem(sys.modules, name)
    for blocked in ("cross_lingual_toolkit.corpus", "cross_lingual_toolkit.metrics",
                    "matplotlib", "matplotlib.pyplot"):
        monkeypatch.setitem(sys.modules, blocked, None)   # importing it raises ImportError
    core = importlib.import_module("cross_lingual_toolkit.stylo_core")
    samples, _ = core.make_samples(author_text("A", 400, 1).split(), 200)
    words, _ = core.select_features(samples, 10)
    z, _, _, _ = core.zscore(core.relative_frequencies(samples, words))
    assert z.shape[0] == 2
    with pytest.raises(ImportError):
        importlib.import_module("cross_lingual_toolkit.stylometry")


def test_stylometry_re_exports_the_core():
    from cross_lingual_toolkit import stylo_core
    for name in ("make_samples", "select_features", "relative_frequencies", "zscore",
                 "burrows_delta", "cosine_delta", "distance_function", "centroids",
                 "group_distances", "leave_one_out", "general_imposters",
                 "rolling_windows", "rolling_delta"):
        assert getattr(st, name) is getattr(stylo_core, name), name


def test_samples_cut_at_sample_words_and_drop_a_short_remainder():
    tokens = [str(i) for i in range(1040)]
    samples, short = st.make_samples(tokens, 200)
    assert [len(x) for x in samples] == [200] * 5        # remainder 40 < 100 dropped
    assert not short
    samples, _ = st.make_samples([str(i) for i in range(1100)], 200)
    assert [len(x) for x in samples] == [200] * 5 + [100]   # half a sample is kept


def test_a_group_shorter_than_one_sample_is_one_short_sample():
    samples, short = st.make_samples(["a"] * 50, 200)
    assert short and [len(x) for x in samples] == [50]
    assert st.make_samples([], 200) == ([], True)


def test_strip_patterns_remove_a_formula_before_counting():
    text = "Salutem plurimam dicit.\nthe garden and the stone\nVale semper.\n"
    patterns = st.compile_patterns([r"^Salutem.*$", r"^Vale.*$"])
    tokens = corpus.tokenize(st.strip_text(text, patterns), remove_stopwords=False)
    words, _ = st.select_features([tokens], 100)
    assert "salutem" not in words and "vale" not in words
    assert "garden" in words


# ---------------------------------------------------------------------------
# A study on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    for group, author, seed in (("Early", "A", 1), ("Late", "A", 2)):
        (root / "letters" / group).mkdir(parents=True)
        for i in range(1, 4):
            (root / "letters" / group / f"{i:03d}.txt").write_text(
                author_text(author, 700, seed * 10 + i), encoding="utf-8")
    for name, author, seed in (("b", "B", 3), ("c", "C", 4), ("d", "D", 5), ("e", "E", 6)):
        (root / f"{name}.txt").write_text(author_text(author, 2000, seed), encoding="utf-8")
    (root / "mixed.txt").write_text(author_text("A", 3000, 7) + " " + author_text("B", 3000, 8),
                                    encoding="utf-8")

    def single(path, lang="en"):
        return {"type": "single_file", "path": path, "author": "x", "work": path,
                "language": lang}

    config = {
        "corpus_root": str(root),
        "languages": {"en": {}, "fr": {}},
        "corpora": {
            "letters": {"type": "collection", "path": "letters", "author": "a",
                        "work": "letters", "language": "en", "groups": ["Early", "Late"]},
            "b_work": single("b.txt"), "c_work": single("c.txt"), "d_work": single("d.txt"),
            "e_work": single("e.txt"), "mixed": single("mixed.txt"),
            "french": single("b.txt", "fr"),
        },
        "stylometry": {
            "_comment": "a comment",
            "groups": {
                "_comment": "a comment among the groups",
                "A_early": ["letters:Early"],
                "A_late": ["letters:Late/001", "letters:Late/002", "letters:Late/003"],
                "B": ["b_work"], "C": ["c_work"], "D": ["d_work"], "E": ["e_work"],
            },
            "mfw": 30,
            "sample_words": 200,
            "imposters": ["C", "D", "E"],
            "verification": [{"questioned": "A_late", "candidate": "A_early"},
                             {"questioned": "A_late", "candidate": "B"}],
            "iterations": 20,
            "rolling": {"corpus": "mixed", "references": ["A_early", "B"],
                        "window_words": 1000, "step_words": 500},
        },
    }
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return corpus.load_config(str(path))


def _validate(config, check_paths=True):
    return corpus.validate_config(config, require=st.REQUIRES, check_paths=check_paths)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_study_fixture_is_valid(study):
    assert _validate(study) == []


def test_selectors_resolve_a_corpus_a_group_and_an_item(study):
    cache = {}
    assert st.selector_texts(study, "b_work", cache) == [
        (open(os.path.join(study["corpus_root"], "b.txt"), encoding="utf-8").read())]
    assert len(st.selector_texts(study, "letters", cache)) == 6
    early = st.selector_texts(study, "letters:Early", cache)
    assert len(early) == 3
    one = st.selector_texts(study, "letters:Late/002", cache)
    assert one == [open(os.path.join(study["corpus_root"], "letters", "Late", "002.txt"),
                        encoding="utf-8").read()]
    assert st.parse_selector("letters:Late/002") == ("letters", "Late", "002")


def test_groups_file_adds_to_groups(study, tmp_path):
    (tmp_path / "groups.csv").write_text(
        "group,selector\nB,letters:Early/001\nNew,c_work\n\n", encoding="utf-8")
    study["stylometry"]["groups_file"] = "groups.csv"   # next to the config
    groups = st.stylometry_groups(study)
    assert groups["B"] == ["b_work", "letters:Early/001"]
    assert groups["New"] == ["c_work"]
    assert list(groups)[:2] == ["A_early", "A_late"]
    # a file-defined group can be referenced, and the overlap is a warning
    study["stylometry"]["imposters"] = ["C", "D", "E", "New"]
    study["stylometry"]["verification"] = []
    warnings = _validate(study)
    assert any('"A_early" and "B" share 1 text selection(s): "letters:Early/001"' in w
               for w in warnings)
    assert any('"C" and "New"' in w for w in warnings)


def test_overlapping_groups_are_one_warning_per_pair(study):
    sty = study["stylometry"]
    sty["groups"]["A_all"] = ["letters:Late"]     # covers all three A_late items
    sty["groups"]["A_late"] = ["letters:Late/001", "letters:Late/002", "letters:Late/003"]
    warnings = [w for w in _validate(study) if '"A_late" and "A_all"' in w]
    assert len(warnings) == 1
    assert 'share 3 text selection(s): "letters:Late/001", "letters:Late/002", ' \
        '"letters:Late/003"' in warnings[0]
    assert "more" not in warnings[0]


def test_a_missing_groups_file_is_a_problem(study):
    study["stylometry"]["groups_file"] = "nowhere.csv"
    with pytest.raises(ValueError, match="nowhere.csv"):
        _validate(study)


def test_a_missing_groups_file_is_not_a_problem_without_path_checks(study):
    study["stylometry"]["groups_file"] = "nowhere.csv"
    assert _validate(study, check_paths=False) == []


def test_an_existing_groups_file_is_still_validated_without_path_checks(study, tmp_path):
    (tmp_path / "groups.csv").write_text("group,selector\nNew,nope\n", encoding="utf-8")
    study["stylometry"]["groups_file"] = "groups.csv"
    with pytest.raises(ValueError, match='group "New": selector "nope"'):
        _validate(study, check_paths=False)


def test_unknown_selectors_are_reported_with_their_group(study):
    groups = study["stylometry"]["groups"]
    groups["B"] = ["nope"]
    groups["C"] = ["letters:Middle"]
    groups["D"] = ["letters:Late/999"]
    with pytest.raises(ValueError) as exc:
        _validate(study)
    message = str(exc.value)
    assert 'group "B": selector "nope"' in message
    assert 'group "C": selector "letters:Middle"' in message
    assert 'group "D": selector "letters:Late/999"' in message and "no such item" in message


def test_item_existence_is_only_checked_with_paths(study):
    study["stylometry"]["groups"]["D"] = ["letters:Late/999"]
    assert _validate(study, check_paths=False) == []


def test_the_same_text_twice_in_one_group_is_a_warning(study):
    study["stylometry"]["groups"]["A_early"] = ["letters:Early", "letters:Early/001"]
    warnings = _validate(study)
    assert any('"A_early" includes the same text twice' in w for w in warnings)


def test_validation_reports_every_problem_together(study):
    sty = study["stylometry"]
    sty["groups"]["F"] = ["french"]
    sty["strip_patterns"] = ["(unclosed"]
    sty["mfw"] = 0
    sty["sample_words"] = -5
    sty["iterations"] = 1.5
    sty["feature_fraction"] = 1.5
    sty["imposters"] = ["C", "Nobody"]
    sty["verification"] = [{"questioned": "A_late", "candidate": "A_late"},
                           {"questioned": "Ghost", "candidate": "C"}]
    sty["rolling"]["window_words"] = 0
    sty["distance"] = "euclid"
    sty["imposters"] = []
    with pytest.raises(ValueError) as exc:
        _validate(study)
    message = str(exc.value)
    for fragment in ("only works within one language", "en: ", "fr: french",
                     "(unclosed", "not a valid regular expression",
                     'stylometry["mfw"]', 'stylometry["sample_words"]',
                     'stylometry["iterations"]', "feature_fraction",
                     "same group", "'Ghost'", "imposters\"] is empty",
                     'rolling"]["window_words"]', "euclid"):
        assert fragment in message, fragment


def test_imposter_problems_are_reported_together(study):
    # kept apart from the test above, which needs an empty imposters list
    study["stylometry"]["imposters"] = ["C", "Nobody"]
    study["stylometry"]["verification"] = [{"questioned": "C", "candidate": "A_late"}]
    with pytest.raises(ValueError) as exc:
        _validate(study)
    assert 'undefined group "Nobody"' in str(exc.value)
    assert '"C" is also in stylometry["imposters"]' in str(exc.value)


def test_verification_without_imposters_is_a_problem(study):
    study["stylometry"]["imposters"] = []
    with pytest.raises(ValueError, match="imposters.*is empty"):
        _validate(study)


def test_rolling_corpus_in_another_language_is_a_problem(study):
    study["stylometry"]["rolling"]["corpus"] = "french"
    with pytest.raises(ValueError, match="one language"):
        _validate(study)


def test_rolling_corpus_outside_the_corpora_restriction_is_a_problem(study):
    study["stylometry"]["corpora"] = ["letters", "b_work", "c_work", "d_work", "e_work"]
    with pytest.raises(ValueError, match='rolling"\\]\\["corpus"\\] "mixed" is not in'):
        _validate(study)
    study["stylometry"]["corpora"].append("mixed")
    _validate(study)


def test_non_string_names_are_problems_not_crashes(study):
    study["stylometry"]["rolling"]["corpus"] = ["mixed"]
    study["stylometry"]["verification"] = [{"questioned": ["A_late"], "candidate": "B"}]
    with pytest.raises(ValueError) as exc:
        _validate(study)
    assert "rolling" in str(exc.value) and "['A_late']" in str(exc.value)


def test_a_missing_section_is_a_problem(study):
    del study["stylometry"]
    with pytest.raises(ValueError, match='no "stylometry" section'):
        _validate(study)


# ---------------------------------------------------------------------------
# Attribution, imposters, rolling
# ---------------------------------------------------------------------------

def test_leave_one_out_separates_two_authors():
    z, labels = _z({"A": author_text("A", 2000, 1), "B": author_text("B", 2000, 2)})
    for distance in (st.cosine_delta, st.burrows_delta):
        rows, confusion, accuracy = st.leave_one_out(z, labels, ["A", "B"], distance)
        assert accuracy == 1.0
        assert confusion.tolist() == [[10, 0], [0, 10]]
        assert all(true == nearest for true, nearest, _ in rows)


def test_general_imposters_scores_same_author_high_and_other_author_low():
    texts = {"A1": author_text("A", 2000, 1), "A2": author_text("A", 2000, 2),
             "A3": author_text("A", 2000, 3), "B": author_text("B", 2000, 4),
             "C": author_text("C", 2000, 5), "D": author_text("D", 2000, 6)}
    z, labels = _z(texts)
    args = (st.cosine_delta, 50, 0.5, 42)
    other_pool = z[np.isin(labels, ["C", "D"])]
    high = st.general_imposters(z[labels == "A1"], z[labels == "A2"], other_pool, *args)
    assert high >= 0.9
    # the questioned author's own texts among the imposters beat a wrong candidate
    pool = z[np.isin(labels, ["A3", "C"])]
    low = st.general_imposters(z[labels == "A1"], z[labels == "B"], pool, *args)
    assert low <= 0.1
    again = st.general_imposters(z[labels == "A1"], z[labels == "A2"], other_pool, *args)
    assert again == high


def test_rolling_delta_moves_from_one_author_to_the_other():
    texts = {"A": author_text("A", 2000, 1), "B": author_text("B", 2000, 2)}
    samples, labels = [], []
    for name, text in texts.items():
        s, _ = st.make_samples(text.split(), 200)
        samples += s
        labels += [name] * len(s)
    words, _ = st.select_features(samples, 50)
    z, mean, sd, keep = st.zscore(st.relative_frequencies(samples, words))
    cents = st.centroids(z, labels, ["A", "B"])
    mixed = (author_text("A", 3000, 3) + " " + author_text("B", 3000, 4)).split()
    windows = st.rolling_windows(mixed, 1000, 500)
    assert [s for s, _ in windows] == list(range(0, 5001, 500))
    scored = st.rolling_delta(windows, words, keep, mean, sd, cents, st.cosine_delta)
    first, last = scored[0][1], scored[-1][1]
    assert first["A"] < first["B"]
    assert last["B"] < last["A"]


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

def test_main_writes_every_output(study, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["xlt-stylometry", "--config", study["_config_path"]])
    st.main()
    out = capsys.readouterr().out
    for name in ("distances", "attribution", "features", "verification", "rolling"):
        assert os.path.exists(tmp_path / "results" / f"stylometry_{name}.csv"), name
    for name in ("distances", "rolling"):
        assert os.path.exists(tmp_path / "visualizations" / f"stylometry_{name}.png"), name
    assert "not a p-value" in out
    assert "Accuracy: " in out

    distances = _read_csv(tmp_path / "results" / "stylometry_distances.csv")
    assert [r["group"] for r in distances] == ["A_early", "A_late", "B", "C", "D", "E"]
    verification = {r["candidate"]: r for r in
                    _read_csv(tmp_path / "results" / "stylometry_verification.csv")}
    assert float(verification["A_early"]["score"]) > float(verification["B"]["score"])
    assert verification["A_early"]["n_questioned_samples"] == "11"   # 2100 tokens: 10 full + one half sample
    assert verification["A_early"]["iterations"] == "20"
    features = _read_csv(tmp_path / "results" / "stylometry_features.csv")
    assert len(features) == len(FUNCTION + CONTENT)       # fewer types than mfw=30
    rolling = _read_csv(tmp_path / "results" / "stylometry_rolling.csv")
    assert list(rolling[0]) == ["window", "start_token", "distance_A_early", "distance_B"]


def test_a_short_group_and_a_rolling_reference_overlap_are_warned(study, tmp_path, capsys):
    (tmp_path / "corpus" / "tiny.txt").write_text(author_text("C", 60, 9), encoding="utf-8")
    study["corpora"]["tiny"] = {"type": "single_file", "path": "tiny.txt", "author": "x",
                                "work": "t", "language": "en"}
    study["stylometry"]["groups"]["Tiny"] = ["tiny"]
    study["stylometry"]["groups"]["B"] = ["b_work", "mixed"]
    study["stylometry"]["verification"] = []
    _validate(study)
    st.analyze(study)
    out = capsys.readouterr().out
    assert 'group "Tiny" has 60 tokens' in out
    assert 'also part of reference group "B"' in out


def test_a_small_imposter_pool_is_warned(study, capsys):
    study["stylometry"]["imposters"] = ["C"]
    st.analyze(study)
    assert "verification score is weak" in capsys.readouterr().out


def test_main_exits_1_with_a_one_line_error_on_a_bad_config(study, monkeypatch, capsys):
    study["stylometry"]["mfw"] = 0
    path = study["_config_path"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in study.items() if k != "_config_path"}, f)
    monkeypatch.setattr("sys.argv", ["xlt-stylometry", "--config", path])
    with pytest.raises(SystemExit) as exc:
        st.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("ERROR: ") and 'stylometry["mfw"]' in err
