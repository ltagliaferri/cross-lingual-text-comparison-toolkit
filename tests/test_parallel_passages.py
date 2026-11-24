"""
Parallel-passage search (parallel_passages.py): sentence splitting,
locations, best/runner-up matching, run merging, DP alignment, config
validation, and an end-to-end run with a stub embedding model.
"""

import csv
import hashlib
import json
import os
import re

import numpy as np
import pytest

from cross_lingual_toolkit import corpus, parallel_passages as pp


SPLIT = re.compile(pp.DEFAULT_SENTENCE_SPLIT)


def _texts(sentences):
    return [s[0] for s in sentences]


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def test_default_split_breaks_after_each_terminator_followed_by_whitespace():
    text = "One. Two? Three! Four; Five: Six.Seven"
    assert _texts(pp.split_sentences(text, SPLIT, 0, 100)) == \
        ["One.", "Two?", "Three!", "Four;", "Five:", "Six.Seven"]


def test_default_split_breaks_after_cjk_terminators_with_or_without_whitespace():
    text = "我们走了。他来了！你好吗？ 再见。"
    assert _texts(pp.split_sentences(text, SPLIT, 0, 100)) == \
        ["我们走了。", "他来了！", "你好吗？", "再见。"]


def test_paragraphs_split_and_whitespace_collapses():
    text = "First  line\ncontinues here\n\n\nSecond paragraph"
    out = pp.split_sentences(text, SPLIT, 0, 100)
    assert [(s, p) for s, _, p in out] == \
        [("First line continues here", 1), ("Second paragraph", 2)]


def test_short_sentences_are_skipped_and_long_ones_wrapped():
    text = "Tiny. " + " ".join(["word"] * 30) + "."
    out = _texts(pp.split_sentences(text, SPLIT, 10, 50))
    assert "Tiny." not in out
    assert len(out) == 3 and all(len(s) <= 50 for s in out)
    assert " ".join(out) == " ".join(["word"] * 30) + "."


def test_a_custom_split_pattern_is_used():
    out = _texts(pp.split_sentences("a | b | c", re.compile(r"\s*\|\s*"), 0, 100))
    assert out == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

BOOK = ("[I.1] Heading paragraph one.\n\nFirst sentence here. Second sentence.\n\n"
        "[I.2]\n\nThird sentence here.")


def test_location_is_the_group_of_the_last_match_up_to_the_sentence():
    out = pp.corpus_sentences("book", BOOK, SPLIT, 0, 100, re.compile(r"\[([IVX]+\.\d+)\]"))
    assert [(x["location"], x["text"]) for x in out] == [
        ("I.1", "[I.1] Heading paragraph one."), ("I.1", "First sentence here."),
        ("I.1", "Second sentence."), ("I.2", "[I.2]"), ("I.2", "Third sentence here.")]
    assert [x["index"] for x in out] == [0, 1, 2, 3, 4]
    assert {x["text_id"] for x in out} == {"book"}


def test_location_without_a_group_is_the_whole_match_and_falls_back_before_it():
    text = "Before any heading.\n\nChapter 2\n\nAfter it."
    out = pp.corpus_sentences("book", text, SPLIT, 0, 100, re.compile(r"Chapter \d+"))
    assert [x["location"] for x in out] == ["p1", "Chapter 2", "Chapter 2"]


def test_location_group_that_did_not_take_part_uses_the_whole_match():
    text = "Preface\n\nIntro sentence.\n\nChapter 2\n\nAfter it."
    out = pp.corpus_sentences("book", text, SPLIT, 0, 100,
                              re.compile(r"^(?:(Chapter \d+)|Preface)", re.MULTILINE))
    assert [x["location"] for x in out] == ["Preface", "Preface", "Chapter 2", "Chapter 2"]


def test_empty_location_matches_are_ignored():
    text = "Chapter 1\n\nFirst sentence.\n\nSecond sentence.\n\nChapter 2\n\nThird sentence."
    out = pp.corpus_sentences("book", text, SPLIT, 0, 100,
                              re.compile(r"^(Chapter \d+)?", re.MULTILINE))
    assert [x["location"] for x in out] == ["Chapter 1"] * 3 + ["Chapter 2"] * 2


def test_a_failing_location_pattern_is_a_value_error_naming_the_key():
    class Broken:
        def finditer(self, text):
            raise TypeError("boom")
    with pytest.raises(ValueError, match=r'\["location_patterns"\]\["book"\].*boom'):
        pp.corpus_sentences("book", "Some text.", SPLIT, 0, 100, Broken())


def test_single_file_without_pattern_uses_paragraph_index():
    out = pp.corpus_sentences("book", "A one. A two.\n\nB one.", SPLIT, 0, 100)
    assert [x["location"] for x in out] == ["p1", "p1", "p2"]


def test_collection_items_are_the_location_and_text_id():
    items = [{"group": "Vol1", "item_num": "001", "text": "Alpha one. Alpha two."},
             {"group": "Vol1", "item_num": "002", "text": "Beta one."}]
    out = pp.corpus_sentences("letters", items, SPLIT, 0, 100)
    assert [(x["location"], x["text_id"], x["index"]) for x in out] == \
        [("Vol1/001", "Vol1/001", 0), ("Vol1/001", "Vol1/001", 1), ("Vol1/002", "Vol1/002", 2)]


def test_text_is_nfc_normalized():
    out = pp.corpus_sentences("x", "café au lait.", SPLIT, 0, 100)
    assert out[0]["text"] == "café au lait."


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_best_and_runner_up_across_blocks():
    rng = np.random.default_rng(0)
    source = pp.normalize_rows(rng.normal(size=(23, 8)))
    target = pp.normalize_rows(rng.normal(size=(17, 8)))
    best, score, second, second_score = pp.best_matches(source, target, block=5)
    sims = source @ target.T
    order = np.argsort(-sims, axis=1)
    assert best.tolist() == order[:, 0].tolist()
    assert second.tolist() == order[:, 1].tolist()
    assert np.allclose(score, sims.max(axis=1))
    assert np.allclose(second_score, sims[np.arange(23), order[:, 1]])
    assert (score >= second_score).all()


def test_best_match_spans_several_targets_stacked():
    source = np.eye(3)
    t1, t2 = np.array([[1.0, 0, 0], [0, 0.6, 0.8]]), np.array([[0, 1.0, 0], [0, 0, 1.0]])
    best, score, second, _ = pp.best_matches(source, np.vstack([t1, t2]))
    assert best.tolist() == [0, 2, 3]
    assert second.tolist()[1:] == [1, 1]


def test_a_single_target_sentence_has_no_runner_up():
    best, score, second, second_score = pp.best_matches(np.eye(2), np.array([[1.0, 0]]))
    assert best.tolist() == [0, 0] and second.tolist() == [-1, -1]
    assert np.isnan(second_score).all()


# ---------------------------------------------------------------------------
# Passages
# ---------------------------------------------------------------------------

def _find(scores, texts, idx, threshold=0.7, gap=2, window=10, min_n=2):
    return pp.find_passages(scores, texts, idx, threshold, gap, window, min_n)


def test_consecutive_matches_form_one_passage():
    out = _find([0.9, 0.8, 0.85], ["a"] * 3, [10, 11, 12])
    assert len(out) == 1
    assert out[0] == {"source_first": 0, "source_last": 2, "matches": [0, 1, 2],
                      "target_text": "a", "target_first": 10, "target_last": 12}


def test_merge_gap_bounds_unmatched_sentences_between_matches():
    scores = [0.9, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9]
    out = _find(scores, ["a"] * 9, list(range(9)), gap=2)
    assert [p["matches"] for p in out] == [[0, 3], [7, 8]]


def test_target_window_and_a_step_back_of_one():
    # 0 -> 5 is within the window; 5 -> 4 is a step back of one; 4 -> 20 is too far
    out = _find([0.9] * 4, ["a"] * 4, [0, 5, 4, 20], window=5)
    assert [p["matches"] for p in out] == [[0, 1, 2]]
    # a step back of two breaks the run
    out = _find([0.9] * 4, ["a"] * 4, [5, 6, 4, 5], window=5)
    assert [p["matches"] for p in out] == [[0, 1], [2, 3]]


def test_two_target_texts_give_two_passages():
    out = _find([0.9] * 4, ["a", "a", "b", "b"], [0, 1, 2, 3])
    assert [(p["matches"], p["target_text"]) for p in out] == [([0, 1], "a"), ([2, 3], "b")]


def test_a_stray_match_into_another_text_does_not_split_a_run():
    out = _find([0.9] * 5, ["a", "a", "b", "a", "a"], [0, 1, 50, 2, 3])
    assert [(p["matches"], p["target_text"]) for p in out] == [([0, 1, 3, 4], "a")]


def test_a_match_into_the_same_text_breaking_the_window_starts_a_new_run():
    out = _find([0.9] * 5, ["a"] * 5, [0, 1, 40, 41, 20], window=5)
    # 1 -> 40 is too far: a new run [2, 3]; 20 fits neither run: a third run
    assert [p["matches"] for p in out] == [[0, 1], [2, 3]]
    out = _find([0.9] * 5, ["a"] * 5, [0, 1, 40, 41, 20], window=5, min_n=1)
    assert [p["matches"] for p in out] == [[0, 1], [2, 3], [4]]


def test_a_far_off_match_into_the_same_text_does_not_split_a_run():
    out = _find([0.9] * 7, ["a"] * 7, [100, 101, 102, 9000, 103, 104, 105])
    assert [p["matches"] for p in out] == [[0, 1, 2, 4, 5, 6]]
    out = _find([0.9] * 7, ["a"] * 7, [100, 101, 102, 9000, 103, 104, 105], min_n=1)
    assert [p["matches"] for p in out] == [[0, 1, 2, 4, 5, 6], [3]]
    # the old run closes once more than merge_gap source sentences pass without it
    out = _find([0.9] * 6, ["a"] * 6, [100, 101, 9000, 9001, 9002, 102], gap=2)
    assert [p["matches"] for p in out] == [[0, 1], [2, 3, 4]]


def test_overlapping_passages_into_two_texts_are_both_reported_in_order():
    texts = ["b", "a", "b", "a", "b", "a"]
    out = _find([0.9] * 6, texts, [10, 0, 11, 1, 12, 2])
    assert [(p["source_first"], p["source_last"], p["target_text"], p["matches"])
            for p in out] == [(0, 4, "b", [0, 2, 4]), (1, 5, "a", [1, 3, 5])]


def test_merge_gap_counts_source_sentences_whatever_they_matched():
    # sentences 1 and 2 match text "b": two source sentences between the "a" matches
    texts = ["a", "b", "b", "a"]
    assert [p["matches"] for p in _find([0.9] * 4, texts, [0, 9, 10, 1], gap=2, min_n=3)] \
        == []
    assert [(p["matches"], p["target_text"]) for p in _find([0.9] * 4, texts, [0, 9, 10, 1],
                                                              gap=2)] == \
        [([0, 3], "a"), ([1, 2], "b")]
    assert [p["matches"] for p in _find([0.9] * 4, texts, [0, 9, 10, 1], gap=1)] == [[1, 2]]


def test_default_merge_gap_is_five():
    assert pp.DEFAULTS["merge_gap"] == 5
    assert pp.settings({"parallel_passages": {}})["merge_gap"] == 5
    scores = [0.9] + [0.1] * 5 + [0.9] + [0.1] * 6 + [0.9]
    gap = pp.settings({"parallel_passages": {}})["merge_gap"]
    assert [p["matches"] for p in _find(scores, ["a"] * 14, list(range(14)), gap=gap)] == \
        [[0, 6]]


def test_min_passage_sentences_drops_short_runs():
    scores = [0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    assert [p["matches"] for p in _find(scores, ["a"] * 7, list(range(7)), min_n=3)] == \
        [[4, 5, 6]]
    assert len(_find(scores, ["a"] * 7, list(range(7)), min_n=1)) == 2


def test_scores_below_threshold_never_match():
    assert _find([0.69, 0.69], ["a", "a"], [0, 1]) == []


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _sim_from_pairs(pairs):
    """sim() that scores 0.9 for exactly the listed (src_range, tgt_range) pairs."""
    return lambda s, t: 0.9 if (s, t) in pairs else 0.0


def test_alignment_follows_a_clean_diagonal():
    sim = _sim_from_pairs({((i, i + 1), (i, i + 1)) for i in range(4)})
    units = pp.align(4, 4, sim, 0.35)
    assert [u[:4] for u in units] == [(i, i + 1, i, i + 1) for i in range(4)]
    assert all(u[4] == 0.9 for u in units)


def test_alignment_finds_a_one_to_two_split():
    sim = _sim_from_pairs({((0, 1), (0, 1)), ((1, 2), (1, 3)), ((2, 3), (3, 4))})
    assert [u[:4] for u in pp.align(3, 4, sim, 0.35)] == \
        [(0, 1, 0, 1), (1, 2, 1, 3), (2, 3, 3, 4)]


def test_alignment_skips_an_extra_sentence_on_each_side():
    # source 1 and target 2 have no counterpart
    sim = _sim_from_pairs({((0, 1), (0, 1)), ((2, 3), (1, 2)), ((3, 4), (3, 4))})
    units = pp.align(4, 4, sim, 0.35)
    assert [u[:4] for u in units] == \
        [(0, 1, 0, 1), (1, 2, 1, 1), (2, 3, 1, 2), (3, 3, 2, 3), (3, 4, 3, 4)]
    assert [u[4] for u in units] == [0.9, None, 0.9, None, 0.9]


# ---------------------------------------------------------------------------
# Config and the script
# ---------------------------------------------------------------------------

SHARED = [   # "translations": the stub embeds the bracketed key, so pairs match exactly
    "[k1] The river ran over the stones all through the long night.",
    "[k2] In the morning the village woke to find the bridge gone.",
    "[k3] Nobody could remember a flood like it in a hundred years.",
    "[k4] The miller said the water had reached his very doorstep.",
]


def _filler(prefix, n):
    return [f"[{prefix}{i}] Some unrelated sentence of filler number {i} goes here."
            for i in range(n)]


class StubModel:
    """Embeds the leading [key] of a sentence as a hashed one-hot-ish vector."""
    def encode(self, texts, **kwargs):
        out = np.zeros((len(texts), 1024))
        for n, text in enumerate(texts):
            for key in re.findall(r"\[(\w+)\]", text):
                h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
                out[n, h % 1024] += 1.0
                out[n, (h // 1024) % 1024] += 0.5
        return out + 1e-3


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "corpus"
    (root / "letters" / "Vol1").mkdir(parents=True)
    source = "\n\n".join([" ".join(_filler("s", 3)), "Chapter 2", " ".join(SHARED),
                          " ".join(_filler("t", 3))])
    (root / "source.txt").write_text(source, encoding="utf-8")
    (root / "letters" / "Vol1" / "001.txt").write_text(
        " ".join(_filler("u", 4)), encoding="utf-8")
    (root / "letters" / "Vol1" / "002.txt").write_text(
        " ".join(_filler("v", 2) + SHARED + _filler("w", 2)), encoding="utf-8")
    (root / "other.txt").write_text(" ".join(_filler("x", 3)), encoding="utf-8")
    config = {
        "corpus_root": str(root),
        "languages": {"la": {}, "it": {}},
        "corpora": {
            "src": {"type": "single_file", "path": "source.txt", "author": "a",
                    "work": "s", "language": "la"},
            "letters": {"type": "collection", "path": "letters", "author": "b",
                        "work": "l", "language": "it", "groups": ["Vol1"]},
            "other": {"type": "single_file", "path": "other.txt", "author": "c",
                      "work": "o", "language": "it"},
        },
        "parallel_passages": {
            "_comment": "a comment",
            "source": "src",
            "targets": ["letters", "other"],
            "location_patterns": {"src": r"^(Chapter \d+)"},
            "threshold": 0.9,
        },
    }
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return corpus.load_config(str(path))


def _validate(config, check_paths=True):
    return corpus.validate_config(config, require=pp.REQUIRES, check_paths=check_paths)


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_study_fixture_is_valid(study):
    assert _validate(study) == []


def test_validation_reports_every_problem_together(study):
    study["parallel_passages"] = {
        "targets": ["letters", "nope", "letters"],
        "sentence_split": "(unclosed",
        "location_patterns": {"src": "[bad", "ghost": "x"},
        "threshold": 1.5, "merge_gap": -1, "max_sentence_chars": 0,
        "min_passage_sentences": 0, "skip_penalty": -0.1, "align": "yes",
        "target_window": 2.5,
    }
    with pytest.raises(ValueError) as exc:
        _validate(study)
    msg = str(exc.value)
    for fragment in ('["source"] is required', 'unknown corpus "nope"', 'lists "letters" twice',
                     '["sentence_split"] is not a valid', '["location_patterns"]["src"] is not',
                     '["location_patterns"]["ghost"] names no', '["threshold"]', '["merge_gap"]',
                     '["max_sentence_chars"]', '["min_passage_sentences"]', '["skip_penalty"]',
                     '["align"]', '["target_window"]'):
        assert fragment in msg, fragment


def test_validation_rejects_unknown_source_and_source_among_targets(study):
    study["parallel_passages"]["source"] = "ghost"
    with pytest.raises(ValueError, match='"source"] refers to unknown corpus'):
        _validate(study)
    study["parallel_passages"]["source"] = "src"
    study["parallel_passages"]["targets"] = ["src", "letters"]
    with pytest.raises(ValueError, match='includes the source corpus "src"'):
        _validate(study)


def test_validation_requires_non_empty_targets_and_the_section(study):
    study["parallel_passages"]["targets"] = []
    with pytest.raises(ValueError, match='"targets"] must not be empty'):
        _validate(study)
    del study["parallel_passages"]
    with pytest.raises(ValueError, match='no "parallel_passages" section'):
        _validate(study)


def test_target_paths_are_checked(study, tmp_path):
    os.remove(tmp_path / "corpus" / "other.txt")
    with pytest.raises(ValueError, match='corpora\\["other"\\]: file not found'):
        _validate(study)
    _validate(study, check_paths=False)


def test_max_sentence_chars_below_min_is_a_warning_not_a_problem(study):
    study["parallel_passages"]["max_sentence_chars"] = 30
    study["parallel_passages"].pop("min_sentence_chars", None)
    warnings = _validate(study)
    assert any('["max_sentence_chars"] (30) is below parallel_passages["min_sentence_chars"] (40)' in w
               for w in warnings)
    study["parallel_passages"]["min_sentence_chars"] = 30
    assert not any('is below' in w for w in _validate(study))


def test_unknown_keys_are_warned(study):
    study["parallel_passages"]["treshold"] = 0.8
    assert any('"treshold"' in w for w in _validate(study))


def test_main_writes_every_output(study, tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(pp, "get_model", lambda name: calls.append(name) or StubModel())
    monkeypatch.setattr("sys.argv", ["xlt-parallel-passages", "--config", study["_config_path"]])
    pp.main()
    out = capsys.readouterr().out
    assert calls == [pp.DEFAULT_EMBEDDING_MODEL]

    results = tmp_path / "results"
    sentences = _read_csv(results / "parallel_sentences.csv")
    assert list(sentences[0]) == ["source_idx", "source_location", "score", "runner_up",
                                  "margin", "target_corpus", "target_location", "target_idx",
                                  "source_text", "target_text"]
    assert [r["source_idx"] for r in sentences] == [str(i) for i in range(10)]
    shared = sentences[3:7]
    assert all(r["source_location"] == "Chapter 2" for r in shared)
    assert all(float(r["score"]) > 0.99 and r["target_location"] == "Vol1/002"
               for r in shared)
    assert [r["target_idx"] for r in shared] == ["6", "7", "8", "9"]
    assert sentences[0]["source_location"] == "p1"

    passages = _read_csv(results / "parallel_passages.csv")
    assert list(passages[0]) == ["passage", "source_first", "source_last", "source_location",
                                 "n_matches", "mean_score", "max_score", "target_corpus",
                                 "target_location", "target_first", "target_last",
                                 "source_text", "target_text"]
    assert len(passages) == 1
    p = passages[0]
    assert (p["source_first"], p["source_last"], p["n_matches"], p["target_corpus"],
            p["target_first"], p["target_last"]) == ("3", "6", "4", "letters", "6", "9")

    alignment = _read_csv(results / "parallel_alignment.csv")
    assert list(alignment[0]) == ["passage", "unit", "source_first", "source_last",
                                  "target_first", "target_last", "similarity",
                                  "source_text", "target_text"]
    # the units cover source 3-6 and target 4-11 (2 context sentences each
    # side, inside item Vol1/002) in order, each sentence exactly once
    covered_s = [i for r in alignment if r["source_first"]
                 for i in range(int(r["source_first"]), int(r["source_last"]) + 1)]
    covered_t = [i for r in alignment if r["target_first"]
                 for i in range(int(r["target_first"]), int(r["target_last"]) + 1)]
    assert covered_s == [3, 4, 5, 6] and covered_t == list(range(4, 12))
    for r in alignment:
        both = bool(r["source_text"]) and bool(r["target_text"])
        assert (r["similarity"] != "") == both

    summary = (results / "parallel_summary.txt").read_text(encoding="utf-8")
    for fragment in ("10 sentences", "99.9%", "0.80: 4", "Passages", "Chapter 2",
                     "not proof of translation"):
        assert fragment in summary, fragment
    assert (tmp_path / "visualizations" / "parallel_passages.png").exists()
    for cid in ("src", "letters", "other"):
        assert (tmp_path / "embeddings" / f"parallel_{cid}_sentences.npy").exists()
    assert "Passages" in out


def test_align_false_writes_no_alignment(study, tmp_path, monkeypatch):
    study["parallel_passages"]["align"] = False
    monkeypatch.setattr(pp, "get_model", lambda name: StubModel())
    pp.analyze(study)
    assert not (tmp_path / "results" / "parallel_alignment.csv").exists()
    assert (tmp_path / "results" / "parallel_passages.csv").exists()


def test_main_exits_1_with_a_one_line_error_on_a_bad_config(study, monkeypatch, capsys):
    study["parallel_passages"]["threshold"] = 2
    path = study["_config_path"]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in study.items() if k != "_config_path"}, f)
    monkeypatch.setattr("sys.argv", ["xlt-parallel-passages", "--config", path])
    with pytest.raises(SystemExit) as exc:
        pp.main()
    assert exc.value.code == 1
    assert capsys.readouterr().err.startswith("ERROR: ")


def _covered_targets(alignment):
    return [i for r in alignment if r["target_first"]
            for i in range(int(r["target_first"]), int(r["target_last"]) + 1)]


def test_alignment_context_stays_inside_the_target_text(study, tmp_path, monkeypatch):
    # the passage is target 6-9 in Vol1/002 (4-11); 5 sentences of context
    # would reach back into Vol1/001 (0-3)
    study["parallel_passages"]["align_context"] = 5
    monkeypatch.setattr(pp, "get_model", lambda name: StubModel())
    pp.analyze(study)
    alignment = _read_csv(tmp_path / "results" / "parallel_alignment.csv")
    assert _covered_targets(alignment) == list(range(4, 12))
    assert not any("[u" in r["target_text"] for r in alignment)


def test_the_configured_embedding_model_is_loaded(study, monkeypatch):
    study["embedding_model"] = "some/other-model"
    calls = []
    monkeypatch.setattr(pp, "get_model", lambda name: calls.append(name) or StubModel())
    pp.analyze(study)
    assert calls == ["some/other-model"]
