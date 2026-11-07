"""
Per-script config validation (validate_config(require=...)), on-disk
corpus checks, the shared CLI wrapper, and the other corpus.py helpers
added in the 0.1.0 review.
"""

import json
import os
import unicodedata

import pytest

from cross_lingual_toolkit import corpus


# ---------------------------------------------------------------------------
# A config that passes every script's requirements (paths unchecked)
# ---------------------------------------------------------------------------

def _full_config():
    return {
        "languages": {"en": {}, "fr": {"stanza_model": "fr"}},
        "corpora": {
            "work": {"type": "single_file", "path": "work.txt", "author": "a",
                     "work": "w", "language": "en"},
            "letters": {"type": "collection", "path": "letters", "author": "a",
                        "work": "l", "language": "en", "groups": ["V1"]},
            "other": {"type": "single_file", "path": "other.txt", "author": "b",
                      "work": "o", "language": "fr"},
        },
        "study": {
            "single_work_corpus": "work", "collection_corpus": "letters",
            "comparison_corpus": "other", "primary_corpora": ["work", "letters"],
            "primary_language": "en", "comparison_language": "fr",
        },
        "term_frequency_clusters": {"c": {"label": "C", "terms": ["x"]}},
        "cross_corpus_concept_clusters": {"c": {"label": "C", "en": ["x"], "fr": ["y"]}},
        "adjective_anchor_terms": {"a": ["x"]},
        "dependency_agents": {"g": {"label": "G", "lemmas": ["x"], "role": "nsubj"}},
        "cross_corpus_dependency": {
            "primary": {"agent_a_lemmas": ["x"], "agent_b_lemmas": ["y"]},
            "comparison": {"agent_a_forms": ["x"], "agent_b_forms": ["y"],
                           "name_forms": ["z"]},
        },
    }


def _all_requires():
    from cross_lingual_toolkit import (
        adjective_window, analysis_a_umap, analysis_b_topics, cross_corpus_dependency,
        cross_corpus_frequency, dependency_parsing, embed_corpus, lexical_richness,
        term_frequency,
    )
    return [m.REQUIRES for m in (
        adjective_window, analysis_a_umap, analysis_b_topics, cross_corpus_dependency,
        cross_corpus_frequency, dependency_parsing, embed_corpus, lexical_richness,
        term_frequency)]


@pytest.mark.parametrize("require", _all_requires())
def test_full_config_meets_every_scripts_requirements(require):
    assert corpus.validate_config(_full_config(), require=require, check_paths=False) == []


TF = ("single_work_corpus:single_file", "collection_corpus:collection",
      "term_frequency_clusters")
XF = ("single_work_corpus:single_file", "comparison_corpus:single_file",
      "cross_corpus_concept_clusters")
XD = ("single_work_corpus:single_file", "comparison_corpus:single_file",
      "cross_corpus_dependency")
DP = ("primary_corpora", "primary_language", "dependency_agents")
LR = ("lexical_richness",)


def _drop_role(role):
    def f(c):
        del c["study"][role]
    return f


def _set(path, value):
    def f(c):
        node = c
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return f


def _delete(*path):
    def f(c):
        node = c
        for key in path[:-1]:
            node = node[key]
        del node[path[-1]]
    return f


CASES = [
    ("single_work_corpus", TF, _drop_role("single_work_corpus")),
    ("collection_corpus", TF, _drop_role("collection_corpus")),
    ("comparison_corpus", XF, _drop_role("comparison_corpus")),
    ("primary_corpora", DP, _drop_role("primary_corpora")),
    ("primary_language", DP, _drop_role("primary_language")),
    ("term_frequency_clusters", TF, _delete("term_frequency_clusters")),
    ("cross_corpus_concept_clusters", XF, _delete("cross_corpus_concept_clusters")),
    ("dependency_agents", DP, _delete("dependency_agents")),
    ('"label"', TF, _delete("term_frequency_clusters", "c", "label")),
    ('"label"', XF, _delete("cross_corpus_concept_clusters", "c", "label")),
    ('["terms"]', TF, _set(["term_frequency_clusters", "c", "terms"], "fear")),
    ('["fr"]', XF, _set(["cross_corpus_concept_clusters", "c", "fr"], "peur")),
    ('"fr" forms', XF, _delete("cross_corpus_concept_clusters", "c", "fr")),
    ("stanza_model", XD, _delete("languages", "fr", "stanza_model")),
    ('cross_corpus_dependency["primary"]', XD, _delete("cross_corpus_dependency", "primary")),
    ('cross_corpus_dependency["comparison"]', XD,
     _delete("cross_corpus_dependency", "comparison")),
    ("a collection corpus", XF, _set(["study", "comparison_corpus"], "letters")),
    ("a collection corpus", TF, _set(["study", "single_work_corpus"], "letters")),
    ("a single_file corpus", TF, _set(["study", "collection_corpus"], "work")),
    ("mattr_window", LR, _set(["lexical_richness"], {"mattr_window": 0})),
    ("smoothing_window", LR, _set(["lexical_richness"], {"smoothing_window": -5})),
    ("n_permutations", LR, _set(["analysis"], {"n_permutations": 0})),
]


@pytest.mark.parametrize("needle,require,mutate", CASES,
                         ids=[f"{i}-{c[0]}" for i, c in enumerate(CASES)])
def test_validate_config_reports_per_script_problems(needle, require, mutate):
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    mutate(cfg)
    with pytest.raises(ValueError) as exc:
        corpus.validate_config(cfg, require=require, check_paths=False)
    assert needle in str(exc.value)


def test_min_topic_size_must_be_at_least_two():
    # HDBSCAN cannot form a cluster of one; 1 used to pass and fail at fit time
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    cfg["topic_modeling"] = {"min_topic_size": 1}
    with pytest.raises(ValueError, match=r'min_topic_size.*at least 2'):
        corpus.validate_config(cfg, require=LR, check_paths=False)
    cfg["topic_modeling"] = {"min_topic_size": 2}
    assert corpus.validate_config(cfg, require=LR, check_paths=False) == []


def test_collection_item_split_threshold_may_be_zero():
    # 0 was valid before the per-script checks existed: chunk every item
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    cfg["chunking"] = {"collection_item_split_threshold": 0}
    assert corpus.validate_config(cfg, require=LR, check_paths=False) == []


def test_chunking_min_chars_may_be_zero_but_not_negative():
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    cfg["chunking"] = {"min_chars": 0}
    assert corpus.validate_config(cfg, require=LR, check_paths=False) == []
    cfg["chunking"] = {"min_chars": -1}
    with pytest.raises(ValueError) as exc:
        corpus.validate_config(cfg, require=LR, check_paths=False)
    assert "min_chars" in str(exc.value)


@pytest.mark.parametrize("value", ["10", 2.5, True])
def test_integer_settings_reject_non_integers(value):
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    cfg["chunking"] = {"max_chars": value}
    with pytest.raises(ValueError) as exc:
        corpus.validate_config(cfg, require=LR, check_paths=False)
    assert 'chunking["max_chars"]' in str(exc.value)


@pytest.mark.parametrize("value", [10, 1e3, 1000.0, 1e4])
def test_integer_settings_accept_whole_numbers(value):
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    cfg["chunking"] = {"max_chars": value}
    assert corpus.validate_config(cfg, require=LR, check_paths=False) == []


def test_whole_float_window_validates_and_runs_in_cross_corpus_dependency():
    # regression: window=10.0 passed validation, then crashed when used as
    # a slice index in cross_corpus_dependency
    from types import SimpleNamespace
    from cross_lingual_toolkit import cross_corpus_dependency as ccd

    cfg = _full_config()
    cfg["lexical_richness"] = {}
    xcfg = cfg["cross_corpus_dependency"]
    xcfg["window"] = 10.0
    assert corpus.validate_config(cfg, require=LR, check_paths=False) == []

    # the anchor sits more than `window` tokens in, so the slice bounds come
    # from the window itself rather than from max(0, ...) / min(len, ...)
    filler = 12

    # spaCy-like: agent B "y" right after an adjective
    noun = SimpleNamespace(lemma_="y", pos_="NOUN", dep_="nsubj")
    noun.head = noun
    toks = [SimpleNamespace(lemma_="w", pos_="DET", dep_="det", head=noun)] * filler
    toks += [SimpleNamespace(lemma_="good", pos_="ADJ", dep_="amod", head=noun), noun]
    _, _, _, b_adjs = ccd.process_primary(lambda t: toks, "y good", xcfg, set())
    assert b_adjs == {"good": 1}

    # Stanza-like: name form "z" right after an adjective
    words = [SimpleNamespace(id=i + 1, text="w", lemma="w", upos="DET", deprel="det", head=0)
             for i in range(filler)]
    words += [SimpleNamespace(id=filler + 1, text="bonus", lemma="bonus", upos="ADJ",
                              deprel="amod", head=filler + 2),
              SimpleNamespace(id=filler + 2, text="z", lemma="z", upos="NOUN",
                              deprel="nsubj", head=0)]
    doc = SimpleNamespace(sentences=[SimpleNamespace(words=words)])
    *_, name_adjs = ccd.process_comparison(lambda t: doc, "z bonus", xcfg)
    assert name_adjs == {"bonus": 1}


def test_integer_settings_reject_whole_float_below_minimum():
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    cfg["chunking"] = {"max_chars": 0.0}
    with pytest.raises(ValueError) as exc:
        corpus.validate_config(cfg, require=LR, check_paths=False)
    assert 'chunking["max_chars"]' in str(exc.value)


def test_validate_config_reports_every_problem_at_once():
    cfg = _full_config()
    del cfg["study"]["collection_corpus"]
    del cfg["term_frequency_clusters"]["c"]["label"]
    cfg["lexical_richness"] = {"mattr_window": 0}
    with pytest.raises(ValueError) as exc:
        corpus.validate_config(cfg, require=TF, check_paths=False)
    message = str(exc.value)
    assert "collection_corpus" in message
    assert '"label"' in message
    assert "mattr_window" in message


def test_validate_config_without_require_keeps_the_base_checks_only():
    cfg = _full_config()
    del cfg["term_frequency_clusters"]
    assert corpus.validate_config(cfg) == []


def test_template_config_is_structurally_valid():
    # its corpus paths are placeholders, so the on-disk checks are skipped
    here = os.path.dirname(os.path.abspath(__file__))
    template = corpus.load_config(os.path.join(here, "..", "config", "template_study.json"))
    assert corpus.validate_config(template) == []
    for require in _all_requires():
        corpus.validate_config(template, require=require, check_paths=False)


# ---------------------------------------------------------------------------
# Corpus paths on disk
# ---------------------------------------------------------------------------

def _disk_config(tmp_path):
    (tmp_path / "work.txt").write_text("text", encoding="utf-8")
    (tmp_path / "other.txt").write_text("texte", encoding="utf-8")
    (tmp_path / "letters" / "V1").mkdir(parents=True)
    (tmp_path / "letters" / "V1" / "1.txt").write_text("a letter", encoding="utf-8")
    cfg = _full_config()
    cfg["corpus_root"] = str(tmp_path)
    return cfg


def test_path_checks_pass_on_a_real_layout(tmp_path):
    assert corpus.validate_config(_disk_config(tmp_path), require=("corpora",)) == []


def test_path_checks_report_a_missing_file(tmp_path):
    cfg = _disk_config(tmp_path)
    cfg["corpora"]["work"]["path"] = "path/to/your/single_work.txt"
    with pytest.raises(ValueError, match="file not found"):
        corpus.validate_config(cfg, require=TF)


def test_path_checks_report_a_directory_declared_single_file(tmp_path):
    cfg = _disk_config(tmp_path)
    cfg["corpora"]["work"]["path"] = "letters"
    with pytest.raises(ValueError, match="is a directory"):
        corpus.validate_config(cfg, require=TF)


def test_path_checks_report_a_file_declared_collection(tmp_path):
    cfg = _disk_config(tmp_path)
    cfg["corpora"]["letters"]["path"] = "work.txt"
    with pytest.raises(ValueError, match="not a directory"):
        corpus.validate_config(cfg, require=TF)


def test_path_checks_report_a_missing_group_directory(tmp_path):
    cfg = _disk_config(tmp_path)
    cfg["corpora"]["letters"]["groups"] = ["V1", "V2_typo"]
    with pytest.raises(ValueError, match="V2_typo"):
        corpus.validate_config(cfg, require=TF)


def test_path_checks_warn_when_groups_is_probably_missing(tmp_path):
    cfg = _disk_config(tmp_path)
    del cfg["corpora"]["letters"]["groups"]
    warnings = corpus.validate_config(cfg, require=TF)
    assert any('"groups"' in w and "V1" in w for w in warnings)


def test_path_checks_honour_the_corpus_root_override(tmp_path):
    cfg = _disk_config(tmp_path)
    cfg["corpus_root"] = str(tmp_path / "nowhere")
    with pytest.raises(ValueError, match="file not found"):
        corpus.validate_config(cfg, require=TF)


# ---------------------------------------------------------------------------
# --output-root precedence
# ---------------------------------------------------------------------------

def test_cli_output_root_beats_config_output_root(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    assert corpus.output_root({"output": {"root": a}, "_output_root": b}) == b


def test_load_config_from_args_output_root_overrides_config_root(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"output": {"root": str(tmp_path / "from_config")}}),
                    encoding="utf-8")
    args = corpus.add_config_args().parse_args(
        ["--config", str(path), "--output-root", str(tmp_path / "from_cli")])
    cfg = corpus.load_config_from_args(args, validate=False)
    results_dir, viz_dir, embed_dir = corpus.output_dirs(cfg)
    for d in (results_dir, viz_dir, embed_dir):
        assert d.startswith(str(tmp_path / "from_cli"))


# ---------------------------------------------------------------------------
# Unicode and case normalization
# ---------------------------------------------------------------------------

def test_tokenize_keeps_decomposed_accents_inside_words():
    assert corpus.tokenize(unicodedata.normalize("NFD", "école café")) == ["école", "café"]


def test_capitalised_stopword_removes_the_lowercase_token(tmp_path, monkeypatch):
    (tmp_path / "sw.txt").write_text("The\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    stops = corpus.load_stopwords({"languages": {"en": {"stopwords_file": "sw.txt"}}}, "en")
    assert corpus.tokenize("The garden", stopwords=stops) == ["garden"]


# ---------------------------------------------------------------------------
# Untokenizable configured terms
# ---------------------------------------------------------------------------

def test_warn_untokenizable_names_each_unmatchable_term(capsys):
    found = corpus.warn_untokenizable(["l'amour", "a", "the", "garden"], "clusters",
                                      stopwords={"the"})
    assert [t for t, _ in found] == ["l'amour", "a", "the"]
    out = capsys.readouterr().out
    for term in ("l'amour", '"a"', '"the"'):
        assert term in out
    assert "garden" not in out


def test_warn_untokenizable_is_quiet_for_plain_terms(capsys):
    assert corpus.warn_untokenizable(["garden", "Stone"], "x") == []
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# split_for_parser()
# ---------------------------------------------------------------------------

def test_split_for_parser_cuts_at_boundaries_and_loses_nothing():
    paragraph = ("The quick brown fox jumps over the lazy dog. " * 7).strip()
    text = "\n\n".join([paragraph] * 40) + "\nTrailing line without a stop"
    for max_chars in (100, 333, 1000):
        pieces = corpus.split_for_parser(text, max_chars)
        assert "".join(pieces) == text
        assert all(len(p) <= max_chars for p in pieces)
        for left, right in zip(pieces, pieces[1:]):
            # never inside a word: one side of every cut is whitespace
            assert left[-1].isspace() or right[0].isspace()


def test_split_for_parser_prefers_paragraph_breaks():
    text = "a" * 30 + ". b b b.\n\n" + "c " * 20
    pieces = corpus.split_for_parser(text, 45)
    assert pieces[0].endswith("\n\n")


def test_split_for_parser_short_text_is_one_piece():
    assert corpus.split_for_parser("short", 100) == ["short"]
    assert corpus.split_for_parser("", 100) == []


# ---------------------------------------------------------------------------
# run_cli(): clean errors, non-zero exit
# ---------------------------------------------------------------------------

def test_main_with_an_invalid_config_exits_non_zero_without_a_traceback(
        tmp_path, monkeypatch, capsys):
    from cross_lingual_toolkit import term_frequency
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"corpora": {}}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["xlt-term-frequency", "--config", str(path)])
    with pytest.raises(SystemExit) as exc:
        term_frequency.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "ERROR:" in err
    assert "Traceback" not in err


def test_main_with_a_missing_config_exits_non_zero(tmp_path, monkeypatch, capsys):
    from cross_lingual_toolkit import lexical_richness
    monkeypatch.setattr("sys.argv", ["x", "--config", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit) as exc:
        lexical_richness.main()
    assert exc.value.code == 1
    assert "ERROR:" in capsys.readouterr().err


def test_run_cli_propagates_an_analysis_error_status(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(_full_config()), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        corpus.run_cli(lambda config: 1, argv=["--config", str(path)])
    assert exc.value.code == 1


def test_run_cli_returns_normally_on_success(tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(_full_config()), encoding="utf-8")
    seen = []
    corpus.run_cli(seen.append, argv=["--config", str(path)])
    assert seen and seen[0]["study"]["single_work_corpus"] == "work"


# ---------------------------------------------------------------------------
# Hardening review 2: wrong JSON types are reported, not tracebacks
# ---------------------------------------------------------------------------

ALL = tuple(sorted({item for req in _all_requires() for item in req}))

TYPE_CASES = [
    ('"study" must be an object', _set(["study"], None)),
    ('"study" must be an object', _set(["study"], ["work"])),
    ('"output" must be an object', _set(["output"], None)),
    ('"output" must be an object', _set(["output"], "out")),
    ('"languages" must be an object', _set(["languages"], None)),
    ('"languages" must be an object', _set(["languages"], ["en"])),
    ('languages["en"] must be an object', _set(["languages", "en"], "en")),
    ('"corpora" must be an object', _set(["corpora"], ["work"])),
    ('corpora["work"] must be an object', _set(["corpora", "work"], "work.txt")),
    ('study["primary_corpora"]', _set(["study", "primary_corpora"], {"work": 1})),
    ('study["primary_corpora"]', _set(["study", "primary_corpora"], [["work"]])),
    ('study["comparison_corpus"] must be a string',
     _set(["study", "comparison_corpus"], ["other"])),
    ('study["primary_language"] must be a string', _set(["study", "primary_language"], 5)),
    ('corpora["work"]["path"] must be a string', _set(["corpora", "work", "path"], 5)),
    ('corpora["work"]["language"] must be a string',
     _set(["corpora", "work", "language"], ["en"])),
    ('corpora["letters"]["groups"]', _set(["corpora", "letters", "groups"], "V1")),
    ('corpora["letters"]["groups"]', _set(["corpora", "letters", "groups"], [1, 2])),
    ('"term_frequency_clusters" must be an object', _set(["term_frequency_clusters"], ["c"])),
    ('"cross_corpus_concept_clusters" must be an object',
     _set(["cross_corpus_concept_clusters"], ["c"])),
    ('"chunking" must be an object', _set(["chunking"], [300, 1600])),
    ('"analysis" must be an object', _set(["analysis"], [])),
]


@pytest.mark.parametrize("needle,mutate", TYPE_CASES,
                         ids=[f"{i}-{c[0]}" for i, c in enumerate(TYPE_CASES)])
def test_validate_config_reports_wrong_json_types(needle, mutate):
    cfg = _full_config()
    mutate(cfg)
    try:   # base checks only: may or may not object, but must not crash
        corpus.validate_config(cfg, check_paths=False)
    except ValueError:
        pass
    for check_paths in (False, True):
        with pytest.raises(ValueError) as exc:
            corpus.validate_config(cfg, require=ALL, check_paths=check_paths)
        assert needle in str(exc.value)


# _check_list item types, adjective_suffixes, chunking integers

LIST_CASES = [
    ('["terms"]', TF, _set(["term_frequency_clusters", "c", "terms"], [5, None])),
    ('["terms"]', TF, _set(["term_frequency_clusters", "c", "terms"], ["fear", "  "])),
    ('["terms"]', TF, _set(["term_frequency_clusters", "c", "terms"], ["fear", "\u00ad"])),
    ('["fr"]', XF, _set(["cross_corpus_concept_clusters", "c", "fr"], [""])),
    ('["name_forms"]', XD, _set(["cross_corpus_dependency", "comparison", "name_forms"], [""])),
    ('["lemmas"]', DP, _set(["dependency_agents", "g", "lemmas"], [None])),
    ('["adjective_suffixes"]', LR, _set(["languages", "en", "adjective_suffixes"], "oso")),
    ('["adjective_suffixes"]', LR, _set(["languages", "en", "adjective_suffixes"], ["oso", ""])),
    ('["keep_short_tokens"]', LR, _set(["languages", "en", "keep_short_tokens"], "eoa")),
    ("collection_item_min_chars", LR,
     _set(["chunking"], {"collection_item_min_chars": -1})),
    ("collection_item_split_threshold", LR,
     _set(["chunking"], {"collection_item_split_threshold": -1})),
    ("collection_item_split_threshold", LR,
     _set(["chunking"], {"collection_item_split_threshold": "1800"})),
]


@pytest.mark.parametrize("needle,require,mutate", LIST_CASES,
                         ids=[f"{i}-{c[0]}" for i, c in enumerate(LIST_CASES)])
def test_validate_config_checks_list_items_and_chunking_ints(needle, require, mutate):
    cfg = _full_config()
    cfg["lexical_richness"] = {}
    mutate(cfg)
    with pytest.raises(ValueError) as exc:
        corpus.validate_config(cfg, require=require, check_paths=False)
    assert needle in str(exc.value)


def test_collection_item_min_chars_may_be_zero():
    cfg = _full_config()
    cfg["chunking"] = {"collection_item_min_chars": 0,
                       "collection_item_split_threshold": 1800}
    assert corpus.validate_config(cfg, require=("corpora",), check_paths=False) == []


def test_adjective_suffixes_are_returned_normalized():
    cfg = {"languages": {"it": {"adjective_suffixes": [" OSO", "ale"]}}}
    assert corpus.adjective_suffixes(cfg, "it") == ("oso", "ale")
