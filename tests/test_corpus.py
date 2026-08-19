import json
import os

import pytest

from cross_lingual_toolkit import corpus


# ---------------------------------------------------------------------------
# tokenize()
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    assert corpus.tokenize("Hello world!") == ["hello", "world"]


def test_tokenize_drops_single_chars_by_default():
    # apostrophe-splitting turns "a'bcd" into "a" + "bcd" — the spurious
    # single-char "a" should be dropped along with any other single-char
    # token, since no keep_short set was passed.
    assert corpus.tokenize("a'bcd e fg") == ["bcd", "fg"]


def test_tokenize_keep_short_exempts_configured_tokens():
    keep_short = {"a", "e"}
    tokens = corpus.tokenize("a'bcd e fg", keep_short=keep_short)
    assert tokens == ["a", "bcd", "e", "fg"]


def test_tokenize_removes_stopwords_when_enabled():
    tokens = corpus.tokenize("foo bar foo baz", stopwords={"foo"})
    assert tokens == ["bar", "baz"]


def test_tokenize_keeps_stopwords_when_disabled():
    tokens = corpus.tokenize("foo bar", remove_stopwords=False, stopwords={"foo"})
    assert tokens == ["foo", "bar"]


def test_tokenize_drops_digits_and_splits_on_punctuation():
    # digits, underscores and hyphens are all separators — the docstring
    # used to promise this while the code kept digits and underscores,
    # which made the two corpora in a cross-corpus run incomparable.
    assert corpus.tokenize("Nel 1300 mid-word under_score.") == [
        "nel", "mid", "word", "under", "score",
    ]


def test_tokenize_preserves_non_ascii_letters():
    assert corpus.tokenize("café, à la carte größer naïve") == [
        "café", "la", "carte", "größer", "naïve",
    ]


# ---------------------------------------------------------------------------
# config_entries()
# ---------------------------------------------------------------------------

def test_config_entries_skips_underscore_comment_keys():
    section = {
        "_comment": "Used by term_frequency.py",
        "real_cluster": {"label": "Real", "terms": ["a"]},
    }
    assert corpus.config_entries(section) == {
        "real_cluster": {"label": "Real", "terms": ["a"]}
    }


def test_config_entries_handles_missing_section():
    assert corpus.config_entries(None) == {}
    assert corpus.config_entries({}) == {}


# ---------------------------------------------------------------------------
# load_config()
# ---------------------------------------------------------------------------

def test_load_config_reads_valid_json(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"study": {"primary_language": "en"}}), encoding="utf-8")
    cfg = corpus.load_config(str(path))
    assert cfg["study"]["primary_language"] == "en"


def test_load_config_records_its_own_path(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    cfg = corpus.load_config(str(path))
    assert cfg["_config_path"] == str(path.resolve())


def test_load_config_missing_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Pass --config"):
        corpus.load_config(str(missing))


def test_load_config_finds_config_under_cwd(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "study.json").write_text(
        json.dumps({"embedding_model": "x"}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert corpus.load_config()["embedding_model"] == "x"


# ---------------------------------------------------------------------------
# validate_config()
# ---------------------------------------------------------------------------

def _valid_config():
    return {
        "corpora": {
            "a": {"type": "single_file", "path": "a.txt", "author": "x",
                  "work": "w", "language": "en"},
        },
        "languages": {"en": {}},
        "study": {"single_work_corpus": "a", "primary_language": "en"},
    }


def test_validate_config_accepts_a_well_formed_config():
    assert corpus.validate_config(_valid_config()) == []


def test_validate_config_rejects_unknown_corpus_id_in_study():
    cfg = _valid_config()
    cfg["study"]["comparison_corpus"] = "typo"
    with pytest.raises(ValueError, match='unknown corpus "typo"'):
        corpus.validate_config(cfg)


def test_validate_config_rejects_missing_required_corpus_key():
    cfg = _valid_config()
    del cfg["corpora"]["a"]["language"]
    with pytest.raises(ValueError, match='missing required key "language"'):
        corpus.validate_config(cfg)


def test_validate_config_rejects_unknown_corpus_type():
    cfg = _valid_config()
    cfg["corpora"]["a"]["type"] = "nonsense"
    with pytest.raises(ValueError, match="expected"):
        corpus.validate_config(cfg)


def test_validate_config_ignores_comment_keys():
    cfg = _valid_config()
    cfg["corpora"]["_comment"] = "not a corpus"
    cfg["languages"]["_comment"] = "not a language"
    assert corpus.validate_config(cfg) == []


def test_validate_config_warns_but_passes_on_unconfigured_language():
    cfg = _valid_config()
    del cfg["languages"]["en"]
    warnings = corpus.validate_config(cfg)
    assert any('language "en"' in w for w in warnings)


# ---------------------------------------------------------------------------
# load_stopwords()
# ---------------------------------------------------------------------------

def test_load_stopwords_none_configured_returns_empty_set():
    cfg = {"languages": {"en": {}}}
    assert corpus.load_stopwords(cfg, "en") == set()


def test_load_stopwords_ignores_comments_and_blank_lines(tmp_path, monkeypatch):
    stop_dir = tmp_path / "stopwords"
    stop_dir.mkdir()
    (stop_dir / "sw.txt").write_text(
        "# a comment\n\nthe\nand\n  \n   # an indented comment\nof\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    cfg = {"languages": {"en": {"stopwords_file": "sw.txt"}}}
    assert corpus.load_stopwords(cfg, "en") == {"the", "and", "of"}


def test_load_stopwords_resolves_next_to_the_config_file(tmp_path, monkeypatch):
    # the working directory holds nothing; the list lives beside the config
    config_dir = tmp_path / "study"
    (config_dir / "stopwords").mkdir(parents=True)
    (config_dir / "stopwords" / "sw.txt").write_text("the\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = {
        "_config_path": str(config_dir / "study.json"),
        "languages": {"en": {"stopwords_file": "sw.txt"}},
    }
    assert corpus.load_stopwords(cfg, "en") == {"the"}


def test_load_stopwords_finds_lists_bundled_with_the_package(tmp_path, monkeypatch):
    # nothing in the working directory: this must still resolve, which is
    # what makes a non-editable install usable.
    monkeypatch.chdir(tmp_path)
    cfg = {"languages": {"la": {"stopwords_file": "stopwords_la.txt"}}}
    assert "et" in corpus.load_stopwords(cfg, "la")


def test_load_stopwords_missing_file_lists_what_was_tried(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"languages": {"en": {"stopwords_file": "nope.txt"}}}
    with pytest.raises(FileNotFoundError, match="Tried:"):
        corpus.load_stopwords(cfg, "en")


# ---------------------------------------------------------------------------
# keep_short_tokens() / adjective_suffixes()
# ---------------------------------------------------------------------------

def test_keep_short_tokens_reads_from_config():
    cfg = {"languages": {"it": {"keep_short_tokens": ["e", "o", "a", "è"]}}}
    assert corpus.keep_short_tokens(cfg, "it") == {"e", "o", "a", "è"}


def test_keep_short_tokens_default_empty():
    assert corpus.keep_short_tokens({}, "la") == set()


def test_adjective_suffixes_reads_from_config():
    cfg = {"languages": {"it": {"adjective_suffixes": ["oso", "osa"]}}}
    assert corpus.adjective_suffixes(cfg, "it") == ("oso", "osa")


def test_adjective_suffixes_default_empty():
    assert corpus.adjective_suffixes({}, "la") == ()


# ---------------------------------------------------------------------------
# load_single_file_corpus() / load_collection_corpus() / load_source_corpus()
# ---------------------------------------------------------------------------

def test_load_single_file_corpus_reads_text(tmp_path):
    (tmp_path / "work.txt").write_text("Hello world.", encoding="utf-8")
    cfg = {"corpora": {"w": {"path": "work.txt"}}}
    text = corpus.load_single_file_corpus(cfg, "w", corpus_root_override=str(tmp_path))
    assert text == "Hello world."


def test_load_single_file_corpus_strips_after_marker(tmp_path):
    (tmp_path / "work.txt").write_text("Keep this.\nEND\nDrop this.", encoding="utf-8")
    cfg = {"corpora": {"w": {"path": "work.txt", "strip_after_marker": "END"}}}
    text = corpus.load_single_file_corpus(cfg, "w", corpus_root_override=str(tmp_path))
    assert text == "Keep this.\n"


def test_load_collection_corpus_walks_groups_sorted(tmp_path):
    (tmp_path / "coll" / "GroupA").mkdir(parents=True)
    (tmp_path / "coll" / "GroupB").mkdir(parents=True)
    (tmp_path / "coll" / "GroupA" / "2.txt").write_text("second", encoding="utf-8")
    (tmp_path / "coll" / "GroupA" / "1.txt").write_text("first", encoding="utf-8")
    (tmp_path / "coll" / "GroupB" / "1.txt").write_text("other group", encoding="utf-8")

    cfg = {"corpora": {"c": {"path": "coll", "groups": ["GroupA", "GroupB"]}}}
    items = corpus.load_collection_corpus(cfg, "c", corpus_root_override=str(tmp_path))

    assert [i["item_num"] for i in items] == ["1", "2", "1"]
    assert [i["group_num"] for i in items] == [1, 1, 2]
    assert items[0]["text"] == "first"
    assert items[2]["group"] == "GroupB"


def test_load_source_corpus_dispatches_on_type(tmp_path):
    (tmp_path / "work.txt").write_text("text", encoding="utf-8")
    cfg = {"corpora": {"w": {"type": "single_file", "path": "work.txt"}}}
    assert corpus.load_source_corpus(cfg, "w", corpus_root_override=str(tmp_path)) == "text"


def test_load_source_corpus_unknown_type_raises():
    cfg = {"corpora": {"w": {"type": "nonsense"}}}
    with pytest.raises(ValueError, match="Unknown corpus type"):
        corpus.load_source_corpus(cfg, "w")


# ---------------------------------------------------------------------------
# corpus_label() / word_freq() / corpus_root()
# ---------------------------------------------------------------------------

def test_corpus_label_uses_configured_label():
    cfg = {"corpora": {"w": {"label": "My Work"}}}
    assert corpus.corpus_label(cfg, "w") == "My Work"


def test_corpus_label_falls_back_to_capitalized_id():
    cfg = {"corpora": {"mywork": {}}}
    assert corpus.corpus_label(cfg, "mywork") == "Mywork"


def test_word_freq_counts_tokens():
    freq = corpus.word_freq(["a", "b", "a"])
    assert freq["a"] == 2
    assert freq["b"] == 1


def test_corpus_root_override_wins():
    cfg = {"corpus_root": "/from/config"}
    assert corpus.corpus_root(cfg, override="/from/cli") == "/from/cli"


def test_corpus_root_falls_back_to_config_then_cwd(tmp_path, monkeypatch):
    assert corpus.corpus_root({"corpus_root": "/from/config"}) == "/from/config"
    monkeypatch.chdir(tmp_path)
    assert corpus.corpus_root({}) == os.getcwd()


# ---------------------------------------------------------------------------
# output_dirs()
# ---------------------------------------------------------------------------

def test_output_dirs_uses_configured_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {"output": {"results_dir": "r", "visualizations_dir": "v", "embeddings_dir": "e"}}
    results_dir, viz_dir, embed_dir = corpus.output_dirs(cfg)
    assert results_dir == os.path.join(os.getcwd(), "r")
    assert viz_dir == os.path.join(os.getcwd(), "v")
    assert embed_dir == os.path.join(os.getcwd(), "e")


def test_output_dirs_default_to_the_working_directory(tmp_path, monkeypatch):
    # never the install location: for a non-editable install that would be
    # inside site-packages.
    monkeypatch.chdir(tmp_path)
    results_dir, viz_dir, embed_dir = corpus.output_dirs({})
    assert results_dir == os.path.join(str(tmp_path), "results")
    assert viz_dir.endswith("visualizations")
    assert embed_dir.endswith("embeddings")


def test_output_dirs_respects_explicit_root(tmp_path):
    cfg = {"_output_root": str(tmp_path)}
    results_dir, _, _ = corpus.output_dirs(cfg)
    assert results_dir == os.path.join(str(tmp_path), "results")


def test_output_dirs_keeps_absolute_dir_names(tmp_path):
    cfg = {"output": {"results_dir": str(tmp_path / "elsewhere")}}
    results_dir, _, _ = corpus.output_dirs(cfg)
    assert results_dir == str(tmp_path / "elsewhere")


# ---------------------------------------------------------------------------
# load_config_from_args()
# ---------------------------------------------------------------------------

def test_load_config_from_args_applies_path_overrides(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({}), encoding="utf-8")
    parser = corpus.add_config_args()
    args = parser.parse_args([
        "--config", str(path),
        "--corpus-root", "/corpora",
        "--output-root", "/out",
    ])
    cfg = corpus.load_config_from_args(args, validate=False)
    assert cfg["corpus_root"] == "/corpora"
    assert corpus.output_dirs(cfg)[0] == os.path.join("/out", "results")
