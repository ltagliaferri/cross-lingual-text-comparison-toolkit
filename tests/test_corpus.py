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


# ---------------------------------------------------------------------------
# load_config()
# ---------------------------------------------------------------------------

def test_load_config_reads_valid_json(tmp_path):
    path = tmp_path / "study.json"
    path.write_text(json.dumps({"study": {"primary_language": "en"}}), encoding="utf-8")
    cfg = corpus.load_config(str(path))
    assert cfg["study"]["primary_language"] == "en"


def test_load_config_missing_file_raises_helpful_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="Pass --config"):
        corpus.load_config(str(missing))


# ---------------------------------------------------------------------------
# load_stopwords()
# ---------------------------------------------------------------------------

def test_load_stopwords_none_configured_returns_empty_set():
    cfg = {"languages": {"en": {}}}
    assert corpus.load_stopwords(cfg, "en") == set()


def test_load_stopwords_ignores_comments_and_blank_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "_STOPWORDS_DIR", str(tmp_path))
    (tmp_path / "sw.txt").write_text(
        "# a comment\n\nthe\nand\n  \n# another comment\nof\n", encoding="utf-8"
    )
    cfg = {"languages": {"en": {"stopwords_file": "sw.txt"}}}
    assert corpus.load_stopwords(cfg, "en") == {"the", "and", "of"}


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


def test_corpus_root_falls_back_to_config_then_default():
    assert corpus.corpus_root({"corpus_root": "/from/config"}) == "/from/config"
    assert corpus.corpus_root({}) == corpus.DEFAULT_CORPUS_ROOT


# ---------------------------------------------------------------------------
# output_dirs()
# ---------------------------------------------------------------------------

def test_output_dirs_uses_configured_names():
    cfg = {"output": {"results_dir": "r", "visualizations_dir": "v", "embeddings_dir": "e"}}
    results_dir, viz_dir, embed_dir = corpus.output_dirs(cfg)
    assert results_dir == os.path.join(corpus._REPO_ROOT, "r")
    assert viz_dir == os.path.join(corpus._REPO_ROOT, "v")
    assert embed_dir == os.path.join(corpus._REPO_ROOT, "e")


def test_output_dirs_defaults():
    results_dir, viz_dir, embed_dir = corpus.output_dirs({})
    assert results_dir.endswith("results")
    assert viz_dir.endswith("visualizations")
    assert embed_dir.endswith("embeddings")
