"""
End-to-end runs of the analyses that need no downloaded model, over a
tiny synthetic two-language corpus written into tmp_path.

These are deliberately small — small enough that a short text falls under
the MATTR window and a config carries "_comment" annotations, which is
exactly the shape that used to crash.
"""

import csv
import json
import os

import pytest

from cross_lingual_toolkit import cross_corpus_frequency, lexical_richness, term_frequency


PRIMARY = """
The garden was quiet and the light was warm.
The garden held a quiet light, and the warm stone kept the heat.

A second paragraph about the same quiet garden, written in 1300,
with a warm light over the stone wall.
"""

COMPARISON = """
El jardin estaba tranquilo y la luz era calida.
El jardin guardaba una luz tranquila, y la piedra calida retenia el calor.
"""

LETTERS = {
    "Volume1": {
        "1": "A short letter about the quiet garden and its warm light.",
        "2": "Another letter, longer, about the stone and the warm quiet light of the garden.",
    },
    "Volume2": {
        "1": "A later letter concerning the garden, the light, and the quiet stone.",
    },
}


@pytest.fixture
def study(tmp_path, monkeypatch):
    """Write a miniature corpus + config, and run from tmp_path."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "primary.txt").write_text(PRIMARY, encoding="utf-8")
    (corpus_dir / "comparison.txt").write_text(COMPARISON, encoding="utf-8")
    for group, items in LETTERS.items():
        (corpus_dir / "letters" / group).mkdir(parents=True)
        for name, text in items.items():
            (corpus_dir / "letters" / group / f"{name}.txt").write_text(text, encoding="utf-8")

    stop_dir = tmp_path / "stopwords"
    stop_dir.mkdir()
    (stop_dir / "en.txt").write_text("# articles\nthe\na\nand\nwas\nof\n", encoding="utf-8")
    (stop_dir / "es.txt").write_text("# articulos\nel\nla\nuna\ny\nera\n", encoding="utf-8")

    config = {
        "_comment": "annotated like the template config",
        "corpus_root": str(corpus_dir),
        "languages": {
            "_comment": "a comment among the languages",
            "en": {"stopwords_file": "en.txt"},
            "es": {"stopwords_file": "es.txt"},
        },
        "corpora": {
            "_comment": "a comment among the corpora",
            "primary": {"label": "Primary", "author": "a", "work": "w1",
                        "language": "en", "type": "single_file", "path": "primary.txt"},
            "letters": {"label": "Letters", "author": "a", "work": "w2",
                        "language": "en", "type": "collection", "path": "letters",
                        "groups": ["Volume1", "Volume2"]},
            "comparison": {"label": "Comparison", "author": "b", "work": "w3",
                           "language": "es", "type": "single_file", "path": "comparison.txt"},
        },
        "study": {
            "primary_corpora": ["primary", "letters"],
            "single_work_corpus": "primary",
            "collection_corpus": "letters",
            "comparison_corpus": "comparison",
            "primary_language": "en",
            "comparison_language": "es",
        },
        "term_frequency_clusters": {
            "_comment": "the annotation that used to crash count_clusters",
            "light": {"label": "Light", "terms": ["light", "warm"]},
            "place": {"label": "Place", "terms": ["garden", "stone"]},
        },
        "cross_corpus_concept_clusters": {
            "_comment": "same annotation, cross-lingual",
            "light": {"label": "Light", "en": ["light", "warm"],
                      "es": ["luz", "calida"]},
            "place": {"label": "Place", "en": ["garden", "stone"],
                      "es": ["jardin", "piedra"]},
        },
        "lexical_richness": {"mattr_window": 20, "smoothing_window": 5},
    }
    config_path = tmp_path / "study.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    from cross_lingual_toolkit.corpus import load_config, validate_config
    loaded = load_config(str(config_path))
    validate_config(loaded)
    return loaded


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_term_frequency_runs_on_an_annotated_config(study, tmp_path):
    term_frequency.analyze(study)

    rows = _read_csv(tmp_path / "results" / "term_frequency.csv")
    assert [r["text"] for r in rows] == ["Primary", "Volume1", "Volume2", "Letters (all)"]
    # "light" and "warm" both occur in the primary text
    assert int(rows[0]["light_raw"]) > 0
    assert float(rows[0]["light_per10k"]) > 0
    assert os.path.exists(tmp_path / "visualizations" / "term_freq_corpora.png")
    assert os.path.exists(tmp_path / "visualizations" / "term_freq_volumes.png")


def test_lexical_richness_runs_on_a_text_shorter_than_the_mattr_window(study, tmp_path):
    # the primary text is well under 500 tokens; with the default window
    # this used to raise "'float' object is not iterable".
    study["lexical_richness"]["mattr_window"] = 500
    lexical_richness.analyze(study)

    mattr_rows = _read_csv(tmp_path / "results" / "lexical_richness_single_work.csv")
    assert len(mattr_rows) == 1                      # one whole-text TTR
    assert 0.0 < float(mattr_rows[0]["mattr"]) <= 1.0
    assert os.path.exists(tmp_path / "visualizations" / "mattr_single_work.png")

    per_item = _read_csv(tmp_path / "results" / "lexical_richness_collection.csv")
    assert len(per_item) == 3                        # three letters across two volumes


def test_lexical_richness_windows_a_longer_text(study, tmp_path):
    lexical_richness.analyze(study)                  # mattr_window=20 from the fixture
    rows = _read_csv(tmp_path / "results" / "lexical_richness_single_work.csv")
    assert len(rows) > 1


def test_cross_corpus_frequency_normalizes_both_sides_comparably(study, tmp_path):
    cross_corpus_frequency.analyze(study)

    rows = {r["cluster"]: r for r in
            _read_csv(tmp_path / "results" / "cross_freq_comparison.csv")}
    assert set(rows) == {"light", "place"}
    # both corpora mention both concepts, so every rate is non-zero — the
    # comparison side is tokenized like the primary one, not left raw.
    for row in rows.values():
        assert float(row["primary_per10k"]) > 0
        assert float(row["comparison_per10k"]) > 0

    dunning = _read_csv(tmp_path / "results" / "cross_dunning_clusters.csv")
    assert {r["cluster"] for r in dunning} == {"light", "place"}
    assert os.path.exists(tmp_path / "visualizations" / "cross_dunning.png")


def test_cross_corpus_frequency_warns_on_one_sided_stopwords(study, tmp_path, capsys):
    del study["languages"]["es"]["stopwords_file"]
    cross_corpus_frequency.analyze(study)
    assert "stopwords_file" in capsys.readouterr().out


def test_analyses_write_under_an_explicit_output_root(study, tmp_path):
    out = tmp_path / "elsewhere"
    study["_output_root"] = str(out)
    term_frequency.analyze(study)
    assert os.path.exists(out / "results" / "term_frequency.csv")
