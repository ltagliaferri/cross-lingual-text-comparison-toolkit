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


# ---------------------------------------------------------------------------
# embed_corpus (model mocked)
# ---------------------------------------------------------------------------

def test_embed_corpus_defaults_to_labse(study, monkeypatch):
    from cross_lingual_toolkit import embed_corpus
    import numpy as np
    study.pop("embedding_model", None)
    calls = []
    monkeypatch.setattr(embed_corpus, "get_model", lambda name: calls.append(name) or object())
    monkeypatch.setattr(embed_corpus, "embed_chunks",
                        lambda chunks, model, path, **kw: np.zeros((len(chunks), 2)))
    embed_corpus.run(study)
    assert calls == ["sentence-transformers/LaBSE"]


def test_prepare_collection_skips_items_below_the_minimum_length(tmp_path, capsys):
    from cross_lingual_toolkit import embed_corpus
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "1.txt").write_text("tiny", encoding="utf-8")
    (tmp_path / "c" / "2.txt").write_text("A real letter. " * 27, encoding="utf-8")
    config = {
        "corpus_root": str(tmp_path),
        "corpora": {"c": {"type": "collection", "path": "c", "author": "a",
                          "work": "w", "language": "en"}},
    }
    chunks = embed_corpus.prepare_collection(config, "c")
    assert len(chunks) == 1
    assert chunks[0]["item_num"] == "2"
    assert "skipped 1 item" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Figure inspection helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def saved_figures(monkeypatch):
    """Record every figure saved, keyed by file name."""
    from matplotlib.figure import Figure
    saved = {}
    original = Figure.savefig

    def spy(self, fname, *args, **kwargs):
        result = original(self, fname, *args, **kwargs)
        saved[os.path.basename(str(fname))] = self
        return result

    monkeypatch.setattr(Figure, "savefig", spy)
    return saved


def _tick_labels(ax):
    return [t.get_text() for t in ax.get_xticklabels()]


def _legend_labels(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def test_plots_label_groups_by_their_configured_names(study, tmp_path, saved_figures):
    term_frequency.analyze(study)
    ax = saved_figures["term_freq_volumes.png"].axes[0]
    assert "Volume1" in _tick_labels(ax) and "Volume2" in _tick_labels(ax)

    lexical_richness.analyze(study)
    box_ax = saved_figures["ttr_by_group.png"].axes[0]
    assert _tick_labels(box_ax) == ["Volume1", "Volume2"]
    scatter_ax = saved_figures["ttr_scatter.png"].axes[0]
    assert _legend_labels(scatter_ax) == ["Volume1", "Volume2"]


def test_lexical_richness_boxplot_avoids_version_specific_label_kwargs(
        study, tmp_path, monkeypatch, saved_figures):
    # boxplot(tick_labels=...) only exists from matplotlib 3.9, labels=
    # is deprecated there; pyproject allows >= 3.5
    from matplotlib.axes import Axes
    calls = []
    original = Axes.boxplot

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "boxplot", spy)
    lexical_richness.analyze(study)
    assert calls
    assert all("tick_labels" not in kw and "labels" not in kw for kw in calls)
    assert _tick_labels(saved_figures["ttr_by_group.png"].axes[0]) == ["Volume1", "Volume2"]


def test_lexical_richness_leaves_empty_items_out_of_the_statistics(study, tmp_path, capsys):
    letters = tmp_path / "corpus" / "letters" / "Volume2"
    (letters / "2.txt").write_text("", encoding="utf-8")
    lexical_richness.analyze(study)

    rows = _read_csv(tmp_path / "results" / "lexical_richness_collection.csv")
    empty = [r for r in rows if r["group"] == "Volume2" and r["item_num"] == "2"]
    assert len(empty) == 1
    assert empty[0]["ttr"] == "" and empty[0]["content_ttr"] == ""

    out = capsys.readouterr().out
    assert "Skipped 1 item" in out
    # Volume2's only real letter is all-distinct content words: TTR 1.0,
    # and the empty file must not pull the mean to 0.5
    volume2 = [line for line in out.splitlines() if line.strip().startswith("Volume2")][0]
    assert "n=  1" in volume2 and "mean=1.000" in volume2


def test_lexical_richness_prints_a_single_mattr_score(study, capsys):
    lexical_richness.analyze(study)
    assert "MATTR (window 20" in capsys.readouterr().out


def test_term_frequency_leaves_an_empty_group_blank(study, tmp_path, saved_figures):
    (tmp_path / "corpus" / "letters" / "Volume2" / "1.txt").write_text("", encoding="utf-8")
    term_frequency.analyze(study)
    rows = {r["text"]: r for r in _read_csv(tmp_path / "results" / "term_frequency.csv")}
    assert rows["Volume2"]["total_tokens"] == "0"
    assert rows["Volume2"]["light_per10k"] == ""
    assert rows["Volume2"]["place_per10k"] == ""
    assert float(rows["Volume1"]["light_per10k"]) > 0


def test_term_frequency_warns_about_unmatchable_terms(study, capsys):
    study["term_frequency_clusters"]["light"]["terms"] += ["l'amour", "the"]
    term_frequency.analyze(study)
    out = capsys.readouterr().out
    assert "\"l'amour\" can never match" in out
    assert '"the" can never match' in out
    assert '"light" can never match' not in out


def test_cross_corpus_frequency_exact_rates(study, tmp_path):
    # By hand, from PRIMARY / COMPARISON above (single-character "a"/"y"
    # and the digits drop out; stopwords from the fixture lists removed):
    #   primary: 37 tokens - 7 "the", 2 "and", 2 "was" = 26 content tokens;
    #            light 3 + warm 3 = 6
    #   comparison: 20 tokens - 3 "el", 2 "la", 1 "una", 1 "era" = 13;
    #            luz 2 + calida 2 = 4
    cross_corpus_frequency.analyze(study)
    rows = {r["cluster"]: r for r in
            _read_csv(tmp_path / "results" / "cross_freq_comparison.csv")}
    assert int(rows["light"]["primary_raw"]) == 6
    assert float(rows["light"]["primary_per10k"]) == pytest.approx(2307.69)
    assert int(rows["light"]["comparison_raw"]) == 4
    assert float(rows["light"]["comparison_per10k"]) == pytest.approx(3076.92)


def test_cross_corpus_frequency_credits_a_zero_g2_to_neither(study, tmp_path):
    study["cross_corpus_concept_clusters"]["absent"] = {
        "label": "Absent", "en": ["zebra"], "es": ["cebra"]}
    cross_corpus_frequency.analyze(study)
    rows = {r["cluster"]: r for r in
            _read_csv(tmp_path / "results" / "cross_dunning_clusters.csv")}
    assert float(rows["absent"]["dunning_g2"]) == 0
    assert rows["absent"]["favours"] == "neither"


def test_cross_corpus_frequency_matches_capitalised_forms(study, tmp_path):
    study["cross_corpus_concept_clusters"]["light"]["en"] = ["Light", "WARM"]
    cross_corpus_frequency.analyze(study)
    rows = {r["cluster"]: r for r in
            _read_csv(tmp_path / "results" / "cross_freq_comparison.csv")}
    assert int(rows["light"]["primary_raw"]) == 6


def test_an_analysis_error_path_exits_non_zero(study, tmp_path, monkeypatch, capsys):
    # the comparison text tokenizes to nothing -> "ERROR: ..." and status 1
    (tmp_path / "corpus" / "comparison.txt").write_text("1300, 1301.", encoding="utf-8")
    config_path = tmp_path / "study2.json"
    data = {k: v for k, v in study.items() if k != "_config_path"}
    config_path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["x", "--config", str(config_path)])
    with pytest.raises(SystemExit) as exc:
        cross_corpus_frequency.main()
    assert exc.value.code == 1
    assert "ERROR:" in capsys.readouterr().out
