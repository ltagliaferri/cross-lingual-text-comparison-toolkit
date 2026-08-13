# Cross-Lingual Text Comparison Toolkit

A pipeline for comparing texts across language boundaries, using
multilingual sentence embeddings to bridge languages that break
traditional stylometry (function-word counting requires a shared lexicon;
embeddings don't). Built for cross-lingual authorship questions, but the
underlying tools — embedding-space clustering, bilingual concept
frequency, cross-lingual dependency comparison, topic modeling — apply
just as well to translation studies, genre comparison, or dating texts.

Every path, language model, stopword list, and domain-specific term list is
driven by a JSON study config — nothing about the corpus or its languages
is hardcoded into the scripts. Point `--config` at your own JSON file,
following the schema described below, to run the pipeline on your own
corpus and language pair. Nothing here is specific to historical text: the
underlying embedding model was trained predominantly on modern text, so a
modern cross-lingual question is, if anything, an easier case than the
historical-language one this toolkit was built for.

## What's included

Just the pipeline code — no config, no corpus, no results/data, no
figures. To use it, write a study config (see **Config** below) and
supply your own corpus.

```
cross-lingual-text-comparison-toolkit/
├── pyproject.toml
└── src/
    └── cross_lingual_toolkit/
        └── *.py                    # the pipeline — see table below
```

Running a script creates `results/`, `visualizations/`, and `embeddings/`
at the repo root (gitignored).

| Module | Installed command | Purpose |
|---|---|---|
| `corpus.py` | *(library, no CLI)* | Config-driven text loading, tokenization, stopwords |
| `embed.py` | *(library, no CLI)* | Chunking, embedding, similarity/permutation-test helpers, embedded-corpus registry |
| `embed_corpus.py` | `xlt-embed-corpus` | Builds chunked corpus JSON + cached embeddings for every corpus in the config — run first |
| `term_frequency.py` | `xlt-term-frequency` | Normalized frequency of configured term clusters: single work vs. dated collection |
| `lexical_richness.py` | `xlt-lexical-richness` | Type-token ratio and MATTR (moving-average TTR) |
| `adjective_window.py` | `xlt-adjective-window` | Adjectives co-occurring within a token window of configured anchor terms (spaCy) |
| `dependency_parsing.py` | `xlt-dependency-parsing` | Dependency-parse patterns around configured agents/verbs (spaCy) |
| `cross_corpus_frequency.py` | `xlt-cross-corpus-frequency` | Bilingual concept-cluster frequency + Dunning log-likelihood across languages |
| `cross_corpus_dependency.py` | `xlt-cross-corpus-dependency` | Cross-lingual dependency comparison (spaCy for the primary language, Stanza for the comparison language) |
| `analysis_a_umap.py` | `xlt-analysis-umap` | Embedding-space clustering by author/work, UMAP projection, permutation test |
| `analysis_b_topics.py` | `xlt-analysis-topics` | Cross-lingual topic modeling (BERTopic over the embedding space) |

## Method

A multilingual sentence embedding model (default:
[LaBSE](https://huggingface.co/sentence-transformers/LaBSE)) maps text
chunks from any of its training languages into a shared geometric space,
trained so that semantically equivalent passages in different languages
land near each other. This makes it possible to ask distance-based
questions — do two corpora cluster together or apart, by author, work, or
period? — across a language boundary that word-frequency stylometry
cannot cross.

Before drawing conclusions from embedding distances, it's worth validating
the model on known translation pairs in your corpus (does a passage land
closer to its known translation than to unrelated passages?) — this matters
most for historical-language corpora, since these models are trained
predominantly on modern text.

## Config

Everything study-specific lives in one JSON file, passed via `--config`.
Top-level keys:

- `corpora` — one entry per named corpus: `type` (`single_file` or
  `collection`, e.g. letters grouped into volumes), `path` (relative to
  `corpus_root`), `author`, `work`, `language`, `label`, optional
  `strip_after_marker` (drop everything from a marker line onward, e.g. an
  editorial apparatus section).
- `study` — which corpus ids play which role: `single_work_corpus`,
  `collection_corpus`, `comparison_corpus`, `primary_corpora` (combined for
  the single-language analyses), `primary_language`, `comparison_language`.
- `languages` — spaCy model name (+ fallback) and/or Stanza model code per
  language code, and language-specific tokenization behavior:
  `stopwords_file` (a filename resolved against the top-level `stopwords/`
  directory — put your word list there), `keep_short_tokens`
  (single-character words that are real content, e.g. Italian
  "e"/"o"/"a"/"è", exempt from the tokenizer's length filter), and
  `adjective_suffixes` (a fallback POS heuristic `adjective_window.py`
  uses only when spaCy is unavailable — leave unset to skip the heuristic
  rather than guess).
- `term_frequency_clusters`, `adjective_anchor_terms`, `dependency_agents`,
  `cross_corpus_concept_clusters`, `cross_corpus_dependency`,
  `topic_modeling` — the domain-specific term lists each script needs. All
  keys inside these are your own vocabulary, not fixed categories.
- `colors`, `chunking`, `embedding_model`, `output` — cosmetic/pipeline
  defaults.

**Language codes:** pick a short code per language (the bundled example
uses `it`/`la`; ISO 639-1 two-letter codes are a reasonable default, e.g.
`en`, `fr`, `de`) and use it consistently everywhere a language is
referenced: as a key in `languages`, as the value of `corpora[...]['language']`
and `study['primary_language']`/`['comparison_language']`, and as a key
inside any per-language dict your own config adds (e.g.
`cross_corpus_concept_clusters[cluster]['it']`/`['la']`, or
`cross_corpus_dependency`'s `primary`/`comparison` sections, which are
keyed by role rather than by code but still correspond 1:1 with
`primary_language`/`comparison_language`). The scripts always look languages
up by whatever code you chose — nothing is hardcoded to `it`/`la`.

**Stopword lists:** plain text, one token per line — blank lines and lines
starting with `#` are ignored, so you can group/comment them (see
`stopwords/stopwords_it_medieval.txt` and `stopwords/stopwords_la.txt` for
examples). Entirely optional: a language with no `stopwords_file`
configured just gets an empty stopword set, not an error. If you already
have a spaCy model installed for your language, its built-in list is a
reasonable starting point:
```python
import spacy
print(sorted(spacy.load("en_core_web_sm").Defaults.stop_words))
```

Every script accepts `--config path/to/your_study.json` and
`--corpus-root path/to/corpus/root` (the directory containing the paths
referenced in `corpora[...]['path']`).

## Adding your own texts

1. **Prepare your files.** Two supported shapes, one per corpus:
   - `single_file` — one plain `.txt` file (a continuous work).
   - `collection` — a directory of dated/grouped items: one subdirectory
     per group, each containing one `.txt` file per item (e.g. one letter
     per file, volumes as groups).
2. **Pick a short code for each language** in your corpus (e.g. `en`,
   `fr` — see **Language codes** above). You'll use these consistently
   throughout the config.
3. **Copy the template** rather than writing a config from scratch:
   ```bash
   cp config/template_study.json config/my_study.json
   ```
   Fill in a `languages` entry per code (spaCy/Stanza model names,
   stopwords file if you have one), a `corpora` entry per text (`path`
   resolved against wherever you set as `corpus_root`), and `study`,
   assigning your corpus ids to whichever roles your analysis needs (see
   next step).
4. **Not every script needs every role.** `term_frequency.py` and
   `lexical_richness.py` specifically compare a `single_work_corpus`
   against a `collection_corpus` — skip them if your corpus doesn't have
   that shape. Everything else (`embed_corpus.py`, both
   `analysis_*.py`, `cross_corpus_*.py`, `adjective_window.py`,
   `dependency_parsing.py`) only needs `single_work_corpus`/
   `comparison_corpus` and `primary_language`/`comparison_language`.
5. **Point at your files:** set `corpus_root` in the config, or pass
   `--corpus-root /path/to/your/texts` at runtime — your texts don't need
   to live inside this repo.
6. **Run `embed_corpus.py` first** (see **Usage** below) — every other
   script reads from the embeddings it caches.

## Setup

```bash
pip install -e .
```

(Editable install — recommended, since you'll likely be adapting the
config/scripts to your own study. Drop `-e` for a plain install if not.)

Then install whatever spaCy and/or Stanza models your config's
`languages` section references, e.g.:

```bash
python -m spacy download it_core_news_lg
python -c "import stanza; stanza.download('la')"
```

## Usage

```bash
xlt-embed-corpus --config path/to/your_study.json --corpus-root /path/to/your/corpus        # run first
xlt-analysis-umap --config path/to/your_study.json --corpus-root /path/to/your/corpus
xlt-analysis-topics --config path/to/your_study.json --corpus-root /path/to/your/corpus
# ...etc — see the command column in the script table above; all accept --config and --corpus-root
```

Without installing, run any module directly instead:
`python -m cross_lingual_toolkit.embed_corpus --config ... --corpus-root ...`
(from `src/`, or with `src/` on `PYTHONPATH`).

Each script writes to `results/` (CSV/text) and `visualizations/` (PNG) at
the repo root.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the corpus-independent logic — tokenization, config loading,
chunking, Dunning G², TTR/MATTR, dependency-window helpers — with small
synthetic inputs, not real corpus
data or downloaded spaCy/Stanza models.

## License

MIT — see [LICENSE](LICENSE).
