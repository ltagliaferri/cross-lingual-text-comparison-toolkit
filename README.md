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

The pipeline code, a config template to copy, and two example stopword
lists — no corpus, no results, no figures. To use it, write a study
config (see **Config** below) and supply your own corpus.

```
cross-lingual-text-comparison-toolkit/
├── pyproject.toml
├── config/
│   └── template_study.json         # copy this to start your own study
├── src/
│   └── cross_lingual_toolkit/
│       ├── *.py                    # the pipeline — see table below
│       └── stopwords/*.txt         # example lists, installed with the package
└── tests/
```

Running a script creates `results/`, `visualizations/`, and `embeddings/`
in the current working directory — pass `--output-root` (or set
`output.root` in the config) to put them elsewhere. `--output-root` wins
over `output.root`.

| Module | Installed command | Purpose |
|---|---|---|
| `corpus.py` | *(library, no CLI)* | Config-driven text loading, tokenization, stopwords, config validation |
| `metrics.py` | *(library, no CLI)* | Corpus-independent measures (TTR/MATTR, Dunning G², cluster counts, dependency-label variants) — no plotting or NLP imports |
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

Multilingual encoders truncate long input. LaBSE reads 256 word-pieces
and silently ignores the rest, and medieval Italian and Latin come to about
3.9 characters per word-piece, so chunks of 800 characters fill the window
while chunks of 1,600 lose roughly their second half (measured on a
three-corpus study: at 1,600 characters, 73 to 87 percent of chunks
overran the window; at 800, none did). The template default is therefore
`chunking.max_chars: 800`, and `xlt-embed-corpus` prints a warning with a
suggested `max_chars` for your corpus whenever chunks exceed the model's
window. Chunks are built from blank-line-separated paragraphs, never
longer than `chunking.max_chars`:
an over-long paragraph is split at sentence ends, then line breaks, then
whitespace. `chunking.min_chars` only drops a whole text shorter than
it; chunks of a longer text (the last one included) can be shorter than
`min_chars` when a paragraph break and `max_chars` force a flush; `0`
keeps everything. All of this describes
single-file corpora. In a collection corpus, items shorter than
`chunking.collection_item_min_chars` (default 200) are skipped, since very
short items give noisy embeddings; an item up to
`chunking.collection_item_split_threshold` (default 900) is kept whole, so
it can be longer than `max_chars`; and a longer item is chunked as above but
with `collection_item_min_chars` in place of `chunking.min_chars`.

**Permutation test** (`analysis_a_umap.py`). The statistic is the mean of
the two corpora's within-corpus similarities minus their between-corpus
similarity, so it does not depend on which corpus is listed first.
Corpus labels are shuffled `analysis.n_permutations` times and
p = (count + 1) / (n + 1), so p is never exactly 0. Each corpus needs at
least two chunks. Read the result with two caveats in mind: chunks of one
text are not independent (neighbours share vocabulary and topic), which
overstates significance; and the two corpora always differ in language
(and often genre), so a small p alone does not show an authorial
difference.

**Log-likelihood (G²)** (`cross_corpus_frequency.py`). The score is the
two-cell log-likelihood standard in corpus linguistics (Rayson & Garside
2000), which sums over each cluster's two observed counts rather than all
four cells of Dunning's full 2×2 table, and is signed (positive = the
primary corpus uses the cluster more). As a rough guide, compare |G²|
with chi-square at 1 df: 3.84 (p < .05), 6.63 (p < .01), 10.83
(p < .001). Those values assume independent tokens and a single test;
with many clusters, and words that cluster in real text, treat them as
lower bounds. A G² of 0 (equal rates, or no occurrences on either side) is
reported as favouring "neither".

**Counting.** Configured terms, forms, lemmas and stopwords are matched
case-insensitively and after Unicode NFC normalization, exactly as the
text is tokenized (lowercase, NFC, split on anything that is not a
letter; combining marks such as Devanagari vowel signs or Arabic and
Hebrew pointing, and the joiners ZWNJ and ZWJ, count as part of the word;
soft hyphens (U+00AD) are removed, so a word split by one counts as
the unbroken word). A configured term the tokenizer can never produce — one with an
apostrophe or hyphen (`l'amour`, `well-being`), a single character not in
`keep_short_tokens`, or a word on the stopword list — would always count
zero, so the scripts print a `WARNING` naming it. Term-frequency clusters
(`term_frequency_clusters`) are counted independently, so one token can
count toward several clusters; cross-corpus concept clusters credit each
token to the first cluster that lists it only. A term listed twice in one
cluster is counted once.

**Lexical richness.** `lexical_richness.py` prints MATTR (the mean TTR
over every `lexical_richness.mattr_window`-token window) for the single
work; the `mattr` column of `lexical_richness_single_work.csv` holds the
per-window TTRs it averages. The single-work windows keep stopwords,
while the collection box plot and scatter use content-word TTR, so the
two numbers are not directly comparable. Collection items with no content
tokens get blank TTR cells and are left out of the plots and summary.

## Config

Everything study-specific lives in one JSON file, passed via `--config`.
Top-level keys:

- `corpus_root` — the directory corpus paths are resolved against
  (overridden by `--corpus-root`; defaults to the working directory).
- `corpora` — one entry per named corpus: `type` (`single_file` or
  `collection`, e.g. letters grouped into volumes), `path` (relative to
  `corpus_root`), `author`, `work`, `language`, `label`, optional `color`
  (used in the UMAP plots), optional `strip_after_marker` (drop everything
  from a marker line onward, e.g. an editorial apparatus section), and,
  for a collection, optional `groups`: the subdirectories of `path`, in
  order, each holding one `.txt` file per item. Omit `groups` when the
  `.txt` files sit directly in `path`. Plots and CSVs label each group by
  its name.
- `study` — which corpus ids play which role: `single_work_corpus`,
  `collection_corpus`, `comparison_corpus`, `primary_corpora` (combined for
  the single-language analyses), `primary_language`, `comparison_language`.
- `languages` — spaCy model name (+ fallback) and/or Stanza model code per
  language code, and language-specific tokenization behavior:
  `stopwords_file` (a filename resolved as described under **Stopword
  lists** below — next to the config, then the working directory, then the
  bundled lists — or an absolute path), `keep_short_tokens`
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
  defaults, all optional (`embedding_model` defaults to
  `sentence-transformers/LaBSE`). `output.root` (or `--output-root`, which
  takes precedence) sets where `results/`, `visualizations/` and
  `embeddings/` are written; by default that is the working directory.
- `analysis`, `lexical_richness`, `adjective_window`, `dependency_parsing`,
  `topic_modeling` — per-script tuning that used to be hardcoded:
  permutation count, MATTR window, context-window size, spaCy chunk size,
  and BERTopic's `min_topic_size`. All optional, all with defaults.

**Comment keys:** any key starting with `_` is treated as an annotation
and skipped wherever the toolkit iterates a section, so you can document
your own config inline the way `template_study.json` does.

**Config validation:** every script checks the config before doing any
work — that the corpus ids named in `study` exist, that each corpus
declares the keys the loaders need, that the languages line up, that the
study roles and config sections *that script* uses are present and well
formed (every cluster has a `label`, term lists are lists, a concept
cluster has forms for both languages, numeric settings are positive — or
non-negative for the `min_chars` settings and the split threshold, where 0
is meaningful), and
that the corpus files and `groups` directories it reads exist on disk —
and reports all the problems at once instead of failing partway through.
Config errors and missing optional dependencies print a one-line
`ERROR:` and exit with status 1.

**Language codes:** pick a short code per language (ISO 639-1 two-letter
codes are a reasonable default, e.g. `en`, `fr`, `de`; the template config
uses `en`/`fr` as placeholders) and use it consistently everywhere a
language is referenced: as a key in `languages`, as the value of
`corpora[...]['language']` and
`study['primary_language']`/`['comparison_language']`, and as a key
inside any per-language dict your own config adds (e.g.
`cross_corpus_concept_clusters[cluster]['en']`/`['fr']`, or
`cross_corpus_dependency`'s `primary`/`comparison` sections, which are
keyed by role rather than by code but still correspond 1:1 with
`primary_language`/`comparison_language`). The scripts always look
languages up by whatever code you chose — no code is hardcoded anywhere.

**Stopword lists:** plain text, one token per line — blank lines and lines
starting with `#` are ignored, so you can group/comment them (see
`src/cross_lingual_toolkit/stopwords/` for two examples, which install
with the package and can be referenced by filename from any config).
`stopwords_file` is looked for in a `stopwords/` directory next to your
config file, then under the working directory, then among the bundled
lists; an absolute path also works. Entirely optional: a language with no
`stopwords_file` configured just gets an empty stopword set, not an
error. One caveat for `cross_corpus_frequency.py`: configure a list for
both languages or for neither, since removing stopwords from only one
side of a cross-corpus comparison biases every rate and G² score toward
that corpus (the script warns if you do). Instead of a file name,
`stopwords_file` also accepts `"spacy"`, which uses spaCy's built-in
stopword list for the language code (about 70 languages; needs the
`parsing` extra but no model download), or `"spacy:<lang>"` to name
spaCy's code when your config uses its own (e.g. `"spacy:it"` for a
language you call `it_med`). A real file named `spacy` is shadowed by the
sentinel; rename it. An unknown code or a missing spaCy is reported when
the config is validated.

Every script accepts `--config path/to/your_study.json`,
`--corpus-root path/to/corpus/root` (the directory containing the paths
referenced in `corpora[...]['path']`), and `--output-root path/to/output`.

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
4. **Not every script needs every role.** Each script checks for the
   roles and sections it uses (see **Config validation**):
   - `term_frequency.py` and `lexical_richness.py` compare a
     `single_work_corpus` (a `single_file`) against a `collection_corpus`
     (a `collection`) — skip them if your corpus doesn't have that shape.
   - `cross_corpus_frequency.py`, `cross_corpus_dependency.py` and both
     `analysis_*.py` read `single_work_corpus` and `comparison_corpus`
     (both `single_file` for the two `cross_corpus_*` scripts).
   - `adjective_window.py` and `dependency_parsing.py` read
     `primary_corpora` and `primary_language`.
   - `embed_corpus.py` embeds every entry in `corpora`;
     `analysis_a_umap.py` plots the corpora beyond the two compared ones
     as context.
5. **Point at your files:** set `corpus_root` in the config, or pass
   `--corpus-root /path/to/your/texts` at runtime — your texts don't need
   to live inside this repo.
6. **Run `embed_corpus.py` first** (see **Usage** below) — every other
   script reads from the embeddings it caches. A cached
   `embeddings/<corpus_id>.npy` is reused only if the hash in the
   `.npy.key` file next to it matches the current chunk texts and
   `embedding_model`, so changing a text, the `chunking` settings or the
   model re-embeds that corpus. The analysis scripts
   stop with an error if a corpus's chunk and embedding counts disagree.

## Setup

The core install is small — matplotlib, numpy, scipy, scikit-learn — and
covers `term_frequency.py`, `lexical_richness.py` and
`cross_corpus_frequency.py`. The model-heavy stacks are extras, so add
only what your analysis needs:

```bash
pip install -e .                  # core: frequency + lexical richness
pip install -e ".[embeddings]"    # + embed_corpus, analysis_a_umap
pip install -e ".[parsing]"       # + adjective_window, both dependency scripts
pip install -e ".[topics]"        # + analysis_b_topics
pip install -e ".[all]"           # everything
```

(Editable install — recommended, since you'll likely be adapting the
config/scripts to your own study. Drop `-e` for a plain install if not;
outputs and stopword lists resolve the same way either way.)

Then install whatever spaCy and/or Stanza models your config's
`languages` section references, e.g.:

```bash
python -m spacy download en_core_web_lg
python -c "import stanza; stanza.download('fr')"
```

(These match the template config's `en`/`fr` placeholders — substitute
the models your own `languages` section names.)

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

Each script writes to `results/` (CSV/text) and `visualizations/` (PNG)
under the working directory, or under `--output-root` (or the config's
`output.root`) if you set one.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the corpus-independent logic — tokenization (including combining
marks and joiners), per-script config validation and on-disk path checks,
stopword resolution, chunking and parser splitting, the embedding cache,
the permutation test, Dunning G², TTR/MATTR, dependency-label and
context-window helpers — plus end-to-end runs of the model-free analyses
over a small synthetic two-language corpus, the spaCy/Stanza extraction
code driven by stub pipelines, and the CLI wrappers' error reporting and
exit codes. No real corpus data and no downloaded spaCy/Stanza models are
involved.

`tests/test_metrics.py` imports only `cross_lingual_toolkit.metrics`, so
that half of the suite runs with nothing but `pytest` installed.

## Future work

Deferred review items and open questions are listed in
[docs/future-work.md](docs/future-work.md).

## License

MIT — see [LICENSE](LICENSE).
