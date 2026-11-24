# Cross-Lingual Text Comparison Toolkit

A pipeline for comparing texts across language boundaries, using
multilingual sentence embeddings to bridge languages that break
traditional stylometry (function-word counting requires a shared lexicon;
embeddings don't). Within a single language, it also offers that
stylometry (`xlt-stylometry`: Burrows's and Cosine Delta, General
Imposters), as the same-language complement to the cross-lingual tools.
Built for cross-lingual authorship questions, but the
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
| `stylometry.py` | `xlt-stylometry` | Same-language stylometry: most-frequent-word Delta between groups, leave-one-out attribution, General Imposters verification, rolling Delta |
| `parallel_passages.py` | `xlt-parallel-passages` | Cross-lingual parallel-passage search: best match for every source sentence in the target corpora, merged into passages and aligned sentence by sentence |

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

**Stylometry (same language)** (`stylometry.py`). The embedding
analyses measure meaning, not style: LaBSE is trained to make translations
look alike, so it discards the habits of expression authorship rests on
and in practice tracks topic and genre. Stylometry measures those habits
directly, as the relative frequencies of the most frequent words (mostly
function words). It therefore only works *within one language*: every
corpus the `stylometry` section uses (and the rolling corpus) must share
one `language`, and validation says so otherwise. Use the embedding
analyses to compare across languages.

You define groups of texts in `stylometry.groups`, e.g. letters from
different periods, scribes or attributed authors. Each group's texts are
concatenated in config order (collection items in their sorted load
order), cleaned with `strip_patterns`, tokenized as everywhere else but
*without* stopword removal (function words are the signal), and cut into
consecutive samples of `sample_words` tokens. The features are the `mfw`
most frequent tokens over all samples of all groups (minus
`exclude_words`; ties broken alphabetically), as relative frequencies
per sample, z-scored over all samples (population SD; a feature with zero
SD is dropped and the count printed). Burrows's Delta is the mean
absolute difference of two z-score vectors; Cosine Delta (Smith and
Aldridge 2011; Evert et al. 2017) is their cosine distance (1 − cosine
similarity). It is the default, and the more robust of the two.
The numerical core (samples, features, Delta, attribution, General
Imposters, rolling Delta) is in `stylo_core.py`, which needs only numpy
and the standard library; `stylometry.py` adds the config, I/O and plots.

- **Distances.** Group-to-group distance is the distance between group
  centroids (the mean of a group's sample z-vectors).
- **Leave-one-out attribution.** Each sample is assigned to the nearest
  group centroid, its own group's centroid computed without it; the
  script prints the confusion matrix and accuracy. High accuracy means the
  groups are stylistically distinguishable. As is common practice, the
  z-scores are computed once over all samples, the held-out one included;
  only the centroid leaves it out.
- **General Imposters verification** (Koppel and Winter 2014; Kestemont
  et al. 2016). For each `verification` pair: in each of `iterations`
  rounds, sample `feature_fraction` of the features and as many samples
  from the `imposters` groups as the candidate has (all, if fewer). A
  questioned sample wins the round when its nearest candidate sample is
  closer than every sampled imposter. The score is wins / (iterations ×
  questioned samples), in [0, 1]; the swapped score (candidate as
  questioned) and their mean are reported too, since the two groups may
  differ in size.
- **Rolling Delta** (optional `rolling`). Windows of `window_words`
  tokens every `step_words` across one corpus, each z-scored with the
  group samples' means and SDs and compared with each reference group's
  centroid. A warning is printed if the rolling corpus is also part of a
  reference group, which inflates its closeness to that group.

Config keys (all optional except `groups`):

| Key | Meaning | Default |
|---|---|---|
| `groups` | `{group_name: [selector, ...]}`. A selector is a corpus id (the whole corpus), `"<corpus_id>:<group>"` (one configured group of a collection, e.g. `"letters:Volume1"`), or `"<corpus_id>:<group>/<item>"` (one item, its file name without `.txt`, e.g. `"letters:Volume1/017"`). Group and item selectors need a collection with `groups` | required |
| `groups_file` | CSV with a `group,selector` header whose rows add to (never replace) `groups`, for long lists. A relative path is looked for next to the config file, then under the working directory (like `stopwords_file`, without the bundled lists) | none |
| `corpora` | a restriction: the only corpus ids that selectors and `rolling.corpus` may use (each must be a configured corpus) | no restriction |
| `mfw` | number of most frequent words used as features | 200 |
| `sample_words` | tokens per sample; a trailing remainder shorter than half a sample is dropped | 2000 |
| `strip_patterns` | regular expressions removed (replaced by a space) from every text before tokenizing, after NFC normalization, with `re.MULTILINE` | `[]` |
| `exclude_words` | words never used as features (e.g. proper names, which track topic) | `[]` |
| `distance` | `"cosine_delta"` or `"burrows_delta"` | `"cosine_delta"` |
| `imposters` | groups forming the imposter pool | `[]` |
| `verification` | `[{"questioned": group, "candidate": group}, ...]`; the two must differ and neither may be an imposter; needs `imposters` | `[]` |
| `iterations` | General Imposters rounds | 100 |
| `feature_fraction` | share of features sampled per round, in (0, 1] | 0.5 |
| `rolling` | `{"corpus": id, "references": [group, ...], "window_words": 2000, "step_words": 500}` | none |
| `random_seed` | seed for every random choice | 42 |

A text assigned to two groups, or twice to one group, is a warning, not
an error, since overlapping groups can be intended; it does make the
groups look more alike.

Outputs: `results/stylometry_distances.csv` (group × group),
`results/stylometry_attribution.csv` (per sample: group, nearest group,
distance to each group; blank where a group has no other sample),
`results/stylometry_features.csv` (the MFW list with overall relative
frequency, and whether a zero SD dropped it),
`results/stylometry_verification.csv`, `results/stylometry_rolling.csv`,
`visualizations/stylometry_distances.png` (heatmap) and
`visualizations/stylometry_rolling.png`.

Cautions:

- **Editions.** Punctuation and spelling in edited historical texts are
  often the modern editor's, so the features are words only (no
  punctuation, no sentence length), and texts from different editions may
  still differ for editorial reasons.
- **Formulas.** Fixed openings and closings (a letter's salutation and
  farewell) inflate a group's consistency; strip them with
  `strip_patterns`.
- **Sample size.** Delta is unreliable below a few thousand words per
  sample (Eder 2015). A group too short for one full sample is kept as one
  short sample, with a printed warning.
- **Same language only.** See above.
- **The imposters score is a proportion, not a test.** It is not a
  p-value, and it is only as good as the imposter pool, which should match
  the questioned and candidate texts in language, period and genre as
  closely as possible. With fewer than three imposter groups or ten
  imposter samples it is weak, and the script prints a warning.

**Parallel-passage search** (`parallel_passages.py`). Finds where one
corpus (the `source`) translates, quotes or paraphrases others (the
`targets`) in another language: an author's own translation embedded in a
third work, say, or a biographer quoting a subject's letters. It reads the
raw texts with the same loaders as every other script (so
`xlt-embed-corpus` need not have run), splits each text or collection item
into blank-line-separated paragraphs and then into sentences at
`sentence_split` (whitespace collapsed, NFC), and embeds every sentence
with `embedding_model`, caching them as
`embeddings/parallel_<corpus_id>_sentences.npy` (re-embedded when the
sentences or the model change). For each source sentence it keeps the
best and runner-up target sentence across all targets by cosine
similarity, computed in blocks of 512 rows so memory stays bounded.

Matches (score ≥ `threshold`) are merged into runs separately for each
target text (the same collection item, or the same single-file corpus).
Walking the source in order, a match joins the current run into its
target text if at most `merge_gap` source sentences lie since that run's
last match (whatever those sentences matched, so a match into another
text does not end the run) and its target index moves forward by at most
`target_window` sentences (or back by one, since the two sides split
sentences differently); otherwise it starts a new run in that text, while
the old run stays open for later matches that fit it. Runs
with at least `min_passage_sentences` matches are passages. With `align`,
each passage's source span is aligned with its target span (extended by
`align_context` sentences each side, within the same target text) by a
monotonic dynamic programme over groups of 1–3 sentences per side (steps
1–1, 1–2, 2–1, 1–3, 3–1, 2–2, 2–3, 3–2), each scored as the cosine of the
joined groups' embeddings × (sentences on both sides) / 2; leaving one
sentence unaligned costs `skip_penalty`.

Config keys (all optional except `source` and `targets`):

| Key | Meaning | Default |
|---|---|---|
| `source` | corpus id whose sentences are searched for | required |
| `targets` | non-empty list of corpus ids to search in; must not include `source` | required |
| `min_sentence_chars` | sentences shorter than this are skipped | 40 |
| `max_sentence_chars` | longer sentences are wrapped (at whitespace) into pieces of at most this length; a value below `min_sentence_chars` is accepted with a warning | 600 |
| `sentence_split` | regular expression splitting a paragraph into sentences | after `.` `?` `!` `;` `:` followed by whitespace, and after `。` `！` `？` with or without whitespace |
| `location_patterns` | `{corpus_id: regex}`. A sentence's location is the last match of the regex (`re.MULTILINE`) in its text up to and including the sentence: group 1 if the regex has a group, else the whole match, e.g. chapter headings | the collection item (`group/item`), or `p<paragraph>` in a single file; also used before a pattern's first match |
| `threshold` | minimum cosine similarity for a sentence match to count as a match and join a passage (0–1) | 0.70 |
| `merge_gap` | most source sentences between two consecutive matches of one passage (counted in the source, whatever they matched; translators condense, so several source sentences in a row can fall below threshold inside a real translation) | 5 |
| `target_window` | most target sentences a passage may move forward between consecutive matches (a step back of one is allowed) | 10 |
| `min_passage_sentences` | matched source sentences a run needs to become a passage | 2 |
| `align` | align each passage sentence by sentence and write `parallel_alignment.csv` | `true` |
| `align_context` | target sentences added each side of a passage's span before aligning | 2 |
| `skip_penalty` | alignment cost of leaving one sentence unaligned | 0.35 |

Outputs: `results/parallel_sentences.csv` (one row per source sentence
in order: `source_idx, source_location, score, runner_up, margin,
target_corpus, target_location, target_idx, source_text, target_text`;
indices count sentences within each corpus), `results/parallel_passages.csv`
(`passage, source_first, source_last, source_location, n_matches,
mean_score, max_score, target_corpus, target_location, target_first,
target_last, source_text, target_text`), `results/parallel_alignment.csv`
(with `align`: `passage, unit, source_first, source_last, target_first,
target_last, similarity, source_text, target_text`; a skipped sentence
has one side empty and no similarity), `results/parallel_summary.txt`
(sentence counts, score quantiles, counts at or above 0.60–0.80, the
passages and their locations) and `visualizations/parallel_passages.png`
(the best score along the source, with the threshold and the passages
shaded). It needs the `embeddings` extra.

Reading the results: validate the settings on a known translation pair
in your corpus first, to see what scores a real translation gets with
your model and texts. Passages, runs of nearby matches into one target
text, are much stronger evidence than isolated matches; isolated
matches between about 0.70 and 0.80 are often shared formulae (stock
phrases, liturgy, scripture) rather than translation. Runs are merged per
target text, so passages into different target texts can overlap in
source span: a source passage may translate one text while a few of its
sentences also match another, as stock phrases or a second witness. A
stock phrase repeated elsewhere in the same target text can still
distort a passage: if it lands within `target_window` of the run it
joins the run, and a long enough stretch of such matches splits it. The
score measures shared meaning, not wording, so paraphrase is found as
well as translation, and a high score is not proof of translation: read
the aligned pairs.

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
  `topic_modeling`, `stylometry`, `parallel_passages` — the domain-specific term lists and settings each script needs. All
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
   - `stylometry.py` reads only its `stylometry` section (see
     **Stylometry (same language)** above), whose corpora must share one
     language.
   - `parallel_passages.py` reads only its `parallel_passages` section
     (see **Parallel-passage search** above).
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
covers `term_frequency.py`, `lexical_richness.py`,
`cross_corpus_frequency.py` and `stylometry.py`. The model-heavy stacks are extras, so add
only what your analysis needs:

```bash
pip install -e .                  # core: frequency, lexical richness, stylometry
pip install -e ".[embeddings]"    # + embed_corpus, analysis_a_umap, parallel_passages
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
the permutation test, Dunning G², TTR/MATTR, Burrows's and Cosine
Delta, General Imposters, parallel-passage matching, run merging and
sentence alignment, dependency-label and context-window helpers — plus end-to-end runs of the model-free analyses
over a small synthetic two-language corpus (the parallel-passage search
with a stub embedding model), the spaCy/Stanza extraction
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
