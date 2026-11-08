# Future work

Items raised during the November 2026 code review that were deliberately
deferred. None is a known bug; each is a judgement call, a behaviour that
needs a real corpus to evaluate, or a low-value edge case. Open an issue (or
a PR) if one of them matters for your study.

## Methodology

- **Casefolding.** Configured terms and text are matched with `lower()`, not
  `casefold()`. Casefolding would map `ß` to `ss` and so stop matching forms
  that keep `ß`. A per-language `casefold` option would let German-corpus
  users opt in.
- **Comparable richness measures.** The single-work MATTR keeps stopwords,
  while the collection box plot and scatter use content-word TTR, so the two
  numbers are not on a common axis. Unifying them changes established
  outputs; the README documents the difference instead.
- **Stopword-list balance.** The scripts warn when one side of a
  cross-lingual comparison has a stopword list and the other does not, but
  not when the two lists differ greatly in size. There is no principled
  threshold; a user-set one could be added.

## Robustness

- **A `--debug` flag.** The CLI wrapper turns `ValueError` and `OSError`
  raised anywhere, including inside an analysis, into a one-line `ERROR:`
  without a traceback. That is right for config mistakes but hides internal
  bugs. A flag that re-raises would help when debugging.
- **Error stream consistency.** Config errors print to stderr; the
  `ERROR:` messages the analyses print on their own early exits go to stdout.
- **Validation scope.** Integer settings are checked for every section present
  in the config, not only the sections the running script reads, so a bad
  `topic_modeling` value also stops `xlt-term-frequency`. Accepted so that all
  problems are reported at once; could be narrowed to the required sections.
- **Very small corpora in BERTopic.** A reviewer suspected the default
  `max_df=0.95` vectorizer setting can fail on a handful of chunks. It was not
  reproduced because BERTopic is not installed in the test environment.
- **Pathological input.** Deeply nested JSON configs raise `RecursionError`
  rather than a clean message; a single unbroken run of text longer than the
  parser limit can be cut between a base character and its combining mark.

## Settings whose behaviour changed

- `adjective_window.window: 0` and `chunking.max_chars: 0` used to run
  (uselessly) and are now rejected by validation.

## Portability to other languages and scripts

The toolkit was built on medieval Italian and Latin. Two of its assumptions
do not carry to other scripts, measured on 2026-11-08 with the LaBSE
tokenizer on one sentence per language:

| Language | Characters per word-piece | Characters that fill LaBSE's 256-piece window | `tokenize()` on one sentence |
|---|---|---|---|
| English | 4.15 | ~1,050 | 21 words |
| German | 3.61 | ~920 | 20 words |
| Russian | 3.41 | ~865 | 14 words |
| Hindi | 2.97 | ~750 | 20 words |
| Arabic | 2.47 | ~625 | 13 words |
| Thai | 2.07 | ~525 | 2 clause-length "words" |
| Japanese | 1.25 | ~320 | 2 clause-length "words" |
| Chinese | 0.94 | ~240 | 2 clause-length "words" |

The embedding analyses (`xlt-embed-corpus`, `xlt-analysis-umap`) never
tokenize and work for every language the model covers. The word-based
analyses (term frequency, lexical richness, adjective windows, cross-corpus
frequency) and BERTopic's vectorizer need word boundaries the current
tokenizer cannot find in Chinese, Japanese or Thai. In priority order:

1. **Size chunks by the model's tokenizer, not by characters.** The
   `chunking.max_chars` default (800) fits Italian and Latin, is generous for
   English and is roughly three times too large for Chinese. Add
   `chunking.max_chars: "auto"` that derives the limit at embed time from the
   loaded model's tokenizer and `max_seq_length` on the actual corpus. The
   truncation warning in `embed.truncation_report()` already computes that
   number; make it the default behaviour.
2. **Unicode sentence terminators.** `embed._split_paragraph`,
   `corpus.split_for_parser` and `cross_corpus_dependency.sentences_containing`
   recognise only `.`, `!` and `?`. Add the CJK full stop and question/
   exclamation marks (。！？), the Devanagari danda (। ॥), the Arabic and
   Armenian question marks (؟ ՞) and the Ethiopic full stop (። ), or make the
   set configurable per language.
3. **Per-language word segmentation.** Add an optional
   `languages.<code>.segmenter` setting that routes `tokenize()` through
   spaCy's tokenizer when one is configured (spaCy ships Chinese, Japanese and
   Thai segmenters). Replace the fixed "drop single-character tokens" rule
   with a per-language `min_token_chars` (default 2), because most Chinese
   words are one character. Until this lands, the README should say plainly
   that the word-based analyses need a space-delimited script, and that
   BERTopic's default vectorizer has the same limit.
4. **Stopwords beyond Italian and Latin.** Only two example lists ship. Allow
   `stopwords_file: "spacy"` to use spaCy's built-in list for the language, so
   a new user is not blocked on writing one.
5. **Model flexibility.** Add an `embedding_max_seq_length` override (LaBSE's
   underlying BERT accepts 512 positions; the sentence-transformers packaging
   stops at 256), and document alternatives: multilingual-e5 needs an
   instruction prefix on each text; BGE-M3 has an 8,192-token window and
   allows far longer chunks. The truncation check already adapts to whichever
   model is loaded.
6. **A translation-pair validation command.** The README tells every user to
   validate the embedding model on known translation pairs from their own
   corpus before trusting distances, and offers no tool for it. An
   `xlt-validate-pairs` command taking aligned passages and reporting how
   often each passage's nearest neighbour is its translation is the one
   step that is identical for every language pair.
7. **A first example study, in a distant language family.** The repository
   ships no corpus at all (the study it was built for is private). A small
   public-domain example (English and Chinese, say) with a checked-in config
   and expected outputs would serve three purposes: a template for new users
   that is not medieval Italian, an integration test for items 1 to 4, and a
   smoke test of the model-dependent paths that the unit tests cannot cover.

Items 1, 2 and 4 are roughly a day's work together. Item 3 is the largest
and most changes what the toolkit can claim to support.
