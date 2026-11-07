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
