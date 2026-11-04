"""
Corpus-independent measures and helpers, kept free of plotting and NLP
imports so they can be tested (and reused) without matplotlib, spaCy or a
downloaded model.

The analysis scripts re-export what they use from here, so existing
imports like `from cross_lingual_toolkit.lexical_richness import ttr`
keep working.
"""

import math
import unicodedata
from collections import Counter


def normalize_term(term):
    """
    Normalize a configured term, form, lemma or stopword the same way
    tokenize() normalizes text: lowercase, then Unicode NFC. Applying this
    to both sides of every comparison means "Fear" in a config matches
    "fear" in a text, and a decomposed (NFD) "é" matches a composed one.
    Lowercasing comes first so that case mappings which produce combining
    marks (Turkish "İ" -> "i" + U+0307) are normalized too. Surrounding
    whitespace is stripped, so " fear" in a config is "fear". Soft hyphens
    (U+00AD, invisible line-break hints common in PDF- and HTML-extracted
    text) are removed, so "co\u00adoperate" matches "cooperate".
    """
    return unicodedata.normalize('NFC', term.lower()).replace('\u00ad', '').strip()


# ---------------------------------------------------------------------------
# Lexical richness
# ---------------------------------------------------------------------------

def ttr(tokens):
    """Type-token ratio: unique tokens / total tokens. 0.0 for empty input."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mattr(tokens, window=500):
    """
    Moving-average TTR: the TTR of each successive window of `window`
    tokens.

    Always returns a list, one value per window start — a text shorter
    than the window yields a single whole-text TTR rather than a bare
    float, so callers can iterate the result unconditionally.
    """
    if window <= 0:
        raise ValueError(f'MATTR window must be positive, got {window}')
    if len(tokens) < window:
        return [ttr(tokens)]
    return [ttr(tokens[i:i + window]) for i in range(0, len(tokens) - window + 1)]


def mattr_score(tokens, window=500):
    """
    MATTR proper (Covington & McFall 2010): the mean of the per-window
    TTRs that mattr() returns. A text shorter than the window gets its
    whole-text TTR.
    """
    values = mattr(tokens, window)
    return sum(values) / len(values)


# ---------------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------------

def normalize(counts, total, per=10_000):
    """
    Normalize raw counts to a rate per `per` tokens. An empty corpus
    (total == 0) yields 0.0 rather than a ZeroDivisionError.
    """
    if not total:
        return {k: 0.0 for k in counts}
    return {k: round(v / total * per, 2) for k, v in counts.items()}


def count_cluster_terms(tokens, clusters):
    """
    Count occurrences of each cluster's `terms` in `tokens`.
    Returns {cluster_key: count}. A token counted for one cluster is
    still available to others (clusters here are not mutually exclusive).

    Terms are matched after normalize_term(), and each distinct term is
    counted once even if a cluster lists it twice.
    """
    freq = Counter(tokens)
    return {key: sum(freq.get(t, 0) for t in {normalize_term(t) for t in data['terms']})
            for key, data in clusters.items()}


def count_cluster_forms(tokens, clusters, lang_key):
    """
    Count occurrences of each cluster's `lang_key` surface forms in
    `tokens`. Unlike count_cluster_terms, a token is attributed to the
    first matching cluster only, so cluster counts sum to at most
    len(tokens). Forms are matched after normalize_term().
    """
    forms = {key: {normalize_term(f) for f in data[lang_key]}
             for key, data in clusters.items()}
    counts = {key: 0 for key in clusters}
    for tok in tokens:
        for key, form_set in forms.items():
            if tok in form_set:
                counts[key] += 1
                break
    return counts


def dunning_g2(a, b, total_a, total_b):
    """
    Return signed G² for a term with count `a` in corpus A (size total_a)
    and count `b` in corpus B (size total_b).
    Positive = over-represented in A; negative = over-represented in B.

    This is the two-cell log-likelihood used in corpus linguistics
    (Rayson & Garside 2000): it sums O·ln(O/E) over the term's two
    observed counts only, not over all four cells of Dunning's (1993)
    full 2x2 table, so it is slightly smaller than the full G² (100 vs. 10
    in 1,000 tokens each gives 85.47 here, 89.76 for the full table).
    Compare it against chi-square with 1 df: 3.84 (p < .05), 6.63
    (p < .01), 10.83 (p < .001) — bearing in mind that tokens are not
    independent and that testing many clusters inflates false positives.

    Both corpora must be non-empty; an empty corpus makes the comparison
    undefined and returns 0.0.
    """
    if not total_a or not total_b:
        return 0.0
    if a == 0 and b == 0:
        return 0.0
    expected_a = total_a * (a + b) / (total_a + total_b)
    expected_b = total_b * (a + b) / (total_a + total_b)

    def term(observed, expected):
        if observed == 0:
            return 0.0
        return observed * math.log(observed / expected)

    g2 = 2 * (term(a, expected_a) + term(b, expected_b))
    if a / total_a < b / total_b:
        g2 = -g2
    return g2


def top_shared_terms(counter_a, counter_b, n):
    """
    The n terms to show when comparing two counters: the most frequent by
    combined count across both, ties broken alphabetically.

    Selection has to run over every term in both counters. Slicing
    Counter.keys() instead returns whatever was *inserted* first, which
    silently drops the most frequent terms from the comparison.
    """
    candidates = set(counter_a) | set(counter_b)
    return sorted(
        candidates,
        key=lambda t: (-(counter_a.get(t, 0) + counter_b.get(t, 0)), t),
    )[:n]


# ---------------------------------------------------------------------------
# Dependency labels
# ---------------------------------------------------------------------------

def dep_variants(role):
    """
    Return every spelling of a dependency label that the supported
    parsers might emit.

    spaCy's English models use the ClearNLP convention ("nsubjpass",
    "dobj"); its other models, and Stanza, use Universal Dependencies
    ("nsubj:pass", "obj"). Matching against this set means a config that
    says "nsubj" or "obj" works for either, instead of silently finding
    nothing.
    """
    aliases = {'obj': ('obj', 'dobj'), 'dobj': ('obj', 'dobj')}
    variants = set()
    for base in aliases.get(role, (role,)):
        variants.update({base, base + 'pass', base + ':pass'})
    return variants


# ---------------------------------------------------------------------------
# Context windows
# ---------------------------------------------------------------------------

def collect_windows(tokens, anchor_forms, window=10):
    """
    Return list of context-window token lists around each occurrence of
    any form in `anchor_forms`. The anchor itself is excluded.
    """
    windows = []
    for i, tok in enumerate(tokens):
        if tok in anchor_forms:
            left  = max(0, i - window)
            right = min(len(tokens), i + window + 1)
            ctx = tokens[left:i] + tokens[i + 1:right]
            windows.append(ctx)
    return windows


def adjectives_from_window_heuristic(window_tokens, stopwords, suffixes):
    """Fallback: count tokens matching configured adjective-suffix heuristics."""
    counter = Counter()
    if not suffixes:
        return counter
    for tok in window_tokens:
        if tok in stopwords:
            continue
        if len(tok) > 4 and tok.endswith(tuple(suffixes)):
            counter[tok] += 1
    return counter
