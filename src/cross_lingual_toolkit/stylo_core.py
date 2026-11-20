"""
The numerical core of stylometry: samples, most-frequent-word features,
z-scores, Burrows's and Cosine Delta, leave-one-out attribution, General
Imposters and rolling Delta.

Imports only the standard library and numpy (no config, no corpus loading,
no plotting), so it can be used and tested on in-memory data on its own.
Tokens are whatever the caller passes; stylometry.py does the tokenizing,
config handling, I/O and plots, and re-exports every name here.
"""

from collections import Counter

import numpy as np


def make_samples(tokens, sample_words):
    """
    Cut tokens into consecutive samples of `sample_words`. A trailing
    remainder of at least half a sample is kept as its own (shorter) sample;
    a shorter one is dropped. Text shorter than one full sample is kept
    whole as a single sample. Returns (samples, short) where `short` says
    the text did not reach one full sample.
    """
    if len(tokens) < sample_words:
        return ([list(tokens)] if tokens else []), True
    samples = [tokens[i:i + sample_words]
               for i in range(0, len(tokens) - sample_words + 1, sample_words)]
    remainder = tokens[len(samples) * sample_words:]
    if remainder and len(remainder) * 2 >= sample_words:
        samples.append(remainder)
    return samples, False


def select_features(samples, mfw, exclude_words=()):
    """
    The `mfw` most frequent tokens over all samples, minus `exclude_words`,
    ties broken alphabetically. Returns (words, counts). `exclude_words`
    are compared as given, so normalize them the way the tokens were.
    """
    exclude = set(exclude_words)
    total = Counter()
    for sample in samples:
        total.update(sample)
    ranked = sorted((w for w in total if w not in exclude), key=lambda w: (-total[w], w))
    words = ranked[:mfw]
    return words, [total[w] for w in words]


def relative_frequencies(samples, words):
    """Samples × features matrix of relative frequencies (count / sample length)."""
    index = {w: j for j, w in enumerate(words)}
    matrix = np.zeros((len(samples), len(words)))
    for i, sample in enumerate(samples):
        for token in sample:
            j = index.get(token)
            if j is not None:
                matrix[i, j] += 1
        if sample:
            matrix[i] /= len(sample)
    return matrix


def zscore(matrix):
    """
    z-score each column over the rows (population SD). Columns with zero SD
    are dropped. Returns (z, mean, sd, keep_mask) with mean/sd of the kept
    columns, so other vectors can be scored on the same scale.
    """
    mean = matrix.mean(axis=0)
    sd = matrix.std(axis=0)
    keep = sd > 0
    return (matrix[:, keep] - mean[keep]) / sd[keep], mean[keep], sd[keep], keep


def burrows_delta(a, b):
    """Mean absolute difference of two z-score vectors."""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def cosine_delta(a, b):
    """Cosine distance (1 − cosine similarity) of two z-score vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / norm)


def distance_function(name):
    return {'burrows_delta': burrows_delta, 'cosine_delta': cosine_delta}[name]


# ---------------------------------------------------------------------------
# Delta between groups
# ---------------------------------------------------------------------------

def centroids(z, labels, names):
    labels = np.asarray(labels)
    return {g: z[labels == g].mean(axis=0) for g in names}


def group_distances(z, labels, names, distance):
    cents = centroids(z, labels, names)
    return np.array([[distance(cents[a], cents[b]) for b in names] for a in names])


def leave_one_out(z, labels, names, distance):
    """
    For each sample, its distance to every group centroid, where its own
    group's centroid leaves it out (NaN if it is its group's only sample).
    Returns (rows, confusion, accuracy): rows are (true, nearest, {group:
    distance}); confusion[i][j] counts true names[i] attributed to names[j].

    A known simplification, and common practice: `z` is z-scored once over
    all samples, the held-out one included; only the centroid leaves it out.
    """
    labels = np.asarray(labels)
    cents = centroids(z, labels, names)
    rows = []
    confusion = np.zeros((len(names), len(names)), dtype=int)
    for i, true in enumerate(labels):
        dists = {}
        for g in names:
            members = labels == g
            if g == true:
                members = members.copy()
                members[i] = False
                if not members.any():
                    dists[g] = float('nan')
                    continue
                cent = z[members].mean(axis=0)
            else:
                cent = cents[g]
            dists[g] = distance(z[i], cent)
        finite = {g: d for g, d in dists.items() if not np.isnan(d)}
        nearest = min(finite, key=lambda g: (finite[g], names.index(g)))
        confusion[names.index(true), names.index(nearest)] += 1
        rows.append((true, nearest, dists))
    accuracy = float(np.trace(confusion) / confusion.sum()) if rows else float('nan')
    return rows, confusion, accuracy


# ---------------------------------------------------------------------------
# General Imposters
# ---------------------------------------------------------------------------

def general_imposters(questioned, candidate, imposters, distance, iterations,
                      feature_fraction, seed):
    """
    Koppel and Winter (2014) / Kestemont et al. (2016). Each iteration
    samples `feature_fraction` of the features and as many imposter samples
    as the candidate has (or all of them, if fewer); a questioned sample
    scores a success when its nearest candidate sample is closer than every
    sampled imposter. Returns successes / (iterations × questioned samples).
    """
    rng = np.random.default_rng(seed)
    n_features = questioned.shape[1]
    k = max(1, int(round(feature_fraction * n_features)))
    n_imp = min(len(candidate), len(imposters))
    successes = 0
    for _ in range(iterations):
        feats = rng.choice(n_features, size=k, replace=False)
        imps = imposters[rng.choice(len(imposters), size=n_imp, replace=False)][:, feats]
        cands = candidate[:, feats]
        for q in questioned[:, feats]:
            d_cand = min(distance(q, c) for c in cands)
            d_imp = min(distance(q, m) for m in imps)
            if d_cand < d_imp:
                successes += 1
    return successes / (iterations * len(questioned))


# ---------------------------------------------------------------------------
# Rolling Delta
# ---------------------------------------------------------------------------

def rolling_windows(tokens, window_words, step_words):
    """[(start, window_tokens)]; a text shorter than one window is one window."""
    if len(tokens) <= window_words:
        return [(0, list(tokens))] if tokens else []
    return [(s, tokens[s:s + window_words])
            for s in range(0, len(tokens) - window_words + 1, step_words)]


def rolling_delta(windows, words, keep, mean, sd, cents, distance):
    """Per window, its distance to each reference centroid (same z-scale)."""
    out = []
    for start, toks in windows:
        freqs = relative_frequencies([toks], words)[0][keep]
        z = (freqs - mean) / sd
        out.append((start, {g: distance(z, c) for g, c in cents.items()}))
    return out
