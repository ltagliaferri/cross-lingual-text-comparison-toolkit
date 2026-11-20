"""
Same-language stylometry: Burrows's Delta, Cosine Delta, General Imposters.

The embedding analyses measure meaning across languages; they cannot
measure authorial style, because a multilingual encoder is trained to make
translations look alike. Stylometry compares the relative frequencies of
the most frequent words (mostly function words), so it only works within
one language. Every corpus this script reads must share one `language`.

Groups of texts come from config['stylometry']['groups'] (and an optional
`groups_file` CSV). The optional `corpora` list restricts which corpora
the selectors and `rolling.corpus` may use; without it there is no
restriction. Each group's texts are concatenated, cut into samples
of `sample_words` tokens, and described by the relative frequencies of the
`mfw` most frequent words over all samples. Stopword lists are NOT applied:
function words are the signal here.

Outputs:
  results/stylometry_distances.csv     – group × group centroid distance
  results/stylometry_attribution.csv   – leave-one-out nearest group per sample
  results/stylometry_features.csv      – the MFW list with overall frequency
  results/stylometry_verification.csv  – General Imposters scores (if configured)
  results/stylometry_rolling.csv       – rolling Delta (if configured)
  visualizations/stylometry_distances.png
  visualizations/stylometry_rolling.png (if configured)
"""

import csv
import os
import re
import unicodedata
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .corpus import (_check_list, _dict_entries, config_entries, corpus_label, corpus_root,
                     ensure_output_dirs, keep_short_tokens, load_source_corpus,
                     run_cli, tokenize)
from .metrics import normalize_term
from .stylo_core import (burrows_delta, centroids, cosine_delta,  # noqa: F401 (re-exported)
                         distance_function, general_imposters, group_distances,
                         leave_one_out, make_samples, relative_frequencies, rolling_delta,
                         rolling_windows, select_features, zscore)

DISTANCES = ('cosine_delta', 'burrows_delta')
DEFAULTS = {
    'mfw': 200,
    'sample_words': 2000,
    'strip_patterns': [],
    'exclude_words': [],
    'distance': 'cosine_delta',
    'imposters': [],
    'verification': [],
    'iterations': 100,
    'feature_fraction': 0.5,
    'random_seed': 42,
}
ROLLING_DEFAULTS = {'window_words': 2000, 'step_words': 500}
# Below these the imposters pool is too small to mean much
MIN_IMPOSTER_GROUPS = 3
MIN_IMPOSTER_SAMPLES = 10


# ---------------------------------------------------------------------------
# Settings, selectors and the groups file
# ---------------------------------------------------------------------------

def settings(config):
    """The stylometry section with defaults filled in (ints coerced)."""
    section = config.get('stylometry') or {}
    s = dict(DEFAULTS)
    s.update({k: v for k, v in section.items() if not k.startswith('_')})
    for key in ('mfw', 'sample_words', 'iterations', 'random_seed'):
        s[key] = int(s[key])
    s['feature_fraction'] = float(s['feature_fraction'])
    rolling = s.get('rolling')
    if rolling:
        r = dict(ROLLING_DEFAULTS)
        r.update(config_entries(rolling))
        r['window_words'] = int(r['window_words'])
        r['step_words'] = int(r['step_words'])
        s['rolling'] = r
    else:
        s['rolling'] = None
    return s


def parse_selector(selector):
    """
    Split a selector into (corpus_id, group, item): "<corpus>",
    "<corpus>:<group>" or "<corpus>:<group>/<item>" (missing parts None).
    """
    corpus_id, sep, rest = selector.partition(':')
    if not sep:
        return corpus_id, None, None
    group, slash, item = rest.partition('/')
    return corpus_id, group, (item if slash else None)


def _covers(a, b):
    """True if selector `a` (parsed) includes every text `b` does."""
    if a[0] != b[0]:
        return False
    if a[1] is None:
        return True
    return a[1] == b[1] and (a[2] is None or a[2] == b[2])


def resolve_groups_file(config, path):
    """
    Resolve a `groups_file`: an absolute path as is; otherwise next to the
    config file, then under the working directory (the stopwords_file
    search, without the bundled lists).
    """
    if os.path.isabs(path):
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f'stylometry groups_file not found: {path}')
    dirs = []
    if config.get('_config_path'):
        dirs.append(os.path.dirname(config['_config_path']))
    dirs.append(os.getcwd())
    tried = [os.path.join(d, path) for d in dirs]
    for candidate in tried:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f'stylometry groups_file "{path}" not found. Tried: '
                            + ', '.join(tried))


def read_groups_file(path):
    """[(group, selector)] rows from a CSV with a group,selector header."""
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = [h.strip() for h in (reader.fieldnames or [])]
        if 'group' not in fields or 'selector' not in fields:
            raise ValueError(f'stylometry groups_file {path} needs a header row with '
                             f'columns "group" and "selector", got {fields}')
        rows = []
        for row in reader:
            row = {(k or '').strip(): (v or '').strip() for k, v in row.items()}
            if row['group'] or row['selector']:
                rows.append((row['group'], row['selector']))
    return rows


def stylometry_groups(config):
    """
    {group: [selector, ...]} from `groups` plus `groups_file` rows, which
    add to (never replace) a group of the same name. Config order first.
    """
    section = config.get('stylometry') or {}
    groups = OrderedDict((name, list(sels))
                         for name, sels in config_entries(section.get('groups')).items())
    if section.get('groups_file'):
        path = resolve_groups_file(config, section['groups_file'])
        for group, selector in read_groups_file(path):
            groups.setdefault(group, []).append(selector)
    return groups


# ---------------------------------------------------------------------------
# Validation (called from corpus.validate_config for require=('stylometry',))
# ---------------------------------------------------------------------------

def _positive_int(value):
    whole = (isinstance(value, int) and not isinstance(value, bool)) or \
        (isinstance(value, float) and value.is_integer())
    return whole and value >= 1


def check_config(config, check_paths, problems, warnings):
    """
    Append every problem with config['stylometry'] to `problems` (and
    overlaps to `warnings`); return the corpus ids it reads, so their
    paths get checked too. Plain integer settings are checked by the
    shared table in corpus.py.
    """
    if 'stylometry' not in config:
        problems.append('config has no "stylometry" section, which this script needs')
        return []
    section = config['stylometry']
    if not isinstance(section, dict):
        return []   # reported by validate_config's section-type check
    corpora = _dict_entries(config.get('corpora'))
    used = []

    # --- groups ---
    raw_groups = section.get('groups')
    groups = OrderedDict()
    if not isinstance(raw_groups, dict) or not config_entries(raw_groups):
        problems.append('stylometry["groups"] must be a non-empty object '
                        '{group_name: [selector, ...]}')
    else:
        for name, sels in config_entries(raw_groups).items():
            if _check_list(sels, f'stylometry["groups"]["{name}"]', problems):
                groups[name] = list(sels)
    if section.get('groups_file') is not None:
        if not isinstance(section['groups_file'], str):
            problems.append('stylometry["groups_file"] must be a string path')
        else:
            try:
                try:
                    path = resolve_groups_file(config, section['groups_file'])
                except FileNotFoundError:
                    if check_paths:
                        raise
                    path = None   # like a corpus file: only missing when paths are checked
                for group, selector in (read_groups_file(path) if path else []):
                    if not group or not selector:
                        problems.append(f'stylometry groups_file {path}: a row has an '
                                        f'empty group or selector ({group!r}, {selector!r})')
                        continue
                    groups.setdefault(group, []).append(selector)
            except (OSError, ValueError, csv.Error) as e:
                problems.append(str(e))

    restrict = section.get('corpora')
    if restrict is not None and _check_list(restrict, 'stylometry["corpora"]', problems):
        for corpus_id in restrict:
            if corpus_id not in corpora:
                problems.append(f'stylometry["corpora"] refers to unknown corpus "{corpus_id}"')
            elif corpus_id not in used:
                used.append(corpus_id)
    else:
        restrict = None

    parsed = {}   # group -> [(selector, parsed)]
    for name, sels in groups.items():
        parsed[name] = []
        for selector in sels:
            corpus_id, group, item = parse_selector(selector)
            where = f'stylometry group "{name}": selector "{selector}"'
            entry = corpora.get(corpus_id)
            if entry is None:
                problems.append(f'{where} names no configured corpus')
                continue
            if restrict is not None and corpus_id not in restrict:
                problems.append(f'{where} uses corpus "{corpus_id}", which is not in '
                                'stylometry["corpora"]')
                continue
            if group is not None:
                configured = entry.get('groups') if isinstance(entry.get('groups'), list) else []
                if entry.get('type') != 'collection' or group not in configured:
                    problems.append(f'{where}: "{group}" is not a configured group of '
                                    f'collection "{corpus_id}"')
                    continue
                if item is not None and not item:
                    problems.append(f'{where} has an empty item name')
                    continue
                if item is not None and check_paths and isinstance(entry.get('path'), str):
                    item_path = os.path.join(corpus_root(config), entry['path'], group,
                                             item + '.txt')
                    if not os.path.isfile(item_path):
                        problems.append(f'{where}: no such item ({item_path})')
                        continue
            parsed[name].append((selector, (corpus_id, group, item)))
            if corpus_id not in used:
                used.append(corpus_id)

    # overlaps: a warning, since overlapping groups may be intended
    for name, sels in parsed.items():
        for i, (sa, a) in enumerate(sels):
            for sb, b in sels[i + 1:]:
                if _covers(a, b) or _covers(b, a):
                    warnings.append(f'stylometry group "{name}" includes the same text '
                                    f'twice ("{sa}" and "{sb}")')
    # one warning per pair of groups; what they share is the narrower
    # selector of each overlapping pair (counted per selector, not on disk)
    names = list(parsed)
    for i, ga in enumerate(names):
        for gb in names[i + 1:]:
            shared = []
            for sa, a in parsed[ga]:
                for sb, b in parsed[gb]:
                    narrower = sb if _covers(a, b) else sa if _covers(b, a) else None
                    if narrower is not None and narrower not in shared:
                        shared.append(narrower)
            if shared:
                examples = ', '.join(f'"{x}"' for x in shared[:3])
                more = f' ... and {len(shared) - 3} more' if len(shared) > 3 else ''
                warnings.append(f'stylometry groups "{ga}" and "{gb}" share {len(shared)} '
                                f'text selection(s): {examples}{more}')

    # --- scalar settings ---
    if 'feature_fraction' in section:
        ff = section['feature_fraction']
        if isinstance(ff, bool) or not isinstance(ff, (int, float)) or not 0 < ff <= 1:
            problems.append(f'stylometry["feature_fraction"] must be a number in (0, 1], '
                            f'got {ff!r}')
    if section.get('distance', DEFAULTS['distance']) not in DISTANCES:
        problems.append(f'stylometry["distance"] must be one of {", ".join(DISTANCES)}, '
                        f'got {section["distance"]!r}')
    patterns = section.get('strip_patterns', [])
    if not isinstance(patterns, list):
        problems.append('stylometry["strip_patterns"] must be a list of regular expressions')
    else:
        for pattern in patterns:
            if not isinstance(pattern, str):
                problems.append(f'stylometry["strip_patterns"]: {pattern!r} is not a string')
                continue
            try:
                re.compile(pattern, re.MULTILINE)
            except re.error as e:
                problems.append(f'stylometry["strip_patterns"]: "{pattern}" is not a valid '
                                f'regular expression ({e})')
    if 'exclude_words' in section:
        _check_list(section['exclude_words'], 'stylometry["exclude_words"]', problems)

    # --- imposters and verification ---
    imposters = section.get('imposters', [])
    if _check_list(imposters, 'stylometry["imposters"]', problems):
        for name in imposters:
            if name not in groups:
                problems.append(f'stylometry["imposters"] names undefined group "{name}"')
    else:
        imposters = []
    verification = section.get('verification', [])
    if not isinstance(verification, list):
        problems.append('stylometry["verification"] must be a list of '
                        '{"questioned": group, "candidate": group} objects')
        verification = []
    if verification and not section.get('imposters'):
        problems.append('stylometry["verification"] is set but stylometry["imposters"] is '
                        'empty: General Imposters needs a pool of other groups')
    for i, pair in enumerate(verification):
        where = f'stylometry["verification"][{i}]'
        if not isinstance(pair, dict):
            problems.append(f'{where} must be an object with "questioned" and "candidate"')
            continue
        for key in ('questioned', 'candidate'):
            value = pair.get(key)
            if value is None:
                problems.append(f'{where} is missing "{key}"')
            elif not isinstance(value, str) or value not in groups:
                problems.append(f'{where}["{key}"] names undefined group {value!r}')
            elif value in imposters:
                problems.append(f'{where}["{key}"] "{value}" is also in stylometry["imposters"]')
        if pair.get('questioned') is not None and pair.get('questioned') == pair.get('candidate'):
            problems.append(f'{where}: questioned and candidate are the same group '
                            f'"{pair["questioned"]}"')

    # --- rolling ---
    rolling = section.get('rolling')
    rolling_corpus = None
    if rolling is not None:
        if not isinstance(rolling, dict):
            problems.append('stylometry["rolling"] must be an object')
        else:
            rolling_corpus = rolling.get('corpus')
            if not isinstance(rolling_corpus, str) or rolling_corpus not in corpora:
                problems.append(f'stylometry["rolling"]["corpus"] must name a configured '
                                f'corpus, got {rolling_corpus!r}')
                rolling_corpus = None
            elif restrict is not None and rolling_corpus not in restrict:
                problems.append(f'stylometry["rolling"]["corpus"] "{rolling_corpus}" is not in '
                                'stylometry["corpora"]')
            refs = rolling.get('references')
            if _check_list(refs, 'stylometry["rolling"]["references"]', problems):
                if not refs:
                    problems.append('stylometry["rolling"]["references"] is empty')
                for name in refs:
                    if name not in groups:
                        problems.append(f'stylometry["rolling"]["references"] names '
                                        f'undefined group "{name}"')
            for key in ('window_words', 'step_words'):
                if key in rolling and not _positive_int(rolling[key]):
                    problems.append(f'stylometry["rolling"]["{key}"] must be a positive '
                                    f'integer, got {rolling[key]!r}')

    # --- one language ---
    langs = OrderedDict()
    for corpus_id in used + ([rolling_corpus] if rolling_corpus else []):
        lang = corpora[corpus_id].get('language')
        if isinstance(lang, str):
            langs.setdefault(lang, []).append(corpus_id)
    if len(langs) > 1:
        detail = '; '.join(f'{lang}: {", ".join(ids)}' for lang, ids in langs.items())
        problems.append(
            'stylometry compares word frequencies, so it only works within one language, '
            f'but its corpora use several ({detail}). Compare across languages with the '
            'embedding analyses (xlt-embed-corpus, xlt-analysis-umap) instead.')

    if rolling_corpus and rolling_corpus not in used:
        used.append(rolling_corpus)
    return used


# ---------------------------------------------------------------------------
# Text and selectors (the numerical core is in stylo_core.py)
# ---------------------------------------------------------------------------

def compile_patterns(patterns):
    return [re.compile(unicodedata.normalize('NFC', p), re.MULTILINE) for p in patterns]


def strip_text(text, patterns):
    """NFC-normalize, then replace every match of each pattern with a space."""
    text = unicodedata.normalize('NFC', text)
    for pattern in patterns:
        text = pattern.sub(' ', text)
    return text


def selector_texts(config, selector, cache=None):
    """
    The texts a selector names, in load order: a single file's text, every
    item of a collection, one group's items, or one item.
    """
    cache = {} if cache is None else cache
    corpus_id, group, item = parse_selector(selector)
    if corpus_id not in cache:
        cache[corpus_id] = load_source_corpus(config, corpus_id)
    loaded = cache[corpus_id]
    if isinstance(loaded, str):
        return [loaded]
    return [it['text'] for it in loaded
            if (group is None or it['group'] == group)
            and (item is None or it['item_num'] == item)]


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

def _write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'  Saved {path}')


def _plot_distances(matrix, names, distance_name, path):
    size = max(5, 0.7 * len(names) + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(matrix, cmap='viridis_r')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                    color='white', fontsize=8)
    fig.colorbar(im, ax=ax, label=distance_name.replace('_', ' '))
    ax.set_title('Stylometric distance between group centroids')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    s = settings(config)
    groups = stylometry_groups(config)
    names = list(groups)
    distance = distance_function(s['distance'])
    patterns = compile_patterns(s['strip_patterns'])

    first_corpus = parse_selector(groups[names[0]][0])[0]
    language = config['corpora'][first_corpus]['language']
    short = keep_short_tokens(config, language)

    def text_tokens(texts):
        return [t for text in texts
                for t in tokenize(strip_text(text, patterns), remove_stopwords=False,
                                  keep_short=short)]

    # --- samples per group ---
    cache = {}
    samples, labels, sample_ids = [], [], []
    for name in names:
        texts = [t for sel in groups[name] for t in selector_texts(config, sel, cache)]
        tokens = text_tokens(texts)
        group_samples, too_short = make_samples(tokens, s['sample_words'])
        if not group_samples:
            print(f'ERROR: stylometry group "{name}" has no tokens — check its selectors.')
            return 1
        if too_short:
            print(f'WARNING: stylometry group "{name}" has {len(tokens)} tokens, fewer than '
                  f'one sample of {s["sample_words"]}; it is kept as a single short sample. '
                  'Delta is unreliable below a few thousand words per sample.')
        for k, sample in enumerate(group_samples, 1):
            samples.append(sample)
            labels.append(name)
            sample_ids.append(f'{name}#{k}')
        print(f'  {name}: {len(tokens)} tokens, {len(group_samples)} sample(s)')

    # --- features and z-scores ---
    words, counts = select_features(samples, s['mfw'],
                                   [normalize_term(w) for w in s['exclude_words']])
    freqs = relative_frequencies(samples, words)
    z, mean, sd, keep = zscore(freqs)
    dropped = int((~keep).sum())
    if dropped:
        print(f'  Dropped {dropped} feature(s) with zero variance across samples.')
    if not keep.any():
        print('ERROR: no stylometric feature varies across samples; add more text or groups.')
        return 1
    print(f'  {int(keep.sum())} features, {len(samples)} samples, '
          f'{s["distance"].replace("_", " ")}')

    total_tokens = sum(len(x) for x in samples)
    _write_csv(os.path.join(results_dir, 'stylometry_features.csv'),
               ['rank', 'word', 'count', 'relative_frequency', 'zero_sd_dropped'],
               [(r, w, c, round(c / total_tokens, 6), int(not kept))
                for r, (w, c, kept) in enumerate(zip(words, counts, keep), 1)])

    # --- group distances ---
    matrix = group_distances(z, labels, names, distance)
    _write_csv(os.path.join(results_dir, 'stylometry_distances.csv'),
               ['group'] + names,
               [[a] + [round(float(v), 6) for v in row] for a, row in zip(names, matrix)])
    _plot_distances(matrix, names, s['distance'],
                    os.path.join(viz_dir, 'stylometry_distances.png'))

    # --- leave-one-out attribution ---
    rows, confusion, accuracy = leave_one_out(z, labels, names, distance)
    _write_csv(os.path.join(results_dir, 'stylometry_attribution.csv'),
               ['sample', 'group', 'nearest_group', 'correct']
               + [f'distance_{g}' for g in names],
               [[sid, true, nearest, int(true == nearest)]
                + ['' if np.isnan(d[g]) else round(d[g], 6) for g in names]
                for sid, (true, nearest, d) in zip(sample_ids, rows)])
    width = max(len(n) for n in names)
    print('\nLeave-one-out attribution (rows: true group, columns: nearest):')
    print(' ' * (width + 2) + '  '.join(f'{n[:8]:>8}' for n in names))
    for name, row in zip(names, confusion):
        print(f'  {name:<{width}}' + '  '.join(f'{v:>8d}' for v in row))
    print(f'Accuracy: {accuracy:.3f} ({int(np.trace(confusion))}/{int(confusion.sum())})')

    # --- General Imposters ---
    labels_arr = np.asarray(labels)
    if s['verification']:
        pool = np.isin(labels_arr, s['imposters'])
        n_pool = int(pool.sum())
        if len(s['imposters']) < MIN_IMPOSTER_GROUPS or n_pool < MIN_IMPOSTER_SAMPLES:
            print(f'WARNING: the imposter pool has {len(s["imposters"])} group(s) and '
                  f'{n_pool} sample(s); with fewer than {MIN_IMPOSTER_GROUPS} groups or '
                  f'{MIN_IMPOSTER_SAMPLES} samples the verification score is weak.')
        out = []
        for pair in s['verification']:
            q_name, c_name = pair['questioned'], pair['candidate']
            q, c = z[labels_arr == q_name], z[labels_arr == c_name]
            args = (z[pool], distance, s['iterations'], s['feature_fraction'],
                    s['random_seed'])
            score = general_imposters(q, c, *args)
            swapped = general_imposters(c, q, *args)
            out.append([q_name, c_name, round(score, 4), round(swapped, 4),
                        round((score + swapped) / 2, 4), len(q), len(c),
                        min(len(c), n_pool), s['iterations']])
        _write_csv(os.path.join(results_dir, 'stylometry_verification.csv'),
                   ['questioned', 'candidate', 'score', 'swapped_score', 'mean_score',
                    'n_questioned_samples', 'n_candidate_samples', 'n_imposter_samples',
                    'iterations'], out)
        print('\nGeneral Imposters (a proportion of wins, not a p-value):')
        for row in out:
            print(f'  {row[0]} -> {row[1]}: score {row[2]:.3f}, swapped {row[3]:.3f}, '
                  f'mean {row[4]:.3f}')

    # --- rolling Delta ---
    rolling = s['rolling']
    if rolling:
        corpus_id, refs = rolling['corpus'], rolling['references']
        for ref in refs:
            if any(parse_selector(sel)[0] == corpus_id for sel in groups[ref]):
                print(f'WARNING: rolling corpus "{corpus_id}" is also part of reference '
                      f'group "{ref}", which inflates its closeness to that group.')
        loaded = load_source_corpus(config, corpus_id)
        texts = [loaded] if isinstance(loaded, str) else [it['text'] for it in loaded]
        tokens = text_tokens(texts)
        windows = rolling_windows(tokens, rolling['window_words'], rolling['step_words'])
        if len(tokens) < rolling['window_words']:
            print(f'WARNING: rolling corpus "{corpus_id}" has {len(tokens)} tokens, fewer '
                  f'than one window of {rolling["window_words"]}; scoring it as one window.')
        cents = centroids(z, labels, names)
        scored = rolling_delta(windows, words, keep, mean, sd,
                               OrderedDict((g, cents[g]) for g in refs), distance)
        _write_csv(os.path.join(results_dir, 'stylometry_rolling.csv'),
                   ['window', 'start_token'] + [f'distance_{g}' for g in refs],
                   [[i, start] + [round(d[g], 6) for g in refs]
                    for i, (start, d) in enumerate(scored)])
        fig, ax = plt.subplots(figsize=(12, 5))
        starts = [start for start, _ in scored]
        for g in refs:
            ax.plot(starts, [d[g] for _, d in scored], marker='.', label=g)
        ax.set_xlabel(f'Window start (token; window = {rolling["window_words"]})')
        ax.set_ylabel(s['distance'].replace('_', ' '))
        ax.set_title(f'Rolling Delta across {corpus_label(config, corpus_id)} '
                     '(lower = closer)')
        ax.legend()
        fig.tight_layout()
        path = os.path.join(viz_dir, 'stylometry_rolling.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Saved {path}')


REQUIRES = ('stylometry',)


def main():
    run_cli(analyze, require=REQUIRES)


if __name__ == '__main__':
    main()
