"""
Parallel-passage search: where does one corpus translate, quote or
paraphrase another, in another language?

Every sentence of the `source` corpus is compared (multilingual sentence
embeddings, cosine similarity) with every sentence of the `targets`
corpora. Each source sentence keeps its best and runner-up target
sentence; consecutive matches that land close together in the same target
text are merged into passages; and each passage can be aligned sentence by
sentence (groups of 1-3 sentences per side) by monotonic dynamic
programming, for a pair-by-pair reading of a translation.

The raw texts are read with the corpus loaders, so `xlt-embed-corpus` need
not have run. Sentence embeddings are cached in the embeddings directory as
parallel_<corpus_id>_sentences.npy (re-embedded when text or model change).

Outputs:
  results/parallel_sentences.csv   – best match for every source sentence
  results/parallel_passages.csv    – merged runs of matches
  results/parallel_alignment.csv   – aligned units per passage (if align)
  results/parallel_summary.txt     – counts, quantiles, passages, caveats
  visualizations/parallel_passages.png
"""

import csv
import os
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .corpus import (_check_list, _dict_entries, config_entries, ensure_output_dirs,
                     load_source_corpus, run_cli)
from .embed import _hard_wrap, embed_chunks, get_model

DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/LaBSE'

# After . ? ! ; : followed by whitespace; after the CJK full stop and
# exclamation/question marks with or without whitespace.
DEFAULT_SENTENCE_SPLIT = r'(?<=[.?!;:])\s+|(?<=[。！？])\s*'

DEFAULTS = {
    'min_sentence_chars': 40,
    'max_sentence_chars': 600,
    'sentence_split': DEFAULT_SENTENCE_SPLIT,
    'location_patterns': {},
    'threshold': 0.70,
    'merge_gap': 5,
    'target_window': 10,
    'min_passage_sentences': 2,
    'align': True,
    'align_context': 2,
    'skip_penalty': 0.35,
}
KEYS = ('source', 'targets') + tuple(DEFAULTS)

# Alignment steps: (source sentences, target sentences) consumed.
ALIGN_STEPS = ((1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2), (2, 3), (3, 2))

SUMMARY_LEVELS = (0.60, 0.65, 0.70, 0.75, 0.80)
QUANTILES = (0.5, 0.9, 0.99, 0.999)

_PARA_BREAK = re.compile(r'\n[ \t]*\n\s*')
_WHITESPACE = re.compile(r'\s+')


# ---------------------------------------------------------------------------
# Settings and validation
# ---------------------------------------------------------------------------

def settings(config):
    """The parallel_passages section with defaults filled in (numbers coerced)."""
    s = dict(DEFAULTS)
    s.update(config_entries(config.get('parallel_passages') or {}))
    for key in ('min_sentence_chars', 'max_sentence_chars', 'merge_gap',
                'target_window', 'min_passage_sentences', 'align_context'):
        s[key] = int(s[key])
    s['threshold'] = float(s['threshold'])
    s['skip_penalty'] = float(s['skip_penalty'])
    s['location_patterns'] = config_entries(s['location_patterns'] or {})
    return s


def _whole(value):
    return (isinstance(value, int) and not isinstance(value, bool)) or \
        (isinstance(value, float) and value.is_integer())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compiles(pattern, where, problems):
    if not isinstance(pattern, str) or not pattern:
        problems.append(f'{where} must be a non-empty regular expression string, '
                        f'got {pattern!r}')
        return False
    try:
        re.compile(pattern)
    except re.error as e:
        problems.append(f'{where} is not a valid regular expression: {e}')
        return False
    return True


def check_config(config, check_paths, problems, warnings):
    """
    Append every problem with config['parallel_passages'] to `problems`
    (unknown keys to `warnings`); return the corpus ids it reads, so their
    paths get checked too.
    """
    if 'parallel_passages' not in config:
        problems.append('config has no "parallel_passages" section, which this script needs')
        return []
    section = config['parallel_passages']
    if not isinstance(section, dict):
        return []   # reported by validate_config's section-type check
    corpora = _dict_entries(config.get('corpora'))
    entries = config_entries(section)
    used = []
    where = 'parallel_passages'

    for key in entries:
        if key not in KEYS:
            warnings.append(f'{where}["{key}"] is not a known setting and is ignored')

    source = entries.get('source')
    if source is None:
        problems.append(f'{where}["source"] is required (the corpus id searched for)')
    elif not isinstance(source, str) or source not in corpora:
        problems.append(f'{where}["source"] refers to unknown corpus {source!r}')
    else:
        used.append(source)

    targets = entries.get('targets')
    if targets is None:
        problems.append(f'{where}["targets"] is required (a list of corpus ids to search in)')
    elif _check_list(targets, f'{where}["targets"]', problems):
        if not targets:
            problems.append(f'{where}["targets"] must not be empty')
        seen = set()
        for corpus_id in targets:
            if corpus_id not in corpora:
                problems.append(f'{where}["targets"] refers to unknown corpus "{corpus_id}"')
            elif corpus_id == source:
                problems.append(f'{where}["targets"] includes the source corpus "{corpus_id}"')
            elif corpus_id in seen:
                problems.append(f'{where}["targets"] lists "{corpus_id}" twice')
            else:
                used.append(corpus_id)
            seen.add(corpus_id)

    for key, minimum in (('min_sentence_chars', 0), ('max_sentence_chars', 1),
                         ('merge_gap', 0), ('target_window', 0),
                         ('min_passage_sentences', 1), ('align_context', 0)):
        if key in entries:
            value = entries[key]
            if not (_whole(value) and value >= minimum):
                kind = 'a positive integer' if minimum else 'a non-negative integer'
                problems.append(f'{where}["{key}"] must be {kind}, got {value!r}')

    low = entries.get('min_sentence_chars', DEFAULTS['min_sentence_chars'])
    high = entries.get('max_sentence_chars', DEFAULTS['max_sentence_chars'])
    if _whole(low) and _whole(high) and 1 <= high < low:
        warnings.append(f'{where}["max_sentence_chars"] ({high}) is below '
                        f'{where}["min_sentence_chars"] ({low}), so every sentence long enough '
                        'to keep is wrapped')

    if 'threshold' in entries:
        value = entries['threshold']
        if not (_number(value) and 0 <= value <= 1):
            problems.append(f'{where}["threshold"] must be a number from 0 to 1, got {value!r}')
    if 'skip_penalty' in entries:
        value = entries['skip_penalty']
        if not (_number(value) and value >= 0):
            problems.append(f'{where}["skip_penalty"] must be a non-negative number, '
                            f'got {value!r}')
    if 'align' in entries and not isinstance(entries['align'], bool):
        problems.append(f'{where}["align"] must be true or false, got {entries["align"]!r}')

    if 'sentence_split' in entries:
        _compiles(entries['sentence_split'], f'{where}["sentence_split"]', problems)

    patterns = entries.get('location_patterns')
    if patterns is not None:
        if not isinstance(patterns, dict):
            problems.append(f'{where}["location_patterns"] must be an object '
                            '{corpus_id: regex}')
        else:
            for corpus_id, pattern in config_entries(patterns).items():
                key = f'{where}["location_patterns"]["{corpus_id}"]'
                if corpus_id not in corpora:
                    problems.append(f'{key} names no configured corpus')
                    continue
                if _compiles(pattern, key, problems) and corpus_id not in used:
                    warnings.append(f'{key}: "{corpus_id}" is neither the source nor a '
                                    'target, so the pattern is unused')
    return used


# ---------------------------------------------------------------------------
# Sentences and locations
# ---------------------------------------------------------------------------

def _pieces(text, pattern, start=0):
    """(piece, start, end) for `text` split at `pattern`, empty pieces dropped."""
    out, pos = [], 0
    for m in pattern.finditer(text):
        if m.end() == 0 or m.start() < pos:
            continue
        out.append((text[pos:m.start()], start + pos, start + m.start()))
        pos = m.end()
    out.append((text[pos:], start + pos, start + len(text)))
    return [(p, a, b) for p, a, b in out if p.strip()]


def split_sentences(text, split_pattern, min_chars, max_chars):
    """
    Split `text` into paragraphs (blank-line separated), then sentences at
    `split_pattern` (a compiled regex). Whitespace is collapsed; sentences
    shorter than `min_chars` are skipped; longer than `max_chars` are
    hard-wrapped. Returns (sentence, end_offset, paragraph_index) tuples,
    paragraph_index counting from 1 and end_offset into `text`.
    """
    out = []
    for p_index, (para, p_start, _) in enumerate(_pieces(text, _PARA_BREAK), 1):
        for sent, _, s_end in _pieces(para, split_pattern, p_start):
            sent = _WHITESPACE.sub(' ', sent).strip()
            if len(sent) < min_chars:
                continue
            for piece in _hard_wrap(sent, max_chars):
                out.append((piece, s_end, p_index))
    return out


def _location_at(matches, end):
    """The label of the last match ending at or before `end`, else None."""
    label = None
    for m_end, m_label in matches:
        if m_end > end:
            break
        label = m_label
    return label


def corpus_sentences(corpus_id, loaded, split_pattern, min_chars, max_chars,
                     location_pattern=None):
    """
    Sentences of one loaded corpus (a single file's text, or a collection's
    item list), in order. Each is a dict: corpus, text_id (the collection
    item `group/item_num`, or the corpus id), location, index (running
    within the corpus) and text. The location is the last match of
    `location_pattern` (group 1 if it has one and it took part in the
    match, else the whole match) in the text up to and
    including the sentence; without a pattern, or before its first match,
    the collection item or `p<paragraph>` of a single file.
    """
    if isinstance(loaded, str):
        texts = [(corpus_id, None, loaded)]
    else:
        texts = [(f"{it['group']}/{it['item_num']}", f"{it['group']}/{it['item_num']}",
                  it['text']) for it in loaded]
    out = []
    for text_id, item_label, text in texts:
        text = unicodedata.normalize('NFC', text)
        matches = []
        if location_pattern is not None:
            try:
                matches = [(m.end(), _WHITESPACE.sub(' ', m.group(1) if m.re.groups
                                                     and m.group(1) is not None
                                                     else m.group(0)).strip())
                           for m in location_pattern.finditer(text)]
                matches = [(m_end, label) for m_end, label in matches if label]
            except Exception as e:
                raise ValueError(f'parallel_passages["location_patterns"]["{corpus_id}"] '
                                 f'failed on the text: {e}') from e
        for sent, end, p_index in split_sentences(text, split_pattern, min_chars, max_chars):
            location = _location_at(matches, end) or item_label or f'p{p_index}'
            out.append({'corpus': corpus_id, 'text_id': text_id, 'location': location,
                        'index': len(out), 'text': sent})
    return out


# ---------------------------------------------------------------------------
# Matching, passages and alignment (pure)
# ---------------------------------------------------------------------------

def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def best_matches(source, target, block=512):
    """
    For each row of `source`, the best and runner-up row of `target` by
    dot product (cosine for normalized rows), computed `block` rows at a
    time. Returns (best_index, best_score, runner_up_index, runner_up_score);
    the runner-up is -1 / nan when `target` has one row.
    """
    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    n = len(source)
    best = np.zeros(n, dtype=int)
    score = np.full(n, np.nan)
    second = np.full(n, -1, dtype=int)
    second_score = np.full(n, np.nan)
    if not len(target):
        raise ValueError('no target sentences to compare with')
    for i in range(0, n, block):
        sims = source[i:i + block] @ target.T
        rows = np.arange(len(sims))
        if sims.shape[1] == 1:
            top = np.zeros((len(sims), 1), dtype=int)
        else:
            top = np.argpartition(-sims, 1, axis=1)[:, :2]
            swap = sims[rows, top[:, 1]] > sims[rows, top[:, 0]]
            top[swap] = top[swap][:, ::-1]
            # ties: prefer the earlier target sentence
            tie = (sims[rows, top[:, 1]] == sims[rows, top[:, 0]]) & (top[:, 1] < top[:, 0])
            top[tie] = top[tie][:, ::-1]
        best[i:i + block] = top[:, 0]
        score[i:i + block] = sims[rows, top[:, 0]]
        if top.shape[1] > 1:
            second[i:i + block] = top[:, 1]
            second_score[i:i + block] = sims[rows, top[:, 1]]
    return best, score, second, second_score


def find_passages(scores, target_texts, target_indices, threshold, merge_gap,
                  target_window, min_passage_sentences):
    """
    Merge matches (score >= threshold) into runs, separately for each
    target text: walking the source in order, a match joins the current
    run into its target text if its source index is at most `merge_gap` + 1
    past the run's last match (whatever the sentences between matched) and
    its target index moves from the run's last one by between -1 and
    +`target_window`; otherwise it starts a new run in that text, and the
    old run stays open (until `merge_gap` + 1 source sentences pass with no
    match joining it), so a stray far-off match does not split a passage.
    The match joins the most recently started open run that fits. A match
    into another text never ends a run, so passages into different target
    texts may overlap in source span. Runs with at least
    `min_passage_sentences` matches are returned, sorted by source_first
    and then target, as dicts: source_first, source_last, matches (source
    indices), target_text, target_first, target_last.
    """
    passages, runs = [], {}

    def close(text, run):
        if len(run) >= min_passage_sentences:
            tgts = [target_indices[i] for i in run]
            passages.append({'source_first': run[0], 'source_last': run[-1],
                             'matches': list(run), 'target_text': text,
                             'target_first': int(min(tgts)), 'target_last': int(max(tgts))})

    for i, score in enumerate(scores):
        if not score >= threshold:
            continue
        text = target_texts[i]
        open_runs = []
        for run in runs.get(text, []):
            if i - run[-1] <= merge_gap + 1:
                open_runs.append(run)
            else:
                close(text, run)
        for run in reversed(open_runs):
            if -1 <= target_indices[i] - target_indices[run[-1]] <= target_window:
                run.append(i)
                break
        else:
            open_runs.append([i])
        runs[text] = open_runs
    for text, open_runs in runs.items():
        for run in open_runs:
            close(text, run)
    passages.sort(key=lambda p: (p['source_first'], p['target_text'], p['target_first']))
    return passages


def align(n_source, n_target, sim, skip_penalty):
    """
    Monotonic alignment of `n_source` with `n_target` sentences by dynamic
    programming. Steps take 1-3 sentences per side (ALIGN_STEPS), scoring
    sim(src_range, tgt_range) x (sentences on both sides) / 2; a skip of
    one sentence on either side costs `skip_penalty`. `sim` takes two
    (start, stop) ranges. Returns units (s0, s1, t0, t1, similarity) with
    half-open ranges; a skip has one empty range and similarity None.
    """
    neg = float('-inf')
    best = np.full((n_source + 1, n_target + 1), neg)
    back = {}
    best[0, 0] = 0.0
    steps = ALIGN_STEPS + ((1, 0), (0, 1))
    for i in range(n_source + 1):
        for j in range(n_target + 1):
            if best[i, j] == neg:
                continue
            for di, dj in steps:
                a, b = i + di, j + dj
                if a > n_source or b > n_target:
                    continue
                if di and dj:
                    s = float(sim((i, a), (j, b)))
                    gain = s * (di + dj) / 2
                else:
                    s, gain = None, -skip_penalty
                if best[i, j] + gain > best[a, b]:
                    best[a, b] = best[i, j] + gain
                    back[(a, b)] = (i, j, s)
    units, cell = [], (n_source, n_target)
    while cell != (0, 0):
        i, j, s = back[cell]
        units.append((i, cell[0], j, cell[1], s))
        cell = (i, j)
    return units[::-1]


# ---------------------------------------------------------------------------
# The script
# ---------------------------------------------------------------------------

def _write_csv(path, header, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f'  Saved {path}')


def _round(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def _span_label(first, last):
    return first if first == last else f'{first} – {last}'


def _encode(model, texts):
    emb = model.encode(texts, batch_size=16, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    return normalize_rows(emb)


def group_similarity(model, source_texts, target_texts):
    """
    sim(src_range, tgt_range) for align(): the cosine of the normalized
    embeddings of the joined sentences, every group of 1-3 sentences per
    side embedded in one batch.
    """
    def groups(texts):
        keys = [(a, a + k) for k in (1, 2, 3) for a in range(len(texts) - k + 1)]
        emb = _encode(model, [' '.join(texts[a:b]) for a, b in keys]) if keys else []
        return {key: emb[n] for n, key in enumerate(keys)}
    src, tgt = groups(source_texts), groups(target_texts)
    return lambda s, t: float(src[s] @ tgt[t])


def _plot(scores, threshold, passages, label, path):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for p in passages:
        ax.axvspan(p['source_first'] - 0.5, p['source_last'] + 0.5, color='tab:orange',
                   alpha=0.3, lw=0)
    ax.plot(np.arange(len(scores)), scores, lw=0.6, color='tab:blue')
    ax.axhline(threshold, color='tab:red', ls='--', lw=1, label=f'threshold {threshold:g}')
    ax.set_xlabel(f'{label} sentence')
    ax.set_ylabel('best cosine similarity')
    ax.set_title(f'Best match for each {label} sentence (passages shaded)')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved {path}')


def analyze(config):
    results_dir, viz_dir, embed_dir = ensure_output_dirs(config)
    s = settings(config)
    section = config['parallel_passages']
    source_id, target_ids = section['source'], list(section['targets'])
    split = re.compile(unicodedata.normalize('NFC', s['sentence_split']))
    locations = {cid: re.compile(unicodedata.normalize('NFC', p), re.MULTILINE)
                 for cid, p in s['location_patterns'].items()}

    print('=== Splitting sentences ===')
    sentences = {}
    for cid in [source_id] + target_ids:
        sentences[cid] = corpus_sentences(cid, load_source_corpus(config, cid), split,
                                          s['min_sentence_chars'], s['max_sentence_chars'],
                                          locations.get(cid))
        print(f'  {cid}: {len(sentences[cid])} sentences')
        if not sentences[cid]:
            raise ValueError(f'corpus "{cid}" has no sentences of at least '
                             f'{s["min_sentence_chars"]} characters')
    src = sentences[source_id]
    tgt = [x for cid in target_ids for x in sentences[cid]]

    model_name = config.get('embedding_model') or DEFAULT_EMBEDDING_MODEL
    print(f'\n=== Embedding sentences ({model_name}) ===')
    model = get_model(model_name)
    emb = {cid: normalize_rows(embed_chunks(
               sentences[cid], model,
               cache_path=os.path.join(embed_dir, f'parallel_{cid}_sentences.npy'),
               model_name=model_name,
               too_long_hint='Lower parallel_passages.max_sentence_chars (about {chars} '
                             'characters fits this corpus) and rerun xlt-parallel-passages.'))
           for cid in [source_id] + target_ids}

    print('\n=== Matching ===')
    best, score, _, second = best_matches(emb[source_id],
                                          np.vstack([emb[cid] for cid in target_ids]))
    matched = [tgt[j] for j in best]
    passages = find_passages(score, [(m['corpus'], m['text_id']) for m in matched],
                             [m['index'] for m in matched], s['threshold'], s['merge_gap'],
                             s['target_window'], s['min_passage_sentences'])

    _write_csv(os.path.join(results_dir, 'parallel_sentences.csv'),
               ['source_idx', 'source_location', 'score', 'runner_up', 'margin',
                'target_corpus', 'target_location', 'target_idx', 'source_text',
                'target_text'],
               [[i, x['location'], _round(score[i]), _round(second[i]),
                 _round(score[i] - second[i]), m['corpus'], m['location'], m['index'],
                 x['text'], m['text']]
                for i, (x, m) in enumerate(zip(src, matched))])

    rows, align_rows = [], []
    for n, p in enumerate(passages, 1):
        corpus_id = p['target_text'][0]
        t_sents = sentences[corpus_id]
        s_span = src[p['source_first']:p['source_last'] + 1]
        t_span = t_sents[p['target_first']:p['target_last'] + 1]
        scores = [float(score[i]) for i in p['matches']]
        p['source_location'] = _span_label(s_span[0]['location'], s_span[-1]['location'])
        p['target_location'] = _span_label(t_span[0]['location'], t_span[-1]['location'])
        rows.append([n, p['source_first'], p['source_last'], p['source_location'],
                     len(p['matches']), _round(np.mean(scores)), _round(max(scores)),
                     corpus_id, p['target_location'], p['target_first'], p['target_last'],
                     ' '.join(x['text'] for x in s_span), ' '.join(x['text'] for x in t_span)])
        if s['align']:
            # extend the target span, staying inside the same target text
            same = [x['index'] for x in t_sents if x['text_id'] == p['target_text'][1]]
            t0 = max(same[0], p['target_first'] - s['align_context'])
            t1 = min(same[-1], p['target_last'] + s['align_context'])
            s_texts = [x['text'] for x in s_span]
            t_texts = [x['text'] for x in t_sents[t0:t1 + 1]]
            sim = group_similarity(model, s_texts, t_texts)
            for u, (a, b, c, d, similarity) in enumerate(
                    align(len(s_texts), len(t_texts), sim, s['skip_penalty']), 1):
                align_rows.append([
                    n, u,
                    p['source_first'] + a if b > a else '',
                    p['source_first'] + b - 1 if b > a else '',
                    t0 + c if d > c else '', t0 + d - 1 if d > c else '',
                    _round(similarity), ' '.join(s_texts[a:b]), ' '.join(t_texts[c:d])])

    _write_csv(os.path.join(results_dir, 'parallel_passages.csv'),
               ['passage', 'source_first', 'source_last', 'source_location', 'n_matches',
                'mean_score', 'max_score', 'target_corpus', 'target_location',
                'target_first', 'target_last', 'source_text', 'target_text'], rows)
    if s['align']:
        _write_csv(os.path.join(results_dir, 'parallel_alignment.csv'),
                   ['passage', 'unit', 'source_first', 'source_last', 'target_first',
                    'target_last', 'similarity', 'source_text', 'target_text'], align_rows)

    quantiles = np.quantile(score, QUANTILES)
    lines = [
        f'Source: {source_id} ({len(src)} sentences); targets: '
        + ', '.join(f'{cid} ({len(sentences[cid])})' for cid in target_ids)
        + f'; {len(tgt)} target sentences',
        'Best-match score quantiles '
        + ', '.join(f'{q * 100:g}%: {v:.3f}' for q, v in zip(QUANTILES, quantiles)),
        'Source sentences with a best match at or above '
        + ', '.join(f'{t:.2f}: {int((score >= t).sum())}' for t in SUMMARY_LEVELS),
        f'Passages (threshold {s["threshold"]:g}, at least {s["min_passage_sentences"]} '
        f'matches): {len(passages)}',
    ]
    for n, p in enumerate(passages, 1):
        lines.append(f'  {n}. {source_id} {p["source_location"]} '
                     f'(sentences {p["source_first"]}-{p["source_last"]}, '
                     f'{len(p["matches"])} matches) -> {p["target_text"][0]} '
                     f'{p["target_location"]}')
    lines += [
        '',
        'Caveat: isolated matches between about 0.70 and 0.80 are often shared '
        'formulae (stock phrases, liturgy, scripture), not translation. Passages '
        'are stronger evidence. The score measures shared meaning, not wording, so '
        'paraphrase is found too, and a high score is not proof of translation.',
    ]
    path = os.path.join(results_dir, 'parallel_summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved {path}')

    _plot(score, s['threshold'], passages, source_id,
          os.path.join(viz_dir, 'parallel_passages.png'))

    print('\n' + '\n'.join(lines))


REQUIRES = ('parallel_passages',)


def main():
    run_cli(analyze, require=REQUIRES)


if __name__ == '__main__':
    main()
