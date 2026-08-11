"""
Lexical richness analysis: a dated collection vs. a single continuous work.

Computes type-token ratio (TTR) and vocabulary size per collection item,
grouped by group (e.g. letter volume). Also computes moving-average TTR
(MATTR) for the single work as a whole, to show how lexical density shifts
across the text.

TTR = unique tokens / total tokens (sensitive to text length).
MATTR = mean TTR over a sliding window of fixed size (length-independent).

Uses config['study']['collection_corpus'] and ['single_work_corpus'].

Outputs:
  results/lexical_richness_collection.csv  – per-item TTR, token count, types
  results/lexical_richness_single_work.csv – MATTR across single-work windows
  visualizations/ttr_by_group.png          – box plot of TTR per group
  visualizations/ttr_scatter.png           – TTR vs. item length scatter
  visualizations/mattr_single_work.png     – MATTR rolling window across the work
"""

import os
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .corpus import (load_config, add_config_args, ensure_output_dirs,
                    load_source_corpus, load_stopwords, tokenize, corpus_label,
                    keep_short_tokens)

MATTR_WINDOW = 500   # tokens per MATTR window


def ttr(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mattr(tokens, window=MATTR_WINDOW):
    """Moving-average TTR: mean TTR over successive windows."""
    if len(tokens) < window:
        return ttr(tokens)
    return [ttr(tokens[i:i + window]) for i in range(0, len(tokens) - window + 1)]


def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    study = config['study']
    collection_id, single_id = study['collection_corpus'], study['single_work_corpus']
    collection_label = corpus_label(config, collection_id)
    single_label      = corpus_label(config, single_id)
    collection_lang   = config['corpora'][collection_id]['language']
    single_lang       = config['corpora'][single_id]['language']
    collection_stops  = load_stopwords(config, collection_lang)
    collection_short  = keep_short_tokens(config, collection_lang)
    single_short      = keep_short_tokens(config, single_lang)

    # --- Collection ---
    print(f'Loading {collection_label}…')
    items = load_source_corpus(config, collection_id)
    rows = []
    for item in items:
        tokens = tokenize(item['text'], remove_stopwords=False, keep_short=collection_short)
        content_tokens = tokenize(item['text'], stopwords=collection_stops, keep_short=collection_short)
        rows.append({
            'group':          item['group'],
            'group_num':      item['group_num'],
            'item_num':       item['item_num'],
            'total_tokens':   len(tokens),
            'unique_tokens':  len(set(tokens)),
            'ttr':            round(ttr(tokens), 4),
            'content_tokens': len(content_tokens),
            'unique_content': len(set(content_tokens)),
            'content_ttr':    round(ttr(content_tokens), 4),
        })

    csv_path = os.path.join(results_dir, 'lexical_richness_collection.csv')
    fieldnames = ['group', 'group_num', 'item_num', 'total_tokens',
                  'unique_tokens', 'ttr', 'content_tokens',
                  'unique_content', 'content_ttr']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'  Saved {csv_path}')

    # --- Box plot: TTR by group ---
    group_nums = sorted(set(r['group_num'] for r in rows))
    group_ttrs = {g: [r['content_ttr'] for r in rows if r['group_num'] == g]
                  for g in group_nums}
    group_labels = [f'Group {g}' for g in group_nums]

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot([group_ttrs[g] for g in group_nums],
                    tick_labels=group_labels, patch_artist=True,
                    medianprops={'color': 'black', 'linewidth': 2})
    colors = plt.cm.Set2.colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xlabel(f'{collection_label} group')
    ax.set_ylabel('Type-token ratio (content words)')
    ax.set_title(f'Lexical richness per {collection_label.lower()} group')
    fig.tight_layout()
    p1 = os.path.join(viz_dir, 'ttr_by_group.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f'  Saved {p1}')

    # --- Scatter: TTR vs. item length ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_scatter = plt.cm.tab10.colors
    for i, g in enumerate(group_nums):
        subset = [r for r in rows if r['group_num'] == g]
        xs = [r['content_tokens'] for r in subset]
        ys = [r['content_ttr']    for r in subset]
        ax.scatter(xs, ys, label=f'Group {g}',
                   color=colors_scatter[i % len(colors_scatter)], alpha=0.7, s=40)

    ax.set_xlabel('Content token count')
    ax.set_ylabel('Type-token ratio (content words)')
    ax.set_title('TTR vs. item length (shorter items inflate TTR)')
    ax.legend(title='Group', fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(viz_dir, 'ttr_scatter.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f'  Saved {p2}')

    # --- Single work MATTR ---
    print(f'Computing {single_label} MATTR…')
    single_tokens = tokenize(load_source_corpus(config, single_id), remove_stopwords=False,
                             keep_short=single_short)
    mattr_vals = mattr(single_tokens, window=MATTR_WINDOW)

    csv2 = os.path.join(results_dir, 'lexical_richness_single_work.csv')
    with open(csv2, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['window_start', 'mattr'])
        for i, val in enumerate(mattr_vals):
            writer.writerow([i, round(val, 4)])
    print(f'  Saved {csv2}')

    window_size = 50
    smoothed = np.convolve(mattr_vals, np.ones(window_size) / window_size, mode='valid')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(mattr_vals, alpha=0.25, color='steelblue', linewidth=0.8, label='Raw MATTR')
    offset = window_size // 2
    ax.plot(range(offset, offset + len(smoothed)), smoothed,
            color='steelblue', linewidth=2, label=f'{window_size}-window smoothed')
    ax.set_xlabel(f'Token position (window size = {MATTR_WINDOW})')
    ax.set_ylabel('TTR within window')
    ax.set_title(f'Lexical richness across the {single_label} (MATTR)')
    ax.legend()
    fig.tight_layout()
    p3 = os.path.join(viz_dir, 'mattr_single_work.png')
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f'  Saved {p3}')

    # --- Summary ---
    print('\nPer-group summary (content-word TTR):')
    for g in group_nums:
        vals = group_ttrs[g]
        print(f'  Group {g}  n={len(vals):3d}  '
              f'mean={np.mean(vals):.3f}  median={np.median(vals):.3f}  '
              f'min={np.min(vals):.3f}  max={np.max(vals):.3f}')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    analyze(config)


if __name__ == '__main__':
    main()
