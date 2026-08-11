"""
Term frequency analysis: a single continuous work vs. a dated collection.

Tracks clusters of study-specific terms (config['term_frequency_clusters'])
and compares their normalized frequency (per 10,000 tokens) across the two
works, and across the collection's groups (e.g. letter volumes).

In the bundled example study this is the Dialogo vs. the Letters, grouped
by volume.

Outputs:
  results/term_frequency.csv           – raw and normalized counts per text
  visualizations/term_freq_corpora.png – single-work vs. collection bar chart
  visualizations/term_freq_volumes.png – per-cluster frequency across groups
"""

import os
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .corpus import (load_config, add_config_args, ensure_output_dirs,
                    load_source_corpus, load_stopwords, tokenize, corpus_label,
                    keep_short_tokens)


def count_clusters(tokens, clusters):
    """Return dict {cluster_key: count} for a token list."""
    freq = Counter(tokens)
    return {key: sum(freq.get(t, 0) for t in data['terms'])
            for key, data in clusters.items()}


def normalize(counts, total_tokens, per=10_000):
    """Normalize raw counts to frequency per `per` tokens."""
    return {k: round(v / total_tokens * per, 2) for k, v in counts.items()}


def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    clusters = config['term_frequency_clusters']
    study = config['study']
    single_id, collection_id = study['single_work_corpus'], study['collection_corpus']

    single_lang     = config['corpora'][single_id]['language']
    collection_lang = config['corpora'][collection_id]['language']
    single_stops     = load_stopwords(config, single_lang)
    collection_stops = load_stopwords(config, collection_lang)
    single_short     = keep_short_tokens(config, single_lang)
    collection_short = keep_short_tokens(config, collection_lang)

    single_label     = corpus_label(config, single_id)
    collection_label = corpus_label(config, collection_id)

    # --- Single work ---
    print(f'Loading {single_label}…')
    single_text   = load_source_corpus(config, single_id)
    single_tokens = tokenize(single_text, stopwords=single_stops, keep_short=single_short)
    single_counts = count_clusters(single_tokens, clusters)
    single_norm   = normalize(single_counts, len(single_tokens))

    # --- Collection, grouped ---
    print(f'Loading {collection_label}…')
    items = load_source_corpus(config, collection_id)
    group_data = {}   # group_num -> {tokens, label}
    all_collection_tokens = []
    for item in items:
        tokens = tokenize(item['text'], stopwords=collection_stops, keep_short=collection_short)
        all_collection_tokens.extend(tokens)
        g = item['group_num']
        if g not in group_data:
            group_data[g] = {'tokens': [], 'label': item['group']}
        group_data[g]['tokens'].extend(tokens)

    for g, data in group_data.items():
        data['counts'] = count_clusters(data['tokens'], clusters)
        data['norm']   = normalize(data['counts'], len(data['tokens']))

    all_collection_counts = count_clusters(all_collection_tokens, clusters)
    all_collection_norm   = normalize(all_collection_counts, len(all_collection_tokens))

    # --- Write CSV ---
    csv_path = os.path.join(results_dir, 'term_frequency.csv')
    cluster_keys   = list(clusters.keys())
    cluster_labels = [clusters[k]['label'] for k in cluster_keys]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['text', 'total_tokens'] + \
                 [f'{k}_raw' for k in cluster_keys] + \
                 [f'{k}_per10k' for k in cluster_keys]
        writer.writerow(header)

        row = [single_label, len(single_tokens)] + \
              [single_counts[k] for k in cluster_keys] + \
              [single_norm[k]   for k in cluster_keys]
        writer.writerow(row)

        for g in sorted(group_data):
            data = group_data[g]
            row = [data['label'], len(data['tokens'])] + \
                  [data['counts'][k] for k in cluster_keys] + \
                  [data['norm'][k]   for k in cluster_keys]
            writer.writerow(row)

        row = [f'{collection_label} (all)', len(all_collection_tokens)] + \
              [all_collection_counts[k] for k in cluster_keys] + \
              [all_collection_norm[k]   for k in cluster_keys]
        writer.writerow(row)

    print(f'  Saved {csv_path}')

    # --- Plot 1: single work vs. collection (grouped bar) ---
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(cluster_keys))
    width = 0.35
    bars1 = ax.bar(x - width / 2,
                   [single_norm[k] for k in cluster_keys],
                   width, label=single_label, color='#4c72b0')
    bars2 = ax.bar(x + width / 2,
                   [all_collection_norm[k] for k in cluster_keys],
                   width, label=f'{collection_label} (all)', color='#dd8452')

    ax.set_xlabel('Term cluster')
    ax.set_ylabel('Frequency per 10,000 tokens')
    ax.set_title(f'Term cluster frequencies: {single_label} vs. {collection_label}')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels, rotation=15, ha='right')
    ax.legend()
    ax.bar_label(bars1, fmt='%.1f', padding=2, fontsize=8)
    ax.bar_label(bars2, fmt='%.1f', padding=2, fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(viz_dir, 'term_freq_corpora.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f'  Saved {p1}')

    # --- Plot 2: cluster frequency across collection groups (line chart) ---
    group_nums = sorted(group_data.keys())
    group_labels = [f'Group {g}' for g in group_nums]

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.tab10.colors
    for i, key in enumerate(cluster_keys):
        vals = [group_data[g]['norm'][key] for g in group_nums]
        ax.plot(group_labels, vals, marker='o', label=clusters[key]['label'],
                color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel(f'{collection_label} group')
    ax.set_ylabel('Frequency per 10,000 tokens')
    ax.set_title(f'Term cluster frequencies across {collection_label.lower()} groups')
    ax.legend(loc='upper right', fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(viz_dir, 'term_freq_volumes.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f'  Saved {p2}')

    # --- Summary to stdout ---
    print(f'\n{single_label} term frequencies (per 10k tokens):')
    for k in cluster_keys:
        print(f'  {clusters[k]["label"]:25s}  {single_norm[k]:6.1f}')
    print(f'\n{collection_label} (combined) term frequencies (per 10k tokens):')
    for k in cluster_keys:
        print(f'  {clusters[k]["label"]:25s}  {all_collection_norm[k]:6.1f}')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    analyze(config)


if __name__ == '__main__':
    main()
