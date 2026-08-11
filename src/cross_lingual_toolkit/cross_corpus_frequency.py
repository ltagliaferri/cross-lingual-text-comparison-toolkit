"""
Cross-corpus term frequency and Dunning log-likelihood comparison between
the study's single-work corpus and its cross-lingual comparison corpus
(config['study']['single_work_corpus'] vs. ['comparison_corpus']).

Because the texts are in different languages, comparison operates on
BILINGUAL CONCEPT CLUSTERS (config['cross_corpus_concept_clusters']): sets
of surface forms in each language that express the same concepts.
Frequencies are normalized per 10,000 tokens and Dunning G² scores
identify which clusters each corpus emphasizes.

Outputs:
  results/cross_freq_comparison.csv    – cluster frequencies per 10k, both corpora
  results/cross_dunning_clusters.csv   – Dunning G² scores per cluster
  visualizations/cross_freq_bar.png    – side-by-side cluster bar chart
  visualizations/cross_dunning.png     – Dunning lollipop chart
"""

import os
import re
import csv
import math
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .corpus import (load_config, add_config_args, ensure_output_dirs,
                    load_source_corpus, load_stopwords, tokenize, corpus_label,
                    keep_short_tokens)

# Each cluster entry in config['cross_corpus_concept_clusters'] is keyed by
# language code (e.g. 'it'/'la'), matching config['corpora'][...]['language'].


def dunning_g2(a, b, total_a, total_b):
    """
    Return signed G² for a term with count `a` in corpus A (size total_a)
    and count `b` in corpus B (size total_b).
    Positive = over-represented in A; negative = over-represented in B.
    """
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


def tokenize_generic(text):
    """Lowercase, split on any non-letter character (Unicode-aware), return tokens >= 2 chars."""
    text = text.lower()
    return [t for t in re.split(r'[\W\d_]+', text, flags=re.UNICODE) if len(t) >= 2]


def count_clusters(tokens, clusters, lang_key):
    freq = Counter(tokens)
    forms = {key: set(data[lang_key]) for key, data in clusters.items()}
    counts = {key: 0 for key in clusters}
    for tok in tokens:
        for key, form_set in forms.items():
            if tok in form_set:
                counts[key] += 1
                break
    return counts


def normalize(counts, total, per=10_000):
    return {k: round(v / total * per, 2) for k, v in counts.items()}


def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    clusters = config['cross_corpus_concept_clusters']
    study = config['study']
    primary_id, comparison_id = study['single_work_corpus'], study['comparison_corpus']
    primary_label     = corpus_label(config, primary_id)
    comparison_label  = corpus_label(config, comparison_id)

    # --- Load primary corpus ---
    print(f'Loading {primary_label}…')
    primary_lang = config['corpora'][primary_id]['language']
    primary_stops = load_stopwords(config, primary_lang)
    primary_short = keep_short_tokens(config, primary_lang)
    primary_text    = load_source_corpus(config, primary_id)
    primary_tokens  = tokenize(primary_text, remove_stopwords=False, keep_short=primary_short)
    primary_content = tokenize(primary_text, stopwords=primary_stops, keep_short=primary_short)
    primary_counts  = count_clusters(primary_content, clusters, primary_lang)
    primary_norm    = normalize(primary_counts, len(primary_content))
    print(f'  {len(primary_tokens):,} raw tokens, {len(primary_content):,} content tokens')

    # --- Load comparison corpus ---
    print(f'Loading {comparison_label}…')
    comparison_lang = config['corpora'][comparison_id]['language']
    comparison_text = load_source_corpus(config, comparison_id)
    comparison_tokens = tokenize_generic(comparison_text)
    comparison_counts = count_clusters(comparison_tokens, clusters, comparison_lang)
    comparison_norm   = normalize(comparison_counts, len(comparison_tokens))
    print(f'  {len(comparison_tokens):,} tokens')

    cluster_keys   = list(clusters.keys())
    cluster_labels = [clusters[k]['label'] for k in cluster_keys]

    # --- Dunning scores ---
    dunning = {}
    for key in cluster_keys:
        dunning[key] = dunning_g2(
            primary_counts[key], comparison_counts[key],
            len(primary_content), len(comparison_tokens),
        )

    # --- Save CSVs ---
    freq_csv = os.path.join(results_dir, 'cross_freq_comparison.csv')
    with open(freq_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['cluster', 'label',
                    'primary_raw', 'primary_per10k',
                    'comparison_raw', 'comparison_per10k'])
        for key in cluster_keys:
            w.writerow([key, clusters[key]['label'],
                        primary_counts[key], primary_norm[key],
                        comparison_counts[key], comparison_norm[key]])
    print(f'  Saved {freq_csv}')

    dun_csv = os.path.join(results_dir, 'cross_dunning_clusters.csv')
    with open(dun_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['cluster', 'label', 'dunning_g2',
                    'favours', 'primary_per10k', 'comparison_per10k'])
        for key in sorted(cluster_keys, key=lambda k: -abs(dunning[k])):
            favour = primary_label if dunning[key] > 0 else comparison_label
            w.writerow([key, clusters[key]['label'],
                        round(dunning[key], 2), favour,
                        primary_norm[key], comparison_norm[key]])
    print(f'  Saved {dun_csv}')

    # --- Plot 1: side-by-side bar chart ---
    x     = np.arange(len(cluster_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - width/2, [primary_norm[k] for k in cluster_keys],
                width, label=primary_label, color='#4c72b0')
    b2 = ax.bar(x + width/2, [comparison_norm[k] for k in cluster_keys],
                width, label=comparison_label, color='#c44e52')
    ax.set_xticks(x)
    ax.set_xticklabels(cluster_labels, rotation=12, ha='right')
    ax.set_ylabel('Frequency per 10,000 tokens')
    ax.set_title(f'Concept cluster frequencies: {primary_label} vs. {comparison_label}')
    ax.legend()
    ax.bar_label(b1, fmt='%.0f', padding=2, fontsize=8)
    ax.bar_label(b2, fmt='%.0f', padding=2, fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(viz_dir, 'cross_freq_bar.png')
    fig.savefig(p1, dpi=150); plt.close(fig)
    print(f'  Saved {p1}')

    # --- Plot 2: Dunning lollipop ---
    sorted_keys = sorted(cluster_keys, key=lambda k: dunning[k])
    scores      = [dunning[k] for k in sorted_keys]
    labels      = [clusters[k]['label'] for k in sorted_keys]
    colours     = ['#4c72b0' if s > 0 else '#c44e52' for s in scores]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hlines(range(len(sorted_keys)), 0, scores, color=colours, linewidth=2)
    ax.scatter(scores, range(len(sorted_keys)), color=colours, s=80, zorder=3)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f'Dunning G² (positive = {primary_label}; negative = {comparison_label})')
    ax.set_title(f'Cluster emphasis: {primary_label} vs. {comparison_label} (Dunning G²)')
    for i, (score, label) in enumerate(zip(scores, labels)):
        side = primary_label if score > 0 else comparison_label
        ax.annotate(f'{side} ({abs(score):.0f})',
                    xy=(score, i), xytext=(5 if score > 0 else -5, 0),
                    textcoords='offset points',
                    va='center', ha='left' if score > 0 else 'right',
                    fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(viz_dir, 'cross_dunning.png')
    fig.savefig(p2, dpi=150); plt.close(fig)
    print(f'  Saved {p2}')

    # --- Console summary ---
    print(f'\n{"Cluster":25s}  {primary_label + "/10k":>13}  {comparison_label + "/10k":>13}  {"G²":>8}  Favours')
    print('-' * 78)
    for key in cluster_keys:
        fav = primary_label if dunning[key] > 0 else comparison_label
        print(f'{clusters[key]["label"]:25s}  '
              f'{primary_norm[key]:13.1f}  {comparison_norm[key]:13.1f}  '
              f'{dunning[key]:8.1f}  {fav}')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    analyze(config)


if __name__ == '__main__':
    main()
