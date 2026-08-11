"""
Adjective-window analysis around anchor terms.

For each anchor term (config['adjective_anchor_terms']), finds all
occurrences across the study's primary corpora (config['study']
['primary_corpora'], combined into one token stream) and collects the
words within a window of ±N tokens. Uses spaCy to identify adjectives
among those words.

Requires: spacy + a model for the primary language
  pip install spacy
  python -m spacy download <model>   (see config['languages'])

Outputs:
  results/adj_window_[term].csv        – ranked adjectives per anchor term
  visualizations/adj_window_[term].png – horizontal bar chart of top adjectives
  visualizations/adj_window_heatmap.png – adjective × anchor term heatmap
"""

import os
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .corpus import (load_config, add_config_args, ensure_output_dirs,
                    load_source_corpus, load_stopwords, tokenize, spacy_model_names,
                    keep_short_tokens, adjective_suffixes)

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

WINDOW_SIZE = 10   # tokens on each side of the anchor


def load_full_corpus(config, stopwords, keep_short):
    """Return tokenized list of all study primary_corpora, combined."""
    tokens = []
    for corpus_id in config['study']['primary_corpora']:
        entry = config['corpora'][corpus_id]
        if entry['type'] == 'single_file':
            tokens.extend(tokenize(load_source_corpus(config, corpus_id),
                                   remove_stopwords=False, keep_short=keep_short))
        else:
            for item in load_source_corpus(config, corpus_id):
                tokens.extend(tokenize(item['text'], remove_stopwords=False,
                                       keep_short=keep_short))
    return tokens


def collect_windows(tokens, anchor_forms, window=WINDOW_SIZE):
    """
    Return list of context-window token lists around each occurrence of
    any form in `anchor_forms`.
    """
    windows = []
    for i, tok in enumerate(tokens):
        if tok in anchor_forms:
            left  = max(0, i - window)
            right = min(len(tokens), i + window + 1)
            ctx = tokens[left:i] + tokens[i + 1:right]
            windows.append(ctx)
    return windows


def adjectives_from_window_spacy(nlp, window_tokens, stopwords):
    """Tag tokens in a window list and return adjective Counter."""
    text = ' '.join(window_tokens)
    doc = nlp(text)
    counter = Counter()
    for token in doc:
        if token.pos_ == 'ADJ' and token.lemma_.lower() not in stopwords:
            counter[token.lemma_.lower()] += 1
    return counter


def adjectives_from_window_heuristic(window_tokens, stopwords, suffixes):
    """Fallback: count tokens matching configured adjective-suffix heuristics."""
    counter = Counter()
    if not suffixes:
        return counter
    for tok in window_tokens:
        if tok in stopwords:
            continue
        if len(tok) > 4 and tok.endswith(suffixes):
            counter[tok] += 1
    return counter


def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    anchor_terms = config['adjective_anchor_terms']
    language = config['study']['primary_language']
    stopwords = load_stopwords(config, language)
    short_tokens = keep_short_tokens(config, language)
    suffixes = adjective_suffixes(config, language)

    nlp = None
    fallback_note = (f'Configure languages.{language}.adjective_suffixes in your '
                      'config to enable the suffix heuristic.' if not suffixes
                      else 'Falling back to configured suffix heuristic.')
    if _SPACY_AVAILABLE:
        primary, fallback = spacy_model_names(config, language)
        try:
            nlp = spacy.load(primary)
            print(f'Using spaCy {primary} for POS tagging.')
        except OSError:
            try:
                nlp = spacy.load(fallback)
                print(f'Using spaCy {fallback} for POS tagging.')
            except OSError:
                print(f'WARNING: spaCy model not found for "{language}". '
                      f'Run: python -m spacy download {primary}\n{fallback_note}')
    else:
        print(f'WARNING: spaCy not installed. Run: pip install spacy\n{fallback_note}')

    print('Loading corpus…')
    tokens = load_full_corpus(config, stopwords, short_tokens)
    print(f'  Total tokens: {len(tokens):,}')

    anchor_results = {}   # anchor_key -> Counter of adjectives

    for anchor_key, anchor_forms in anchor_terms.items():
        print(f'  Analysing windows around "{anchor_key}"…')
        windows = collect_windows(tokens, set(anchor_forms))
        print(f'    {len(windows)} occurrences found')

        combined_counter = Counter()
        for window_tokens in windows:
            if nlp:
                combined_counter.update(
                    adjectives_from_window_spacy(nlp, window_tokens, stopwords)
                )
            else:
                combined_counter.update(
                    adjectives_from_window_heuristic(window_tokens, stopwords, suffixes)
                )

        anchor_results[anchor_key] = combined_counter

        # Per-term CSV
        csv_path = os.path.join(results_dir, f'adj_window_{anchor_key}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['adjective', 'count', 'occurrences_in_corpus',
                             'rate_per_occurrence'])
            n = len(windows)
            for word, count in combined_counter.most_common(50):
                rate = round(count / n, 3) if n else 0
                writer.writerow([word, count, n, rate])
        print(f'    Saved {csv_path}')

        # Per-term bar chart (top 15)
        top = combined_counter.most_common(15)
        if not top:
            continue
        words, counts = zip(*top)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(list(words)[::-1], list(counts)[::-1], color='#4c72b0', alpha=0.85)
        ax.set_xlabel('Count in windows')
        ax.set_title(f'Adjectives near "{anchor_key}" (window ±{WINDOW_SIZE})')
        fig.tight_layout()
        p = os.path.join(viz_dir, f'adj_window_{anchor_key}.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f'    Saved {p}')

    # --- Heatmap: top adjectives × anchor terms ---
    all_adjs = Counter()
    for counter in anchor_results.values():
        all_adjs.update(counter)
    top_adjs = [w for w, _ in all_adjs.most_common(20)]

    if top_adjs:
        anchor_keys = list(anchor_terms.keys())
        matrix = np.zeros((len(top_adjs), len(anchor_keys)))
        for j, akey in enumerate(anchor_keys):
            for i, adj in enumerate(top_adjs):
                matrix[i, j] = anchor_results[akey].get(adj, 0)

        col_max = matrix.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1
        matrix_norm = matrix / col_max

        fig, ax = plt.subplots(figsize=(13, 7))
        im = ax.imshow(matrix_norm, aspect='auto', cmap='Blues')
        ax.set_xticks(range(len(anchor_keys)))
        ax.set_xticklabels(anchor_keys, rotation=45, ha='right')
        ax.set_yticks(range(len(top_adjs)))
        ax.set_yticklabels(top_adjs)
        plt.colorbar(im, ax=ax, label='Normalized frequency')
        ax.set_title('Adjective associations by anchor term (column-normalized)')
        fig.tight_layout()
        p = os.path.join(viz_dir, 'adj_window_heatmap.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f'\nSaved {p}')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    analyze(config)


if __name__ == '__main__':
    main()
