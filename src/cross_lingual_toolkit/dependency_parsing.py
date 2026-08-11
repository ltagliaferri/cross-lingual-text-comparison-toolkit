"""
Dependency parsing: verbal patterns around key agents.

Uses spaCy's dependency parser to answer, for each agent defined in
config['dependency_agents']: what verbs does this agent perform (or
receive), as grammatical subject or object?

Processes each of the study's primary corpora (config['study']
['primary_corpora']) separately.

Requires: spacy + a model for the primary language
  pip install spacy
  python -m spacy download <model>   (see config['languages'])

Outputs:
  results/dep_verbs_[agent]_[corpus].csv          – verb frequencies per agent
  visualizations/dep_verbs_[agent]_[corpus].png   – horizontal bar charts
"""

import os
import sys
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .corpus import (load_config, add_config_args, ensure_output_dirs,
                    load_source_corpus, load_spacy, corpus_label)

try:
    import spacy  # noqa: F401
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False


def extract_verbs(doc, agent_lemmas, dep_role):
    """
    Return Counter of verb lemmas where a token with a lemma in `agent_lemmas`
    has the dependency relation `dep_role` to its head.
    """
    counter = Counter()
    for token in doc:
        if (token.lemma_.lower() in agent_lemmas and
                token.dep_ in (dep_role, dep_role + 'pass')):
            head = token.head
            if head.pos_ in ('VERB', 'AUX'):
                counter[head.lemma_.lower()] += 1
    return counter


def process_text(nlp, text, agents, chunk_size=50_000):
    """Process a large text in chunks and aggregate dependency counters."""
    agent_counters = {key: Counter() for key in agents}

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        doc = nlp(chunk)
        for key, cfg in agents.items():
            agent_counters[key].update(
                extract_verbs(doc, set(cfg['lemmas']), cfg['role'])
            )

    return agent_counters


def plot_verbs(counter, title, path, top_n=15, color='#4c72b0'):
    top = counter.most_common(top_n)
    if not top:
        return
    words, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(list(words)[::-1], list(counts)[::-1], color=color, alpha=0.85)
    ax.set_xlabel('Occurrences')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def corpus_text(config, corpus_id):
    entry = config['corpora'][corpus_id]
    if entry['type'] == 'single_file':
        return load_source_corpus(config, corpus_id)
    return '\n'.join(item['text'] for item in load_source_corpus(config, corpus_id))


def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    agents = config['dependency_agents']
    language = config['study']['primary_language']

    if not _SPACY_AVAILABLE:
        print('ERROR: spaCy is required for dependency parsing.\n'
              'Install with: pip install spacy')
        sys.exit(1)

    try:
        nlp = load_spacy(config, language)
    except OSError as e:
        print(f'ERROR: {e}')
        sys.exit(1)

    for corpus_id in config['study']['primary_corpora']:
        label = corpus_label(config, corpus_id)
        text = corpus_text(config, corpus_id)
        print(f'\nProcessing {label} ({len(text):,} chars)…')
        counters = process_text(nlp, text, agents)

        for key, cfg in agents.items():
            counter = counters[key]

            csv_path = os.path.join(results_dir, f'dep_verbs_{key}_{corpus_id}.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['verb_lemma', 'count'])
                writer.writerows(counter.most_common(30))
            print(f'  Saved {csv_path}')

            p = os.path.join(viz_dir, f'dep_verbs_{key}_{corpus_id}.png')
            plot_verbs(counter,
                       title=f"{cfg['label']} — {label}",
                       path=p,
                       color=cfg.get('color', '#4c72b0'))
            print(f'  Saved {p}')

        # Side-by-side subplot for this corpus
        n = len(agents)
        cols = 2
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
        axes_flat = axes.flat if n > 1 else [axes]
        for ax, (key, cfg) in zip(axes_flat, agents.items()):
            top = counters[key].most_common(12)
            if top:
                words, cnts = zip(*top)
                ax.barh(list(words)[::-1], list(cnts)[::-1],
                        color=cfg.get('color', '#4c72b0'), alpha=0.85)
            ax.set_title(cfg['label'])
            ax.set_xlabel('Occurrences')

        fig.suptitle(f'Verbal patterns by agent — {label}', fontsize=13)
        fig.tight_layout()
        p = os.path.join(viz_dir, f'dep_verbs_overview_{corpus_id}.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f'  Saved {p}')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    analyze(config)


if __name__ == '__main__':
    main()
