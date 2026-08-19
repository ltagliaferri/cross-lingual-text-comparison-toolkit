"""
Analysis B: Cross-lingual topic modeling with BERTopic.

Topics are formed from the embedding space (language-agnostic), so a topic
may contain passages in more than one language about the same concept.
Reveals which topics each author emphasises and which they share.

Uses only the study's primary and comparison corpora (config['study']),
matching Analysis A's core comparison — any other configured corpora
(e.g. a collection like letters) are excluded here.

BERTopic needs min_topic_size chunks before it will form a topic at all;
set config['topic_modeling']['min_topic_size'] (default 8) to suit your
corpus, and expect no topics from a very small one.

Outputs:
  results/b_topic_info.csv            — topic terms and sizes
  results/b_topic_by_author.csv       — topic distribution per author
  results/b_topic_by_work.csv         — topic distribution per work
  visualizations/b_top_topics.png     — top-20 topics bar chart
  visualizations/b_topic_heatmap.png  — author × topic heatmap
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .corpus import (add_config_args, ensure_output_dirs,
                     load_config_from_args, load_stopwords)
from .embed import load_embedded_corpora

DEFAULT_MIN_TOPIC_SIZE = 8


def build_vectorizer_stopwords(config, languages):
    """Combine each language's stopword list + topic-modeling extras/exceptions."""
    all_stopwords = []
    topic_cfg = config.get('topic_modeling', {})
    for lang in languages:
        base = load_stopwords(config, lang)
        lang_cfg = topic_cfg.get(lang) or {}
        keep = set(lang_cfg.get('keep_stopwords', []))
        filtered = [w for w in base if w not in keep]
        all_stopwords.extend(filtered)
        all_stopwords.extend(lang_cfg.get('extra_stopwords', []))
    return all_stopwords


def _topic_words(topic_model, topic_id, n):
    """Top-n terms for a topic. get_topic() returns False for unknown ids."""
    topic = topic_model.get_topic(topic_id)
    if not topic:
        return []
    return [w for w, _ in topic[:n]]


def run(config):
    results_dir, viz_dir, embed_dir = ensure_output_dirs(config)
    study = config['study']
    primary_id, comparison_id = study['single_work_corpus'], study['comparison_corpus']
    primary_lang    = config['corpora'][primary_id]['language']
    comparison_lang = config['corpora'][comparison_id]['language']
    min_topic_size  = int(config.get('topic_modeling', {})
                                .get('min_topic_size', DEFAULT_MIN_TOPIC_SIZE))

    print('Loading chunks and embeddings…')
    data = load_embedded_corpora(embed_dir, [primary_id, comparison_id])

    all_chunks = data[primary_id]['chunks'] + data[comparison_id]['chunks']
    all_emb    = np.vstack([data[primary_id]['embeddings'], data[comparison_id]['embeddings']])

    texts   = [c['text']   for c in all_chunks]
    authors = [c['author'] for c in all_chunks]
    works   = [c['work']   for c in all_chunks]

    # --- BERTopic ---
    if len(texts) < min_topic_size:
        print(f'ERROR: only {len(texts)} chunks, fewer than min_topic_size='
              f'{min_topic_size}. BERTopic cannot form a topic from this. '
              'Lower topic_modeling.min_topic_size, or use a larger corpus / '
              'smaller chunking.max_chars.')
        return

    print(f'Running BERTopic on {len(texts)} chunks…')
    try:
        from bertopic import BERTopic
    except ImportError:
        raise ImportError(
            'BERTopic is required for this analysis. Install it with: '
            'pip install "cross-lingual-toolkit[topics]"'
        )
    from sklearn.feature_extraction.text import CountVectorizer

    all_stopwords = build_vectorizer_stopwords(config, {primary_lang, comparison_lang})

    vectorizer = CountVectorizer(min_df=1, max_df=0.95,
                                 token_pattern=r'(?u)\b\w{3,}\b',
                                 stop_words=all_stopwords)
    topic_model = BERTopic(
        embedding_model=None,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=True,
        language='multilingual',
    )
    topics, _ = topic_model.fit_transform(texts, all_emb)

    topic_info = topic_model.get_topic_info()
    n_topics = len(topic_info[topic_info['Topic'] != -1])
    print(f'  Found {n_topics} topics (excluding outlier topic -1)')
    if n_topics == 0:
        print('  Every chunk was assigned to the outlier topic — nothing to plot. '
              f'Try lowering topic_modeling.min_topic_size (currently {min_topic_size}).')
        return

    # --- Save topic info ---
    csv_path = os.path.join(results_dir, 'b_topic_info.csv')
    topic_info.to_csv(csv_path, index=False)
    print(f'  Saved {csv_path}')

    # --- Topic distribution by author and work ---
    import pandas as pd
    df = pd.DataFrame({
        'topic':  topics,
        'author': authors,
        'work':   works,
    })
    df = df[df['topic'] != -1]  # drop outlier assignments
    if df.empty:
        print('  No non-outlier chunk assignments — nothing to summarize.')
        return

    by_author = (df.groupby(['author', 'topic']).size()
                   .unstack(fill_value=0))
    by_author_prop = by_author.div(by_author.sum(axis=1), axis=0)

    by_work = (df.groupby(['work', 'topic']).size()
                 .unstack(fill_value=0))
    by_work_prop = by_work.div(by_work.sum(axis=1), axis=0)

    by_author_prop.to_csv(os.path.join(results_dir, 'b_topic_by_author.csv'))
    by_work_prop.to_csv(os.path.join(results_dir,   'b_topic_by_work.csv'))

    # --- Plot 1: top-20 topics by size ---
    top_topics = topic_info[topic_info['Topic'] != -1].head(20)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(range(len(top_topics)),
            top_topics['Count'].values[::-1],
            color='#4c72b0', alpha=0.8)
    ax.set_yticks(range(len(top_topics)))
    top_labels = [f"T{tid}: " + ", ".join(_topic_words(topic_model, tid, 4))
                  for tid in top_topics['Topic'].values[::-1]]
    ax.set_yticklabels(top_labels, fontsize=9)
    ax.set_xlabel('Chunk count')
    ax.set_title('Top 20 topics (embeddings + BERTopic, multilingual)')
    fig.tight_layout()
    p = os.path.join(viz_dir, 'b_top_topics.png')
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f'  Saved {p}')

    # --- Plot 2: author × topic heatmap (top 25 topics) ---
    top_n = min(25, len(by_author_prop.columns))
    top_topic_ids = (by_author_prop.sum(axis=0)
                                   .nlargest(top_n).index.tolist())
    heat_data = by_author_prop[top_topic_ids]
    col_labels = [f"T{tid}: " + ", ".join(_topic_words(topic_model, tid, 3))
                  for tid in top_topic_ids]

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(heat_data, xticklabels=col_labels, yticklabels=heat_data.index,
                cmap='YlOrRd', ax=ax, linewidths=0.4)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=8)
    ax.set_title('Topic proportions by author (top 25 topics)')
    fig.tight_layout()
    p = os.path.join(viz_dir, 'b_topic_heatmap.png')
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f'  Saved {p}')

    # --- Console: most distinctive topics, for each pair of authors present ---
    authors_present = sorted(by_author_prop.index)
    for i, author_a in enumerate(authors_present):
        for author_b in authors_present[i + 1:]:
            print(f'\n=== Topics most distinctive for {author_a} (vs. {author_b}) ===')
            diff = (by_author_prop.loc[author_a] - by_author_prop.loc[author_b]).nlargest(10)
            for tid, val in diff.items():
                words = ", ".join(_topic_words(topic_model, int(tid), 6))
                print(f'  T{tid} (+{val:.3f}): {words}')

            print(f'\n=== Topics most distinctive for {author_b} (vs. {author_a}) ===')
            diff2 = (by_author_prop.loc[author_b] - by_author_prop.loc[author_a]).nlargest(10)
            for tid, val in diff2.items():
                words = ", ".join(_topic_words(topic_model, int(tid), 6))
                print(f'  T{tid} (+{val:.3f}): {words}')


def main():
    args = add_config_args().parse_args()
    run(load_config_from_args(args))


if __name__ == '__main__':
    main()
