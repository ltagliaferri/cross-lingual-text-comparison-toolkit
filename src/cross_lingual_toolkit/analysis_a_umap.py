"""
Analysis A: Direct embedding comparison.

Does the study's primary corpus and its cross-lingual comparison corpus
form distinguishable clusters in the shared embedding space? The "core"
comparison is primary vs. comparison (config['study']); any other
configured corpora (e.g. a collection like letters) are included in the
UMAP plot for context but not in the core pairwise/permutation stats,
matching how they're excluded from Analysis B's topic model.

Outputs:
  visualizations/a_umap_authors.png       — UMAP colored by author
  visualizations/a_umap_works.png         — UMAP colored by work
  results/a_similarity_matrix.csv         — pairwise within/between similarities
  results/a_permutation_test.txt          — p-value report
"""

import os
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2

from .corpus import (add_config_args, config_entries, ensure_output_dirs,
                     load_config_from_args, corpus_label)
from .embed import load_embedded_corpora, mean_pairwise_sim, permutation_test


def add_confidence_ellipse(ax, x, y, color, confidence=0.95, lw=1.8, alpha_fill=0.07):
    """Draw a confidence ellipse for 2D point cloud (x, y)."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    scale = np.sqrt(chi2.ppf(confidence, df=2))
    width  = 2 * scale * np.sqrt(eigenvalues[0])
    height = 2 * scale * np.sqrt(eigenvalues[1])
    ax.add_patch(Ellipse(xy=(mean_x, mean_y), width=width, height=height,
                         angle=angle, facecolor=color, alpha=alpha_fill,
                         edgecolor=color, linewidth=lw, linestyle='--', zorder=2))


def build_palette(keys, configured):
    """Map each key to a configured color, falling back to a tab10 cycle."""
    cycle = plt.cm.tab10.colors
    palette = {}
    for i, key in enumerate(keys):
        palette[key] = configured.get(key, cycle[i % len(cycle)])
    return palette


def run(config):
    results_dir, viz_dir, embed_dir = ensure_output_dirs(config)
    study = config['study']
    primary_id, comparison_id = study['single_work_corpus'], study['comparison_corpus']
    corpus_ids = list(config_entries(config['corpora']).keys())
    n_perm = int(config.get('analysis', {}).get('n_permutations', 1000))
    context_ids = [c for c in corpus_ids if c not in (primary_id, comparison_id)]

    print('Loading chunks and embeddings…')
    data = load_embedded_corpora(embed_dir, corpus_ids)

    emb_primary    = data[primary_id]['embeddings']
    emb_comparison = data[comparison_id]['embeddings']

    all_chunks = [c for cid in corpus_ids for c in data[cid]['chunks']]
    all_emb    = np.vstack([data[cid]['embeddings'] for cid in corpus_ids])
    all_corpus_ids = [cid for cid in corpus_ids for _ in data[cid]['chunks']]

    authors = [c['author'] for c in all_chunks]

    author_colors = build_palette(sorted(set(authors)),
                                  config.get('colors', {}).get('authors', {}))
    work_colors   = build_palette(corpus_ids,
                                  {cid: config['corpora'][cid].get('color')
                                   for cid in corpus_ids if config['corpora'][cid].get('color')})
    work_labels = {cid: corpus_label(config, cid) + (' (context)' if cid in context_ids else '')
                   for cid in corpus_ids}

    # --- UMAP (all configured corpora, for context) ---
    print('Running UMAP…')
    try:
        import umap
    except ImportError:
        raise ImportError(
            'umap-learn is required for this analysis. Install it with: '
            'pip install "cross-lingual-toolkit[embeddings]"'
        )
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                        metric='cosine', random_state=42)
    coords = reducer.fit_transform(all_emb)

    primary_label    = corpus_label(config, primary_id)
    comparison_label = corpus_label(config, comparison_id)

    # Plot 1: colored by author
    fig, ax = plt.subplots(figsize=(10, 8))
    for author, color in author_colors.items():
        mask = [i for i, a in enumerate(authors) if a == author]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=12, alpha=0.55, label=author.capitalize())
    ax.set_title(f'{primary_label.split()[0]} vs. {comparison_label.split()[0]} '
                 'in shared embedding space')
    ax.legend(markerscale=2)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    fig.tight_layout()
    p = os.path.join(viz_dir, 'a_umap_authors.png')
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f'  Saved {p}')

    # Plot 2: colored by work
    fig, ax = plt.subplots(figsize=(10, 8))
    for work, color in work_colors.items():
        mask = [i for i, cid in enumerate(all_corpus_ids) if cid == work]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=12, alpha=0.55, label=work_labels[work])
    ax.set_title('Corpus chunks by work — shared embedding space\n'
                 '(Note: check for differing languages, genres, and modes)')
    ax.legend(markerscale=2)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    fig.tight_layout()
    p = os.path.join(viz_dir, 'a_umap_works.png')
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f'  Saved {p}')

    # Plot 3: authors + 95% confidence ellipses
    fig, ax = plt.subplots(figsize=(10, 8))
    for author, color in author_colors.items():
        mask = [i for i, a in enumerate(authors) if a == author]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=12, alpha=0.55, label=author.capitalize(), zorder=3)
        add_confidence_ellipse(ax, coords[mask, 0], coords[mask, 1], color)
    ax.set_title(f'{primary_label.split()[0]} vs. {comparison_label.split()[0]} '
                 'in shared embedding space\n95% confidence ellipses')
    ax.legend(markerscale=2)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    fig.tight_layout()
    p = os.path.join(viz_dir, 'a_umap_authors_ellipse.png')
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f'  Saved {p}')

    # Plot 4: works + 95% confidence ellipses (primary and comparison only)
    fig, ax = plt.subplots(figsize=(10, 8))
    for work, color in work_colors.items():
        mask = [i for i, cid in enumerate(all_corpus_ids) if cid == work]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], s=12, alpha=0.55, label=work_labels[work], zorder=3)
        if work in (primary_id, comparison_id):
            add_confidence_ellipse(ax, coords[mask, 0], coords[mask, 1], color)
    ax.set_title(f'Corpus chunks by work — shared embedding space\n'
                 f'95% confidence ellipses: {primary_label} and {comparison_label}')
    ax.legend(markerscale=2)
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    fig.tight_layout()
    p = os.path.join(viz_dir, 'a_umap_works_ellipse.png')
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f'  Saved {p}')

    # --- Pairwise similarity matrix (primary vs comparison as core) ---
    print('Computing pairwise similarities…')

    pairs = [
        ('within_primary',        emb_primary,    emb_primary,    True),
        ('within_comparison',     emb_comparison, emb_comparison, True),
        ('primary_comparison',    emb_primary,    emb_comparison, False),
    ]
    for cid in context_ids:
        emb_ctx = data[cid]['embeddings']
        pairs.append((f'within_{cid}', emb_ctx, emb_ctx, True))
        pairs.append((f'{cid}_primary', emb_ctx, emb_primary, False))
        pairs.append((f'{cid}_comparison', emb_ctx, emb_comparison, False))

    rows = []
    for label, a, b, same in pairs:
        sim = mean_pairwise_sim(a, b, same=same)
        rows.append({'comparison': label, 'mean_cosine_similarity': round(sim, 4)})
        print(f'  {label:25s}: {sim:.4f}')

    csv_path = os.path.join(results_dir, 'a_similarity_matrix.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['comparison', 'mean_cosine_similarity'])
        w.writeheader(); w.writerows(rows)
    print(f'  Saved {csv_path}')

    # --- Permutation test (primary vs comparison) ---
    print(f'Running permutation test ({n_perm} permutations)…')
    p_val = permutation_test(emb_primary, emb_comparison, n_perm=n_perm)
    report = (
        f"Permutation test: {primary_label} vs. {comparison_label}\n"
        f"  Note: check whether this is a cross-genre/cross-language/cross-mode comparison\n"
        f"  Observed within-{primary_label} sim:   "
        f"{mean_pairwise_sim(emb_primary, emb_primary, same=True):.4f}\n"
        f"  Observed {primary_label}-{comparison_label} sim:  "
        f"{mean_pairwise_sim(emb_primary, emb_comparison):.4f}\n"
        f"  Observed gap: "
        f"{mean_pairwise_sim(emb_primary, emb_primary, same=True) - mean_pairwise_sim(emb_primary, emb_comparison):.4f}\n"
        f"  p-value ({n_perm} permutations): {p_val:.4f}\n"
    )
    print(report)
    txt_path = os.path.join(results_dir, 'a_permutation_test.txt')
    with open(txt_path, 'w') as f:
        f.write(report)
    print(f'  Saved {txt_path}')


def main():
    args = add_config_args().parse_args()
    run(load_config_from_args(args))


if __name__ == '__main__':
    main()
