"""
Cross-corpus dependency parsing comparison between the study's primary
corpus (spaCy) and its cross-lingual comparison corpus (Stanza).

Compares two configured "agents" — arbitrary role names, e.g. a divine name
and a first-person/reflexive term; the bundled example study uses them for
God and the soul — plus the studied person's own name-forms in the
comparison corpus. Agent forms/lemmas are nested under
config['cross_corpus_dependency']['primary'] (spaCy-side, keyed by lemma)
and ['comparison'] (Stanza-side, keyed by surface form):
  1. What verbs does agent A perform, in each corpus?
  2. What verbs does agent B perform, in each corpus?
  3. What is done TO the studied person's name (as object) in the
     comparison corpus, vs. to agent B in the primary corpus?
  4. Adjectives near the studied person's name (comparison corpus) vs.
     near agent B (primary corpus).

For the comparison corpus, Stanza is run only on sentences containing the
relevant anchor terms (not the full text) to keep processing time
reasonable.

Requires:
  pip install stanza  (+ the comparison-language model, see config)
  pip install spacy   (+ the primary-language model, see config)

Outputs:
  results/crossdep_agent_a_verbs.csv      – agent A's verbs, both corpora
  results/crossdep_agent_b_verbs.csv      – agent B's verbs, both corpora
  results/crossdep_agent_b_as_object.csv  – what is done to agent B / the name
  results/crossdep_name_adj.csv           – adjectives near the studied person's name
  visualizations/crossdep_agent_a_verbs.png
  visualizations/crossdep_agent_b_verbs.png
  visualizations/crossdep_agent_b_object.png
  visualizations/crossdep_adj_name_vs_agent_b.png
"""

import os
import re
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .corpus import (load_config, add_config_args, ensure_output_dirs,
                    load_source_corpus, load_stopwords, load_spacy, corpus_label)

try:
    import stanza
    _STANZA_AVAILABLE = True
except ImportError:
    _STANZA_AVAILABLE = False


def sentences_containing(text, terms):
    """Yield sentences (split on '.', '!', '?') containing any term in `terms`."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    for sent in sentences:
        low = sent.lower()
        if any(re.search(r'\b' + re.escape(t) + r'\b', low) for t in terms):
            key = sent[:80]
            if key not in seen:
                seen.add(key)
                yield sent


def plot_comparison(counter_a, counter_b, label_a, label_b, title, path, n=12):
    all_verbs = set(list(counter_a.keys())[:n] + list(counter_b.keys())[:n])
    verbs = sorted(all_verbs,
                   key=lambda v: counter_a.get(v, 0) + counter_b.get(v, 0),
                   reverse=True)[:n]
    x = range(len(verbs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar([i - width/2 for i in x], [counter_a.get(v, 0) for v in verbs],
           width, label=label_a, color='#4c72b0', alpha=0.85)
    ax.bar([i + width/2 for i in x], [counter_b.get(v, 0) for v in verbs],
           width, label=label_b, color='#c44e52', alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(verbs, rotation=25, ha='right')
    ax.set_ylabel('Occurrences')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_csv(path, counter_a, counter_b, label_a, label_b, n=30):
    all_items = set(list(counter_a.keys()) + list(counter_b.keys()))
    rows = sorted(all_items,
                  key=lambda v: counter_a.get(v, 0) + counter_b.get(v, 0),
                  reverse=True)[:n]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['term', label_a, label_b])
        for r in rows:
            w.writerow([r, counter_a.get(r, 0), counter_b.get(r, 0)])


# ---------------------------------------------------------------------------
# Primary-language (spaCy) processing
# ---------------------------------------------------------------------------
def process_primary(nlp, text, cfg, stopwords):
    """
    Extract from the primary corpus:
      - verbs where an agent-A lemma is nsubj
      - verbs where an agent-B lemma is nsubj
      - verbs where an agent-B lemma is obj
      - adjectives in window around agent B
    """
    print('  Processing primary corpus with spaCy…')
    a_lemmas = set(cfg['primary']['agent_a_lemmas'])
    b_lemmas = set(cfg['primary']['agent_b_lemmas'])
    window = cfg['window']

    a_verbs  = Counter()
    b_verbs  = Counter()
    b_obj    = Counter()
    b_adjs   = Counter()

    chunk = 50_000
    for start in range(0, len(text), chunk):
        doc = nlp(text[start:start + chunk])
        for token in doc:
            lem = token.lemma_.lower()

            if lem in a_lemmas and token.dep_ == 'nsubj':
                if token.head.pos_ in ('VERB', 'AUX'):
                    a_verbs[token.head.lemma_.lower()] += 1

            if lem in b_lemmas:
                if token.dep_ == 'nsubj' and token.head.pos_ in ('VERB', 'AUX'):
                    b_verbs[token.head.lemma_.lower()] += 1
                if token.dep_ == 'obj' and token.head.pos_ in ('VERB', 'AUX'):
                    b_obj[token.head.lemma_.lower()] += 1

        toks = [t for t in doc]
        for i, t in enumerate(toks):
            if t.lemma_.lower() in b_lemmas:
                lo = max(0, i - window)
                hi = min(len(toks), i + window + 1)
                for w in toks[lo:i] + toks[i+1:hi]:
                    if w.pos_ == 'ADJ' and w.lemma_.lower() not in stopwords:
                        b_adjs[w.lemma_.lower()] += 1

    return a_verbs, b_verbs, b_obj, b_adjs


# ---------------------------------------------------------------------------
# Comparison-language (Stanza) processing
# ---------------------------------------------------------------------------
def process_comparison(nlp, text, cfg):
    """
    Extract from the comparison corpus:
      - verbs where an agent-A form is nsubj
      - verbs where an agent-B form is nsubj/obj
      - verbs where the studied person's name is obj
      - adjectives in window around the studied person's name
    """
    print('  Processing comparison corpus with Stanza (targeted sentences)…')
    a_forms    = {t.lower() for t in cfg['comparison']['agent_a_forms']}
    b_forms    = {t.lower() for t in cfg['comparison']['agent_b_forms']}
    name_forms = {t.lower() for t in cfg['comparison']['name_forms']}
    window = cfg['window']

    a_verbs   = Counter()
    b_verbs   = Counter()
    name_obj  = Counter()
    name_adjs = Counter()

    all_anchors = a_forms | b_forms | name_forms
    batch, batch_size = [], 80

    def process_batch(sentences):
        for sent_text in sentences:
            try:
                doc = nlp(sent_text)
            except Exception:
                continue
            for sentence in doc.sentences:
                words = sentence.words
                for word in words:
                    surf = word.text.lower()

                    if surf in a_forms and word.deprel in ('nsubj', 'nsubj:pass'):
                        head = next((w for w in words if w.id == word.head), None)
                        if head and head.upos in ('VERB', 'AUX'):
                            a_verbs[(head.lemma or head.text).lower()] += 1

                    if surf in b_forms:
                        head = next((w for w in words if w.id == word.head), None)
                        if head and head.upos in ('VERB', 'AUX'):
                            if word.deprel in ('nsubj', 'nsubj:pass', 'obj'):
                                b_verbs[(head.lemma or head.text).lower()] += 1

                    if surf in name_forms and word.deprel == 'obj':
                        head = next((w for w in words if w.id == word.head), None)
                        if head and head.upos in ('VERB', 'AUX'):
                            name_obj[(head.lemma or head.text).lower()] += 1

                for i, word in enumerate(words):
                    if word.text.lower() in name_forms:
                        lo = max(0, i - window)
                        hi = min(len(words), i + window + 1)
                        for w in words[lo:i] + words[i+1:hi]:
                            if w.upos == 'ADJ':
                                name_adjs[(w.lemma or w.text).lower()] += 1

    for sent in sentences_containing(text, all_anchors):
        batch.append(sent)
        if len(batch) >= batch_size:
            process_batch(batch)
            batch = []
    if batch:
        process_batch(batch)

    return a_verbs, b_verbs, name_obj, name_adjs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    cfg = config['cross_corpus_dependency']
    study = config['study']
    primary_id, comparison_id = study['single_work_corpus'], study['comparison_corpus']
    primary_lang    = config['corpora'][primary_id]['language']
    comparison_lang = config['corpora'][comparison_id]['language']
    primary_label     = corpus_label(config, primary_id)
    comparison_label  = corpus_label(config, comparison_id)
    a_label = cfg.get('agent_a_label', 'Agent A')
    b_label = cfg.get('agent_b_label', 'Agent B')
    stopwords = load_stopwords(config, primary_lang)

    print('Loading spaCy model…')
    try:
        nlp_primary = load_spacy(config, primary_lang)
    except OSError as e:
        print(f'ERROR: {e}'); return

    if not _STANZA_AVAILABLE:
        print('ERROR: stanza is required. Install with: pip install stanza')
        return

    print('Loading Stanza pipeline…')
    stanza_model = config['languages'][comparison_lang]['stanza_model']
    try:
        nlp_comparison = stanza.Pipeline(stanza_model,
                                         processors='tokenize,lemma,pos,depparse',
                                         verbose=False)
    except Exception as e:
        print(f'ERROR: {e}'); return

    primary_text    = load_source_corpus(config, primary_id)
    comparison_text = load_source_corpus(config, comparison_id)

    a_p, b_subj_p, b_obj_p, b_adjs_p = \
        process_primary(nlp_primary, primary_text, cfg, stopwords)
    a_c, b_c, name_obj_c, name_adjs_c = \
        process_comparison(nlp_comparison, comparison_text, cfg)

    label_p = f'{primary_label} ({primary_lang} lemma)'
    label_c = f'{comparison_label} ({comparison_lang} lemma)'

    # --- Agent A's verbs ---
    save_csv(os.path.join(results_dir, 'crossdep_agent_a_verbs.csv'),
             a_p, a_c, label_p, label_c)
    plot_comparison(a_p, a_c, f'{primary_label} — {a_label}', f'{comparison_label} — {a_label}',
                    f"{a_label}'s verbs: {primary_label} vs. {comparison_label}",
                    os.path.join(viz_dir, 'crossdep_agent_a_verbs.png'))
    print(f'  Saved {a_label} verbs')

    # --- Agent B verbs (subject) ---
    save_csv(os.path.join(results_dir, 'crossdep_agent_b_verbs.csv'),
             b_subj_p, b_c, f'{primary_label} – {b_label} (subject)', f'{comparison_label} – {b_label} (subject)')
    plot_comparison(b_subj_p, b_c, f'{primary_label} — {b_label} subj.', f'{comparison_label} — {b_label} subj.',
                    f'{b_label} as subject: {primary_label} vs. {comparison_label}',
                    os.path.join(viz_dir, 'crossdep_agent_b_verbs.png'))
    print(f'  Saved {b_label} verbs')

    # --- What is done TO agent B / the studied person's name ---
    save_csv(os.path.join(results_dir, 'crossdep_agent_b_as_object.csv'),
             b_obj_p, name_obj_c, f'{primary_label} – {b_label} (object)', f'{comparison_label} – name (object)')
    plot_comparison(b_obj_p, name_obj_c, f'{primary_label} — {b_label} obj.', f'{comparison_label} — name obj.',
                    f'Done TO {b_label.lower()} / studied person: {primary_label} vs. {comparison_label}',
                    os.path.join(viz_dir, 'crossdep_agent_b_object.png'))
    print(f'  Saved {b_label}-as-object verbs')

    # --- Adjectives near agent B (primary) vs. the name (comparison) ---
    for term in cfg.get('primary', {}).get('adj_filter', []):
        b_adjs_p.pop(term, None)
    for term in cfg.get('comparison', {}).get('adj_filter', []):
        name_adjs_c.pop(term, None)

    save_csv(os.path.join(results_dir, 'crossdep_name_adj.csv'),
             b_adjs_p, name_adjs_c,
             f'{primary_label} – adj. near {b_label.lower()}', f'{comparison_label} – adj. near name')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, counter, title, color in [
        (axes[0], b_adjs_p, f'Adj. near {b_label.lower()} — {primary_label}', '#4c72b0'),
        (axes[1], name_adjs_c, f'Adj. near name — {comparison_label}', '#c44e52'),
    ]:
        top = counter.most_common(14)
        if top:
            words, counts = zip(*top)
            ax.barh(list(words)[::-1], list(counts)[::-1], color=color, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel('Count')

    fig.suptitle(f'How {b_label.lower()} / the studied person is described: '
                 'self-authorship vs. comparison text', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(viz_dir, 'crossdep_adj_name_vs_agent_b.png'), dpi=150)
    plt.close(fig)
    print('  Saved adjective comparison')

    # --- Console summary ---
    print(f"\n=== {a_label}'s top verbs ===")
    print(f'  {primary_label:35s}  {comparison_label:35s}')
    for (w1, c1), (w2, c2) in zip(a_p.most_common(8), a_c.most_common(8)):
        print(f'  {w1}({c1}){" "*(33-len(w1)-len(str(c1)))}  {w2}({c2})')

    print(f'\n=== Adj. near {b_label.lower()} ({primary_label}) vs. name ({comparison_label}) ===')
    print(f'  {primary_label:35s}  {comparison_label:35s}')
    for (w1, c1), (w2, c2) in zip(b_adjs_p.most_common(8), name_adjs_c.most_common(8)):
        print(f'  {w1}({c1}){" "*(33-len(w1)-len(str(c1)))}  {w2}({c2})')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    analyze(config)


if __name__ == '__main__':
    main()
