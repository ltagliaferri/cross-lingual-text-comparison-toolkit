"""
Speaker segmentation for an alternating-speaker dialogic text.

The configured corpus (config['speaker_segmentation']['corpus']) alternates
between two speakers plus narrative passages. This script uses a
state-machine over paragraphs: a paragraph beginning with ": -"
(attribution + colon + dash) triggers a speaker switch, and the
attribution phrase before the colon identifies who is speaking, matched
against configured cue phrases for each speaker.

This convention (and the cue phrases / narrative-transition markers) is
specific to the text this was built for — the *Dialogo* of Catherine of
Siena — and will need re-tuning in config for a different text's dialogue
formatting.

Outputs:
  results/[speaker]_speech.txt     – each speaker's concatenated speech
  results/speaker_stats.csv        – word counts, type counts, top-20 content words
  visualizations/speaker_terms.png – top-15 content words per speaker (bar charts)
"""

import os
import re
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .corpus import (load_config, add_config_args, ensure_output_dirs,
                    load_source_corpus, load_stopwords, tokenize, keep_short_tokens)


def classify_attribution(text, cfg):
    """Return a speaker key, or None, given an attribution phrase."""
    t = text.lower()
    for key in ('speaker_2', 'speaker_1'):
        for cue in cfg.get(f'{key}_cues', []):
            if cue in t:
                return key
    return None


def segment(text, cfg):
    """
    Split the text into speaker segments, using the state machine described
    in the module docstring. Returns dict {speaker_key: [paragraphs]}.
    """
    paras = [p.strip() for p in text.split('\n') if p.strip()]
    speakers = ('speaker_1', 'speaker_2', 'narrative')
    segments = {key: [] for key in speakers}
    current_speaker = cfg.get('default_speaker', 'speaker_1')
    in_speech_block = False

    narrative_starts = re.compile(cfg['narrative_starts_pattern'], re.IGNORECASE)
    heading_prefix = cfg.get('chapter_heading_prefix')
    heading_max_len = cfg.get('chapter_heading_max_len', 400)

    for para in paras:
        # 1. Chapter headings reset to narrative context
        if heading_prefix and para.startswith(heading_prefix) and len(para) < heading_max_len:
            segments['narrative'].append(para)
            in_speech_block = False
            current_speaker = cfg.get('default_speaker', 'speaker_1')
            continue

        # 2. Mixed paragraph with attribution + speech: "Text: - Speech"
        speech_intro = re.search(r':\s{0,3}-\s', para)
        if speech_intro:
            split_pos   = speech_intro.start()
            attribution = para[:split_pos]
            speech_body = para[speech_intro.end():]

            speaker = classify_attribution(attribution, cfg)
            if speaker:
                current_speaker = speaker

            if attribution.strip():
                segments['narrative'].append(attribution.strip())
            if speech_body.strip():
                segments[current_speaker].append(speech_body.strip())
                in_speech_block = True
            continue

        # 3. Paragraph that starts with ' - ': direct speech paragraph
        if re.match(r'^\s*[-–]\s', para):
            clean = re.sub(r'^\s*[-–]\s*', '', para).strip()
            if clean:
                segments[current_speaker].append(clean)
            in_speech_block = True
            continue

        # 4. Plain paragraph (no leading dash, no attribution+dash)
        if in_speech_block and not narrative_starts.match(para):
            segments[current_speaker].append(para)
        else:
            segments['narrative'].append(para)
            in_speech_block = False

    return segments


def top_words(paragraphs, stopwords, keep_short, n=20):
    """Return top-n content words across a list of paragraphs."""
    tokens = tokenize(' '.join(paragraphs), stopwords=stopwords, keep_short=keep_short)
    return Counter(tokens).most_common(n)


def analyze(config):
    results_dir, viz_dir, _ = ensure_output_dirs(config)
    cfg = config['speaker_segmentation']
    corpus_id = cfg['corpus']
    language = config['corpora'][corpus_id]['language']
    stopwords = load_stopwords(config, language)
    keep_short = keep_short_tokens(config, language)
    labels = cfg['speaker_labels']
    colors = cfg['colors']
    speakers = ('speaker_1', 'speaker_2', 'narrative')

    print(f'Loading and segmenting {corpus_id}…')
    text = load_source_corpus(config, corpus_id)
    segs = segment(text, cfg)

    # --- Save plain-text files ---
    for key in speakers:
        path = os.path.join(results_dir, f'{key}_speech.txt')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(segs[key]))
        print(f'  Saved {path}')

    # --- Stats CSV ---
    csv_path = os.path.join(results_dir, 'speaker_stats.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['speaker', 'paragraphs', 'total_tokens',
                         'unique_tokens', 'ttr', 'top_20_content_words'])
        for key in speakers:
            combined = ' '.join(segs[key])
            all_tokens  = tokenize(combined, remove_stopwords=False, keep_short=keep_short)
            top = '; '.join(f'{w}({c})' for w, c in top_words(segs[key], stopwords, keep_short))
            ttr = round(len(set(all_tokens)) / len(all_tokens), 4) if all_tokens else 0
            writer.writerow([key, len(segs[key]), len(all_tokens),
                             len(set(all_tokens)), ttr, top])
    print(f'  Saved {csv_path}')

    # --- Visualization: top-15 words per speaker ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, key in zip(axes, speakers):
        top = top_words(segs[key], stopwords, keep_short, n=15)
        words, counts = zip(*top) if top else ([], [])
        ax.barh(list(words)[::-1], list(counts)[::-1],
                color=colors[key], alpha=0.85)
        n_paras = len(segs[key])
        n_tok   = len(tokenize(' '.join(segs[key]), remove_stopwords=False, keep_short=keep_short))
        ax.set_xlabel('Token count')
        ax.set_title(f'{labels[key]}\n({n_paras} paragraphs, {n_tok:,} tokens)')

    fig.suptitle(f'Top 15 content words by speaker — {corpus_id}', fontsize=13)
    fig.tight_layout()
    p = os.path.join(viz_dir, 'speaker_terms.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f'  Saved {p}')

    # --- Summary ---
    print('\nSegmentation summary:')
    for key in speakers:
        combined = ' '.join(segs[key])
        toks = tokenize(combined, remove_stopwords=False, keep_short=keep_short)
        print(f'  {labels[key]:25s}  {len(segs[key]):4d} paragraphs  {len(toks):7,} tokens')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    analyze(config)


if __name__ == '__main__':
    main()
