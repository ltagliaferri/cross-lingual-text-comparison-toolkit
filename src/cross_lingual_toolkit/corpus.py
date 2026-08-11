"""
Generic corpus loading, tokenization, and config access.

Every path, stopword list, and language model name is driven by a study
config JSON (default: config/study.json — this toolkit's worked example,
Catherine of Siena vs. Raymond of Capua, Italian vs. Latin). Point a
script's --config flag at your own config file, following the same
schema, to reuse the toolkit on a different corpus and language pair.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))            # .../src/cross_lingual_toolkit
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))           # repo root
_CONFIG_DIR = os.path.join(_REPO_ROOT, 'config')
_STOPWORDS_DIR = os.path.join(_REPO_ROOT, 'stopwords')
DEFAULT_CONFIG_PATH = os.path.join(_CONFIG_DIR, 'study.json')

# Default corpus root: the repo root (i.e. the directory containing the
# corpus paths referenced by config['corpora'][...]['path'], unless
# overridden via config['corpus_root'] or --corpus-root).
DEFAULT_CORPUS_ROOT = _REPO_ROOT


def load_config(config_path=None):
    """
    Load a study config JSON. Defaults to this repo's bundled example
    config — only present when running from a clone of this repo, not
    when the package is installed elsewhere, so pass --config explicitly
    in that case.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No config file at "{path}". Pass --config path/to/your_study.json '
            '(see config/template_study.json in the repo for a starting point).'
        )
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def add_config_args(parser=None):
    """Add --config/--corpus-root flags to (or create) an argparse parser."""
    parser = parser or argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='Path to a study config JSON (default: bundled example study)')
    parser.add_argument('--corpus-root', default=None,
                        help='Override the corpus root directory referenced by the config')
    return parser


def output_dirs(config):
    """Return (results_dir, viz_dir, embed_dir) absolute paths for this config."""
    out = config.get('output', {})
    return (
        os.path.join(_REPO_ROOT, out.get('results_dir', 'results')),
        os.path.join(_REPO_ROOT, out.get('visualizations_dir', 'visualizations')),
        os.path.join(_REPO_ROOT, out.get('embeddings_dir', 'embeddings')),
    )


def ensure_output_dirs(config):
    results_dir, viz_dir, embed_dir = output_dirs(config)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    return results_dir, viz_dir, embed_dir


def corpus_root(config, override=None):
    return override or config.get('corpus_root') or DEFAULT_CORPUS_ROOT


def load_stopwords(config, language):
    """
    Load the stopword set configured for `language` (empty set if none
    configured). `stopwords_file` is a filename resolved against the
    top-level stopwords/ directory, not the config directory.
    """
    lang_cfg = config.get('languages', {}).get(language, {})
    rel_path = lang_cfg.get('stopwords_file')
    if not rel_path:
        return set()
    path = os.path.join(_STOPWORDS_DIR, rel_path)
    with open(path, encoding='utf-8') as f:
        return {line.strip() for line in f
                if line.strip() and not line.startswith('#')}


def keep_short_tokens(config, language):
    """
    Return the set of single-character tokens configured as meaningful for
    `language` (e.g. Italian 'e', 'o', 'a', 'è' — all real words) and thus
    exempt from tokenize()'s length filter. Empty by default.
    """
    lang_cfg = config.get('languages', {}).get(language, {})
    return set(lang_cfg.get('keep_short_tokens', []))


def adjective_suffixes(config, language):
    """
    Return the tuple of adjective-morphology suffixes configured for
    `language`, used by adjective_window.py as a fallback POS heuristic when
    spaCy is unavailable. Empty by default (heuristic finds nothing).
    """
    lang_cfg = config.get('languages', {}).get(language, {})
    return tuple(lang_cfg.get('adjective_suffixes', []))


def spacy_model_names(config, language):
    """Return (primary, fallback) spaCy model names for `language`."""
    lang_cfg = config.get('languages', {}).get(language, {})
    return lang_cfg.get('spacy_model'), lang_cfg.get('spacy_model_fallback')


def load_spacy(config, language):
    """Load the configured spaCy model for `language`, falling back if needed."""
    import spacy
    primary, fallback = spacy_model_names(config, language)
    try:
        nlp = spacy.load(primary)
        print(f'Loaded spaCy {primary}')
        return nlp
    except OSError:
        if fallback:
            try:
                nlp = spacy.load(fallback)
                print(f'Loaded spaCy {fallback} (fallback)')
                return nlp
            except OSError:
                pass
        raise OSError(
            f'No spaCy model found for language "{language}". '
            f'Run: python -m spacy download {primary}'
        )


def load_single_file_corpus(config, corpus_id, corpus_root_override=None):
    """Load a single-file corpus (e.g. a treatise). Returns raw text."""
    entry = config['corpora'][corpus_id]
    root = corpus_root(config, corpus_root_override)
    path = os.path.join(root, entry['path'])
    with open(path, encoding='utf-8') as f:
        text = f.read()

    marker = entry.get('strip_after_marker')
    if marker:
        lines = text.splitlines(keepends=True)
        cutoff = next((i for i, ln in enumerate(lines) if ln.strip() == marker),
                       len(lines))
        text = ''.join(lines[:cutoff])
    return text


def load_collection_corpus(config, corpus_id, corpus_root_override=None):
    """
    Load a directory-of-dated-items corpus (e.g. letters grouped into
    volumes). Returns a list of dicts:
      {group, group_num, item_num, path, text}
    """
    entry = config['corpora'][corpus_id]
    root = os.path.join(corpus_root(config, corpus_root_override), entry['path'])
    groups = entry.get('groups') or [None]

    items = []
    for group_num, group in enumerate(groups, 1):
        group_dir = os.path.join(root, group) if group else root
        for fpath in sorted(glob.glob(os.path.join(group_dir, '*.txt'))):
            with open(fpath, encoding='utf-8') as f:
                text = f.read()
            item_num = os.path.splitext(os.path.basename(fpath))[0]
            items.append({
                'group':      group or entry['path'],
                'group_num':  group_num,
                'item_num':   item_num,
                'path':       fpath,
                'text':       text,
            })
    return items


def load_source_corpus(config, corpus_id, corpus_root_override=None):
    """Load any configured source corpus by id, dispatching on its declared type."""
    entry = config['corpora'][corpus_id]
    if entry['type'] == 'single_file':
        return load_single_file_corpus(config, corpus_id, corpus_root_override)
    if entry['type'] == 'collection':
        return load_collection_corpus(config, corpus_id, corpus_root_override)
    raise ValueError(f'Unknown corpus type "{entry["type"]}" for corpus "{corpus_id}"')


def corpus_label(config, corpus_id):
    entry = config['corpora'][corpus_id]
    return entry.get('label', corpus_id.capitalize())


def tokenize(text, remove_stopwords=True, stopwords=None, keep_short=None):
    """
    Lowercase, strip punctuation, split on whitespace.
    Handles apostrophe elisions by splitting on them.
    `keep_short` is a set of single-character tokens to retain despite the
    length filter — language-specific, see keep_short_tokens().
    Returns list of tokens.
    """
    text = text.lower()
    # split elisions: "l'anima" -> "l anima"
    text = re.sub(r"[''']", ' ', text)
    # remove punctuation except hyphens within words
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    tokens = text.split()
    # remove pure digits and single characters, except configured exceptions
    keep_short = keep_short or ()
    tokens = [t for t in tokens if len(t) > 1 or t in keep_short]
    if remove_stopwords and stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def word_freq(tokens):
    """Return Counter of token frequencies."""
    return Counter(tokens)
