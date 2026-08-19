"""
Generic corpus loading, tokenization, and config access.

Every path, stopword list, and language model name is driven by a study
config JSON. Point a script's --config flag at your own config file,
following the schema described in the README, to reuse the toolkit on a
different corpus and language pair.

Path resolution is deliberately install-agnostic, so the toolkit behaves
the same run from a clone as from a plain `pip install`:

  * outputs go under the current working directory (or --output-root),
  * corpus paths resolve against config['corpus_root'] / --corpus-root,
  * stopword files are searched for next to the config file, under the
    working directory, and finally among the lists bundled with the
    package.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))            # .../cross_lingual_toolkit
_BUNDLED_STOPWORDS_DIR = os.path.join(_HERE, 'stopwords')     # ships with the package

# Only meaningful when running from a source clone; used as a last-resort
# config location, never for reading or writing study data.
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, 'config', 'study.json')


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def config_entries(section):
    """
    Return a config section's real entries, dropping "_"-prefixed keys.

    Sections like term_frequency_clusters are keyed by the study's own
    vocabulary, so there is no fixed schema to distinguish data from the
    "_comment" annotations the template config carries. The convention is
    that a leading underscore marks a comment, everywhere a section is
    iterated as data.
    """
    if not section:
        return {}
    return {k: v for k, v in section.items() if not k.startswith('_')}


def _config_candidates(config_path=None):
    if config_path:
        return [config_path]
    return [os.path.join(os.getcwd(), 'config', 'study.json'), DEFAULT_CONFIG_PATH]


def load_config(config_path=None):
    """
    Load a study config JSON. With no explicit path, looks for
    config/study.json under the working directory, then in a source clone
    of this repo. Records the resolved path in the returned config so
    stopword files can be found relative to it.
    """
    candidates = _config_candidates(config_path)
    for path in candidates:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                config = json.load(f)
            config['_config_path'] = os.path.abspath(path)
            return config

    tried = '\n  '.join(candidates)
    raise FileNotFoundError(
        f'No config file found. Tried:\n  {tried}\n'
        'Pass --config path/to/your_study.json '
        '(see config/template_study.json in the repo for a starting point).'
    )


def add_config_args(parser=None):
    """Add --config/--corpus-root/--output-root flags to (or create) a parser."""
    parser = parser or argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help='Path to a study config JSON '
                             '(default: ./config/study.json)')
    parser.add_argument('--corpus-root', default=None,
                        help='Override the corpus root directory referenced by the config')
    parser.add_argument('--output-root', default=None,
                        help='Directory to write results/, visualizations/ and '
                             'embeddings/ into (default: the working directory)')
    return parser


def load_config_from_args(args, validate=True):
    """Load the config named by parsed CLI args and apply the path overrides."""
    config = load_config(args.config)
    if getattr(args, 'corpus_root', None):
        config['corpus_root'] = args.corpus_root
    if getattr(args, 'output_root', None):
        config['_output_root'] = args.output_root
    if validate:
        validate_config(config)
    return config


def validate_config(config):
    """
    Check the structural invariants every script depends on, so a typo
    surfaces as one clear message instead of a KeyError deep in an
    analysis. Returns a list of non-fatal warnings; raises ValueError on
    anything that would certainly break.
    """
    problems, warnings = [], []

    corpora = config_entries(config.get('corpora'))
    if not corpora:
        problems.append('config has no "corpora" section (or it is empty)')

    for corpus_id, entry in corpora.items():
        for key in ('type', 'path', 'author', 'work', 'language'):
            if key not in entry:
                problems.append(f'corpora["{corpus_id}"] is missing required key "{key}"')
        if entry.get('type') not in (None, 'single_file', 'collection'):
            problems.append(
                f'corpora["{corpus_id}"]["type"] is "{entry["type"]}"; '
                'expected "single_file" or "collection"'
            )

    study = config.get('study', {})
    for role in ('single_work_corpus', 'collection_corpus', 'comparison_corpus'):
        corpus_id = study.get(role)
        if corpus_id and corpus_id not in corpora:
            problems.append(f'study["{role}"] refers to unknown corpus "{corpus_id}"')
    for corpus_id in study.get('primary_corpora', []):
        if corpus_id not in corpora:
            problems.append(f'study["primary_corpora"] refers to unknown corpus "{corpus_id}"')

    declared = {entry['language'] for entry in corpora.values() if 'language' in entry}
    for role in ('primary_language', 'comparison_language'):
        lang = study.get(role)
        if lang and lang not in declared:
            problems.append(
                f'study["{role}"] is "{lang}", which no corpus declares as its language '
                f'(corpora use: {", ".join(sorted(declared)) or "none"})'
            )

    languages = config_entries(config.get('languages'))
    for lang in sorted(declared):
        if lang not in languages:
            warnings.append(
                f'language "{lang}" has no entry in "languages" — it will get no '
                'stopwords and no spaCy/Stanza model'
            )

    for cluster_key, cluster in config_entries(
            config.get('cross_corpus_concept_clusters')).items():
        for lang in sorted(declared):
            if lang in languages and lang not in cluster:
                warnings.append(
                    f'cross_corpus_concept_clusters["{cluster_key}"] has no '
                    f'"{lang}" forms'
                )

    if problems:
        raise ValueError('Invalid study config:\n  - ' + '\n  - '.join(problems))
    for warning in warnings:
        print(f'WARNING: {warning}')
    return warnings


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def output_root(config):
    """
    Base directory for generated output. Explicit config/CLI setting wins;
    otherwise the working directory — never the install location, which
    for a non-editable install is inside site-packages.
    """
    return (config.get('output', {}).get('root')
            or config.get('_output_root')
            or os.getcwd())


def output_dirs(config):
    """Return (results_dir, viz_dir, embed_dir) absolute paths for this config."""
    root = output_root(config)
    out = config.get('output', {})
    return (
        os.path.join(root, out.get('results_dir', 'results')),
        os.path.join(root, out.get('visualizations_dir', 'visualizations')),
        os.path.join(root, out.get('embeddings_dir', 'embeddings')),
    )


def ensure_output_dirs(config):
    results_dir, viz_dir, embed_dir = output_dirs(config)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(embed_dir, exist_ok=True)
    return results_dir, viz_dir, embed_dir


def corpus_root(config, override=None):
    return override or config.get('corpus_root') or os.getcwd()


def stopwords_search_dirs(config):
    """
    Directories searched, in order, for a configured `stopwords_file`:
    alongside the config file, under the working directory, then the
    lists bundled with the package.
    """
    dirs = []
    config_path = config.get('_config_path')
    if config_path:
        config_dir = os.path.dirname(config_path)
        dirs += [
            os.path.join(config_dir, 'stopwords'),
            os.path.join(os.path.dirname(config_dir), 'stopwords'),
            config_dir,
        ]
    cwd = os.getcwd()
    dirs += [os.path.join(cwd, 'stopwords'), cwd, _BUNDLED_STOPWORDS_DIR]

    seen, unique = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


def resolve_stopwords_path(config, rel_path):
    """Resolve a configured stopwords_file to an existing path, or raise."""
    if os.path.isabs(rel_path):
        if os.path.exists(rel_path):
            return rel_path
        raise FileNotFoundError(f'Stopword file not found: {rel_path}')

    tried = []
    for directory in stopwords_search_dirs(config):
        candidate = os.path.join(directory, rel_path)
        tried.append(candidate)
        if os.path.exists(candidate):
            return candidate

    listing = '\n  '.join(tried)
    raise FileNotFoundError(
        f'Stopword file "{rel_path}" not found. Tried:\n  {listing}\n'
        'Use an absolute path, or put the list in a "stopwords" directory '
        'next to your config file.'
    )


# ---------------------------------------------------------------------------
# Language settings
# ---------------------------------------------------------------------------

def load_stopwords(config, language):
    """
    Load the stopword set configured for `language` (empty set if none
    configured). See resolve_stopwords_path() for where files are looked
    for. Blank lines and comment lines (leading "#", indented or not) are
    ignored.
    """
    lang_cfg = config.get('languages', {}).get(language, {})
    rel_path = lang_cfg.get('stopwords_file')
    if not rel_path:
        return set()
    path = resolve_stopwords_path(config, rel_path)
    with open(path, encoding='utf-8') as f:
        return {line.strip() for line in f
                if line.strip() and not line.strip().startswith('#')}


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
    try:
        import spacy
    except ImportError:
        raise ImportError(
            'spaCy is required for this analysis. Install it with: '
            'pip install "cross-lingual-toolkit[parsing]"'
        )
    primary, fallback = spacy_model_names(config, language)
    if not primary and not fallback:
        raise OSError(
            f'No spaCy model configured for language "{language}". '
            f'Set languages.{language}.spacy_model in your config.'
        )
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


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize(text, remove_stopwords=True, stopwords=None, keep_short=None):
    """
    Lowercase, then split on everything that is not a Unicode letter:
    punctuation, digits, underscores and hyphens are all separators, so
    "mid-word" becomes two tokens and "1300" drops out entirely.
    Apostrophe elisions split the same way ("l'amico" -> "l", "amico").

    Single-character tokens are dropped unless listed in `keep_short` —
    language-specific, see keep_short_tokens().

    Returns list of tokens.
    """
    text = text.lower()
    # any run of non-letters is a separator (Unicode-aware, so accented
    # letters survive); this drops digits and underscores along with
    # punctuation, keeping every corpus on the same footing.
    tokens = [t for t in re.split(r'[\W\d_]+', text, flags=re.UNICODE) if t]
    keep_short = keep_short or ()
    tokens = [t for t in tokens if len(t) > 1 or t in keep_short]
    if remove_stopwords and stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def word_freq(tokens):
    """Return Counter of token frequencies."""
    return Counter(tokens)
