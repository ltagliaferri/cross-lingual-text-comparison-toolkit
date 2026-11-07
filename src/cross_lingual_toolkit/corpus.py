"""
Generic corpus loading, tokenization, and config access.

Every path, stopword list, and language model name is driven by a study
config JSON. Point a script's --config flag at your own config file,
following the schema described in the README, to reuse the toolkit on a
different corpus and language pair.

Path resolution is deliberately install-agnostic, so the toolkit behaves
the same run from a clone as from a plain `pip install`:

  * outputs go under --output-root, else config['output']['root'], else
    the current working directory,
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
import sys
import unicodedata
from collections import Counter

from .metrics import normalize_term   # re-exported, and used by tokenize()

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
            # utf-8-sig: a byte-order mark (common from Windows editors)
            # is skipped instead of breaking json.load
            with open(path, encoding='utf-8-sig') as f:
                config = json.load(f)
            if not isinstance(config, dict):
                raise ValueError(f'{path}: a study config must be a JSON object, '
                                 f'got {type(config).__name__}')
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


def load_config_from_args(args, validate=True, require=None, check_paths=True):
    """
    Load the config named by parsed CLI args and apply the path overrides.
    `require` / `check_paths` are passed to validate_config().
    """
    config = load_config(args.config)
    corpus_root_arg = getattr(args, 'corpus_root', None)
    if corpus_root_arg is not None:
        if not corpus_root_arg.strip():
            raise ValueError('--corpus-root was given an empty path')
        config['corpus_root'] = corpus_root_arg
    if getattr(args, 'output_root', None):
        config['_output_root'] = args.output_root
    if validate:
        validate_config(config, require=require, check_paths=check_paths)
    return config


# Study roles a script can require. A role written "role:type" (e.g.
# "single_work_corpus:single_file") also requires that corpus type.
STUDY_ROLES = ('single_work_corpus', 'collection_corpus', 'comparison_corpus',
               'primary_corpora', 'primary_language', 'comparison_language')

# Positive-integer settings: (section, key).
_POSITIVE_INT_SETTINGS = (
    ('lexical_richness', 'mattr_window'),
    ('lexical_richness', 'smoothing_window'),
    ('analysis', 'n_permutations'),
    ('adjective_window', 'window'),
    ('dependency_parsing', 'chunk_chars'),
    ('cross_corpus_dependency', 'window'),
    ('cross_corpus_dependency', 'chunk_chars'),
    ('chunking', 'max_chars'),
)

# 0 means "keep every chunk / item" (or, for the split threshold, "chunk
# every item"), so only negatives are rejected here.
_NON_NEGATIVE_INT_SETTINGS = (
    ('chunking', 'min_chars'),
    ('chunking', 'collection_item_min_chars'),
    ('chunking', 'collection_item_split_threshold'),
)

# Settings with a higher floor: (section, key, minimum, why).
_MINIMUM_INT_SETTINGS = (
    # HDBSCAN's min_cluster_size must be at least 2; 1 passes BERTopic's
    # own checks and fails at fit time
    ('topic_modeling', 'min_topic_size', 2),
)

# Settings sections that must be JSON objects when present.
_SETTINGS_SECTIONS = ('output', 'chunking', 'lexical_richness', 'analysis',
                      'adjective_window', 'dependency_parsing', 'topic_modeling')


def _check_list(value, where, problems):
    """A list whose items are all non-empty (after stripping) strings."""
    if not isinstance(value, list):
        problems.append(f'{where} must be a list of strings, got {type(value).__name__}')
        return False
    # normalize_term() is what every reader applies, so a string that it
    # empties (whitespace, or only soft hyphens) is empty here too
    bad = [item for item in value
           if not isinstance(item, str) or not normalize_term(item)]
    if bad:
        problems.append(f'{where} must contain only non-empty strings, got '
                        + ', '.join(repr(item) for item in bad))
        return False
    return True


def _dict_entries(section):
    """config_entries() of a section, keeping only object-valued entries."""
    if not isinstance(section, dict):
        return {}
    return {k: v for k, v in config_entries(section).items() if isinstance(v, dict)}


def _check_section(config, name, langs, problems):
    """Per-section shape checks for the sections a script requires."""
    section = config.get(name)
    if section is None:
        problems.append(f'config has no "{name}" section, which this script needs')
        return
    if not isinstance(section, dict):
        problems.append(f'"{name}" must be an object')
        return
    entries = config_entries(section)

    if name in ('term_frequency_clusters', 'cross_corpus_concept_clusters',
                'dependency_agents', 'adjective_anchor_terms') and not entries:
        problems.append(f'"{name}" defines no entries '
                        '(keys starting with "_" are treated as comments)')

    if name == 'term_frequency_clusters':
        for key, cluster in entries.items():
            where = f'{name}["{key}"]'
            if not isinstance(cluster, dict):
                problems.append(f'{where} must be an object with "label" and "terms"')
                continue
            if 'label' not in cluster:
                problems.append(f'{where} is missing "label"')
            if 'terms' not in cluster:
                problems.append(f'{where} is missing "terms"')
            else:
                _check_list(cluster['terms'], f'{where}["terms"]', problems)

    elif name == 'cross_corpus_concept_clusters':
        for key, cluster in entries.items():
            where = f'{name}["{key}"]'
            if not isinstance(cluster, dict):
                problems.append(f'{where} must be an object with "label" and per-language forms')
                continue
            if 'label' not in cluster:
                problems.append(f'{where} is missing "label"')
            for lang in langs:
                if lang not in cluster:
                    problems.append(f'{where} has no "{lang}" forms')
                else:
                    _check_list(cluster[lang], f'{where}["{lang}"]', problems)

    elif name == 'adjective_anchor_terms':
        for key, forms in entries.items():
            _check_list(forms, f'{name}["{key}"]', problems)

    elif name == 'dependency_agents':
        for key, agent in entries.items():
            where = f'{name}["{key}"]'
            if not isinstance(agent, dict):
                problems.append(f'{where} must be an object with "label", "lemmas" and "role"')
                continue
            for k in ('label', 'lemmas', 'role'):
                if k not in agent:
                    problems.append(f'{where} is missing "{k}"')
            if 'lemmas' in agent:
                _check_list(agent['lemmas'], f'{where}["lemmas"]', problems)

    elif name == 'cross_corpus_dependency':
        for side, keys in (('primary', ('agent_a_lemmas', 'agent_b_lemmas')),
                           ('comparison', ('agent_a_forms', 'agent_b_forms', 'name_forms'))):
            sub = section.get(side)
            where = f'{name}["{side}"]'
            if not isinstance(sub, dict):
                problems.append(f'{where} is missing (or not an object)')
                continue
            for k in keys:
                if k not in sub:
                    problems.append(f'{where} is missing "{k}"')
                else:
                    _check_list(sub[k], f'{where}["{k}"]', problems)
            if 'adj_filter' in sub:
                _check_list(sub['adj_filter'], f'{where}["adj_filter"]', problems)


def _check_corpus_on_disk(config, corpus_id, problems, warnings):
    """Check that a corpus's path exists and matches its declared type."""
    entry = _dict_entries(config.get('corpora')).get(corpus_id) or {}
    if not isinstance(entry.get('path'), str) or \
            entry.get('type') not in ('single_file', 'collection'):
        return   # already reported by the structural checks
    path = os.path.join(corpus_root(config), entry['path'])
    where = f'corpora["{corpus_id}"]'
    if entry['type'] == 'single_file':
        if os.path.isdir(path):
            problems.append(f'{where} is a single_file but its path is a directory: {path}')
        elif not os.path.isfile(path):
            problems.append(f'{where}: file not found: {path} '
                            '(set corpus_root or pass --corpus-root)')
        return

    if not os.path.isdir(path):
        kind = 'a file, not a directory' if os.path.isfile(path) else 'not found'
        problems.append(f'{where} is a collection but its path is {kind}: {path} '
                        '(set corpus_root or pass --corpus-root)')
        return
    groups = entry.get('groups')
    if groups and isinstance(groups, list):   # wrong types reported earlier
        for group in (g for g in groups if isinstance(g, str)):
            if not os.path.isdir(os.path.join(path, group)):
                problems.append(f'{where}["groups"] lists "{group}", but there is no '
                                f'subdirectory {os.path.join(path, group)}')
    elif not glob.glob(os.path.join(path, '*.txt')):
        subdirs = sorted(d for d in os.listdir(path)
                         if os.path.isdir(os.path.join(path, d)))
        if subdirs:
            warnings.append(
                f'{where} has no .txt files directly in {path}, but has '
                f'subdirectories ({", ".join(subdirs[:5])}) — did you mean to list '
                'them in "groups"?'
            )


def _check_requirements(config, require, check_paths, problems, warnings):
    # wrong-typed containers were reported by validate_config(); skip them
    corpora = _dict_entries(config.get('corpora'))
    study = config.get('study') if isinstance(config.get('study'), dict) else {}
    languages = _dict_entries(config.get('languages'))
    used_corpora = []
    sections = []

    for item in require:
        name, _, wanted_type = item.partition(':')
        if name == 'corpora':
            used_corpora.extend(corpora)
            continue
        if name not in STUDY_ROLES:
            sections.append(name)
            continue
        value = study.get(name)
        if not value:
            problems.append(f'study["{name}"] is not set, but this script needs it')
            continue
        if name == 'primary_corpora':
            if not isinstance(value, list):
                problems.append('study["primary_corpora"] must be a list of corpus ids')
                continue
            used_corpora.extend(c for c in value if isinstance(c, str) and c in corpora)
        elif name.endswith('_corpus') and isinstance(value, str) and value in corpora:
            used_corpora.append(value)
            actual = corpora[value].get('type')
            if wanted_type and actual and actual != wanted_type:
                problems.append(f'study["{name}"] is "{value}", a {actual} corpus; '
                                f'this script needs a {wanted_type}')

    # languages of the two sides of a cross-corpus comparison
    cross_langs = []
    def role_language(role):
        corpus_id = study.get(role)
        if not isinstance(corpus_id, str):
            return None
        lang = corpora.get(corpus_id, {}).get('language')
        return lang if isinstance(lang, str) else None

    for role in ('single_work_corpus', 'comparison_corpus'):
        lang = role_language(role)
        if lang and lang not in cross_langs:
            cross_langs.append(lang)

    for name in sections:
        _check_section(config, name, cross_langs, problems)

    if 'cross_corpus_dependency' in sections:
        comp_lang = role_language('comparison_corpus')
        if comp_lang and not languages.get(comp_lang, {}).get('stanza_model'):
            problems.append(f'languages["{comp_lang}"] has no "stanza_model", which '
                            'cross_corpus_dependency needs for the comparison corpus')

    int_settings = ([(s, k, 1, 'a positive integer') for s, k in _POSITIVE_INT_SETTINGS]
                    + [(s, k, 0, 'a non-negative integer') for s, k in _NON_NEGATIVE_INT_SETTINGS]
                    + [(s, k, m, f'an integer of at least {m}') for s, k, m in _MINIMUM_INT_SETTINGS])
    for section, key, minimum, kind in int_settings:
        sub = config.get(section)
        if isinstance(sub, dict) and key in sub:
            value = sub[key]
            # JSON parses 1e3 or 1000.0 as floats; whole numbers are fine
            # because every reader of these settings wraps the value in
            # int() (keep it that way when adding a setting here)
            whole = (isinstance(value, int) and not isinstance(value, bool)) or \
                (isinstance(value, float) and value.is_integer())
            ok = whole and value >= minimum
            if not ok:
                problems.append(f'{section}["{key}"] must be {kind}, got {value!r}')

    if check_paths:
        seen = set()
        for corpus_id in used_corpora:
            if corpus_id not in seen:
                seen.add(corpus_id)
                _check_corpus_on_disk(config, corpus_id, problems, warnings)


def validate_config(config, require=None, check_paths=True):
    """
    Check the structural invariants every script depends on, so a typo
    surfaces as one clear message instead of a KeyError deep in an
    analysis. Returns a list of non-fatal warnings; raises ValueError on
    anything that would certainly break, listing every problem at once.

    `require` lists what the calling script needs on top of that: study
    roles (STUDY_ROLES, optionally "role:corpus_type"), config sections
    (checked for shape, e.g. every cluster has a "label" and list-valued
    terms), or "corpora" for every configured corpus. With `require` set
    and `check_paths` true, the corpora those roles name are also checked
    against disk (after any --corpus-root override). With `require=None`
    only the base checks run.
    """
    problems, warnings = [], []

    def object_or_empty(key):
        """config[key] if it is an object; {} (reporting a problem) otherwise."""
        if key not in config:
            return {}
        value = config[key]
        if not isinstance(value, dict):
            problems.append(f'"{key}" must be an object, got {type(value).__name__}')
            return {}
        return value

    corpora = config_entries(object_or_empty('corpora'))
    if not corpora and isinstance(config.get('corpora', {}), dict):   # else reported
        problems.append('config has no "corpora" section (or it is empty)')

    for corpus_id, entry in list(corpora.items()):
        if not isinstance(entry, dict):
            problems.append(f'corpora["{corpus_id}"] must be an object, '
                            f'got {type(entry).__name__}')
            del corpora[corpus_id]
            continue
        for key in ('type', 'path', 'author', 'work', 'language'):
            if key not in entry:
                problems.append(f'corpora["{corpus_id}"] is missing required key "{key}"')
            elif key in ('path', 'language') and not isinstance(entry[key], str):
                problems.append(f'corpora["{corpus_id}"]["{key}"] must be a string, '
                                f'got {entry[key]!r}')
        if entry.get('type') not in (None, 'single_file', 'collection'):
            problems.append(
                f'corpora["{corpus_id}"]["type"] is {entry["type"]!r}; '
                'expected "single_file" or "collection"'
            )
        if entry.get('groups') is not None:
            _check_list(entry['groups'], f'corpora["{corpus_id}"]["groups"]', problems)

    study = object_or_empty('study')
    for role in ('single_work_corpus', 'collection_corpus', 'comparison_corpus',
                 'primary_language', 'comparison_language'):
        value = study.get(role)
        if value is not None and not isinstance(value, str):
            problems.append(f'study["{role}"] must be a string, got {value!r}')
    for role in ('single_work_corpus', 'collection_corpus', 'comparison_corpus'):
        corpus_id = study.get(role)
        if corpus_id and isinstance(corpus_id, str) and corpus_id not in corpora:
            problems.append(f'study["{role}"] refers to unknown corpus "{corpus_id}"')
    primary = study.get('primary_corpora', [])
    if _check_list(primary, 'study["primary_corpora"]', problems):
        for corpus_id in primary:
            if corpus_id not in corpora:
                problems.append(f'study["primary_corpora"] refers to unknown corpus "{corpus_id}"')

    declared = {entry['language'] for entry in corpora.values()
                if isinstance(entry.get('language'), str)}
    for role in ('primary_language', 'comparison_language'):
        lang = study.get(role)
        if lang and isinstance(lang, str) and lang not in declared:
            problems.append(
                f'study["{role}"] is "{lang}", which no corpus declares as its language '
                f'(corpora use: {", ".join(sorted(declared)) or "none"})'
            )

    languages = config_entries(object_or_empty('languages'))
    for lang, lang_cfg in list(languages.items()):
        where = f'languages["{lang}"]'
        if not isinstance(lang_cfg, dict):
            problems.append(f'{where} must be an object, got {type(lang_cfg).__name__}')
            del languages[lang]
            continue
        for key in ('keep_short_tokens', 'adjective_suffixes'):
            if key in lang_cfg:
                _check_list(lang_cfg[key], f'{where}["{key}"]', problems)
        if lang_cfg.get('stopwords_file') is not None and \
                not isinstance(lang_cfg['stopwords_file'], str):
            problems.append(f'{where}["stopwords_file"] must be a string, '
                            f'got {lang_cfg["stopwords_file"]!r}')

    for section in _SETTINGS_SECTIONS:
        object_or_empty(section)

    for lang in sorted(declared):
        if lang not in languages:
            warnings.append(
                f'language "{lang}" has no entry in "languages" — it will get no '
                'stopwords and no spaCy/Stanza model'
            )

    # (a wrong-typed section is reported by _check_section when required)
    for cluster_key, cluster in _dict_entries(
            config.get('cross_corpus_concept_clusters')).items():
        for lang in sorted(declared):
            if lang in languages and lang not in cluster:
                warnings.append(
                    f'cross_corpus_concept_clusters["{cluster_key}"] has no '
                    f'"{lang}" forms'
                )

    if require is not None:
        _check_requirements(config, require, check_paths, problems, warnings)

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
    Base directory for generated output: the CLI's --output-root beats the
    config's output.root, which beats the working directory — never the
    install location, which for a non-editable install is inside
    site-packages.
    """
    return (config.get('_output_root')
            or (config.get('output') or {}).get('root')
            or os.getcwd())


def output_dirs(config):
    """Return (results_dir, viz_dir, embed_dir) absolute paths for this config."""
    root = output_root(config)
    out = config.get('output') or {}
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
    ignored. Entries are normalized with normalize_term(), so "The" in a
    list removes "the" from the (lowercased) text.
    """
    lang_cfg = config.get('languages', {}).get(language, {})
    rel_path = lang_cfg.get('stopwords_file')
    if not rel_path:
        return set()
    path = resolve_stopwords_path(config, rel_path)
    with open(path, encoding='utf-8-sig') as f:   # skip a byte-order mark
        return {normalize_term(line.strip()) for line in f
                if line.strip() and not line.strip().startswith('#')}


def keep_short_tokens(config, language):
    """
    Return the set of single-character tokens configured as meaningful for
    `language` (e.g. Italian 'e', 'o', 'a', 'è' — all real words) and thus
    exempt from tokenize()'s length filter. Empty by default.
    """
    lang_cfg = config.get('languages', {}).get(language, {})
    return {normalize_term(t) for t in lang_cfg.get('keep_short_tokens', [])}


def adjective_suffixes(config, language):
    """
    Return the tuple of adjective-morphology suffixes configured for
    `language`, used by adjective_window.py as a fallback POS heuristic when
    spaCy is unavailable. Empty by default (heuristic finds nothing).
    """
    lang_cfg = config.get('languages', {}).get(language, {})
    return tuple(normalize_term(s) for s in lang_cfg.get('adjective_suffixes', []))


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

def _read_text(path):
    """Read a UTF-8 corpus file, naming the file if it is not valid UTF-8."""
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError as e:
        raise ValueError(f'{path} is not valid UTF-8 ({e}); re-save it as UTF-8') from e


def load_single_file_corpus(config, corpus_id, corpus_root_override=None):
    """Load a single-file corpus (e.g. a treatise). Returns raw text."""
    entry = config['corpora'][corpus_id]
    root = corpus_root(config, corpus_root_override)
    path = os.path.join(root, entry['path'])
    text = _read_text(path)

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
            text = _read_text(fpath)
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

def _word_mark_class():
    """
    Character-class body for what Python's \\w leaves out but belongs
    inside a word: combining marks (Unicode categories Mn, Mc, Me — e.g.
    Devanagari vowel signs, Arabic and Hebrew pointing, the U+0307 that
    lowercasing Turkish "İ" produces) and the joiners ZWNJ and ZWJ
    (Persian compounds, Indic conjuncts). Soft hyphens never reach the
    tokenizer: normalize_term() removes them.

    Combining marks occur only in planes 0-1 and the variation selectors
    supplement (U+E0100-U+E01EF), so only those are scanned, which keeps
    import fast.
    """
    ranges, start, prev = [], None, None
    for cp in (*range(0x20000), *range(0xE0000, 0xE0200)):
        if unicodedata.category(chr(cp)) in ('Mn', 'Mc', 'Me'):
            if start is None:
                start = cp
            elif cp != prev + 1:
                ranges.append((start, prev))
                start = cp
            prev = cp
    ranges.append((start, prev))
    ranges.append((0x200C, 0x200D))
    return ''.join(f'\\U{a:08x}' if a == b else f'\\U{a:08x}-\\U{b:08x}'
                   for a, b in ranges)


# Inside-a-word characters beyond \w, for regexes that need a word
# boundary (see cross_corpus_dependency.sentences_containing).
WORD_MARKS = _word_mark_class()
# A run of anything that is not a letter or in-word mark: punctuation,
# whitespace, digits and underscores.
_TOKEN_SEPARATOR = re.compile(rf'(?:[^\w{WORD_MARKS}]|[\d_])+')


def tokenize(text, remove_stopwords=True, stopwords=None, keep_short=None):
    """
    Lowercase, normalize to Unicode NFC and remove soft hyphens
    (normalize_term()), so decomposed accents, common in text from macOS or
    PDF extraction, stay part of their word, then split on everything that is not a Unicode
    letter or an in-word combining mark / joiner (WORD_MARKS):
    punctuation, digits, underscores and visible hyphens are all separators, so
    "mid-word" becomes two tokens and "1300" drops out entirely.
    Apostrophe elisions split the same way ("l'amico" -> "l", "amico").

    Single-character tokens are dropped unless listed in `keep_short` —
    language-specific, see keep_short_tokens().

    Returns list of tokens.
    """
    text = normalize_term(text)
    # any run of non-letters is a separator (Unicode-aware, so accented
    # letters and combining marks survive); this drops digits and
    # underscores along with punctuation, keeping every corpus on the
    # same footing.
    tokens = [t for t in _TOKEN_SEPARATOR.split(text) if t]
    keep_short = keep_short or ()
    tokens = [t for t in tokens if len(t) > 1 or t in keep_short]
    if remove_stopwords and stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens


def word_freq(tokens):
    """Return Counter of token frequencies."""
    return Counter(tokens)


def untokenizable_terms(terms, stopwords=None, keep_short=None):
    """
    Return [(term, reason)] for configured terms that tokenize() can never
    produce as a single token — split on an apostrophe or hyphen, dropped
    as a single character, or removed as a stopword — and so would always
    count zero.
    """
    found = []
    for term in terms:
        want = normalize_term(term)
        got = tokenize(term, remove_stopwords=False, keep_short=keep_short)
        if len(got) > 1:
            reason = f'the tokenizer splits it into {got}'
        elif not got:
            reason = ('the tokenizer drops it (single characters need '
                      'keep_short_tokens; digits and punctuation are separators)')
        elif got[0] != want:
            reason = f'the tokenizer turns it into "{got[0]}"'
        elif stopwords and want in stopwords:
            reason = 'it is in the stopword list, which is removed before counting'
        else:
            continue
        found.append((term, reason))
    return found


def warn_untokenizable(terms, where, stopwords=None, keep_short=None):
    """Print a WARNING for each term untokenizable_terms() flags; return them."""
    found = untokenizable_terms(terms, stopwords=stopwords, keep_short=keep_short)
    for term, reason in found:
        print(f'WARNING: {where}: "{term}" can never match — {reason}.')
    return found


# ---------------------------------------------------------------------------
# Splitting long texts for a parser
# ---------------------------------------------------------------------------

_PARA_BREAK = re.compile(r'\n[ \t]*\n\s*')
_SENT_END   = re.compile(r'[.!?]["\'\u201d\u2019)\]]*\s+')
_WHITESPACE = re.compile(r'\s+')


def split_for_parser(text, max_chars):
    """
    Split `text` into pieces of at most `max_chars` characters for a
    parser with an input limit (spaCy). Each cut falls at the last
    paragraph break before the limit; failing that, the last sentence end;
    failing that, the last whitespace — so no word or (where avoidable)
    sentence is cut in half. Joining the pieces reproduces `text` exactly.

    The one exception is a run with no whitespace that is longer than
    `max_chars`: it is cut at exactly `max_chars`, which can separate a
    base character from a combining mark that follows it.
    """
    if max_chars <= 0:
        raise ValueError(f'max_chars must be positive, got {max_chars}')
    pieces, start = [], 0
    while len(text) - start > max_chars:
        window = text[start:start + max_chars]
        cut = 0
        for pattern in (_PARA_BREAK, _SENT_END, _WHITESPACE):
            ends = [m.end() for m in pattern.finditer(window)]
            if ends and ends[-1] > 0:
                cut = ends[-1]
                break
        if cut <= 0:
            cut = max_chars   # one unbroken run longer than the limit
        pieces.append(text[start:start + cut])
        start += cut
    if start < len(text):
        pieces.append(text[start:])
    return pieces


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_cli(analysis, require=None, check_paths=True, argv=None):
    """
    Shared main() for every script: parse the standard flags, load and
    validate the config, run `analysis(config)`, and turn the expected
    failures (bad config, missing file, missing optional dependency) into
    a one-line "ERROR: ..." on stderr and exit status 1 instead of a
    traceback. A truthy return value from `analysis` is used as the exit
    status.
    """
    args = add_config_args().parse_args(argv)
    try:
        config = load_config_from_args(args, require=require, check_paths=check_paths)
        status = analysis(config)
    except (ValueError, FileNotFoundError, ImportError, OSError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)
    if status:
        sys.exit(status)
