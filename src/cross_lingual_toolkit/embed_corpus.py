"""
Prepare corpus chunks and generate embeddings for every corpus in the
study config.

Run this first — all analysis scripts load from the cached files it
produces, in <embeddings_dir>/<corpus_id>.json and <corpus_id>.npy,
one pair per corpus defined in config['corpora'].
"""

import os

from .corpus import (config_entries, ensure_output_dirs, load_source_corpus,
                     corpus_label, run_cli)
from .embed import chunk_by_paragraph, save_corpus, get_model, embed_chunks

DEFAULT_EMBEDDING_MODEL = 'sentence-transformers/LaBSE'


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------

def prepare_single_file(config, corpus_id):
    entry = config['corpora'][corpus_id]
    chunking = config.get('chunking', {})
    text = load_source_corpus(config, corpus_id)
    parts = chunk_by_paragraph(text,
                               min_chars=int(chunking.get('min_chars', 300)),
                               max_chars=int(chunking.get('max_chars', 800)))
    return [
        {
            'chunk_id': f'{corpus_id}_{i:04d}',
            'author':   entry['author'],
            'work':     entry['work'],
            'language': entry['language'],
            'text':     part,
        }
        for i, part in enumerate(parts)
    ]


def prepare_collection(config, corpus_id):
    entry = config['corpora'][corpus_id]
    chunking = config.get('chunking', {})
    split_threshold = int(chunking.get('collection_item_split_threshold', 900))
    item_min_chars  = int(chunking.get('collection_item_min_chars', 200))
    max_chars       = int(chunking.get('max_chars', 800))

    items = load_source_corpus(config, corpus_id)
    chunks = []
    skipped = 0
    for item in items:
        text = item['text'].strip()
        if not text:
            continue
        if len(text) < item_min_chars:
            # too short to embed reliably; would add noise to the
            # within-collection similarity and the UMAP
            skipped += 1
            continue
        if len(text) > split_threshold:
            parts = chunk_by_paragraph(text, min_chars=item_min_chars, max_chars=max_chars)
        else:
            parts = [text]
        for i, part in enumerate(parts):
            chunks.append({
                'chunk_id':  f"{corpus_id}_{item['group_num']}_{item['item_num']}_{i:02d}",
                'author':    entry['author'],
                'work':      entry['work'],
                'language':  entry['language'],
                'group':     item['group'],
                'group_num': item['group_num'],
                'item_num':  item['item_num'],
                'text':      part,
            })
    if skipped:
        print(f'  {corpus_id}: skipped {skipped} item(s) shorter than '
              f'chunking.collection_item_min_chars ({item_min_chars} characters)')
    return chunks


def prepare_corpus(config, corpus_id):
    entry = config['corpora'][corpus_id]
    if entry['type'] == 'single_file':
        return prepare_single_file(config, corpus_id)
    if entry['type'] == 'collection':
        return prepare_collection(config, corpus_id)
    raise ValueError(f'Unknown corpus type "{entry["type"]}" for corpus "{corpus_id}"')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(config):
    _, _, embed_dir = ensure_output_dirs(config)

    print('=== Preparing corpus chunks ===')
    all_chunks = {}
    for corpus_id in config_entries(config['corpora']):
        chunks = prepare_corpus(config, corpus_id)
        if not chunks:
            print(f'  WARNING: {corpus_id} produced no chunks — its text may be '
                  'shorter than chunking.min_chars, or the path may be wrong.')
        all_chunks[corpus_id] = chunks
        print(f'  {corpus_label(config, corpus_id):15s} ({corpus_id}): {len(chunks):4d} chunks')
    print(f'  Total: {sum(len(c) for c in all_chunks.values()):4d} chunks')

    for corpus_id, chunks in all_chunks.items():
        save_corpus(chunks, os.path.join(embed_dir, f'{corpus_id}.json'))
    print('  Corpus JSON saved.')

    model_name = config.get('embedding_model') or DEFAULT_EMBEDDING_MODEL
    print(f"\n=== Generating embeddings ({model_name}) ===")
    model = get_model(model_name)

    for corpus_id, chunks in all_chunks.items():
        embed_chunks(chunks, model, os.path.join(embed_dir, f'{corpus_id}.npy'),
                     model_name=model_name)

    print('\nDone. Run the analysis scripts next.')


REQUIRES = ('corpora',)


def main():
    run_cli(run, require=REQUIRES)


if __name__ == '__main__':
    main()
