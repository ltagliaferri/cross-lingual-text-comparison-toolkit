"""
Prepare corpus chunks and generate embeddings for every corpus in the
study config.

Run this first — all analysis scripts load from the cached files it
produces, in <embeddings_dir>/<corpus_id>.json and <corpus_id>.npy,
one pair per corpus defined in config['corpora'].
"""

import os

from .corpus import load_config, add_config_args, ensure_output_dirs, load_source_corpus, corpus_label
from .embed import chunk_by_paragraph, save_corpus, get_model, embed_chunks


# ---------------------------------------------------------------------------
# Corpus preparation
# ---------------------------------------------------------------------------

def prepare_single_file(config, corpus_id):
    entry = config['corpora'][corpus_id]
    chunking = config.get('chunking', {})
    text = load_source_corpus(config, corpus_id)
    parts = chunk_by_paragraph(text,
                               min_chars=chunking.get('min_chars', 300),
                               max_chars=chunking.get('max_chars', 1600))
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
    split_threshold = chunking.get('collection_item_split_threshold', 1800)
    item_min_chars  = chunking.get('collection_item_min_chars', 200)
    max_chars       = chunking.get('max_chars', 1600)

    items = load_source_corpus(config, corpus_id)
    chunks = []
    for item in items:
        text = item['text'].strip()
        if not text:
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
    for corpus_id in config['corpora']:
        chunks = prepare_corpus(config, corpus_id)
        all_chunks[corpus_id] = chunks
        print(f'  {corpus_label(config, corpus_id):15s} ({corpus_id}): {len(chunks):4d} chunks')
    print(f'  Total: {sum(len(c) for c in all_chunks.values()):4d} chunks')

    for corpus_id, chunks in all_chunks.items():
        save_corpus(chunks, os.path.join(embed_dir, f'{corpus_id}.json'))
    print('  Corpus JSON saved.')

    print(f"\n=== Generating embeddings ({config['embedding_model']}) ===")
    model = get_model(config['embedding_model'])

    for corpus_id, chunks in all_chunks.items():
        embed_chunks(chunks, model, os.path.join(embed_dir, f'{corpus_id}.npy'))

    print('\nDone. Run the analysis scripts next.')


def main():
    args = add_config_args().parse_args()
    config = load_config(args.config)
    if args.corpus_root:
        config['corpus_root'] = args.corpus_root
    run(config)


if __name__ == '__main__':
    main()
