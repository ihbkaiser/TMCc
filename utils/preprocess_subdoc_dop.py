#!/usr/bin/env python3
"""
Script: build_subdoc_phrase_bow.py

From raw documents (one per line), splits each into Q sub-documents (equal token chunks),
extracts two-word phrases per subdoc using a given phrase vocab, and builds a 3D Bag-of-Phrases
count array of shape [N_docs, Q, V_phrases]. Saves a compressed .npz with key 'data'.

Usage:
    python build_subdoc_phrase_bow.py \
        --docs /path/to/train_texts.txt \
        --phrase-vocab /path/to/vocab_phrase.txt \
        --n-splits Q \
        --output /path/to/train_sub_phrase_bow.npz

Example:
    # Suppose train_texts.txt has 2 docs:
    # "abc coverage ability conduct ability federal"
    # "able afford abc coverage able detect"
    # vocab_phrase.txt lists those five two-word phrases.

    python build_subdoc_phrase_bow.py \
        --docs train_texts.txt \
        --phrase-vocab vocab_phrase.txt \
        --n-splits 2 \
        --output train_sub_phrase_bow.npz

    # This will split each doc into 2 subdocs, extract phrase counts, and save shape (2,2,5).

"""
import argparse
import re
import numpy as np
from tqdm import tqdm

def load_phrases(vocab_file):
    """
    Load two-word phrases (lowercased), sort by descending length,
    and return list plus phrase->index mapping.
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        phrases = [line.strip().lower() for line in f if line.strip()]
    # match longer phrases first (though all are two words here)
    phrases.sort(key=lambda p: len(p.split()), reverse=True)
    vocab = {phrase: idx for idx, phrase in enumerate(phrases)}
    return phrases, vocab


def extract_phrases(text, phrases, vocab):
    """
    Find all non-overlapping phrase indices in `text`.
    """
    t = text.lower()
    found = []
    for phrase in phrases:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, t):
            found.append(vocab[phrase])
            # remove to avoid overlaps
            t = re.sub(pattern, ' ', t)
    return found


def split_document(text, n_splits):
    """
    Split `text` into `n_splits` contiguous token chunks.
    """
    tokens = text.strip().split()
    total = len(tokens)
    if total == 0:
        return [''] * n_splits
    chunk = (total + n_splits - 1) // n_splits
    subdocs = []
    for i in range(n_splits):
        start = i * chunk
        end = min((i + 1) * chunk, total)
        subdocs.append(' '.join(tokens[start:end]))
    return subdocs


def main():
    parser = argparse.ArgumentParser(description="Build per-subdoc phrase-bow from raw docs.")
    parser.add_argument('--docs',         default="datasets/20NG/train_texts.txt", help="Path to raw docs file (one doc per line)")
    parser.add_argument('--phrase-vocab', default="datasets/20NG/vocab_phrase.txt", help="Two-word phrase vocab file")
    parser.add_argument('--n-splits',     type=int, default=5, help="Number of sub-doc splits per document")
    parser.add_argument('--output',       default="datasets/20NG/train_sub_bop.npz", help="Path to save output .npz (key 'data')")
    args = parser.parse_args()

    # Load phrase list and mapping
    phrases, vocab = load_phrases(args.phrase_vocab)
    Vp = len(vocab)
    Q = args.n_splits

    # Read documents
    with open(args.docs, 'r', encoding='utf-8') as f:
        docs = [line.strip() for line in f]
    N = len(docs)

    # Initialize array: [N, Q, Vp]
    data = np.zeros((N, Q, Vp), dtype=np.int32)

    # Process each document
    for i, doc in tqdm(enumerate(docs)):
        subdocs = split_document(doc, Q)
        for q, sub in enumerate(subdocs):
            idxs = extract_phrases(sub, phrases, vocab)
            for p_idx in idxs:
                data[i, q, p_idx] += 1

    # Save result
    np.savez_compressed(args.output, data=data)
    print(f"Saved phrase-bow array of shape {data.shape} to {args.output}")

if __name__ == '__main__':
    main()
