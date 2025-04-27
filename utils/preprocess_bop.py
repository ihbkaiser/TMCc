#!/usr/bin/env python3
"""
Script: build_phrase_bow.py

Builds a Bag-of-Phrases sparse CSR matrix (.npz) from a pre-extracted phrases file.
Assumes each line in --docs-phrases contains only two-word phrases, space-separated.

Usage:
    python build_phrase_bow.py \
      --docs-phrases /path/to/train_dop.txt \
      --vocab /path/to/vocab_phrase.txt \
      --output-bow /path/to/train_dop_bow.npz
"""
import argparse
import scipy.sparse as sp
import numpy as np
from tqdm import tqdm

def load_vocab(vocab_file):
    """
    Load phrase vocabulary (lowercased) into a dict {phrase: index}.
    """
    with open(vocab_file, 'r', encoding='utf-8') as f:
        phrases = [line.strip().lower() for line in f if line.strip()]
    # Sort for consistent ordering
    phrases.sort()
    return {phrase: idx for idx, phrase in enumerate(phrases)}


def process_docs_phrases(docs_phrases_file, vocab):
    """
    Reads the pre-extracted phrases file and builds a list of lists of phrase indices.
    Each phrase is exactly two words: tokens are grouped in pairs.
    """
    doc_indices = []
    with open(docs_phrases_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().lower().split()
            idxs = []
            # group every two tokens into one phrase
            for w1, w2 in zip(tokens[0::2], tokens[1::2]):
                phrase = f"{w1} {w2}"
                if phrase in vocab:
                    idxs.append(vocab[phrase])
            doc_indices.append(idxs)
    return doc_indices


def build_bow_matrix(doc_phrase_indices, n_docs, n_features):
    """
    Build a CSR sparse matrix (n_docs x n_features) from index lists.
    """
    indptr = [0]
    indices = []
    data = []

    for doc in tqdm(doc_phrase_indices):
        counts = {}
        for idx in doc:
            counts[idx] = counts.get(idx, 0) + 1
        for feat_idx, cnt in counts.items():
            indices.append(feat_idx)
            data.append(cnt)
        indptr.append(len(indices))

    return sp.csr_matrix((data, indices, indptr), shape=(n_docs, n_features), dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Build Bag-of-Phrases from pre-extracted phrase doc file.")
    parser.add_argument('--docs-phrases', default="datasets/20NG/train_dop.txt", help="Path to pre-extracted phrases file (one doc per line)")
    parser.add_argument('--vocab',        default="datasets/20NG/vocab_phrase.txt", help="Path to vocab_phrase.txt")
    parser.add_argument('--output-bow',   default="datasets/20NG/train_bop.npz", help="Path to save output .npz file")
    args = parser.parse_args()

    # Load vocabulary
    vocab = load_vocab(args.vocab)
    n_features = len(vocab)

    # Process documents to indices
    doc_phrase_indices = process_docs_phrases(args.docs_phrases, vocab)
    n_docs = len(doc_phrase_indices)

    # Build and save sparse BOW matrix
    X = build_bow_matrix(doc_phrase_indices, n_docs, n_features)
    sp.save_npz(args.output_bow, X)
    print(f"Saved Bag-of-Phrases matrix of shape {X.shape} to {args.output_bow}")

if __name__ == '__main__':
    main()
