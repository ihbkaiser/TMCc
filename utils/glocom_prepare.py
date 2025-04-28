#!/usr/bin/env python3
"""
prepare_data_glo_com.py — Prepare sparse train/test data for GloCOM

For each document d:
  1. Tokenize → sliding‐window segments of length window_size with stride
  2. Build BoW per segment → csr_matrix of shape (Q_d, V)
  3. Build BoW for full doc → ndarray shape (V,)
  4. Concat horizontally [subdoc_BoW | parent_BoW] → csr_matrix (Q_d, 2V)
  5. Stack over all docs → final csr_matrix (sum_d Q_d, 2V)

Outputs:
  - train.npz containing sparse matrix “data”
  - test.npz  containing sparse matrix “data”
  - requires an existing vocab.txt (one token per line)
"""
import os
import argparse
import numpy as np
from gensim.utils import simple_preprocess
from scipy import sparse
from tqdm import tqdm

def read_texts(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def generate_sliding_windows(tokens, window_size, stride):
    L = len(tokens)
    segments = []
    for start in range(0, L, stride):
        end = start + window_size
        if start != 0 and end > L:
            start = max(0, L - window_size)
            end = L
        segments.append(tokens[start:end])
        if end >= L:
            break
    return segments

def build_subdoc_bows(texts, vocab, window_size, stride):
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    bows_list = []
    for doc in tqdm(texts, desc="Building subdoc BoW", unit="doc"):
        tokens = simple_preprocess(doc)
        segments = generate_sliding_windows(tokens, window_size, stride)
        mat = np.zeros((len(segments), V), dtype=np.float32)
        for i_seg, seg in enumerate(segments):
            for w in seg:
                idx = word2idx.get(w)
                if idx is not None:
                    mat[i_seg, idx] += 1.0
        bows_list.append(sparse.csr_matrix(mat))
    return bows_list

def build_parent_bows(texts, vocab):
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    parent_list = []
    for doc in tqdm(texts, desc="Building parent BoW", unit="doc"):
        tokens = simple_preprocess(doc)
        vec = np.zeros(V, dtype=np.float32)
        for w in tokens:
            idx = word2idx.get(w)
            if idx is not None:
                vec[idx] += 1.0
        parent_list.append(vec)
    return parent_list

def concat_bows_sparse(parent_bows, subbows_list):
    parts = []
    for p_vec, s_csr in zip(parent_bows, subbows_list):
        Q_d = s_csr.shape[0]
        # repeat parent vector Q_d times as sparse CSR
        p_row = sparse.csr_matrix(p_vec.reshape(1, -1))   # (1, V)
        P = sparse.vstack([p_row] * Q_d, format='csr')    # (Q_d, V)
        # concat horizontally: [subdoc | parent]
        part = sparse.hstack([s_csr, P], format='csr')    # (Q_d, 2V)
        parts.append(part)
    # stack all docs vertically
    return sparse.vstack(parts, format='csr')             # (sum_d Q_d, 2V)
def build_eval_sparse(parent_bows, V):
    """
    Tạo ma trận CSR shape (num_docs, 2V) mà cột 0:V = 0, cột V:2V = parent BoW
    """
    zeros = sparse.csr_matrix((1, V), dtype=np.float32)  # hàng zeros length V
    parts = []
    for p_vec in parent_bows:
        # parent vector → CSR (1, V)
        p_row = sparse.csr_matrix(p_vec.reshape(1, -1))
        # nửa đầu zeros, nửa sau p_row
        row = sparse.hstack([zeros, p_row], format='csr')
        parts.append(row)
    return sparse.vstack(parts, format='csr') 
def main():
    parser = argparse.ArgumentParser(description="Prepare GloCOM input data")
    parser.add_argument("--train_txt", type=str, default="tm_datasets/20NG/train_texts.txt",
                        help="Path to train_texts.txt")
    parser.add_argument("--test_txt", type=str, default="tm_datasets/20NG/test_texts.txt",
                        help="Path to test_texts.txt")
    parser.add_argument("--vocab_txt", type=str, default="tm_datasets/20NG/vocab.txt",
                        help="Path to vocab.txt (one token per line)")
    parser.add_argument("--window_size", type=int, default=50,
                        help="Sliding window length")
    parser.add_argument("--stride", type=int, default=40,
                        help="Sliding window stride")
    parser.add_argument("--train_out", type=str, default="tm_datasets/20NG/glocom_data/train.npz",
                        help="Output .npz for train data")
    parser.add_argument("--test_out", type=str, default="tm_datasets/20NG/glocom_data/test.npz",
                        help="Output .npz for test data")
    parser.add_argument("--test_eval_out", type=str, default="tm_datasets/20NG/glocom_data/test_eval.npz",
                        help="Output .npz for test eval data")
    parser.add_argument("--train_eval_out", type=str, default="tm_datasets/20NG/glocom_data/train_eval.npz",
                        help="Output .npz for train eval data")
    args = parser.parse_args()

    # Load texts and vocab
    train_docs = read_texts(args.train_txt)
    test_docs  = read_texts(args.test_txt)
    vocab      = read_texts(args.vocab_txt)
    V = len(vocab)
    # Build BoWs
    train_subbows = build_subdoc_bows(train_docs, vocab,
                                      args.window_size, args.stride)
    test_subbows  = build_subdoc_bows(test_docs,  vocab,
                                      args.window_size, args.stride)
    train_parent = build_parent_bows(train_docs, vocab)
    test_parent  = build_parent_bows(test_docs,  vocab)

    # Concat and save sparse matrices
    train_sparse = concat_bows_sparse(train_parent, train_subbows)
    test_sparse  = concat_bows_sparse(test_parent,  test_subbows)
    train_eval = build_eval_sparse(train_parent, V)
    test_eval  = build_eval_sparse(test_parent,  V)

    sparse.save_npz(args.train_out, train_sparse)
    sparse.save_npz(args.test_out,  test_sparse)
    sparse.save_npz(args.train_eval_out, train_eval)
    sparse.save_npz(args.test_eval_out, test_eval)


    print(f"Saved train data → {args.train_out}, shape {train_sparse.shape}")
    print(f"Saved test  data → {args.test_out},  shape {test_sparse.shape}")
    print(f"Saved train eval (parent)→ train_eval.npz,   shape {train_eval.shape}")
    print(f"Saved test  eval (parent)→ test_eval.npz,    shape {test_eval.shape}")

if __name__ == "__main__":
    main()
