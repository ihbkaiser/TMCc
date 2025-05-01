# generate_subdocs.py — Generate fixed‑Q sub-document BoW arrays without padding
"""
Split each document into exactly Q sub‑documents (segments) and build BoW per segment.
Outputs are saved as NumPy NPZ archives for fast loading with metadata:
  - train_sub.npz: array 'data' of shape [N_train, Q, Vw], int Q, int Vw
  - test_sub.npz:  array 'data' of shape [N_test, Q, Vw]

Usage:
    python utils/subdoc_prepare.py --data_path datasets/20NG --num_subdocs 5 --output_dir datasets/20NG
"""
import os
import argparse
import numpy as np
from gensim.utils import simple_preprocess
from file_utils import read_text


def split_into_segments(tokens: list[str], num_segments: int) -> list[list[str]]:
    """
    Split tokens into exactly num_segments segments using np.array_split,
    balancing nearly equally without dropping or zero‑padding.
    Returns list of length num_segments (some segments may be shorter).
    """
    return list(np.array_split(tokens, num_segments))


def build_subdoc_bows(texts: list[str], vocab: list[str], num_segments: int) -> list[np.ndarray]:

    word2idx = {w: i for i, w in enumerate(vocab)}
    bows_list: list[np.ndarray] = []
    for doc in texts:
        tokens = simple_preprocess(doc)
        segments = split_into_segments(tokens, num_segments)
        Q = len(segments)
        V = len(vocab)
        sub_bows = np.zeros((Q, V), dtype=np.float32)
        for s, seg_tokens in enumerate(segments):
            for w in seg_tokens:
                idx = word2idx.get(w)
                if idx is not None:
                    sub_bows[s, idx] += 1.0
        bows_list.append(sub_bows)
    return bows_list


def main(args):
    dp = args.data_path
    out = args.output_dir or dp
    os.makedirs(out, exist_ok=True)

    # Load raw text and vocab
    train_texts = read_text(os.path.join(dp, 'train_texts.txt'))
    test_texts  = read_text(os.path.join(dp, 'test_texts.txt'))
    vocab       = read_text(os.path.join(dp, 'vocab.txt'))

    Q = args.num_subdocs
    print(f"Generating {Q} sub-docs per document and building BoW...")

    train_sub = build_subdoc_bows(train_texts, vocab, Q)
    test_sub  = build_subdoc_bows(test_texts,  vocab, Q)

    # Stack into dense arrays
    train_arr = np.stack(train_sub, axis=0)  # [N_train, Q, Vw]
    test_arr  = np.stack(test_sub,  axis=0)  # [N_test, Q, Vw]
    Vw = train_arr.shape[2]

    print(f"→ train_arr: {train_arr.shape}")
    print(f"→ test_arr:  {test_arr.shape}")

    # Save as NPZ with metadata
    train_path = os.path.join(out, 'train_sub.npz')
    test_path  = os.path.join(out, 'test_sub.npz')
    np.savez(train_path, data=train_arr, Q=Q, Vw=Vw)
    np.savez(test_path,  data=test_arr,  Q=Q, Vw=Vw)

    print(f"Saved dense subdoc arrays to:\n  {train_path}\n  {test_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate fixed‑Q sub-doc BoW (dense NPZ)")
    parser.add_argument('--data_path',   type=str, default='tm_datasets/20NG',
                        help='Folder with train_texts.txt & vocab.txt')
    parser.add_argument('--num_subdocs', type=int, default=5,
                        help='Fixed number Q of sub-documents per doc')
    parser.add_argument('--output_dir',  type=str, default='tm_datasets/20NG',
                        help='Directory to save outputs')
    args = parser.parse_args()
    main(args)
