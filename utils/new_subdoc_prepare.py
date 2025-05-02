import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F
from scipy import sparse
from gensim.utils import simple_preprocess
import numpy as np

from utils.file_utils import read_text

train_texts = read_text("tm_datasets/20NG/train_texts.txt")
test_texts = read_text("tm_datasets/20NG/test_texts.txt")
vocab = read_text("tm_datasets/20NG/vocab.txt")

def generate_sliding_windows(
    tokens: List[str], window_size: int, stride: int
) -> List[str]:
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
    q = len(segments)
    return segments, q


def build_subdoc_bows(
    texts: List[str], vocab: List[str], window_size: int, stride: int
) -> Tuple[List[np.ndarray], int]:
    word2idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    bows_list = []
    q_max = 0

    for doc in tqdm(texts, desc="Building subdoc BoW", unit="doc"):
        tokens = simple_preprocess(doc)
        segments, q = generate_sliding_windows(tokens, window_size, stride)
        q_max = max(q_max, q)
        mat = np.zeros((q, V), dtype=np.float32)
        for i_seg, seg in enumerate(segments):
            for w in seg:
                idx = word2idx.get(w)
                if idx is not None:
                    mat[i_seg, idx] += 1.0
        bows_list.append(mat)

    return bows_list, q_max


def pad_bows_to_fixed_q(bows_list: List[np.ndarray], target_q: int) -> np.ndarray:
    V = bows_list[0].shape[1]
    padded = []
    for mat in bows_list:
        q = mat.shape[0]
        if q < target_q:
            pad = np.zeros((target_q - q, V), dtype=np.float32)
            mat = np.vstack([mat, pad])
        padded.append(mat)
    return padded


window_size = 50
stride = 40

train_bows, q_train = build_subdoc_bows(train_texts, vocab, window_size, stride)
test_bows, q_test = build_subdoc_bows(test_texts, vocab, window_size, stride)

Q = max(q_train, q_test)
Vw = len(vocab)

print(f"Padding to Q={Q}, Vw={Vw}...")

train_arr = pad_bows_to_fixed_q(train_bows, Q)
test_arr = pad_bows_to_fixed_q(test_bows, Q)

train_arr = np.stack(train_arr, axis=0)  # shape: (N, target_q, V)
test_arr = np.stack(train_arr, axis=0)  # shape: (N, target_q, V)

# Save BoW arrays
np.savez("tm_datasets/20NG/new_train_sub.npz", data=train_arr, Q=Q, Vw=Vw)
np.savez("tm_datasets/20NG/new_test_sub.npz", data=test_arr, Q=Q, Vw=Vw)

print(f"Saved BoW arrays: {train_arr.shape}, {test_arr.shape}")
