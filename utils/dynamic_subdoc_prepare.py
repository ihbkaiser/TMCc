#!/usr/bin/env python3
import os
import argparse
import numpy as np
from gensim.utils import simple_preprocess
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F

def get_optimal_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def sbert_embeddings(
    texts,      # List[str]
    batch_size: int = 32,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(get_optimal_device())
    loader = DataLoader(texts, batch_size=batch_size, shuffle=False, collate_fn=list)
    all_embs = []
    for batch in tqdm(loader, desc="SBERT segments"):
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        emb = mean_pooling(out, enc["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)  # (total_segments, C)

def generate_sliding_windows(tokens, window_size, stride):
    segments = []
    L = len(tokens)
    for start in range(0, L, stride):
        end = start + window_size
        if start != 0 and end > L:
            start, end = max(0, L-window_size), L
        segments.append(tokens[start:end])
        if end >= L:
            break
    return segments

def build_fixed_subdocs(
    texts,       # List[str]
    vocab,       # List[str]
    window_size,
    stride,
    model_name,
    batch_size,
):
    # prepare vocab→idx
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}

    seg_texts = []
    bows = []
    segments_per_doc = []

    # 1) tokenize & segment + build raw bows
    for doc in tqdm(texts, desc="Segmenting docs"):
        tokens = simple_preprocess(doc)
        segs = generate_sliding_windows(tokens, window_size, stride)
        segments_per_doc.append(len(segs))
        for seg in segs:
            seg_texts.append(" ".join(seg))
            vec = np.zeros(V, dtype=np.float32)
            for w in seg:
                idx = w2i.get(w)
                if idx is not None:
                    vec[idx] += 1.0
            bows.append(vec)
    bows = np.stack(bows, axis=0)  # (M, V)

    # 2) SBERT‐embed all segments
    embs = sbert_embeddings(seg_texts, batch_size, model_name)  # (M, C)
    C = embs.shape[1]

    # 3) pad to fixed Q_max
    Q_max = max(segments_per_doc)
    N = len(texts)
    bow_tensor = np.zeros((N, Q_max, V), dtype=np.float32)
    emb_tensor = np.zeros((N, Q_max, C), dtype=np.float32)

    idx = 0
    for i, q in enumerate(segments_per_doc):
        bow_tensor[i, :q, :] = bows[idx: idx+q]
        emb_tensor[i, :q, :] = embs[idx: idx+q].numpy()
        idx += q

    return bow_tensor, emb_tensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',     type=str, default="tm_datasets/WOS_medium",
                        help="dir with train_texts.txt, test_texts.txt, vocab.txt")
    parser.add_argument('--output_dir',    type=str, default='tm_datasets/WOS_medium/dynamic_subdoc',
                        help="where to save the .npz files")
    parser.add_argument('--window_size',   type=int, default=30)
    parser.add_argument('--stride',        type=int, default=24)
    parser.add_argument('--batch_size',    type=int, default=32)
    parser.add_argument('--model_name',    type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    # prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    train_texts = open(os.path.join(args.data_path, 'train_texts.txt')).read().splitlines()
    test_texts  = open(os.path.join(args.data_path, 'test_texts.txt')).read().splitlines()
    vocab       = open(os.path.join(args.data_path, 'vocab.txt')).read().splitlines()

    # train
    bow_tr, emb_tr = build_fixed_subdocs(
        train_texts, vocab,
        args.window_size, args.stride,
        args.model_name, args.batch_size
    )
    np.savez_compressed(
        os.path.join(args.output_dir, 'train_sub.npz'), data=bow_tr)
    np.savez_compressed(
        os.path.join(args.output_dir, 'train_sub_contextual.npz'), data=emb_tr)

    # test
    bow_te, emb_te = build_fixed_subdocs(
        test_texts, vocab,
        args.window_size, args.stride,
        args.model_name, args.batch_size
    )
    np.savez_compressed(
        os.path.join(args.output_dir, 'test_sub.npz'), data=bow_te)
    np.savez_compressed(
        os.path.join(args.output_dir, 'test_sub_contextual.npz'), data=emb_te)

    print(f"Saved .npz files to {args.output_dir}")

if __name__ == "__main__":
    main()
