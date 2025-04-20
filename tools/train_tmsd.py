# train_htm_phrase.py — Training HTMPhrase model (doc/subdoc/phrase modes)
"""
Train HTMPhrase (Hierarchical Topic VAE with phrase‑pair likelihood and optional sub‑document adapters).
Supports three modes based on available data:
  • doc‑only: uses train_bow.npz/test_bow.npz
  • +phrase: also uses train_phrase.npz/test_phrase.npz + phrase_pairs.npy
  • +subdoc: also uses train_sub.npy/test_sub.npy (or .npz) for adapter

Expected files under --data_path:
  - train_bow.npz, test_bow.npz        : BoW (word) [N, Vw]
  - train_phrase.npz, test_phrase.npz  : BoW (phrase) [N, Vp] (optional)
  - phrase_pairs.npy                   : [Vp,2] int indices into vocab_word.txt (required if phrase)
  - train_sub.npy or train_sub.npz,   : BoW sub‑docs [N, S, Vw] (optional)
    test_sub.npy or test_sub.npz
  - word_embeddings.npz               : [Vw, D]
  - vocab_word.txt, vocab_phrase.txt  : word and phrase vocab for evaluation
  - train_labels.txt, test_labels.txt : integer labels for clustering
  - train_texts.txt                    : raw train texts for coherence
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tmsd import TMSD
from topmost import eva
from utils.file_utils import read_text


def load_npz(path: str) -> np.ndarray:
    return scipy.sparse.load_npz(path).toarray().astype('float32')


def load_optional(path_base: str) -> np.ndarray | None:
    """Try loading .npy or .npz at path_base; return None if neither exists."""
    npy = path_base + '.npy'
    npz = path_base + '.npz'
    if os.path.exists(npy):
        return np.load(npy, allow_pickle=True)
    if os.path.exists(npz):
        return load_npz(npz)
    return None


def main(args):
    dp = args.data_path
    # 1) load core BoW
    Xw_train = load_npz(os.path.join(dp, 'train_bow.npz'))
    Xw_test  = load_npz(os.path.join(dp, 'test_bow.npz'))
    vocab_size = Xw_train.shape[1]

    # 2) optional phrase BoW + pairs
    Xp_train = load_optional(os.path.join(dp, 'train_phrase'))
    Xp_test  = load_optional(os.path.join(dp, 'test_phrase'))
    phrase_pairs = None
    if Xp_train is not None:
        pp = os.path.join(dp, 'phrase_pairs.npy')
        assert os.path.exists(pp), 'phrase_pairs.npy required when phrase counts present'
        phrase_pairs = np.load(pp)

    # 3) optional sub‑doc BoW
    Xs_train_obj = load_optional(os.path.join(dp, 'train_sub'))  # [N,S,Vw]
    Xs_test_obj  = load_optional(os.path.join(dp, 'test_sub'))
    if isinstance(Xs_train_obj, np.ndarray) and Xs_train_obj.dtype == object:
        Xs_train = np.stack(Xs_train_obj, axis=0).astype('float32')
    else:
        Xs_train = Xs_train_obj
    if isinstance(Xs_test_obj, np.ndarray) and Xs_test_obj.dtype == object:
        Xs_test = np.stack(Xs_test_obj, axis=0).astype('float32')
    else:
        Xs_test = Xs_test_obj

    # 4) load embeddings
    we = load_npz(os.path.join(dp, 'word_embeddings.npz'))

    # 5) evaluation data
    y_train = np.loadtxt(os.path.join(dp, 'train_labels.txt'), dtype=int)
    y_test  = np.loadtxt(os.path.join(dp, 'test_labels.txt'), dtype=int)
    train_texts = read_text(os.path.join(dp, 'train_texts.txt'))

    # 6) build dataset
    tensors_tr = [torch.from_numpy(Xw_train)]
    tensors_te = [torch.from_numpy(Xw_test)]
    if Xp_train is not None:
        tensors_tr.append(torch.from_numpy(Xp_train)); tensors_te.append(torch.from_numpy(Xp_test))
    if Xs_train is not None:
        tensors_tr.append(torch.from_numpy(Xs_train)); tensors_te.append(torch.from_numpy(Xs_test))

    train_ds = TensorDataset(*tensors_tr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # 7) model instantiation
    model = TMSD(
        vocab_word=Xw_train.shape[1],
        phrase_pairs=phrase_pairs,
        n_topics=args.num_topic,
        enc_hidden=args.enc_hidden,
        dropout=args.dropout,
        beta_temp=args.beta_temp,
        n_cluster=args.num_cluster,
        word_emb=we,
        ot_weight=args.weight_ot,
        ot_alpha=args.alpha_ot,
        adapter_alpha=args.alpha_adapter,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 8) training loop
    for epoch in range(1, args.num_epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            xw = batch[0]
            ptr = 1 if Xp_train is not None else None
            xp  = batch[ptr] if Xp_train is not None else None
            xs  = batch[-1]  if Xs_train is not None else None

            optimizer.zero_grad()
            out = model(x_word=xw, phrase_counts=xp, x_sub=xs)
            out['loss'].backward()
            optimizer.step()
            total_loss += out['loss'].item()

        print(f"Epoch {epoch}/{args.num_epochs} Loss={total_loss/len(train_loader):.4f}")

        # evaluation
        model.eval()
        with torch.no_grad():
            # get doc-topic ignoring sub-doc
            xw_te = torch.from_numpy(Xw_test)
            xp_te = torch.from_numpy(Xp_test) if Xp_test is not None else None
            theta_te = model.get_theta(xw_te, xp_te)
            clus = eva._clustering(theta_te.numpy(), y_test)
            print(f" Clustering: {clus}")

            # coherence & diversity
            vocab_full = read_text(os.path.join(dp, 'vocab.txt'))
            if Xp_train is not None:
                vocab_full += read_text(os.path.join(dp, 'vocab_phrase.txt'))
            topw = model.get_top_words(vocab_full, k=50)
            topic_strs = [' '.join(ws) for ws in topw]
            cv = eva._coherence(train_texts, vocab_full, topic_strs, coherence_type='c_v', topn=20)
            td = eva._diversity(topic_strs)
            print(f" Cv={cv:.4f} TD={td:.4f}\n")

if __name__ == '__main__':
    p = argparse.ArgumentParser("Train HTMPhrase model — flexible modes")
    p.add_argument('--data_path', type=str, default='datasets/20NG')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--enc_hidden', type=int, default=256)
    p.add_argument('--num_topic', type=int, default=50)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--beta_temp', type=float, default=1.0)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--num_cluster', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--weight_ot', type=float, default=250.0)
    p.add_argument('--alpha_ot', type=float, default=20.0)
    p.add_argument('--alpha_adapter', type=float, default=0.5)
    args = p.parse_args()
    main(args)
