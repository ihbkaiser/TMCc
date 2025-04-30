#!/usr/bin/env python3
"""
train_tmsd6.py — Train TMSD6 topic model end to end with sub-documents

Usage:
    python train_tmsd6.py
"""

import os
import sys
import logging
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from scipy import sparse

# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tmsd6 import TMSD6
from topmost import eva
from utils.coherence_wiki import TC_on_wikipedia
from utils.static_utils import print_topic_words


class DatasetHandler:
    """
    Loads sparse .npz doc/train/test and sub-doc files and a vocab.txt,
    converts to dense torch.Tensors, and prepares a DataLoader.
    """
    def __init__(self, data_path, batch_size):
        self.args = SimpleNamespace(data_path=data_path)
        train_doc_path = f"{data_path}/train_bow.npz"
        test_doc_path  = f"{data_path}/test_bow.npz"
        train_sub_path = f"{data_path}/train_sub.npz"
        test_sub_path  = f"{data_path}/test_sub.npz"
        vocab_path     = f"{data_path}/vocab.txt"

        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [w.strip() for w in f if w.strip()]

        # Load document-level sparse data
        train_sp = sparse.load_npz(train_doc_path)
        test_sp  = sparse.load_npz(test_doc_path)

        train_sub_np = np.load(train_sub_path)['data'].astype(np.float32)
        test_sub_np  = np.load(test_sub_path)['data'].astype(np.float32)

        # Convert docs to dense numpy → torch.Tensor
        X_train      = train_sp.toarray().astype(np.float32)
        X_test       = test_sp.toarray().astype(np.float32)



        sub_train = torch.from_numpy(train_sub_np)  #[N_train, S, V]
        sub_test  = torch.from_numpy(test_sub_np)   #[N_train, S, V]

        # Store labels if available
        self.y_train = np.loadtxt(f'{data_path}/train_labels.txt', dtype=int)
        self.y_test  = np.loadtxt(f'{data_path}/test_labels.txt', dtype=int)

        # Torch tensors for docs
        self.train_data      = torch.from_numpy(X_train)
        self.test_data       = torch.from_numpy(X_test)
    
        self.train_dataloader = DataLoader(
            TensorDataset(self.train_data, sub_train),
            batch_size=batch_size,
            shuffle=True
        )


class BasicTrainer:
    def __init__(
        self,
        model,
        epochs: int = 200,
        learning_rate: float = 2e-3,
        batch_size: int = 200,
        lr_scheduler: str = None,
        lr_step_size: int = 125,
        log_interval: int = 5,
    ):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

        self.logger = logging.getLogger('main')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def make_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            return StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5)
        raise NotImplementedError

    def fit_transform(self, ds, num_top_words=15, verbose=False):
        self.train(ds, verbose)
        top_words = self.export_top_words(ds.vocab, num_top_words)
        train_theta = self.test(ds.train_data)
        return top_words, train_theta

    def train(self, ds, verbose=False):
        opt = self.make_optimizer()
        if self.lr_scheduler:
            sch = self.make_lr_scheduler(opt)

        data_size = len(ds.train_dataloader.dataset)
        for epoch in tqdm(range(1, self.epochs+1), desc="Epochs"):
            self.model.train()
            loss_acc = defaultdict(float)

            for doc_batch, sub_batch in ds.train_dataloader:
                x = doc_batch.to(self.device)
                x_sub = sub_batch.to(self.device)
                rst = self.model(x, x_sub)
                loss = rst['loss']

                opt.zero_grad()
                loss.backward()
                opt.step()

                bs = x.size(0)
                for k,v in rst.items(): loss_acc[k] += v.item()*bs

            if self.lr_scheduler: sch.step()
            if verbose and epoch % self.log_interval == 0:
                msg = f"Epoch {epoch:03d}"
                for k,total in loss_acc.items(): msg += f" | {k}: {total/data_size:.4f}"
                print(msg); self.logger.info(msg)

            if epoch % 10 == 0:
                train_t, test_t = self.export_theta(ds)
                clus = eva._clustering(test_t, ds.y_test)
                self.logger.info(f"Clustering result: {clus}")
                tw = self.model.get_top_words(ds.vocab, num_top_words=15)
                _, cv = TC_on_wikipedia(tw, cv_type="C_V"); self.logger.info(f"Coherence Cv: {cv:.4f}")
                td = eva._diversity([' '.join(t) for t in tw]); self.logger.info(f"Diversity TD: {td:.4f}")

    def test(self, data):
        self.model.eval()
        thetas = []
        with torch.no_grad():
            N = data.shape[0]
            for idx in torch.split(torch.arange(N), self.batch_size):
                batch = data[idx].to(self.device)
                theta = self.model.get_theta(batch)
                thetas.extend(theta.cpu().tolist())
        return np.array(thetas)

    def export_beta(self):
        return self.model.get_beta().detach().cpu().numpy()

    def export_top_words(self, vocab, num_top_words):
        beta = self.export_beta()
        return print_topic_words(beta, vocab, num_top_words)

    def export_theta(self, ds):
        test_doc = ds.test_data
        train_doc = ds.train_data
        return self.test(train_doc), self.test(test_doc)

    def save_beta(self, out):
        np.save(os.path.join(out, 'beta.npy'), self.export_beta())

    def save_theta(self, ds, out):
        tr, te = self.export_theta(ds)
        np.save(os.path.join(out,'train_theta.npy'), tr)
        np.save(os.path.join(out,'test_theta.npy'), te)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_path = "tm_datasets/20NG"
    out_dir = "outputs/tmsd6"
    os.makedirs(out_dir, exist_ok=True)

    ds = DatasetHandler(data_path=data_path, batch_size=200)
    vocab_size = len(ds.vocab)
    W_emb = sparse.load_npz(f'{data_path}/word_embeddings.npz').toarray().astype('float32')
    args = SimpleNamespace(
        vocab_size=vocab_size,
        en1_units=200,
        dropout=0.0,
        embed_size=200,
        num_topic=50,
        num_cluster=10,  #for DKM loss
        adapter_alpha=0.1,
        beta_temp=0.2,
        tau=1.0,
        weight_loss_ECR=60.0,
        sinkhorn_alpha=20.0,
        sinkhorn_max_iter=100,
        augment_coef=0.5,
        data_path=data_path,
        word_embeddings=W_emb,
    )

    model = TMSD6(args)
    trainer = BasicTrainer(model, epochs=200, learning_rate=2e-3, batch_size=200,
                           lr_scheduler=None, lr_step_size=125, log_interval=5)
    tw, train_t = trainer.fit_transform(ds, num_top_words=15, verbose=True)
    trainer.save_beta(out_dir)
    trainer.save_theta(ds, out_dir)

    print("Training complete. Outputs saved to:", out_dir)
