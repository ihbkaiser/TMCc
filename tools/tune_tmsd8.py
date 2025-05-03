#!/usr/bin/env python3
"""
sweep_tmsd6_full.py — Full hyperparameter sweep for TMSD6 with W&B,
bao gồm cả DatasetHandler và BasicTrainer.
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
from scipy import sparse
from tqdm import tqdm
import wandb

# Đảm bảo project root trên sys.path nếu cần import relative
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tmsd8 import TMSD6
from topmost import eva
from utils.coherence_wiki import TC_on_wikipedia
from utils.static_utils import print_topic_words
from utils.irbo import buubyyboo_dth

class DatasetHandler:
    """
    Loads sparse .npz doc/train/test and sub-doc files and a vocab.txt,
    converts to dense torch.Tensors, and prepares a DataLoader.
    """
    def __init__(self, data_path, batch_size):
        train_doc = f"{data_path}/train_bow.npz"
        test_doc  = f"{data_path}/test_bow.npz"
        train_sub = f"{data_path}/dynamic_subdoc/train_sub.npz"
        test_sub  = f"{data_path}/dynamic_subdoc/test_sub.npz"
        vocab_fp  = f"{data_path}/vocab.txt"

        # Vocab
        with open(vocab_fp, 'r', encoding='utf-8') as f:
            self.vocab = [w.strip() for w in f if w.strip()]

        # Sparse → dense docs
        train_sp = sparse.load_npz(train_doc)
        test_sp  = sparse.load_npz(test_doc)
        X_train  = train_sp.toarray().astype(np.float32)
        X_test   = test_sp.toarray().astype(np.float32)

        # Sub-doc arrays
        train_sub_np = np.load(train_sub)['data'].astype(np.float32)
        test_sub_np  = np.load(test_sub)['data'].astype(np.float32)

        # Labels
        self.y_train = np.loadtxt(f"{data_path}/train_labels.txt", dtype=int)
        self.y_test  = np.loadtxt(f"{data_path}/test_labels.txt", dtype=int)

        # Torch tensors
        self.train_data = torch.from_numpy(X_train)
        self.test_data  = torch.from_numpy(X_test)
        self.train_sub  = torch.from_numpy(train_sub_np)
        self.test_sub   = torch.from_numpy(test_sub_np)

        self.train_dataloader = DataLoader(
            TensorDataset(self.train_data, self.train_sub),
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

    def export_theta(self, ds):
        return self.test(ds.train_data), self.test(ds.test_data)

    def save_beta(self, out):
        beta = self.model.get_beta().detach().cpu().numpy()
        np.save(os.path.join(out, 'beta.npy'), beta)

    def save_theta(self, ds, out):
        tr, te = self.export_theta(ds)
        np.save(os.path.join(out, 'train_theta.npy'), tr)
        np.save(os.path.join(out, 'test_theta.npy'), te)


# -----------------------------------------------------------------------------
# W&B sweep configuration
# -----------------------------------------------------------------------------
sweep_config = {
    'method': 'grid',
    'metric': { 'name': 'Coherence_Cv', 'goal': 'maximize' },
    'parameters': {
        'adapter_alpha':   { 'values': [0.5, 0.2, 0.1] },
        'weight_loss_ECR': { 'values': [60, 100, 250, 400] },
        'lambda_doc':      { 'values': [1.0, 0.5, 0.2, 0.0] },
        'augment_coef':    { 'value': 0.0 },
        'learning_rate':   { 'value': 2e-3 },
        'epochs':          { 'value': 500 },
        'batch_size':      { 'value': 200 },
        'lr_step_size':    { 'value': 125 },
        'log_interval':    { 'value': 5 },
    }
}

def train():
    wandb.init(project="tmsd6-topic-model", config=sweep_config['parameters'])
    config = wandb.config
    data_path = "tm_datasets/20NG"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ds = DatasetHandler(data_path=data_path, batch_size=config.batch_size)

    W_emb = sparse.load_npz(f'{data_path}/word_embeddings.npz').toarray().astype('float32')
    args = SimpleNamespace(
        vocab_size        = len(ds.vocab),
        en1_units         = 200,
        dropout           = 0.0,
        embed_size        = 200,
        num_topic         = 50,
        num_cluster       = 10,
        adapter_alpha     = config.adapter_alpha,
        beta_temp         = 0.2,
        tau               = 1.0,
        weight_loss_ECR   = config.weight_loss_ECR,
        sinkhorn_alpha    = 20.0,
        sinkhorn_max_iter = 1000,
        augment_coef      = config.augment_coef,
        lambda_doc        = config.lambda_doc,
        data_path         = "tm_datasets/20NG",
        word_embeddings   = W_emb,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TMSD6(args).to(device)
    trainer = BasicTrainer(
        model,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        lr_scheduler="StepLR",
        lr_step_size=config.lr_step_size,
        log_interval=config.log_interval
    )

    wandb.watch(model, log="all", log_freq=config.log_interval)

    for epoch in range(1, config.epochs+1):
        model.train()
        loss_acc = defaultdict(float)
        data_size = len(ds.train_dataloader.dataset)

        # train
        for doc_batch, sub_batch in ds.train_dataloader:
            x, x_sub = doc_batch.to(device), sub_batch.to(device)
            rst = model(x, x_sub)
            loss = rst['loss']

            trainer.make_optimizer().zero_grad()
            loss.backward()
            trainer.make_optimizer().step()

            bs = x.size(0)
            for k,v in rst.items():
                loss_acc[k] += v.item() * bs

        # scheduler
        if trainer.lr_scheduler:
            trainer.make_lr_scheduler(trainer.make_optimizer()).step()

        # log train loss
        if epoch % config.log_interval == 0:
            logs = {'epoch': epoch}
            for k,total in loss_acc.items():
                logs[k] = total / data_size
            wandb.log(logs)

        # eval every 5 epochs
        if epoch % 5 == 0:
            _, test_theta = trainer.export_theta(ds)
            clus = eva._clustering(test_theta, ds.y_test)
            top_words = model.get_top_words(ds.vocab, num_top_words=15)
            _, cv = TC_on_wikipedia(top_words, cv_type="C_V")
            td = eva._diversity([' '.join(t) for t in top_words])
            irbo = buubyyboo_dth(top_words, topk=15)
            wandb.log({
                'epoch': epoch,
                'Coherence_Cv': cv,
                'Diversity_TD': td,
                'Purity': clus.get('Purity'),
                'NMI': clus.get('NMI'),
                'IRBO': irbo
            })

    # save outputs
    out_dir = "outputs/tmsd6_sweep"
    os.makedirs(out_dir, exist_ok=True)
    trainer.save_beta(out_dir)
    trainer.save_theta(ds, out_dir)
    wandb.save(os.path.join(out_dir, "*.npy"))


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="tmsd8-topic-model")
    wandb.agent(sweep_id, function=train)
