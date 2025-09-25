#!/usr/bin/env python3
"""
train_sae_ntm.py — Train SAE-NTM topic model end to end with sentences
Usage:
    python train_sae_ntm.py
"""
import os
import sys
import logging
from types import SimpleNamespace
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from scipy import sparse
from sentence_transformers import SentenceTransformer
# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.sae_ntm_2 import SAE_NTM  # Assuming the model is in models/sae_ntm.py
from topmost import eva
from utils.coherence_wiki import TC_on_wikipedia
from utils.static_utils import print_topic_words  # If custom, replace with model.get_top_words
from utils.irbo import buubyyboo_dth
import wandb
from baseline.evaluations.classification import evaluate_classification
import argparse
parser = argparse.ArgumentParser(description="LLM evaluation")
parser.add_argument('--dataset', type=str, default="20NG")
parser.add_argument('--num_topics', type=int, default=50)
parser.add_argument('--project_name', type=str, default='SAE-NTMv2-20ng-50topic')
args = parser.parse_args()
wandb.login(key="25283834ecbe7bd282505b0721ea3adcd8e789d3", relogin=True)
wandb.init(project=args.project_name)
import pickle

class CustomDataset(Dataset):
    def __init__(self, bow):
        self.bow = bow

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        return self.bow[idx], idx  # Return bow and index for embedding lookup

def custom_collate(batch):
    bow = torch.stack([item[0] for item in batch])
    indices = [item[1] for item in batch]
    return bow, indices

class DatasetHandler:
    """
    Loads sparse .npz doc/train/test files, vocab.txt, and raw texts for sentences.
    Prepares sentences by splitting raw documents using NLTK.
    """
    def __init__(self, data_path, batch_size):
        self.args = SimpleNamespace(data_path=data_path)
        train_doc_path = f"{data_path}/train_bow.npz"
        test_doc_path  = f"{data_path}/test_bow.npz"
        train_text_path = f"{data_path}/train_texts.txt"
        test_text_path  = f"{data_path}/test_texts.txt"
        vocab_path     = f"{data_path}/vocab.txt"
        train_sentence_path = f"{data_path}/train_sentences.pkl"
        test_sentence_path  = f"{data_path}/test_sentences.pkl"
        # Load vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [w.strip() for w in f if w.strip()]
        # Load document-level sparse data
        train_sp = sparse.load_npz(train_doc_path)
        test_sp  = sparse.load_npz(test_doc_path)
        # Convert docs to dense numpy → torch.Tensor
        X_train      = train_sp.toarray().astype(np.float32)
        X_test       = test_sp.toarray().astype(np.float32)
        # Load sentences from pickle
        with open(train_sentence_path, "rb") as f:
            train_sentences = pickle.load(f)
        with open(test_sentence_path, "rb") as f:
            test_sentences = pickle.load(f)
        # Precompute embeddings
        sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Train
        flat_train = [s for doc in train_sentences for s in doc]
        self.train_emb_flat = torch.tensor(sentence_transformer.encode(flat_train), dtype=torch.float32, device=device)
        self.train_emb_lengths = [len(doc) for doc in train_sentences]
        # Test
        flat_test = [s for doc in test_sentences for s in doc]
        self.test_emb_flat = torch.tensor(sentence_transformer.encode(flat_test), dtype=torch.float32, device=device)
        self.test_emb_lengths = [len(doc) for doc in test_sentences]
        # Store labels if available
        self.y_train = np.loadtxt(f'{data_path}/train_labels.txt', dtype=int)
        self.y_test  = np.loadtxt(f'{data_path}/test_labels.txt', dtype=int)
        # Torch tensors for docs
        self.train_data      = torch.from_numpy(X_train)
        self.test_data       = torch.from_numpy(X_test)
    
        self.train_dataloader = DataLoader(
            CustomDataset(self.train_data),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate
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
        train_theta = self.test(ds.train_data, ds.train_emb_flat, ds.train_emb_lengths)
        return top_words, train_theta

    def train(self, ds, verbose=False):
        opt = self.make_optimizer()
        if self.lr_scheduler:
            sch = self.make_lr_scheduler(opt)
        data_size = len(ds.train_dataloader.dataset)
        for epoch in tqdm(range(1, self.epochs+1), desc="Epochs"):
            self.model.train()
            loss_acc = defaultdict(float)
            for doc_batch, idx_batch in ds.train_dataloader:
                emb_batch = []
                start = 0
                for rel_idx in idx_batch:
                    m = ds.train_emb_lengths[rel_idx]
                    emb_batch.append(ds.train_emb_flat[start:start + m])
                    start += m
                x = doc_batch.to(self.device)
                rst = self.model(x, emb_batch)
                loss = rst['loss']
                opt.zero_grad()
                loss.backward()
                opt.step()
                bs = x.size(0)
                for k,v in rst.items(): loss_acc[k] += v.item()*bs
            if self.lr_scheduler:
                sch.step()
            # Log losses to W&B
            log_data = {f"loss/{k}": total / data_size for k, total in loss_acc.items()}
            wandb.log(log_data, step=epoch)
            if verbose and epoch % self.log_interval == 0:
                msg = f"Epoch {epoch:03d}"
                for k, total in loss_acc.items():
                    msg += f" | {k}: {total/data_size:.4f}"
                print(msg)
                self.logger.info(msg)
            if epoch % 50 == 0:
                train_t, test_t = self.export_theta(ds)
                clus = eva._clustering(test_t, ds.y_test)
                self.logger.info(f"Clustering result: {clus}")
                tw = self.model.get_top_words(ds.vocab, num_top_words=15)
                metric_data = evaluate_classification(train_t, test_t, ds.y_train, ds.y_test)
                self.logger.info(f"Classification results: {metric_data}")
                _, cv = TC_on_wikipedia(tw, cv_type="C_V"); self.logger.info(f"Coherence Cv: {cv:.4f}")
                td = eva._diversity([' '.join(t) for t in tw]); self.logger.info(f"Diversity TD: {td:.4f}")
                irbo = buubyyboo_dth(tw, topk=15)
                metric_data["Diversity_TD"] = td
                metric_data["IRBO"] = irbo
                metric_data["CV"] = cv
                if isinstance(clus, dict):
                    for ck, cvl in clus.items():
                        metric_data[f"clustering/{ck}"] = cvl
                wandb.log(metric_data, step=epoch)

    def test(self, data, emb_flat, emb_lengths):
        self.model.eval()
        thetas = []
        with torch.no_grad():
            N = data.shape[0]
            start_emb = 0
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch = data[start:end].to(self.device)
                emb_batch = []
                for i in range(start, end):
                    m = emb_lengths[i]
                    emb_batch.append(emb_flat[start_emb:start_emb + m])
                    start_emb += m
                theta = self.model.get_theta(batch, emb_batch)
                thetas.extend(theta.cpu().tolist())
        return np.array(thetas)

    def export_beta(self):
        # SAE-NTM uses decoder_linear.weight as beta (num_topic x vocab_size), transpose for V x K if needed
        return self.model.decoder_linear.weight.detach().cpu().numpy().T  # To match typical beta (V x K)

    def export_top_words(self, vocab, num_top_words):
        # Use model's built-in method
        return self.model.get_top_words(vocab, num_top_words)

    def export_theta(self, ds):
        train_t = self.test(ds.train_data, ds.train_emb_flat, ds.train_emb_lengths)
        test_t = self.test(ds.test_data, ds.test_emb_flat, ds.test_emb_lengths)
        return train_t, test_t

    def save_beta(self, out):
        np.save(os.path.join(out, f'beta.npy'), self.export_beta())

    def save_theta(self, ds, out):
        tr, te = self.export_theta(ds)
        np.save(os.path.join(out,f'train_theta.npy'), tr)
        np.save(os.path.join(out,f'test_theta.npy'), te)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = args.dataset
    num_topics = args.num_topics
    data_path = f"/home/ducanh/Credit/TM-clusterrin/tm_datasets/{dataset}"  # Adjust path as needed
    out_dir = f"/home/ducanh/Credit/TM-clusterrin/outputs/sae_ntm/{dataset}_{num_topics}_new"
    os.makedirs(out_dir, exist_ok=True)
    ds = DatasetHandler(data_path=data_path, batch_size=200)
    vocab_size = len(ds.vocab)
    args_model = SimpleNamespace(
        vocab_size=vocab_size,
        en1_units=200,
        dropout=0.0,
        num_topic=num_topics
    )
    model = SAE_NTM(args_model)
    trainer = BasicTrainer(model, epochs=500, learning_rate=2e-3, batch_size=200,
                           lr_scheduler="StepLR", lr_step_size=125, log_interval=5)
    tw, train_t = trainer.fit_transform(ds, num_top_words=15, verbose=True)
    trainer.save_beta(out_dir)
    trainer.save_theta(ds, out_dir)
    print("Training complete. Outputs saved to:", out_dir)