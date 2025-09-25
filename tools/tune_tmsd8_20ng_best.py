

import os
import sys
import logging
import argparse
from types import SimpleNamespace
from collections import defaultdict
from dotenv import load_dotenv
# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from scipy import sparse
from utils.irbo import buubyyboo_dth
import wandb

from models.tmsd8 import TMSD6
from topmost import eva
from utils.coherence_wiki import TC_on_wikipedia
from utils.static_utils import print_topic_words

load_dotenv()

wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
# Parse command-line arguments
parser = argparse.ArgumentParser(description="W&B sweep runner for TMSD6")
parser.add_argument('--data_path', type=str, default="tm_datasets/WOS_medium", help="path to dataset")
parser.add_argument('--num_topics', type=int, default=50, help="number of topics")
parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="compute device")
parser.add_argument('--project_name', type=str, default='WoS_medium_100')
args = parser.parse_args()

# Sweep configuration for W&B
sweep_config = {
    'method': 'grid',
    'metric': { 'name': 'Coherence_Cv', 'goal': 'maximize' },
    'parameters': {
        # alpha for KL adapter
        'adapter_alpha':   { 'values': [0.1] },
        
        # weight for ECR loss
        'weight_loss_ECR': { 'values': [150] },
        
        # lambda for document full theta reconstruction
        'lambda_doc':      { 'values': [1] },
        
        # augmentation coefficient
        'augment_coef':    { 'values': [0.0] },
        
        # Learning rate
        'learning_rate':   { 'values': [0.002] },
        
        # Fixed parameters
        'epochs':          { 'value': 500 },
        'batch_size':      { 'value': 200 },
        'lr_step_size':    { 'value': 125 },
        'log_interval':    { 'value': 5 },
        'num_topics':      { 'value': args.num_topics }
    }
}

class DatasetHandler:
    """
    Loads sparse .npz doc/train/test and sub-doc files and a vocab.txt,
    converts to dense torch.Tensors, and prepares a DataLoader.
    """
    def __init__(self, data_path, batch_size):
        train_doc_path = f"{data_path}/train_bow.npz"
        test_doc_path  = f"{data_path}/test_bow.npz"
        # train_sub_path = f"{data_path}/dynamic_subdoc/train_sub.npz"
        # test_sub_path  = f"{data_path}/dynamic_subdoc/test_sub.npz"
        train_sub_path = f"{data_path}/sub_sentence/train_subsent.npz"
        test_sub_path  = f"{data_path}/sub_sentence/test_subsent.npz"
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

        sub_train = torch.from_numpy(train_sub_np)
        sub_test  = torch.from_numpy(test_sub_np)

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
        device: str = None
    ):
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lr_scheduler = lr_scheduler
        self.lr_step_size = lr_step_size
        self.log_interval = log_interval

        self.logger = logging.getLogger('main')
        # Use provided device or default
        self.device = args.device
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
                for k, v in rst.items():
                    loss_acc[k] += v.item() * bs

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
                _, cv = TC_on_wikipedia(tw, cv_type="C_V"); self.logger.info(f"Coherence Cv: {cv:.4f}")
                td = eva._diversity([' '.join(t) for t in tw]); self.logger.info(f"Diversity TD: {td:.4f}")
                irbo = buubyyboo_dth(tw, topk=15)
                metric_data = {"Cv": cv, "Diversity_TD": td, "IRBO": irbo}
                if isinstance(clus, dict):
                    for ck, cvl in clus.items():
                        metric_data[f"{ck}"] = cvl
                wandb.log(metric_data, step=epoch)
            # if epoch == self.epochs:
            #     train_t, test_t = self.export_theta(ds)
            #     clus = eva._clustering(test_t, ds.y_test)
            #     self.logger.info(f"Final clustering result: {clus}")
            #     tw = self.model.get_top_words(ds.vocab, num_top_words=15)
            #     _, cv = TC_on_wikipedia(tw, cv_type="C_V"); self.logger.info(f"Final Coherence Cv: {cv:.4f}")
            #     td = eva._diversity([' '.join(t) for t in tw]); self.logger.info(f"Final Diversity TD: {td:.4f}")
            #     irbo = buubyyboo_dth(tw, topk=15)
            #     metric_data = {"Final_Coherence_Cv": cv, "Final_Diversity_TD": td, "Final_IRBO": irbo}
            #     if isinstance(clus, dict):
            #         for ck, cvl in clus.items():
            #             metric_data[f"final_clustering/{ck}"] = cvl
            #     wandb.log(metric_data, step=epoch)

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
        return self.test(ds.train_data), self.test(ds.test_data)

    def save_beta(self, out):
        np.save(os.path.join(out, 'beta.npy'), self.export_beta())

    def save_theta(self, ds, out):
        tr, te = self.export_theta(ds)
        np.save(os.path.join(out,'train_theta.npy'), tr)
        np.save(os.path.join(out,'test_theta.npy'), te)

# Define the training run for W&B agent
def train():
    wandb.init()
    config = wandb.config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_path = args.data_path
    out_dir = "outputs/tmsd6"
    os.makedirs(out_dir, exist_ok=True)

    ds = DatasetHandler(data_path=data_path, batch_size=config.batch_size)
    vocab_size = len(ds.vocab)
    W_emb = sparse.load_npz(f'{data_path}/word_embeddings.npz').toarray().astype('float32')
    model_args = SimpleNamespace(
        vocab_size=vocab_size,
        en1_units=200,
        dropout=0.0,
        embed_size=200,
        num_topic=config.num_topics,
        num_cluster=10,
        adapter_alpha=config.adapter_alpha,
        beta_temp=0.2,
        tau=1.0,
        weight_loss_ECR=config.weight_loss_ECR,
        sinkhorn_alpha=20.0,
        sinkhorn_max_iter=1000,
        augment_coef=config.augment_coef,
        data_path=data_path,
        word_embeddings=W_emb,
        lambda_doc=config.lambda_doc
    )

    model = TMSD6(model_args)
    trainer = BasicTrainer(
        model,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        lr_scheduler="StepLR",
        lr_step_size=config.lr_step_size,
        log_interval=config.log_interval,
        device=args.device
    )
    tw, train_t = trainer.fit_transform(ds, num_top_words=15, verbose=True)
    trainer.save_beta(out_dir)
    trainer.save_theta(ds, out_dir)

    print("Training complete. Outputs saved to:", out_dir)

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project=args.project_name)
    wandb.agent(sweep_id, function=train)