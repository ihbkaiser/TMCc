#!/usr/bin/env python3
"""
train_glocom.py — Train GloCOM topic model end to end

Usage:
    python train_glocom.py
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from collections import defaultdict
from scipy import sparse
from models.glocom import GloCOM
from topmost import eva
from utils.coherence_wiki import TC_on_wikipedia
from utils.static_utils import print_topic_words

class DatasetHandler:
    """
    Loads sparse .npz train/test files and a vocab.txt,
    converts to dense torch.Tensors, and prepares a DataLoader.
    """
    def __init__(self, data_path, batch_size):
        # load vocabulary
        train_path = f"{data_path}/glocom_data/train.npz"
        test_path = f"{data_path}/glocom_data/test.npz"
        train_eval_path = f"{data_path}/glocom_data/train_eval.npz"
        test_eval_path = f"{data_path}/glocom_data/test_eval.npz"
        vocab_path = f"{data_path}/vocab.txt"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = [w.strip() for w in f if w.strip()]

        # load sparse .npz → convert to dense numpy → torch.Tensor
        train_sparse = sparse.load_npz(train_path)
        test_sparse  = sparse.load_npz(test_path)
        train_eval_sparse = sparse.load_npz(train_eval_path)
        test_eval_sparse = sparse.load_npz(test_eval_path)

        X_train = train_sparse.toarray().astype(np.float32)
        X_test  = test_sparse.toarray().astype(np.float32)
        X_eval_train = train_eval_sparse.toarray().astype(np.float32)
        X_eval_test  = test_eval_sparse.toarray().astype(np.float32)
        y_train = np.loadtxt(f'{data_path}/train_labels.txt', dtype=int)
        y_test  = np.loadtxt(f'{data_path}/test_labels.txt', dtype=int)

        self.train_data = torch.from_numpy(X_train)
        self.test_data  = torch.from_numpy(X_test)
        self.train_eval_data = torch.from_numpy(X_eval_train)
        self.test_eval_data = torch.from_numpy(X_eval_test)
        self.y_train = y_train
        self.y_test = y_test

        # DataLoader for training
        self.train_dataloader = DataLoader(
            TensorDataset(self.train_data),
            batch_size=batch_size,
            shuffle=True
        )


class BasicTrainer:
    def __init__(
        self,
        model,
        epochs: int = 1,
        learning_rate: float = 0.002,
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

        # set device and move model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def make_optimizer(self):
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.learning_rate,
        )

    def make_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "StepLR":
            return StepLR(optimizer, step_size=self.lr_step_size, gamma=0.5, verbose=False)
        else:
            raise NotImplementedError(f"LR scheduler {self.lr_scheduler}")

    def fit_transform(self, dataset_handler, num_top_words=15, verbose=False):
        self.train(dataset_handler, verbose=verbose)
        top_words = self.export_top_words(dataset_handler.vocab, num_top_words)
        train_theta = self.test(dataset_handler.train_data)
        return top_words, train_theta

    def train(self, dataset_handler, verbose=False):
        optimizer = self.make_optimizer()
        if self.lr_scheduler:
            print("===> using lr_scheduler")
            self.logger.info("===> using lr_scheduler")
            lr_scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(dataset_handler.train_dataloader.dataset)
        for epoch in tqdm(range(1, self.epochs + 1), desc="Epochs"):
            self.model.train()
            loss_accum = defaultdict(float)

            for batch in dataset_handler.train_dataloader:
                x = batch[0].to(self.device)
                rst = self.model(x)  # expects dict with 'loss', 'loss_TM', 'loss_ECR'
                loss = rst['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                for k, v in rst.items():
                    loss_accum[k] += v.item() * bs

            if self.lr_scheduler:
                lr_scheduler.step()
            


            if verbose and epoch % self.log_interval == 0:
                msg = f"Epoch {epoch:03d}"
                for k, total in loss_accum.items():
                    msg += f" | {k}: {total / data_size:.4f}"
                print(msg)
                self.logger.info(msg)
            # evaluation
            if epoch % 10 == 0:
                train_doc_topic, test_doc_topic = self.export_theta(dataset_handler)
                clus_res = eva._clustering(test_doc_topic, dataset_handler.y_test)
                self.logger.info(f'Clustering: {clus_res}')
                top_words = self.model.get_top_words(dataset_handler.vocab, num_top_words=15)
                _, cv = TC_on_wikipedia(top_words, cv_type="C_V")
                logging.info(f'  Coherence Cv: {cv:.4f}')

                    # diversity TD
                topic_strs = [' '.join(t) for t in top_words]
                td = eva._diversity(topic_strs)
                logging.info(f'  Diversity TD: {td:.4f}')

            beta = self.export_beta()
    def test(self, input_data):
        self.model.eval()
        theta_list = []
        with torch.no_grad():
            N = input_data.shape[0]
            for idx in torch.split(torch.arange(N), self.batch_size):
                batch = input_data[idx].to(self.device)
                theta = self.model.get_global_theta(batch)
                theta_list.extend(theta.cpu().tolist())
        return np.array(theta_list)

    def export_beta(self):
        return self.model.get_beta().detach().cpu().numpy()

    def export_top_words(self, vocab, num_top_words=15):
        beta = self.export_beta()
        return print_topic_words(beta, vocab, num_top_words)

    def export_theta(self, dataset_handler):
        train_eval_data = dataset_handler.train_eval_data.to(self.device)
        test_eval_data = dataset_handler.test_eval_data.to(self.device)
        train_theta = self.test(train_eval_data)
        test_theta = self.test(test_eval_data)
        return train_theta, test_theta

    def save_beta(self, dir_path):
        beta = self.export_beta()
        np.save(os.path.join(dir_path, 'beta.npy'), beta)
        return beta

    def save_top_words(self, vocab, num_top_words, dir_path):
        top_words = self.export_top_words(vocab, num_top_words)
        path = os.path.join(dir_path, f'top_words_{num_top_words}.txt')
        with open(path, 'w') as f:
            for line in top_words:
                f.write(line + '\n')
        return top_words

    def save_theta(self, dataset_handler, dir_path):
        train_theta, test_theta = self.export_theta(dataset_handler)
        np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
        np.save(os.path.join(dir_path, 'test_theta.npy'),  test_theta)
        np.save(os.path.join(dir_path, 'train_argmax_theta.npy'),
                np.argmax(train_theta, axis=1))
        np.save(os.path.join(dir_path, 'test_argmax_theta.npy'),
                np.argmax(test_theta, axis=1))
        return train_theta, test_theta

    def save_embeddings(self, dir_path):
        if hasattr(self.model, 'word_embeddings'):
            W = self.model.word_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'word_embeddings.npy'), W)
            self.logger.info(f'word_embeddings shape: {W.shape}')
        if hasattr(self.model, 'topic_embeddings'):
            T = self.model.topic_embeddings.detach().cpu().numpy()
            np.save(os.path.join(dir_path, 'topic_embeddings.npy'), T)
            self.logger.info(f'topic_embeddings shape: {T.shape}')
            D = sparse.distance.cdist(T, T)
            np.save(os.path.join(dir_path, 'topic_dist.npy'), D)
        return


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    dataset_path = "tm_datasets/20NG"
    W_emb = sparse.load_npz(f'{dataset_path}/word_embeddings.npz').toarray().astype('float32')

    # Paths — adjust as needed
    output_dir = "outputs/glocom"
    os.makedirs(output_dir, exist_ok=True)

    # load dataset
    ds = DatasetHandler(
        data_path=dataset_path,
        batch_size=200
    )

    # instantiate model
    vocab_size = len(ds.vocab)
    model = GloCOM(
        vocab_size=vocab_size,
        num_topics=50,
        en_units=200,
        dropout=0.0,
        pretrained_WE=None,
        embed_size=200,
        sinkhorn_alpha=20.0,
        sinkhorn_max_iter=100,
        beta_temp=0.2,
        aug_coef=0.5,
        prior_var=0.1,
        weight_loss_ECR=60.0
    )

    # train
    trainer = BasicTrainer(
        model=model,
        epochs=200,
        learning_rate=2e-3,
        batch_size=200,
        lr_scheduler=None,
        lr_step_size=125,
        log_interval=5
    )
    top_words, train_theta = trainer.fit_transform(
        ds,
        num_top_words=15,
        verbose=True
    )

    # save outputs
    trainer.save_beta(output_dir)
    # trainer.save_top_words(ds.vocab, 15, output_dir)
    trainer.save_theta(ds, output_dir)
    # trainer.save_embeddings(output_dir)

    print("Training complete. Outputs saved to:", output_dir)
