# train_tmsd3.py
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import logging

from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tmsd3 import TMSD
from utils.file_utils import read_text
from utils.coherence_wiki import TC_on_wikipedia
from topmost import eva

# Prepare timestamps and filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"train_tmsd3_{timestamp}.log"
beta_filename = f"beta_tmsd3_{timestamp}.npy"

# Configure logging
handlers = [logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)

class Trainer:
    def __init__(self, args):
        self.args = args
        path = args.data_path

        # 1) Load core BoW
        X_train = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        X_test  = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')

        # 2) Optional phrase BoW + pairs
        Xp_train = None
        phrase_pairs = None
        pp = os.path.join(path, 'train_phrase.npz')
        if os.path.exists(pp):
            Xp_train = scipy.sparse.load_npz(pp).toarray().astype('float32')
            phrase_pairs = np.load(os.path.join(path, 'phrase_pairs.npy'))

        # 3) Optional sub-doc BoW
        Xs_train = None
        sp = os.path.join(path, 'train_sub.npz')
        if os.path.exists(sp):
            # note: load 'data' field like train_tmsd2
            Xs_train = np.load(f'{path}/train_sub.npz')['data'].astype('float32')

        # 4) Load embeddings
        W_emb = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')

        # 5) Evaluation data
        y_train = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
        y_test  = np.loadtxt(f'{path}/test_labels.txt', dtype=int)
        vocab   = read_text(f'{path}/vocab.txt')
        texts   = read_text(f'{path}/train_texts.txt')

        # Store for training/eval
        self.train_tensor  = torch.FloatTensor(X_train)
        self.test_tensor   = torch.FloatTensor(X_test)
        self.Xp_train      = Xp_train
        self.Xs_train      = Xs_train
        self.train_labels  = y_train
        self.test_labels   = y_test
        self.vocab         = vocab
        self.texts         = texts

        # 6) Model instantiation arguments
        class ModelArgs: pass
        margs = ModelArgs()
        margs.vocab_size      = X_train.shape[1]
        margs.num_topic       = args.num_topic
        margs.enc_hidden      = args.enc_hidden
        margs.dropout         = args.dropout
        margs.beta_temp       = args.beta_temp
        margs.num_cluster     = args.num_cluster
        margs.word_embeddings = W_emb
        margs.ot_weight       = args.weight_ot
        margs.ot_alpha        = args.sinkhorn_alpha
        margs.OT_max_iter     = args.OT_max_iter
        margs.stopThr         = args.stopThr
        margs.adapter_alpha   = args.adapter_alpha
        margs.use_subdoc      = args.use_subdoc
        margs.use_phrase      = Xp_train is not None
        margs.phrase_pairs    = phrase_pairs

        self.model = TMSD(margs)

    def make_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.lr)

    def make_scheduler(self, optimizer):
        if getattr(self.args, 'lr_step_size', None):
            return StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.5, verbose=True)
        return None

    def train(self):
        # Prepare dataset
        if self.args.use_subdoc:
            train_sub = np.load(f'{self.args.data_path}/train_sub.npz')['data'].astype('float32')
            train_dataset = TensorDataset(
                self.train_tensor,
                torch.FloatTensor(train_sub)
            )
        elif getattr(self.args, 'use_phrase', False):
            train_dataset = TensorDataset(
                self.train_tensor,
                torch.FloatTensor(self.Xp_train)
            )
        else:
            train_dataset = TensorDataset(self.train_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        optimizer = self.make_optimizer()
        scheduler = self.make_scheduler(optimizer)
        data_size = len(train_loader.dataset)

        for epoch in range(1, self.args.num_epochs + 1):
            self.model.train()
            # loss_sums = defaultdict(float)

            for batch in train_loader:
                optimizer.zero_grad()
                if self.args.use_subdoc:
                    x, x_sub = batch
                    out = self.model(x_word=x, phrase_counts=None, x_sub=x_sub)
                elif getattr(self.args, 'use_phrase', False):
                    x, x_phrase = batch
                    out = self.model(x_word=x, phrase_counts=x_phrase, x_sub=None)
                else:
                    x, = batch
                    out = self.model(x_word=x)

                out['loss'].backward()
                optimizer.step()

                # bs = x.size(0)
                # for k, v in out.items():
                #     loss_sums[k] += v.item() * bs

            if scheduler:
                scheduler.step()

            # Log training losses
            msg = f'Epoch {epoch:03d}'
            logging.info(msg)
            # for k, total in loss_sums.items():
            #     msg += f' | {k}: {total/data_size:.4f}'
            # logging.info(msg)

            # Evaluation
            self.model.eval()
            with torch.no_grad():
                theta_te = self.model.get_theta(self.test_tensor)
                clus = eva._clustering(theta_te.numpy(), self.test_labels)
                logging.info(f'  Clustering: {clus}')

                topw = self.model.get_top_words(self.vocab, 15)
                _, cv = TC_on_wikipedia(topw, cv_type='C_V')
                td = eva._diversity([' '.join(ws) for ws in topw])
                logging.info(f'  Coherence Cv: {cv:.4f} | Diversity TD: {td:.4f}')

        # Save beta and logs
        beta = self.model.get_beta().detach().cpu().numpy()
        np.save(beta_filename, beta)
        logging.info(f'Saved beta to {beta_filename}')
        logging.info(f'Training log saved to {log_filename}')
        return beta

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str,   default='datasets/20NG')
    parser.add_argument('--batch_size',     type=int,   default=200)
    parser.add_argument('--enc_hidden',     type=int,   default=200)
    parser.add_argument('--num_topic',      type=int,   default=50)
    parser.add_argument('--dropout',        type=float, default=0)
    parser.add_argument('--beta_temp',      type=float, default=0.2)
    parser.add_argument('--num_cluster',    type=int,   default=10)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--num_epochs',     type=int,   default=500)
    parser.add_argument('--weight_ot',      type=float, default=100.0)
    parser.add_argument('--sinkhorn_alpha', type=float, default=20.0)
    parser.add_argument('--OT_max_iter',    type=int,   default=1000)
    parser.add_argument('--stopThr',        type=float, default=5e-3)
    parser.add_argument('--adapter_alpha',  type=float, default=0.1)
    parser.add_argument('--use_subdoc',     type=bool,  default=True)
    parser.add_argument('--lr_step_size',   type=int,   default=125)
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
