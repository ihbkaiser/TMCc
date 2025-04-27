import os
import sys
import json
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import logging

# allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tmsd5 import VAE_OT
from utils.file_utils import read_text
from utils.coherence_wiki import TC_on_wikipedia
from topmost import eva

# custom dataset to handle BoW, subdoc, and BoP
from torch.utils.data import Dataset

def collate_fn(batch):
    # batch is list of tuples (x, x_sub, x_phrase)
    xs = torch.stack([item[0] for item in batch], dim=0)
    # detect subdoc presence
    if batch[0][1] is not None:
        x_subs = torch.stack([item[1] for item in batch], dim=0)
    else:
        x_subs = None
    # detect phrase presence
    if batch[0][2] is not None:
        x_phrs = torch.stack([item[2] for item in batch], dim=0)
    else:
        x_phrs = None
    return xs, x_subs, x_phrs

class TMSDDataset(Dataset):
    def __init__(self, bow, sub=None, phrase=None):
        self.bow = bow
        self.sub = sub
        self.phrase = phrase
        self.N = bow.size(0)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.bow[idx]
        x_sub = self.sub[idx] if self.sub is not None else None
        x_phrase = self.phrase[idx] if self.phrase is not None else None
        return x, x_sub, x_phrase

# timestamps
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_{timestamp}.log"
output_beta_filename = f"beta_{timestamp}.npy"

# logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)]
)

class Trainer:
    def __init__(self, args):
        self.args = args
        path = args.data_path
        # --- load document-level data ---
        X_train = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        X_test  = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        self.train_tensor = torch.FloatTensor(X_train)
        self.test_tensor  = torch.FloatTensor(X_test)

        # --- load sub-document data if used ---
        self.train_sub = None
        if args.use_subdoc:
            X_sub = np.load(f'{path}/train_sub.npz')['data'].astype('float32')
            self.train_sub = torch.FloatTensor(X_sub)

        # --- load phrase-level data if used ---
        self.train_phrase = None
        if args.use_phrase:
            # load Bag-of-Phrases sparse matrix
            X_phrase = scipy.sparse.load_npz(f'{path}/train_bop.npz').toarray().astype('float32')
            # align number of documents by padding missing phrase rows with zeros
            N_docs = X_train.shape[0]
            P_docs, Vp = X_phrase.shape
            if P_docs < N_docs:
                pad = np.zeros((N_docs - P_docs, Vp), dtype='float32')
                X_phrase = np.vstack([X_phrase, pad])
                logging.info(f"Padded train_bop from {P_docs} to {N_docs} documents with zeros.")
            elif P_docs > N_docs:
                logging.warning(f"train_bop.npz has {P_docs} rows; ignoring extra {P_docs - N_docs} rows.")
                X_phrase = X_phrase[:N_docs]
            self.train_phrase = torch.FloatTensor(X_phrase)

        # --- load labels, vocab, texts ---
        y_train = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
        y_test  = np.loadtxt(f'{path}/test_labels.txt', dtype=int)
        vocab   = read_text(f'{path}/vocab.txt')
        texts   = read_text(f'{path}/train_texts.txt')

        self.train_labels = y_train
        self.test_labels  = y_test
        self.vocab        = vocab
        self.texts        = texts

        # --- build phrase embeddings by averaging word embeddings ---
        # --- Validate document counts match across inputs ---
        N = self.train_tensor.size(0)
        if self.train_sub is not None and self.train_sub.size(0) != N:
            raise RuntimeError(f"train_sub.npz first dimension ({self.train_sub.size(0)}) != train_bow.npz ({N})")
        if self.train_phrase is not None and self.train_phrase.size(0) != N:
            raise RuntimeError(f"train_bop.npz first dimension ({self.train_phrase.size(0)}) != train_bow.npz ({N})")

        phrase_map = json.load(open(args.phrase_map))  # {phrase_id: [word_ids]}
        assert args.phrase_map, "--phrase_map is required for phrase embedding"
        phrase_map = json.load(open(args.phrase_map))  # {phrase_id: [word_ids]}
        # load word embeddings
        W_emb = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')
        phrase_embs = []
        for pid in sorted(phrase_map, key=int):
            ids = phrase_map[pid]
            phrase_embs.append(W_emb[ids].mean(axis=0))
        phrase_embs = np.vstack(phrase_embs)

        # --- model arguments ---
        class ModelArgs: pass
        margs = ModelArgs()
        margs.vocab_size        = X_train.shape[1]
        margs.phrase_vocab_size = phrase_embs.shape[0]
        margs.en1_units         = args.en1_units
        margs.num_topic         = args.num_topic
        margs.dropout           = args.dropout
        margs.beta_temp         = args.beta_temp
        margs.tau               = args.tau
        margs.num_cluster       = args.num_cluster
        margs.word_embeddings   = W_emb
        margs.phrase_embeddings = phrase_embs
        margs.weight_loss_ECR   = args.weight_loss_ECR
        margs.sinkhorn_alpha    = args.sinkhorn_alpha
        margs.OT_max_iter       = args.OT_max_iter
        margs.stopThr           = args.stopThr
        margs.adapter_alpha     = args.adapter_alpha

        self.model = VAE_OT(margs)

    def make_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.lr)

    def make_lr_scheduler(self, optimizer):
        return StepLR(optimizer, step_size=self.args.lr_step_size, gamma=0.5, verbose=True)

    def train(self):
                # prepare dataset with custom Dataset
        dataset = TMSDDataset(self.train_tensor, self.train_sub, self.train_phrase)
        loader  = DataLoader(dataset,
                             batch_size=self.args.batch_size,
                             shuffle=True,
                             collate_fn=collate_fn)

        optimizer = self.make_optimizer()
        if getattr(self.args, 'lr_step_size', None):
            logging.info("===> Warning: using lr_scheduler")
            scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(loader.dataset)
        for epoch in range(1, self.args.num_epochs+1):
            self.model.train()
            loss_sums = defaultdict(float)
            for batch in loader:
                x = batch[0].to(self.model.mu2.device)
                x_sub = batch[1].to(self.model.mu2.device) if self.train_sub is not None else None
                x_phrase = batch[2].to(self.model.mu2.device) if self.train_phrase is not None else None

                out = self.model(x, x_sub=x_sub, x_phrase=x_phrase)
                loss = out['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = x.size(0)
                for k,v in out.items():
                    loss_sums[k] += v.item() * bs

            if getattr(self, 'scheduler', None):
                scheduler.step()

            # log train losses
            msg = f'Epoch {epoch:03d}'
            for k,total in loss_sums.items():
                msg += f' | {k}: {total/data_size:.4f}'
            logging.info(msg)

            # validation / evaluation
            self.model.eval()
            if epoch % 100 == 0:
                with torch.no_grad():
                    doc_topic = self.model.get_theta(self.test_tensor)
                    clus_res  = eva._clustering(doc_topic, self.test_labels)
                    logging.info(f'  Clustering: {clus_res}')
                    top_words = self.model.get_top_words(self.vocab)
                    _,cv = TC_on_wikipedia(top_words, cv_type="C_V")
                    logging.info(f'  Coherence Cv: {cv:.4f}')
                    topic_strs = [' '.join(t) for t in top_words]
                    td = eva._diversity(topic_strs)
                    logging.info(f'  Diversity TD: {td:.4f}')

        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str, default='datasets/20NG')
    parser.add_argument('--batch_size',     type=int, default=200)
    parser.add_argument('--en1_units',      type=int, default=200)
    parser.add_argument('--num_topic',      type=int, default=50)
    parser.add_argument('--dropout',        type=float, default=0)
    parser.add_argument('--beta_temp',      type=float, default=0.2)
    parser.add_argument('--tau',            type=float, default=1.0)
    parser.add_argument('--num_cluster',    type=int, default=10)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--num_epochs',     type=int, default=500)
    parser.add_argument('--weight_loss_ECR',type=float, default=100)
    parser.add_argument('--sinkhorn_alpha', type=float, default=20)
    parser.add_argument('--OT_max_iter',    type=int, default=1000)
    parser.add_argument('--stopThr',        type=float, default=0.5e-2)
    parser.add_argument('--adapter_alpha',  type=float, default=0.1)
    parser.add_argument('--use_subdoc',     type=bool, default=True)
    parser.add_argument('--use_phrase',     type=bool, default=True)
    parser.add_argument('--phrase_map', type=str, default='datasets/20NG/phrase_map.json',
                        help='JSON mapping phrase_id to list of word_ids in vocab')
    parser.add_argument('--lr_step_size',   type=int, default=125)
    args = parser.parse_args()

    trainer = Trainer(args)
    beta = trainer.train()
    np.save(output_beta_filename, beta)
    logging.info(f"Saved beta to {output_beta_filename}")
    logging.info(f"Training log saved to {log_filename}")