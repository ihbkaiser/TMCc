import os
import sys
import glob
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import logging  # for logging to file

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tmsd2 import VAE_OT
from models.contextual_tmsd import ContextualTMSD
from utils.file_utils import read_text
from utils.coherence_wiki import TC_on_wikipedia
from topmost import eva

# Prepare timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"training_{timestamp}.log"
output_beta_filename = f"beta_{timestamp}.npy"

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

class DocDataset(Dataset):
    """
    Loads per-document BoW and embeddings. Returns (x_doc, x_sub_bow, x_sub_emb).
    """
    def __init__(self, data_path, prefix):
        subdir = os.path.join(data_path, 'contextual_data')
        X_train = scipy.sparse.load_npz(f'{data_path}/train_bow.npz').toarray().astype('float32')
        X_test  = scipy.sparse.load_npz(f'{data_path}/test_bow.npz').toarray().astype('float32')
        self.train_tensor = torch.FloatTensor(X_train).to(args.device)
        self.test_tensor  = torch.FloatTensor(X_test).to(args.device)
        self.train_labels = np.loadtxt(f'{data_path}/train_labels.txt', dtype=int)
        self.test_labels  = np.loadtxt(f'{data_path}/test_labels.txt', dtype=int)
        self.bow_files = sorted(glob.glob(f"{subdir}/{prefix}_*_sub_*.npz"))
        self.emb_files = [bf.replace('_sub_', '_sub_emb_').replace('.npz', '.npy')
                          for bf in self.bow_files]
        assert len(self.bow_files) == len(self.emb_files), "BoW and embedding file mismatch"
        vocab_path = os.path.join(data_path, 'vocab.txt')
        self.vocab = read_text(vocab_path)
        self.V = len(self.vocab)
        self.sub_bows = []
        self.sub_embs = []
        for bf, ef in zip(self.bow_files, self.emb_files):
            bow = scipy.sparse.load_npz(bf).toarray().astype('float32')  # [Q_i, V]
            emb = np.load(ef).astype('float32')                           # [Q_i, C]
            self.sub_bows.append(torch.from_numpy(bow))
            self.sub_embs.append(torch.from_numpy(emb))

    def __len__(self):
        return len(self.sub_bows)

    def __getitem__(self, idx):
        x_sub_bow = self.sub_bows[idx]
        x_sub_emb = self.sub_embs[idx]
        x_doc = x_sub_bow.sum(dim=0)  # [V]
        return x_doc, x_sub_bow, x_sub_emb

class Trainer:
    def __init__(self, args):
        self.args = args
        self.dataset = DocDataset(args.data_path, args.prefix)
        W_emb   = scipy.sparse.load_npz(f'{args.data_path}/word_embeddings.npz').toarray().astype('float32')
        class ModelArgs: pass
        margs = ModelArgs()
        margs.vocab_size      = self.dataset.V
        margs.sub_embed_dim   = self.dataset.sub_embs[0].size(1)
        margs.sub_proj_hidden = args.sub_proj_hidden
        margs.en1_units       = args.en1_units
        margs.num_topic       = args.num_topic
        margs.dropout         = args.dropout
        margs.beta_temp       = args.beta_temp
        margs.tau             = args.tau
        margs.num_cluster     = args.num_cluster
        margs.word_embeddings = W_emb
        margs.weight_loss_ECR = args.weight_loss_ECR
        margs.sinkhorn_alpha  = args.sinkhorn_alpha
        margs.OT_max_iter     = args.OT_max_iter
        margs.stopThr         = args.stopThr
        margs.adapter_alpha   = args.adapter_alpha
        self.model = ContextualTMSD(margs).to(args.device)

    def make_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.lr)

    def make_lr_scheduler(self, optimizer):
        return StepLR(optimizer, step_size=self.args.lr_step_size,
                      gamma=0.5, verbose=True)

    def train(self):
        # collate_fn with padding to max Q in batch
        def collate_fn(batch):
            docs, bows, embs = zip(*batch)
            docs = torch.stack(docs, 0)  # [B, V]
            Qs = [b.size(0) for b in bows]
            Qmax = max(Qs)
            B, V = docs.size()
            C = embs[0].size(1)
            bows_pad = torch.zeros(B, Qmax, V)
            embs_pad = torch.zeros(B, Qmax, C)
            for i, (b, e) in enumerate(zip(bows, embs)):
                Qi = b.size(0)
                bows_pad[i, :Qi] = b
                embs_pad[i, :Qi] = e
            return docs.to(self.args.device), bows_pad.to(self.args.device), embs_pad.to(self.args.device)

        loader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            collate_fn=collate_fn
        )

        optimizer = self.make_optimizer()
        if self.args.lr_step_size:
            logging.info("===> Warning: using lr_scheduler")
            scheduler = self.make_lr_scheduler(optimizer)

        total_samples = 0
        for epoch in range(1, self.args.num_epochs + 1):
            self.model.train()
            loss_sums = defaultdict(float)

            for docs, bows, embs in loader:
                out = self.model(docs, bows, embs)
                loss = out['loss']
                optimizer.zero_grad(); loss.backward(); optimizer.step()

                bs = docs.size(0)
                total_samples += bs
                for k, v in out.items(): loss_sums[k] += v.item() * bs

            if self.args.lr_step_size:
                scheduler.step()

            log_msg = f'Epoch {epoch:03d}'
            for k, total in loss_sums.items(): log_msg += f' | {k}: {total/total_samples:.4f}'
            logging.info(log_msg)

            # evaluate coherence & diversity
            self.model.eval()
            with torch.no_grad():
                top_words = self.model.get_top_words(self.dataset.vocab, num_top_words=15)
                _, cv = TC_on_wikipedia(top_words, cv_type="C_V")
                td = eva._diversity([' '.join(t) for t in top_words])
                logging.info(f'  Coherence Cv: {cv:.4f} | Diversity TD: {td:.4f}')
                doc_topic = self.model.get_theta(self.dataset.test_tensor)
                clus_res = eva._clustering(doc_topic.cpu().numpy(), self.dataset.test_labels)
                logging.info(f'  Clustering: {clus_res}')
                

        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train ContextualTMSD without padding, supports batch_size>1")
    parser.add_argument('--data_path',      type=str,   default="datasets/20NG")
    parser.add_argument('--prefix',         type=str,   default="20NG")
    parser.add_argument('--batch_size',     type=int,   default=200)
    parser.add_argument('--shuffle',        type=bool,  default=True)
    parser.add_argument('--device',         type=str,   default='cuda:0')
    parser.add_argument('--num_epochs',     type=int,   default=500)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--sub_proj_hidden',type=int,   default=256)
    parser.add_argument('--en1_units',      type=int,   default=256)
    parser.add_argument('--num_topic',      type=int,   default=50)
    parser.add_argument('--dropout',        type=float, default=0.0)
    parser.add_argument('--beta_temp',      type=float, default=0.2)
    parser.add_argument('--tau',            type=float, default=1.0)
    parser.add_argument('--num_cluster',    type=int,   default=10)
    parser.add_argument('--weight_loss_ECR',type=float, default=100.0)
    parser.add_argument('--sinkhorn_alpha', type=float, default=20.0)
    parser.add_argument('--OT_max_iter',    type=int,   default=1000)
    parser.add_argument('--stopThr',        type=float, default=0.5e-2)
    parser.add_argument('--adapter_alpha',  type=float, default=0.1)
    parser.add_argument('--lr_step_size',   type=int,   default=125,
                        help='If set, StepLR every this many epochs')
    parser.add_argument('--save_dir',       type=str,   default='checkpoints')
    args = parser.parse_args()

    trainer = Trainer(args)
    beta = trainer.train()

    np.save(output_beta_filename, beta)
    logging.info(f"Saved beta to {output_beta_filename}")
    logging.info(f"Training log saved to {log_filename}")
