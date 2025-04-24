import os
import sys
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
import logging  # for logging to file

# assume bạn đã thêm path để import các module bên ngoài
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.tmsd2 import VAE_OT
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

class Trainer:
    def __init__(self, args):
        self.args = args

        # --- load data once ---
        path = args.data_path
        X_train = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
        X_test  = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
        W_emb   = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')
        y_train = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
        y_test  = np.loadtxt(f'{path}/test_labels.txt', dtype=int)
        vocab   = read_text(f'{path}/vocab.txt')
        texts   = read_text(f'{path}/train_texts.txt')

        self.train_tensor = torch.FloatTensor(X_train)
        self.test_tensor  = torch.FloatTensor(X_test)
        self.train_labels = y_train
        self.test_labels  = y_test
        self.vocab        = vocab
        self.texts        = texts

        # --- model args ---
        class ModelArgs: pass
        margs = ModelArgs()
        margs.vocab_size     = X_train.shape[1]
        margs.en1_units      = args.en1_units
        margs.num_topic      = args.num_topic
        margs.dropout        = args.dropout
        margs.beta_temp      = args.beta_temp
        margs.tau            = args.tau
        margs.num_cluster    = args.num_cluster
        margs.word_embeddings= W_emb
        margs.weight_loss_ECR= args.weight_loss_ECR
        margs.sinkhorn_alpha = args.sinkhorn_alpha
        margs.OT_max_iter    = args.OT_max_iter
        margs.stopThr        = args.stopThr
        margs.adapter_alpha  = args.adapter_alpha

        self.model = VAE_OT(margs)

    def make_optimizer(self):
        return optim.Adam(
            params=self.model.parameters(),
            lr=self.args.lr
        )

    def make_lr_scheduler(self, optimizer):
        # only if lr_step_size được cung cấp
        return StepLR(
            optimizer,
            step_size=self.args.lr_step_size,
            gamma=0.5,
            verbose=True
        )

    def train(self):
        # DataLoader
        if self.args.use_subdoc:
            train_sub = np.load(f'{self.args.data_path}/train_sub.npz')['data'].astype('float32')
            train_dataset = TensorDataset(self.train_tensor,
                                          torch.FloatTensor(train_sub))
        else:
            train_dataset = TensorDataset(self.train_tensor)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size,
                                  shuffle=True)

        optimizer = self.make_optimizer()
        if getattr(self.args, 'lr_step_size', None) is not None:
            logging.info("===> Warning: using lr_scheduler")
            scheduler = self.make_lr_scheduler(optimizer)

        data_size = len(train_loader.dataset)
        for epoch in range(1, self.args.num_epochs + 1):
            self.model.train()
            loss_sums = defaultdict(float)

            for batch in train_loader:
                # unpack batch
                if self.args.use_subdoc:
                    x, x_sub = batch
                    out = self.model(x, x_sub)
                else:
                    x, = batch
                    out = self.model(x)

                loss = out['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # accumulate losses * batch_size
                bs = x.size(0)
                for k, v in out.items():
                    loss_sums[k] += v.item() * bs

            if getattr(self, 'scheduler', None):
                scheduler.step()

            # log train losses
            log_msg = f'Epoch {epoch:03d}'
            for k, total in loss_sums.items():
                log_msg += f' | {k}: {total / data_size: .4f}'
            logging.info(log_msg)

            # --- validation / evaluation ---
            self.model.eval()
            with torch.no_grad():
                # clustering
                doc_topic = self.model.get_theta(self.test_tensor)
                clus_res = eva._clustering(doc_topic, self.test_labels)
                logging.info(f'  Clustering: {clus_res}')

                # coherence Cv
                top_words = self.model.get_top_words(self.vocab, num_top_words=15)
                _, cv = TC_on_wikipedia(top_words, cv_type="C_V")
                logging.info(f'  Coherence Cv: {cv:.4f}')

                # diversity TD
                topic_strs = [' '.join(t) for t in top_words]
                td = eva._diversity(topic_strs)
                logging.info(f'  Diversity TD: {td:.4f}')

        # cuối cùng trả về beta
        beta = self.model.get_beta().detach().cpu().numpy()
        return beta

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',      type=str,   default='datasets/20NG')
    parser.add_argument('--batch_size',     type=int,   default=200)
    parser.add_argument('--en1_units',      type=int,   default=200)
    parser.add_argument('--num_topic',      type=int,   default=50)
    parser.add_argument('--dropout',        type=float, default=0)
    parser.add_argument('--beta_temp',      type=float, default=0.2)
    parser.add_argument('--tau',            type=float, default=1.0)
    parser.add_argument('--num_cluster',    type=int,   default=10)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--num_epochs',     type=int,   default=500)
    parser.add_argument('--weight_loss_ECR',type=float, default=100)
    parser.add_argument('--sinkhorn_alpha', type=float, default=20)
    parser.add_argument('--OT_max_iter',    type=int,   default=1000)
    parser.add_argument('--stopThr',        type=float, default=0.5e-2)
    parser.add_argument('--adapter_alpha',  type=float, default=0.1)
    parser.add_argument('--use_subdoc',     type=bool,  default=True)
    parser.add_argument('--lr_step_size',   type=int,   default=125,
                        help='Nếu chỉ định, sẽ tạo StepLR với step_size này')
    args = parser.parse_args()

    trainer = Trainer(args)
    beta = trainer.train()

    # Save beta with a timestamped filename
    np.save(output_beta_filename, beta)
    logging.info(f"Saved beta to {output_beta_filename}")
    logging.info(f"Training log saved to {log_filename}")
