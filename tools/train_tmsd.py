import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tmsd import TMSD
from utils.file_utils import read_text
from topmost import eva

def train(args):
    # 1) Load BOW data
    path = args.data_path
    train_data = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
    test_data  = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')

    # 2) Optionally load sub-document slices
    if args.use_subdoc:
        train_sub = np.load(f'{path}/train_sub.npz')['data'].astype('float32')
        train_sub_tensor = torch.FloatTensor(train_sub)
        train_tensor = torch.FloatTensor(train_data)
        train_dataset = TensorDataset(train_tensor, train_sub_tensor)
    else:
        train_tensor = torch.FloatTensor(train_data)
        train_dataset = TensorDataset(train_tensor)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 3) Labels and vocabulary for evaluation
    train_labels = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
    test_labels  = np.loadtxt(f'{path}/test_labels.txt', dtype=int)
    vocab        = read_text(f'{path}/vocab.txt')
    train_texts  = read_text(f'{path}/train_texts.txt')

    # 4) Build model args
    class ModelArgs: pass
    margs = ModelArgs()
    margs.vocab_size      = train_data.shape[1]
    margs.en1_units       = args.en1_units
    margs.num_topic       = args.num_topic
    margs.dropout         = args.dropout
    margs.beta_temp       = args.beta_temp
    margs.tau             = args.tau
    margs.num_cluster     = args.num_cluster
    margs.word_embeddings = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')
    # OT params
    margs.weight_loss_ECR = args.weight_loss_ECR
    margs.sinkhorn_alpha  = args.sinkhorn_alpha
    margs.OT_max_iter     = args.OT_max_iter
    margs.stopThr         = args.stopThr
    # subdoc flag
    margs.use_subdoc      = args.use_subdoc
    margs.adapter_alpha   = args.adapter_alpha

    # 5) Instantiate model and optimizer
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = TMSD(margs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 6) Training loop
    for epoch in range(1, args.num_epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            if args.use_subdoc:
                x_batch, x_sub = batch
            else:
                x_batch = batch[0]
                x_sub = None
            x_batch = x_batch.to(device)
            if x_sub is not None:
                x_sub = x_sub.to(device)

            optimizer.zero_grad()
            out = model(x_batch, x_sub)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{args.num_epochs}, Loss = {avg_loss:.4f}")

        # 7) Evaluation
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_data).to(device)
            doc_topic = model.get_theta(test_tensor).cpu().numpy()
            clustering_results = eva._clustering(doc_topic, test_labels)
            print(f"Epoch {epoch} Clustering: {clustering_results}")

            # Topic coherence & diversity
            top_words = model.get_top_words(vocab, num_top_words=50)
            topic_strs = [' '.join(t) for t in top_words]
            cv = eva._coherence(train_texts, vocab, topic_strs, coherence_type='c_v', topn=50)
            td = eva._diversity(topic_strs)
            print(f"Epoch {epoch} Coherence (C_v): {cv:.4f}, Diversity (TD): {td:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/20NG')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--en1_units', type=int, default=256)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--beta_temp', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--num_cluster', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--weight_loss_ECR', type=float, default=100)
    parser.add_argument('--sinkhorn_alpha', type=float, default=20)
    parser.add_argument('--OT_max_iter', type=int, default=1000)
    parser.add_argument('--stopThr', type=float, default=0.5e-2)
    parser.add_argument('--use_subdoc', type=bool, default=False)
    parser.add_argument('--adapter_alpha', type=float, default=0.1)
    args = parser.parse_args()
    train(args)
