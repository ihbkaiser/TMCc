import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae_dkm import VAEDeepKMeans
from utils.file_utils import read_text
from topmost import eva  

def train(args):
    path = args.data_path
    train_data = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
    test_data = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
    word_embeddings = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')
    train_labels = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
    test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)
    vocab = read_text(f'{path}/vocab.txt')

    train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    class ModelArgs:
        pass
    margs = ModelArgs()
    margs.vocab_size = train_data.shape[1]
    margs.en1_units = args.en1_units
    margs.num_topic = args.num_topic
    margs.dropout = args.dropout
    margs.beta_temp = args.beta_temp
    margs.tau = args.tau
    margs.num_cluster = args.num_cluster
    margs.word_embeddings = word_embeddings

    model = VAEDeepKMeans(margs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            x_batch = batch[0]
            optimizer.zero_grad()
            out = model(x_batch)
            loss = out['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss = {avg_loss:.4f}")

        model.eval()
        with torch.no_grad():
            doc_topic_matrix = model.get_theta(test_tensor)
            clustering_results = eva._clustering(doc_topic_matrix, test_labels)
            print(f"Epoch {epoch+1} Clustering results: {clustering_results}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train VAEDeepKMeans model')
    parser.add_argument('--data_path', type=str, default='datasets/20NG')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--en1_units', type=int, default=256)
    parser.add_argument('--num_topic', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--beta_temp', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--num_cluster', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=5)
    args = parser.parse_args()
    train(args)
