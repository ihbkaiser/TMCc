import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vae_ot import VAE_OT
from utils.file_utils import read_text
from utils.coherence_wiki import TC_on_wikipedia
from topmost import eva

def train(args):
    # --- Load data ---
    path = args.data_path
    train_data       = scipy.sparse.load_npz(f'{path}/train_bow.npz').toarray().astype('float32')
    test_data        = scipy.sparse.load_npz(f'{path}/test_bow.npz').toarray().astype('float32')
    word_embeddings  = scipy.sparse.load_npz(f'{path}/word_embeddings.npz').toarray().astype('float32')
    train_labels     = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
    test_labels      = np.loadtxt(f'{path}/test_labels.txt', dtype=int)
    vocab            = read_text(f'{path}/vocab.txt')
    train_texts      = read_text(f'{path}/train_texts.txt')

    # --- Build tensors & loader ---
    train_tensor = torch.FloatTensor(train_data)
    test_tensor  = torch.FloatTensor(test_data)
    train_dataset = TensorDataset(train_tensor)
    train_loader  = DataLoader(train_dataset,
                               batch_size=args.batch_size,
                               shuffle=True)

    # --- Prepare model arguments ---
    class ModelArgs: pass
    margs = ModelArgs()
    margs.vocab_size      = train_data.shape[1]
    margs.en1_units       = args.en1_units
    margs.num_topic       = args.num_topic
    margs.dropout         = args.dropout
    margs.beta_temp       = args.beta_temp
    margs.tau             = args.tau
    margs.num_cluster     = args.num_cluster
    margs.word_embeddings = word_embeddings
    # OT-related
    margs.weight_loss_ECR = args.weight_loss_ECR
    margs.sinkhorn_alpha  = args.sinkhorn_alpha
    margs.OT_max_iter     = args.OT_max_iter
    margs.stopThr         = args.stopThr

    # --- Instantiate model & optimizer ---
    model     = VAE_OT(margs)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --- Training loop ---
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss    = 0.0
        total_batches = 0

        for batch in train_loader:
            x_batch = batch[0]

            optimizer.zero_grad()
            out   = model(x_batch)
            loss  = out['loss']
            loss.backward()
            optimizer.step()

            total_loss    += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        print(f"Epoch {epoch}/{args.num_epochs} — Loss: {avg_loss:.4f}")

        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            # Clustering (Purity/NMI)
            doc_topic_matrix  = model.get_theta(test_tensor)
            clustering_results = eva._clustering(doc_topic_matrix, test_labels)
            print(f"  Clustering results: {clustering_results}")

            # Topic coherence (Cv)
            top_words_list = model.get_top_words(vocab, num_top_words=15)
            _, cv_score    = TC_on_wikipedia(top_words_list, cv_type="C_V")
            print(f"  Topic Coherence (Cv): {cv_score:.4f}")

            # Topic diversity (TD)
            topic_strs = [' '.join(t) for t in top_words_list]
            td_score   = eva._diversity(topic_strs)
            print(f"  Topic Diversity (TD): {td_score:.4f}")

    # Optionally return learned beta
    return model.get_beta().detach().cpu().numpy()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train VAE_OT model with OT loss')
    parser.add_argument('--data_path',      type=str,   default='datasets/20NG')
    parser.add_argument('--batch_size',     type=int,   default=200)
    parser.add_argument('--en1_units',      type=int,   default=200)
    parser.add_argument('--num_topic',      type=int,   default=50)
    parser.add_argument('--dropout',        type=float, default=0)
    parser.add_argument('--beta_temp',      type=float, default=0.2)
    parser.add_argument('--tau',            type=float, default=1.0)
    parser.add_argument('--num_cluster',    type=int,   default=10)
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--num_epochs',     type=int,   default=100)
    parser.add_argument('--weight_loss_ECR',type=float, default=100, help="Weight for the OT loss")
    parser.add_argument('--sinkhorn_alpha', type=float, default=20,  help="Sinkhorn alpha parameter")
    parser.add_argument('--OT_max_iter',    type=int,   default=1000,help="Maximum iterations for Sinkhorn")
    parser.add_argument('--stopThr',        type=float, default=0.5e-2,help="Stop threshold for Sinkhorn")
    args = parser.parse_args()

    beta = train(args)
    # if you want to save beta for later:
    np.save('beta.npy', beta)
