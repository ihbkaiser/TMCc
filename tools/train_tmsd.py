import sys, os
import numpy as np
import torch
import torch.optim as optim
import scipy.sparse
from torch.utils.data import DataLoader, TensorDataset

# make sure your project root is on PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.tmsd import TMSD
from utils.file_utils import read_text
from utils.coherence_wiki import TC_on_wikipedia
from topmost import eva

# Top2Vec imports
from top2vec import Top2Vec
from top2vec.embedding import contextual_token_embeddings, smooth_document_token_embeddings

def compute_ctx_subdoc_embs(train_texts, window_size):
    """
    1) Train a contextual Top2Vec model (splitting each doc into sliding windows).
    2) Extract per-token embeddings, smooth them, and then flatten into a
       (total_tokens × embed_dim) array.
    3) Group those back into per-document lists and pad to a fixed S.
    """
    # --- 1) Fit contextual Top2Vec ---
    t2v = Top2Vec(
        documents=train_texts,
        contextual_top2vec=True,
        embedding_model="all-MiniLM-L6-v2",
        split_documents=True,
        chunk_length=50,
        chunk_overlap_ratio=0.2,
        verbose=False
    )

    # --- 2) Get raw token embeddings & labels ---
    doc_tok_embs, doc_tokens, doc_labels = contextual_token_embeddings(
        train_texts,
        batch_size=32,
        model_max_length=512,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # smooth each doc’s token embeddings
    smoothed = smooth_document_token_embeddings(doc_tok_embs, window_size=window_size)
    flat_embs = np.vstack(smoothed)                     # shape (N_tokens, E)
    labels   = np.concatenate([[i]*emb.shape[0] 
                               for i, emb in enumerate(smoothed)])  # (N_tokens,)

    # --- 3) Group & pad to fixed S per doc ---
    n_docs = len(train_texts)
    counts = np.bincount(labels.astype(int))
    S      = counts.max()
    E      = flat_embs.shape[1]

    M_sub = np.zeros((n_docs, S, E), dtype=np.float32)
    ptrs  = [0]*n_docs
    for emb, d in zip(flat_embs, labels):
        d = int(d)
        idx = ptrs[d]
        M_sub[d, idx] = emb
        ptrs[d] += 1

    return M_sub  # shape (n_docs, S, E)

def train(args):
    # 0) Raw texts (one per doc)
    path         = args.data_path
    train_texts  = read_text(f'{path}/train_texts.txt')
    test_texts   = read_text(f'{path}/test_texts.txt')  # if you need it

    # 1) Precompute contextual subdoc embeddings for the new loss
    if args.use_ctx_loss:
        print("⏳ computing contextual Top2Vec subdoc embeddings…")
        M_sub = compute_ctx_subdoc_embs(train_texts,
                                        window_size=args.c_top2vec_smoothing_window)
        # M_sub: (N_docs, S, E)

    # 2) Load your BOW data
    train_data = scipy.sparse.load_npz(f'{path}/train_bow.npz')\
                         .toarray().astype('float32')
    test_data  = scipy.sparse.load_npz(f'{path}/test_bow.npz')\
                         .toarray().astype('float32')

    # 3) Build Dataset + DataLoader
    if args.use_subdoc:
        train_sub       = np.load(f'{path}/train_sub.npz')['data']\
                              .astype('float32')  # (N_docs, S, V)
        train_tensor    = torch.FloatTensor(train_data)
        train_sub_tensor= torch.FloatTensor(train_sub)
        print("⏳ computing contextual Top2Vec subdoc embeddings…")
        M_sub = compute_ctx_subdoc_embs(train_texts,
                                        window_size=args.c_top2vec_smoothing_window)
        ctx_tensor      = torch.FloatTensor(M_sub)

        train_dataset = TensorDataset(train_tensor,
                                      train_sub_tensor,
                                      ctx_tensor)
    else:
        train_tensor  = torch.FloatTensor(train_data)
        train_dataset = TensorDataset(train_tensor)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)

    # 4) Labels, vocab & texts for evaluation
    train_labels = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
    test_labels  = np.loadtxt(f'{path}/test_labels.txt', dtype=int)
    vocab        = read_text(f'{path}/vocab.txt')

    # 5) Build model args
    class ModelArgs: pass
    margs = ModelArgs()
    margs.vocab_size      = train_data.shape[1]
    margs.en1_units       = args.en1_units
    margs.num_topic       = args.num_topic
    margs.dropout         = args.dropout
    margs.beta_temp       = args.beta_temp
    margs.tau             = args.tau
    margs.num_cluster     = args.num_cluster
    margs.word_embeddings = scipy.sparse.load_npz(f'{path}/word_embeddings.npz')\
                                  .toarray().astype('float32')
    # OT loss params
    margs.weight_loss_ECR = args.weight_loss_ECR
    margs.sinkhorn_alpha  = args.sinkhorn_alpha
    margs.OT_max_iter     = args.OT_max_iter
    margs.stopThr         = args.stopThr
    # subdoc BOW flag
    margs.use_subdoc      = args.use_subdoc
    margs.adapter_alpha   = args.adapter_alpha
    # ** new ** contextual‐Top2Vec regularizer
    margs.use_ctx_loss    = args.use_ctx_loss
    margs.ctx_sigma       = args.ctx_sigma

    # 6) Instantiate model & optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TMSD(margs).to(device)
    optim_ = optim.Adam(model.parameters(), lr=args.lr)

    # 7) Training loop
    for epoch in range(1, args.num_epochs+1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            if args.use_subdoc:
                x_batch, x_sub, m_sub = batch
                m_sub = m_sub.to(device)
            else:
                x_batch, x_sub, m_sub = batch[0], None, None

            x_batch = x_batch.to(device)
            if x_sub is not None:
                x_sub = x_sub.to(device)

            optim_.zero_grad()
            out = model(x_batch, x_sub, m_sub=m_sub)
            loss = out['loss']
            loss.backward()
            optim_.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch:03d}] train loss = {avg_loss:.4f}")

        # 8) Evaluation on test set
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_data).to(device)
            θ = model.get_theta(test_tensor).cpu().numpy()
            clustering_results = eva._clustering(θ, test_labels)
            print(f"[Epoch {epoch:03d}] Clustering → {clustering_results}")

            top_words  = model.get_top_words(vocab, num_top_words=15)
            topic_strs = [' '.join(ws) for ws in top_words]
            # cv = eva._coherence(train_texts, vocab, topic_strs,
            #                     coherence_type='c_v', topn=50)
            _, cv = TC_on_wikipedia(top_words, cv_type="C_V")
            td = eva._diversity(topic_strs)
            print(f"[Epoch {epoch:03d}] Coherence (C_v)={cv:.4f}, Diversity={td:.4f}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',                 type=str,   default='datasets/20NG')
    parser.add_argument('--batch_size',                type=int,   default=64)
    parser.add_argument('--en1_units',                 type=int,   default=256)
    parser.add_argument('--num_topic',                 type=int,   default=50)
    parser.add_argument('--dropout',                   type=float, default=0.2)
    parser.add_argument('--beta_temp',                 type=float, default=1.0)
    parser.add_argument('--tau',                       type=float, default=1.0)
    parser.add_argument('--num_cluster',               type=int,   default=10)
    parser.add_argument('--lr',                        type=float, default=1e-3)
    parser.add_argument('--num_epochs',                type=int,   default=100)
    parser.add_argument('--weight_loss_ECR',           type=float, default=100)
    parser.add_argument('--sinkhorn_alpha',            type=float, default=20)
    parser.add_argument('--OT_max_iter',               type=int,   default=1000)
    parser.add_argument('--stopThr',                   type=float, default=0.5e-2)
    parser.add_argument('--use_subdoc',                type=bool,  default=True)
    parser.add_argument('--adapter_alpha',             type=float, default=0.1)
    # ---- new flags for the contextual Top2Vec loss ----
    parser.add_argument('--use_ctx_loss',              action='store_true',
                        help="enable the W·||θ_i−θ_j|| contextual Top2Vec regularizer")
    parser.add_argument('--ctx_sigma',                 type=float, default=1.0,
                        help="σ in exp(cos(m_i,m_j) / σ) for the contextual loss")
    parser.add_argument('--c_top2vec_smoothing_window',type=int,   default=5,
                        help="smoothing window size for token embeddings")

    args = parser.parse_args()
    train(args)