import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize

def get_cuda_optimal_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_device = None
    best_memory = float('inf')
    for i in range(torch.cuda.device_count()):
        mem_usage = torch.cuda.memory_allocated(i)
        if mem_usage < best_memory:
            best_memory = mem_usage
            best_device = i
    print(f"Optimal device: cuda:{best_device}")
    return torch.device(f"cuda:1")

def get_optimal_device():
    return get_cuda_optimal_device() if torch.cuda.is_available() else torch.device("cpu")

def average_embeddings(documents, batch_size=32, 
                       model_max_length=512, 
                       embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    device = get_optimal_device()
    data_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    avg_embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding vocabulary"):
            batch_inputs = tokenizer(batch, padding="max_length", max_length=model_max_length, 
                                     truncation=True, return_tensors="pt")
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            last_hidden_state = model(**batch_inputs).last_hidden_state
            avg_embedding = last_hidden_state.mean(dim=1)
            avg_embeddings_list.append(avg_embedding.cpu().numpy())
    document_vectors = normalize(np.vstack(avg_embeddings_list))
    return document_vectors

def contextual_token_embeddings(documents, batch_size=32, model_max_length=512, 
                                embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    device = get_optimal_device()
    data_loader = DataLoader(documents, batch_size=batch_size, shuffle=False)
    model.eval()
    model.to(device)
    last_hidden_states = []
    all_attention_masks = []
    all_tokens = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Embedding documents"):
            batch_inputs = tokenizer(batch, 
                                     padding="max_length", 
                                     max_length=model_max_length, 
                                     truncation=True, return_tensors="pt")
            all_attention_masks.extend(batch_inputs['attention_mask'])
            all_tokens.extend(batch_inputs['input_ids'])
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            last_hidden_state = model(**batch_inputs).last_hidden_state
            last_hidden_states.append(last_hidden_state.cpu())
    all_hidden_states = torch.cat(last_hidden_states, dim=0)
    document_token_embeddings = []
    document_tokens = []
    document_labels = []
    for ind, (hidden_state, attention_mask, tokens) in enumerate(zip(all_hidden_states, 
                                                                     all_attention_masks, all_tokens)):
        indices = attention_mask.nonzero(as_tuple=True)[0]
        embeddings = hidden_state[indices]
        tokens = tokens[indices]
        decoded_tokens = [tokenizer.decode(int(token)) for token in tokens]
        document_token_embeddings.append(embeddings.detach().numpy())
        document_tokens.append(decoded_tokens)
        document_labels.extend([ind] * len(decoded_tokens))
    return document_token_embeddings, document_tokens, document_labels

def sliding_window_average(document_token_embeddings, document_tokens, window_size, stride):
    averaged_embeddings = []
    chunk_tokens = []
    for doc, tokens in tqdm(zip(document_token_embeddings, document_tokens), 
                            total=len(document_token_embeddings), desc="Sliding window averaging"):
        doc_averages = []
        for i in range(0, len(doc), stride):
            start = i
            end = i + window_size
            if start != 0 and end > len(doc):
                start = len(doc) - window_size
                end = len(doc)
            window = doc[start:end]
            window_average = np.mean(window, axis=0)
            doc_averages.append(window_average)
            chunk_tokens.append(" ".join(tokens[start:end]))
        averaged_embeddings.append(doc_averages)
    averaged_embeddings = np.vstack(averaged_embeddings)
    averaged_embeddings = normalize(averaged_embeddings)
    return averaged_embeddings, chunk_tokens

def average_adjacent_tokens(token_embeddings, window_size):
    num_tokens, embedding_size = token_embeddings.shape
    averaged_embeddings = np.zeros_like(token_embeddings)
    token_embeddings = normalize(token_embeddings)
    for i in range(num_tokens):
        start_idx = max(0, i - window_size)
        end_idx = min(num_tokens, i + window_size + 1)
        averaged_embeddings[i] = np.mean(token_embeddings[start_idx:end_idx], axis=0)
    return averaged_embeddings

def smooth_document_token_embeddings(document_token_embeddings, window_size=2):
    smoothed_document_embeddings = []
    for doc in tqdm(document_token_embeddings, desc="Smoothing document token embeddings"):
        smoothed_doc = average_adjacent_tokens(doc, window_size=window_size)
        smoothed_document_embeddings.append(smoothed_doc)
    return smoothed_document_embeddings

def get_document_embeddings(corpus):
    corpus_token_embeddings, corpus_tokens, corpus_labels = contextual_token_embeddings(corpus, 
                                                            batch_size=32, 
                                                            model_max_length=512, 
                                                            embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    avg_embeddings, chunk_tokens = sliding_window_average(corpus_token_embeddings, corpus_tokens, 
                                                          window_size=50, stride=40)
    return avg_embeddings, corpus_token_embeddings, corpus_labels

################# Contextual data processing #################
#!/usr/bin/env python3
"""
subdoc_prepare.py — Generate per-document sub-document BoW and embeddings, with progress bars via tqdm

Split each document into overlapping segments by window_size and stride, then for each doc:
  - Build BoW per segment => CSR sparse matrix shape [Q_i, V]
  - Compute segment embeddings per segment => dense NumPy array shape [Q_i, C]

Save individual files per document:
  - BoW:    scipy.sparse.save_npz('{prefix}_sub_{i}.npz', bow_csr)
  - Embeds: np.save('{prefix}_sub_emb_{i}.npy', emb_array)

Usage:
    python utils/subdoc_prepare.py \
      --data_path datasets/20NG --window_size 50 --stride 40 \
      --output_dir datasets/20NG --prefix 20NG \
      --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
      --batch_size 32 --model_max_length 512
"""
import os
import argparse
import numpy as np
from gensim.utils import simple_preprocess
from file_utils import read_text
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from scipy import sparse
from tqdm import tqdm

def get_optimal_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sliding_windows(tokens, window_size, stride):
    segments = []
    L = len(tokens)
    if L == 0:
        return segments
    for start in range(0, L, stride):
        end = start + window_size
        if start != 0 and end > L:
            start = max(0, L - window_size)
            end = L
        segments.append(tokens[start:end])
        if end == L:
            break
    return segments

def build_subdoc_bows(texts, vocab, window_size, stride):
    word2idx = {w: i for i, w in enumerate(vocab)}
    bows_list = []
    V = len(vocab)
    for doc in tqdm(texts, desc="Building BoW", unit="doc"):
        tokens = simple_preprocess(doc)
        segments = generate_sliding_windows(tokens, window_size, stride)
        bow = np.zeros((len(segments), V), dtype=np.float32)
        for i_seg, seg in enumerate(segments):
            for w in seg:
                idx = word2idx.get(w)
                if idx is not None:
                    bow[i_seg, idx] += 1.0
        bows_list.append(sparse.csr_matrix(bow))
    return bows_list

def build_subdoc_embeddings(texts, window_size, stride,
                             embedding_model, batch_size,
                             model_max_length):
    doc_segments = []
    flat_segments = []
    # generate segments per doc
    for doc in tqdm(texts, desc="Generating segments", unit="doc"):
        tokens = simple_preprocess(doc)
        segments = generate_sliding_windows(tokens, window_size, stride)
        doc_segments.append(segments)
        flat_segments.extend([" ".join(seg) for seg in segments])

    device = get_optimal_device()
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to(device)
    model.eval()

    emb_list = []
    # batch-wise embedding
    for i in tqdm(range(0, len(flat_segments), batch_size), desc="Embedding batches", unit="batch"):
        batch = flat_segments[i:i+batch_size]
        enc = tokenizer(batch, padding='max_length', truncation=True,
                        max_length=model_max_length, return_tensors='pt')
        input_ids = enc['input_ids'].to(device)
        attn_mask = enc['attention_mask'].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        emb_list.append(emb)
    emb_all = normalize(np.vstack(emb_list))

    emb_docs = []
    idx = 0
    for segments in tqdm(doc_segments, desc="Splitting embeddings per doc", unit="doc"):
        Q = len(segments)
        emb_docs.append(emb_all[idx:idx+Q])
        idx += Q
    return emb_docs

def main(args):
    dp = args.data_path
    out = args.output_dir or dp
    os.makedirs(out, exist_ok=True)

    train_texts = read_text(os.path.join(dp, 'train_texts.txt'))
    test_texts  = read_text(os.path.join(dp, 'test_texts.txt'))
    vocab       = read_text(os.path.join(dp, 'vocab.txt'))

    w = args.window_size
    s = args.stride
    prefix = os.path.join(out, args.prefix)
    print(f"Window size={w}, stride={s}")

    # Build BoW and embeddings lists
    train_bows = build_subdoc_bows(train_texts, vocab, w, s)
    train_embs = build_subdoc_embeddings(
        train_texts, w, s,
        args.embedding_model, args.batch_size,
        args.model_max_length
    )
    test_bows = build_subdoc_bows(test_texts, vocab, w, s)
    test_embs = build_subdoc_embeddings(
        test_texts, w, s,
        args.embedding_model, args.batch_size,
        args.model_max_length
    )

    # Save per-document files with progress bars
    n_train = len(train_bows)
    n_test  = len(test_bows)
    for i in tqdm(range(n_train), desc="Saving train docs", unit="doc"):
        sparse.save_npz(f"{prefix}_train_sub_{i}.npz", train_bows[i])
        np.save(f"{prefix}_train_sub_emb_{i}.npy", train_embs[i])
    for i in tqdm(range(n_test), desc="Saving test docs", unit="doc"):
        sparse.save_npz(f"{prefix}_test_sub_{i}.npz", test_bows[i])
        np.save(f"{prefix}_test_sub_emb_{i}.npy", test_embs[i])

    print(f"Saved {n_train} train and {n_test} test documents in {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate per-doc sparse BoW + dense embeddings files with progress bars")
    parser.add_argument('--data_path',      type=str, default='datasets/20NG')
    parser.add_argument('--window_size',    type=int, default=50)
    parser.add_argument('--stride',         type=int, default=40)
    parser.add_argument('--prefix',         type=str, default='20NG')
    parser.add_argument('--output_dir',     type=str, default='datasets/20NG/contextual_data')
    parser.add_argument('--embedding_model',type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--model_max_length', type=int, default=512)
    args = parser.parse_args()
    main(args)
