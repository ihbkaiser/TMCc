import string
import re
import numpy as np


def preprocess(doc: str):
    # Lowercase
    doc = doc.lower()
    
    # Split by some ending puncts
    subdocs = re.split(r'(?<=[.?!])\s+', doc)
    subdocs = [subdoc.strip() for subdoc in subdocs if subdoc.strip()]
    
    return subdocs


def join_subdoc(subdocs, n=1):
    translator = str.maketrans("", "", string.punctuation)
    
    # Group every n sentences together
    grouped_subdocs = [" ".join(subdocs[i:i+n]) for i in range(0, len(subdocs), n)]
    
    return [subdoc.translate(translator).split() for subdoc in grouped_subdocs if subdoc]


def build_subsentence(docs, vocab):
    vocab2idx = {w: i for i, w in enumerate(vocab)}
    X = len(docs)
    Q_max = max(len(doc) for doc in docs)   #max subdocs
    V = len(vocab)
    
    bow_tensor = np.zeros((X, Q_max, V), dtype=np.float32)
    
    for doc_idx, doc in enumerate(docs):
        for subdoc_idx, subdoc in enumerate(doc):
            for token in subdoc:
                if token in vocab2idx:
                    bow_tensor[doc_idx, subdoc_idx, vocab2idx[token]] += 1
    return bow_tensor


def build_subsentence_index(docs, vocab):
    vocab2idx = {w: i + 1 for i, w in enumerate(vocab)}
    X = len(docs)
    Q_max = max(len(doc) for doc in docs)
    W_max = max(len(subdoc) for doc in docs for subdoc in doc)
    
    index_tensor = np.zeros((X, Q_max, W_max), dtype=np.int32)
    for doc_idx, doc in enumerate(docs):
        for subdoc_idx, subdoc in enumerate(doc):
            for token_idx, token in enumerate(subdoc):
                index_tensor[doc_idx, subdoc_idx, token_idx] = vocab2idx.get(token, 0)
    
    return index_tensor
    
