import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from utils.document_embedding import get_document_embeddings
DATA_NAME = '20NG'
train_text_path = f'../datasets/{DATA_NAME}/train_texts.txt'
test_text_path = f'../datasets/{DATA_NAME}/test_texts.txt'
vocab_path = f'../datasets/{DATA_NAME}/vocab.txt'
test_label_path = f'../datasets/{DATA_NAME}/test_labels.txt'

with open(train_text_path, 'r') as file:
    t20ng_documents = file.readlines()
t20ng_documents = [doc.strip() for doc in t20ng_documents]
out = get_document_embeddings(t20ng_documents)
np.save('datasets/20NG/document_embeddings.npy', out[0])