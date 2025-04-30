import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch 
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.file_utils import read_text

train_texts = read_text('datasets/20NG/train_texts.txt')
test_texts = read_text('datasets/20NG/test_texts.txt')
print(type(test_texts))
print(len(test_texts))
print(type(test_texts[0]))


def get_optimal_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

def sbert_embeddings(
    documents: list[str],
    batch_size: int = 32,
    model_max_length: int = 512,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
) -> torch.Tensor:
    """
    Compute SBERT embeddings for a list of strings.

    Returns:
        Tensor of shape (len(documents), hidden_size)
    """
    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    device = get_optimal_device()
    model.to(device)
    model.eval()

    # helper for mean-pooling
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state        # (B, T, D)
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = torch.sum(token_embeddings * mask, dim=1)       # (B, D)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)          # (B, D)
        return summed / counts

    loader = DataLoader(documents, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    for batch in tqdm(loader, desc="Encoding batches", unit="batch"):
        # batch is a list of strings
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=model_max_length,
            return_tensors='pt'
        )
        # move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
        emb = mean_pooling(output, encoded['attention_mask'])  # (batch_size, D)
        all_embeddings.append(emb.cpu())

    # concatenate all batches
    return torch.cat(all_embeddings, dim=0)  

train_embeddings = sbert_embeddings(train_texts)  # shape (len(train_texts), D)
test_embeddings  = sbert_embeddings(test_texts)   # shape (len(test_texts),  D)