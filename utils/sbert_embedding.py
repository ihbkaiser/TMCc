import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch.nn.functional as F

from utils.file_utils import read_text

train_texts = read_text("tm_datasets/20NG/train_texts.txt")
test_texts = read_text("tm_datasets/20NG/test_texts.txt")
print(type(test_texts))
print(len(test_texts))
print(type(test_texts[0]))


def get_optimal_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def sbert_embeddings(
    documents: List[str],
    batch_size: int = 32,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> torch.Tensor:
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = get_optimal_device()
    model.to(device)
    model.eval()

    loader = DataLoader(documents, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    for batch in tqdm(loader, desc="Encoding batches", unit="batch"):
        # batch is a list of strings
        encoded = tokenizer(
            list(batch), padding=True, truncation=True, return_tensors="pt"
        )
        # move to device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)
        emb = mean_pooling(output, encoded["attention_mask"])  # (batch_size, D)
        emb = F.normalize(emb, p=2, dim=1)  # (batch_size, D)
        all_embeddings.append(emb.cpu())

    # concatenate all batches
    return torch.cat(all_embeddings, dim=0)


train_embeddings = sbert_embeddings(train_texts)  # shape (len(train_texts), D)
test_embeddings = sbert_embeddings(test_texts)  # shape (len(test_texts),  D)

train_np = train_embeddings.cpu().numpy()
test_np = test_embeddings.cpu().numpy()

import numpy as np

np.savez_compressed("tm_datasets/20NG/contextual_data/train_contextual.npz", train_np)
np.savez_compressed("tm_datasets/20NG/contextual_data/test_contextual.npz", test_np)
