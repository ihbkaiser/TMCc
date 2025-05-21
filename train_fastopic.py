#!/usr/bin/env python3
"""Train a FastTopic model on the 20 Newsgroups dataset."""

import argparse
import logging
import os
import sys
from typing import Tuple

import numpy as np
import torch
import wandb
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastopic import FASTopic
from topmost import eva
from topmost.preprocess import Preprocess
from utils.coherence_wiki import TC_on_wikipedia
from utils.file_utils import read_text
from utils.irbo import buubyyboo_dth

wandb.login(key='25283834ecbe7bd282505b0721ea3adcd8e789d3', relogin=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train FastTopic model on 20NG dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="tm_datasets/20NG",
        help="Path to dataset",
    )
    parser.add_argument(
        "--num_topics",
        type=int,
        default=50,
        help="Number of topics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Compute device",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help="Logging interval (epochs)",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="20NG-FastTopic",
        help="W&B project name",
    )
    return parser.parse_args()


def load_data(data_path: str) -> Tuple[list, list, np.ndarray, np.ndarray, list]:
    """Load and preprocess dataset from text files."""
    logger.info("Loading 20 Newsgroups dataset...")
    try:
        train_docs = read_text(os.path.join(data_path, "train_texts.txt"))
        train_labels = np.array(read_text(os.path.join(data_path, "train_labels.txt")), dtype=int)
        test_docs = read_text(os.path.join(data_path, "test_texts.txt"))
        test_labels = np.array(read_text(os.path.join(data_path, "test_labels.txt")), dtype=int)
        vocabs = read_text(os.path.join(data_path, "vocab.txt"))
        logger.info(f"Train docs: {len(train_docs)}, Test docs: {len(test_docs)}, Vocab size: {len(vocabs)}")
        return train_docs, train_labels, test_docs, test_labels, vocabs         #type: ignore
    except FileNotFoundError as e:
        logger.error(f"Dataset files not found: {e}")
        sys.exit(1)


def train_model(args: argparse.Namespace) -> Tuple[FASTopic, np.ndarray, np.ndarray]:
    """Train FastTopic model and evaluate performance."""
    # Initialize W&B
    wandb.init(project=args.project_name)
    wandb.config.update({
        "num_topics": args.num_topics,
        "epochs": args.epochs,
        "device": args.device,
    })

    # Load data
    train_docs, train_labels, test_docs, test_labels, vocabs = load_data(args.data_path)

    # Preprocess data
    logger.info("Preprocessing data...")
    preprocess = Preprocess(vocab_size=len(vocabs))
    # preprocess = vocabs

    # Initialize model
    logger.info(f"Initializing FastTopic model with {args.num_topics} topics...")
    model = FASTopic(
        num_topics=args.num_topics,
        preprocess=preprocess,         #type: ignore
        device=args.device,
        verbose=True,
    )

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        logger.info(f"Epoch {epoch}/{args.epochs}")

        # Train
        logger.info("Training on train data...")
        if epoch == 1:
            top_words, train_doc_topic_dist = model.fit_transform(train_docs, epochs=1)          #type: ignore
        else:
            model.fit(train_docs, epochs=1)            #type: ignore
            train_doc_topic_dist = model.transform(train_docs)              #type: ignore
 
        # Evaluate
        logger.info("Evaluating on test data...")
        test_doc_topic_dist = model.transform(test_docs)             #type: ignore

        # Log metrics
        if epoch % args.log_interval == 0:
            metrics = compute_metrics(model, test_doc_topic_dist, test_labels)
            wandb.log({"epoch": epoch, **metrics})

    # Save model
    out_dir = "outputs/fastopic"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"fastopic_{args.num_topics}topics.zip")
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Log final metrics
    final_metrics = compute_metrics(model, test_doc_topic_dist, test_labels)           #type: ignore
    logger.info("Final metrics:")
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    return model, train_doc_topic_dist, test_doc_topic_dist          #type: ignore


def compute_metrics(model: FASTopic, test_doc_topic_dist: np.ndarray, test_labels: np.ndarray) -> dict:
    """Compute evaluation metrics for the model."""
    # Clustering metrics
    clustering_metrics = eva._clustering(test_doc_topic_dist, test_labels)
    metrics = {f"clustering/{k}": v for k, v in clustering_metrics.items()}

    # Topic diversity
    top_words = model.get_top_words()
    diversity = eva._diversity([" ".join(t) for t in top_words])
    metrics["Diversity_TD"] = diversity

    # Topic coherence
    _, coherence = TC_on_wikipedia(top_words, cv_type="C_V")
    metrics["Coherence_Cv"] = coherence

    # IRBO
    irbo = buubyyboo_dth(top_words, topk=15)
    metrics["IRBO"] = irbo

    logger.info(f"Clustering result: {clustering_metrics}")
    logger.info(f"Diversity TD: {diversity:.4f}")
    logger.info(f"Coherence Cv: {coherence:.4f}")
    logger.info(f"IRBO: {irbo:.4f}")

    return metrics


def main():
    """Main function to execute training."""
    args = parse_args()
    try:
        train_model(args)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()