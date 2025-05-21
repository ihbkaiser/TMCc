#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from types import SimpleNamespace
from tqdm import tqdm
# Ensure project root is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import wandb
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from fastopic import FASTopic
from topmost import eva 
from topmost.preprocess import Preprocess
from utils.coherence_wiki import TC_on_wikipedia
from utils.irbo import buubyyboo_dth
from utils.file_utils import read_text

# Parse command-line arguments
parser = argparse.ArgumentParser(description="FastTopic trainer for 20NG dataset")
parser.add_argument('--data_path', type=str, default='tm_datasets/20NG')
parser.add_argument('--num_topics', type=int, default=50, help="number of topics")
parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), choices=['cuda', 'cpu'], help="compute device")
parser.add_argument('--epochs', type=int, default=10, help="number of epochs to train")
parser.add_argument('--log_interval', type=int, default=1, help="logging interval (epochs)")
parser.add_argument('--project_name', type=str, default='20NG-FastTopic')
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger('main')

def train():
    """
    Train FastTopic on 20NG dataset.
    """
    wandb.init(project=args.project_name)
    wandb.config.update({
        "num_topics": args.num_topics,
        "epochs": args.epochs,
        "device": args.device,
    })
    
    logger.info("Loading 20 Newsgroups dataset...")
    
    # # Load 20 Newsgroups dataset
    # train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    # test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    # train_docs = train_data['data']               #type: ignore    
    # test_docs = test_data['data']                 #type: ignore
    # train_labels = train_data['target']           #type: ignore
    # test_labels = test_data['target']              #type: ignore
    
    train_docs = read_text(f"{args.data_path}/train_texts.txt")
    train_labels = read_text(f"{args.data_path}/train_labels.txt")
    test_docs = read_text(f"{args.data_path}/test_texts.txt")
    test_labels = read_text(f"{args.data_path}/test_labels.txt")
    
    test_labels = np.array(test_labels, dtype=int)
    train_labels = np.array(train_labels, dtype=int)
    
    
    logger.info(f"Train docs: {len(train_docs)}, Test docs: {len(test_docs)}")
    
    # Preprocess the data
    logger.info("Preprocessing data...")
    preprocess = Preprocess(vocab_size=10000)
    
    # Initialize FastTopic model with the specified parameters
    logger.info(f"Initializing FastTopic model with {args.num_topics} topics...")
    model = FASTopic(
        num_topics=args.num_topics, 
        preprocess=preprocess,        #type: ignore
        device=args.device,
        verbose=True
    )
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train for specified number of epochs
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        
        # Train on training set
        logger.info("Training on train data...")
        if epoch == 1:
            # First epoch: full fit_transform
            top_words, train_doc_topic_dist = model.fit_transform(train_docs, epochs=1)
        else:
            # Continue training for subsequent epochs
            model.fit(train_docs, epochs=1)
            train_doc_topic_dist = model.transform(train_docs)
        
        # Evaluate on test set
        logger.info("Evaluating on test data...")
        test_doc_topic_dist = model.transform(test_docs)
        
        # Evaluate the model
        if epoch % args.log_interval == 0:
            # Get clustering metrics
            clus = eva._clustering(test_doc_topic_dist, test_labels)
            logger.info(f"Clustering result: {clus}")
            
            # Get topic diversity
            tw = model.get_top_words()
            td = eva._diversity([' '.join(t) for t in tw])
            logger.info(f"Diversity TD: {td:.4f}")
            
            # Get topic coherence
            _, cv = TC_on_wikipedia(tw, cv_type="C_V")
            logger.info(f"Coherence Cv: {cv:.4f}")
            
            # Get IRBO
            irbo = buubyyboo_dth(tw, topk=15)
            logger.info(f"IRBO: {irbo:.4f}")
            
            # Log metrics to W&B
            metric_data = {
                "epoch": epoch,
                "Coherence_Cv": cv,
                "Diversity_TD": td,
                "IRBO": irbo
            }
            if isinstance(clus, dict):
                for ck, cvl in clus.items():
                    metric_data[f"clustering/{ck}"] = cvl
            wandb.log(metric_data)
    
    # Save the final model
    out_dir = "outputs/fastopic"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"fastopic_{args.num_topics}topics.zip")
    model.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    # Get final metrics
    final_tw = model.get_top_words()
    final_td = eva._diversity([' '.join(t) for t in final_tw])
    _, final_cv = TC_on_wikipedia(final_tw, cv_type="C_V")
    final_irbo = buubyyboo_dth(final_tw, topk=15)
    
    logger.info("Final metrics:")
    logger.info(f"Topic Coherence (Cv): {final_cv:.4f}")
    logger.info(f"Topic Diversity (TD): {final_td:.4f}")
    logger.info(f"IRBO: {final_irbo:.4f}")
    
    # # Print top words for each topic
    # logger.info("Top words for each topic:")
    # for i, words in enumerate(final_tw):
    #     logger.info(f"Topic {i}: {', '.join(words[:10])}")
    
    return model, train_doc_topic_dist, test_doc_topic_dist         #type: ignore

if __name__ == "__main__":
    train()