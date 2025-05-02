#!/usr/bin/env python3
"""
Script to perform KMeans clustering on document embeddings
and save cluster labels to a .npy file.

Usage:
    python cluster_documents.py \
        --input tm_datasets/20NG/contextual_data/train_contextual.npz \
        --output tm_datasets/20NG/contextual_data/train_labels.npy \
        --n_clusters 50
"""
import argparse
import numpy as np
from sklearn.cluster import KMeans

def main():
    parser = argparse.ArgumentParser(
        description='Cluster document embeddings and save labels.'
    )
    parser.add_argument(
        '--input', type=str,
        default='tm_datasets/20NG/contextual_data/train_contextual.npz',
        help='Path to input .npz file containing embeddings.'
    )
    parser.add_argument(
        '--output', type=str,
        default='tm_datasets/20NG/contextual_data/train_labels.npy',
        help='Path to output .npy file to save cluster labels.'
    )
    parser.add_argument(
        '--n_clusters', type=int, default=50,
        help='Number of clusters for KMeans.'
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Random state for reproducibility.'
    )
    args = parser.parse_args()

    # Load embeddings
    data = np.load(args.input, allow_pickle=True)
    # embeddings assumed under key 'arr_0'
    X = data['arr_0']

    # Run KMeans
    kmeans = KMeans(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        n_init='auto'
    )
    labels = kmeans.fit_predict(X)

    # Save labels to .npy
    np.save(args.output, labels)

    # Report summary
    counts = np.bincount(labels, minlength=args.n_clusters)
    print(f"Saved cluster labels to: {args.output}")
    print("Cluster counts:", counts)

if __name__ == '__main__':
    main()
