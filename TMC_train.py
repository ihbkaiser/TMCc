import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TMC(nn.Module):
    def __init__(self, latent_dim, num_clusters, tau=1.0):
        super(TMC, self).__init__()
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, latent_dim))
        self.tau = tau

    def forward(self, z):
        diff = z.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=2)
        q = F.softmax(-dist_sq / self.tau, dim=1)
        loss = torch.mean(torch.sum(q * dist_sq, dim=1))
        return q, loss

def main(args):
    embeddings = np.load(args.embeddings_path)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(embeddings_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    latent_dim = embeddings.shape[1]
    num_clusters = args.num_clusters
    model = TMC(latent_dim, num_clusters, tau=args.tau)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_epochs = args.num_epochs
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            q, loss = model(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataset):.4f}")
    
    model.eval()
    with torch.no_grad():
        q, _ = model(embeddings_tensor)
        assignments = torch.argmax(q, dim=1).cpu().numpy()
    print(f"Cluster assignments shape: {assignments.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TMC model for clustering embeddings')
    parser.add_argument('--embeddings_path', type=str, required=True, help='Path to the embeddings file (numpy .npy)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--num_clusters', type=int, required=True, help='Number of clusters (topics)')
    parser.add_argument('--tau', type=float, default=1.0, help='Temperature parameter for soft assignment')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the model')
    
    args = parser.parse_args()
    main(args)
