import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class TMC(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=512, num_clusters=10, tau=1.0):
        super(TMC, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.cluster_centers = torch.empty(num_clusters, hidden_dim)
        nn.init.trunc_normal_(self.cluster_centers, std=0.1)
        self.cluster_centers = nn.Parameter(F.normalize(self.cluster_centers, p=2, dim=1))
        
        self.tau = tau

    def forward(self, z):
        hidden = self.linear(z)
        diff = hidden.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
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
    if args.save_path:
        dir_path = os.path.dirname(args.save_path)
        if dir_path: 
            os.makedirs(dir_path, exist_ok=True)
        torch.save(model.state_dict(), args.save_path)
        print(f"Model saved to {args.save_path}")


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
    parser.add_argument('--save_path', type=str, default='tmc_model_ver2.pth', help='Path to save the trained model')

    args = parser.parse_args()
    main(args)
