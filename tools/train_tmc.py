import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from models.tmc import TMC

def train(args):
    embeddings = np.load(args.embeddings_path)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(embeddings_tensor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    latent_dim = embeddings.shape[1]
    model = TMC(input_dim=latent_dim, hidden_dim=args.hidden_dim, num_clusters=args.num_clusters, tau=args.tau)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.num_epochs):
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
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension for TMC')
    parser.add_argument('--save_path', type=str, default='tmc_model_ver2.pth', help='Path to save the trained model')
    args = parser.parse_args()
    train(args)
