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
        # Soft assignment
        q = F.softmax(-dist_sq / self.tau, dim=1)
        loss = torch.mean(torch.sum(q * dist_sq, dim=1))
        return q, loss
