import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEDeepKMeans(nn.Module):
    def __init__(self, args):
        super(VAEDeepKMeans, self).__init__()
        self.args = args
        self.beta_temp = args.beta_temp
        self.tau = args.tau                  
        self.num_cluster = args.num_cluster 
        self.a = np.ones((1, args.num_topic), dtype=np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), axis=1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T +
                                                   (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, axis=1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(args.vocab_size, args.en1_units)
        self.fc12 = nn.Linear(args.en1_units, args.en1_units)
        self.fc21 = nn.Linear(args.en1_units, args.num_topic)
        self.fc22 = nn.Linear(args.en1_units, args.num_topic)
        self.fc1_dropout = nn.Dropout(args.dropout)
        self.theta_dropout = nn.Dropout(args.dropout)

        self.mean_bn = nn.BatchNorm1d(args.num_topic)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size)
        self.decoder_bn.weight.requires_grad = False

        self.word_embeddings = torch.from_numpy(args.word_embeddings).float()
        self.word_embeddings = nn.Parameter(F.normalize(self.word_embeddings))
        self.topic_embeddings = torch.empty((args.num_topic, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))
        self.cluster_centers = nn.Parameter(torch.randn(args.num_cluster, args.num_topic))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def encode(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        loss_KL = self.compute_loss_KL(mu, logvar)
        return theta, loss_KL, z

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(dim=1) - self.args.num_topic)
        return KLD.mean()

    def pairwise_euclidean_distance(self, x, y):
        x_norm = torch.sum(x ** 2, dim=1, keepdim=True)
        y_norm = torch.sum(y ** 2, dim=1)
        return x_norm + y_norm - 2 * torch.matmul(x, y.t())

    def get_beta(self):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=0)
        return beta

    def forward(self, x):
        theta, loss_KL, z = self.encode(x)
        beta = self.get_beta()
        recon = F.softmax(self.decoder_bn(torch.matmul(theta, beta)), dim=-1)
        recon_loss = -(x * recon.log()).sum(dim=1).mean()
        loss_TM = recon_loss + loss_KL

        dist = self.pairwise_euclidean_distance(z, self.cluster_centers)
        p = F.softmax(-dist / self.tau, dim=1)
        distances = torch.sqrt(dist + 1e-8)
        cluster_loss = (p * distances).sum(dim=1).mean()

        loss = loss_TM + cluster_loss
        return {'loss': loss, 'loss_TM': loss_TM, 'loss_cluster': cluster_loss}

    def get_theta(self, x):
        theta, _, _ = self.encode(x)
        return theta
