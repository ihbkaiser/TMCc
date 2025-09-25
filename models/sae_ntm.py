import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from sentence_transformers import SentenceTransformer
'''
SAE-NTM: Sentence-Aware Encoder for Neural Topic Modeling
- Based on the paper description
- Assumes emb_batch is list of tensors (pre-computed embeddings per doc)
'''
class SAE_NTM(nn.Module):
    def __init__(self, args):
        super(SAE_NTM, self).__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_topic = args.num_topic
        self.en1_units = args.en1_units
        self.dropout = nn.Dropout(args.dropout)
        self.a = np.ones((1, args.num_topic), dtype=np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), axis=1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T +
                                                 (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, axis=1)).T))
        self.mu2.requires_grad = False
        self.var2.requires_grad = False
        self.fc11 = nn.Linear(args.vocab_size, args.en1_units)
        self.fc12 = nn.Linear(args.en1_units, args.en1_units)
        self.fc1_dropout = nn.Dropout(args.dropout)
        self.sentence_dim = 384  # Hardcode for 'all-MiniLM-L6-v2'; adjust if model changes
        self.query_proj = nn.Linear(args.en1_units, self.sentence_dim)
        self.key_proj = nn.Linear(self.sentence_dim, self.sentence_dim)
        self.value_proj = nn.Linear(self.sentence_dim, self.sentence_dim)
        self.mean_linear = nn.Linear(self.sentence_dim, args.num_topic)
        self.logvar_linear = nn.Linear(self.sentence_dim, args.num_topic)
        self.mean_bn = nn.BatchNorm1d(args.num_topic)
        self.mean_bn.weight.requires_grad = False
        self.logvar_bn = nn.BatchNorm1d(args.num_topic)
        self.logvar_bn.weight.requires_grad = False
        self.decoder_linear = nn.Linear(args.num_topic, args.vocab_size)
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size)
        self.decoder_bn.weight.requires_grad = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def compute_loss_KL(self, mu, logvar):
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(dim=1) - self.args.num_topic)
        return KLD.mean()

    def forward(self, x, emb_batch):
        device = x.device
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        h_bow = self.fc1_dropout(e1)  # B x en1_units
        d_list = []
        for b in range(x.shape[0]):
            h_sent = emb_batch[b]  # m x D (tensor)
            m = h_sent.shape[0] if len(h_sent.shape) > 1 else 0
            if m == 0:
                d_b = torch.zeros(1, self.sentence_dim, device=device)
            else:
                query = self.query_proj(h_bow[b].unsqueeze(0))  # 1 x D
                key = self.key_proj(h_sent)  # m x D
                scores = torch.matmul(query, key.T) / math.sqrt(self.sentence_dim)  # 1 x m
                alpha = F.softmax(scores, dim=-1)
                value = self.value_proj(h_sent)  # m x D
                d_b = torch.matmul(alpha, value)  # 1 x D
            d_list.append(d_b)
        d_i = torch.cat(d_list, dim=0)  # B x D
        mu = self.mean_bn(self.mean_linear(d_i))
        logvar = self.logvar_bn(self.logvar_linear(d_i))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        logit = self.decoder_linear(theta)
        recon = F.softmax(self.decoder_bn(logit), dim=-1)
        recon_loss = -(x * recon.log()).sum(dim=1).mean()
        loss_KL = self.compute_loss_KL(mu, logvar)
        loss_TM = recon_loss + loss_KL
        return {'loss': loss_TM, 'loss_TM': loss_TM}

    def get_theta(self, x, emb_batch):
        with torch.no_grad():
            device = x.device
            e1 = F.softplus(self.fc11(x))
            e1 = F.softplus(self.fc12(e1))
            h_bow = self.fc1_dropout(e1)
            d_list = []
            for b in range(x.shape[0]):
                h_sent = emb_batch[b]  # m x D (tensor)
                m = h_sent.shape[0] if len(h_sent.shape) > 1 else 0
                if m == 0:
                    d_b = torch.zeros(1, self.sentence_dim, device=device)
                else:
                    query = self.query_proj(h_bow[b].unsqueeze(0))
                    key = self.key_proj(h_sent)
                    scores = torch.matmul(query, key.T) / math.sqrt(self.sentence_dim)
                    alpha = F.softmax(scores, dim=-1)
                    value = self.value_proj(h_sent)
                    d_b = torch.matmul(alpha, value)
                d_list.append(d_b)
            d_i = torch.cat(d_list, dim=0)
            mu = self.mean_bn(self.mean_linear(d_i))
            logvar = self.logvar_bn(self.logvar_linear(d_i))
            z = self.reparameterize(mu, logvar)
            theta = F.softmax(z, dim=1)
        return theta

    def get_top_words(self, vocab, num_top_words=15):
        beta = self.decoder_linear.weight.data.cpu().numpy().T  # K x V
        top_words = []
        for i in range(self.num_topic):
            topic_dist = beta[i]
            top_idx = np.argsort(topic_dist)[-num_top_words:][::-1]
            topic_words = np.array(vocab)[top_idx]
            top_words.append(list(topic_words))
        return top_words