"""
models/tmsd.py — Hierarchical Topic Model (VAE) with optional sub-doc & phrase modeling
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SinkhornOTLoss", "TMSD"]

def pairwise_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(1, keepdim=True) + y.pow(2).sum(1) - 2.0 * (x @ y.t())

def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    if mu.requires_grad:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std
    return mu

class SinkhornOTLoss(nn.Module):
    def __init__(self, weight=100.0, alpha=20.0, max_iter=1000, stop_thr=5e-3):
        super().__init__()
        self.weight = weight
        self.alpha = alpha
        self.max_iter = max_iter
        self.stop_thr = stop_thr
        self.eps = 1e-16
    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        T, V = cost.shape
        device = cost.device
        a = torch.full((T,1), 1.0/T, device=device)
        b = torch.full((V,1), 1.0/V, device=device)
        u = a.clone()
        K = torch.exp(-self.alpha * cost)
        for i in range(self.max_iter):
            v = b / (K.t() @ u + self.eps)
            u = a / (K @ v + self.eps)
            if (i+1) % 50 == 0:
                err = (v*(K.t()@u) - b).abs().sum()
                if err < self.stop_thr:
                    break
        pi = u * (K * v.t())
        return self.weight * (pi * cost).sum()

class TMSD(nn.Module):
    @staticmethod
    def _logistic_normal_prior(n_topics: int) -> Tuple[np.ndarray,np.ndarray]:
        a = np.ones((1,n_topics),dtype=np.float32)
        mu2 = (np.log(a) - np.mean(np.log(a),keepdims=True)).reshape(-1)
        var2 = (((1.0/a)*(1-2.0/n_topics)) + (1.0/n_topics**2)*np.sum(1.0/a,keepdims=True)).reshape(-1)
        return mu2, var2

    def __init__(
        self,
        vocab_word: int,
        phrase_pairs: Optional[np.ndarray] = None,
        n_topics: int = 50,
        enc_hidden: int = 256,
        dropout: float = 0.2,
        beta_temp: float = 1.0,
        n_cluster: int = 10,
        word_emb: np.ndarray | None = None,
        ot_weight: float = 100.0,
        ot_alpha: float = 20.0,
        adapter_alpha: float = 0.5,
        subdoc_phrase: bool = True # if True enable phrase & subdoc modeling
    ) -> None:
        super().__init__()
        self.n_topics = n_topics
        self.beta_temp = beta_temp
        self.alpha2 = adapter_alpha**2
        self.use_subdoc_phrase = subdoc_phrase
        # prepare phrase indices
        phrase_idx_tensor = torch.from_numpy(phrase_pairs).long() if (phrase_pairs is not None and subdoc_phrase) else torch.empty(0,2,dtype=torch.long)
        self.register_buffer("phrase_idx", phrase_idx_tensor)
        self.Vp = phrase_idx_tensor.size(0)
        # document encoder dimension
        doc_dim = vocab_word 
        # document encoder
        self.enc_fc1 = nn.Linear(doc_dim, enc_hidden)
        self.enc_fc2 = nn.Linear(enc_hidden, enc_hidden)
        self.decoder_bn = nn.BatchNorm1d(vocab_word)
        self.decoder_bn.weight.requires_grad = False
        self.enc_mu  = nn.Linear(enc_hidden, n_topics)
        self.enc_logv= nn.Linear(enc_hidden, n_topics)
        self.drop    = nn.Dropout(dropout)
        
        # subdoc adapter (only uses vocab_word dims)
        self.sub_fc1 = nn.Linear(vocab_word, enc_hidden)
        self.sub_fc2 = nn.Linear(enc_hidden, enc_hidden)
        # logistic-normal prior
        mu0,var0 = self._logistic_normal_prior(n_topics)
        self.prior_mu  = nn.Parameter(torch.from_numpy(mu0), requires_grad=False)
        self.prior_var = nn.Parameter(torch.from_numpy(var0), requires_grad=False)
        # embeddings
        if word_emb is None:
            raise ValueError("word_emb must be provided")
        w_emb = torch.from_numpy(word_emb).float()
        self.word_embeddings  = nn.Parameter(F.normalize(w_emb), requires_grad=False)
        self.topic_embeddings = nn.Parameter(F.normalize(torch.randn(n_topics,w_emb.size(1))))
        self.cluster_centres  = nn.Parameter(torch.randn(n_cluster,n_topics))
        # OT loss
        self.ot_loss = SinkhornOTLoss(weight=ot_weight, alpha=ot_alpha)
        self.bn_mu  = nn.BatchNorm1d(n_topics, affine=False)
        self.bn_lv  = nn.BatchNorm1d(n_topics, affine=False)

    def _encode(self, bow: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        h = F.softplus(self.enc_fc1(bow))
        h = F.softplus(self.enc_fc2(h))
        h = self.drop(h)
        mu = self.bn_mu(self.enc_mu(h))
        logv = self.bn_lv(self.enc_logv(h))
        z = reparameterise(mu,logv)
        theta = F.softmax(z,dim=1)
        var = logv.exp()
        kl = 0.5*((var/self.prior_var) + ((mu-self.prior_mu)**2/self.prior_var) + self.prior_var.log() - logv - 1).sum(1)
        return theta, kl, z

    def forward(
        self,
        x_word: torch.Tensor,       # [B, Vw]
        phrase_counts: Optional[torch.Tensor] = None,  # [B, Vp]
        x_sub: Optional[torch.Tensor] = None,         # [B, S, Vw]
    ) -> dict:
        # 1) document-level encoding
        bow_all = x_word
        # if self.use_subdoc_phrase:
        #     assert phrase_counts is not None, "phrase_counts required"
        #     bow_all = torch.cat([x_word, phrase_counts], dim=1)
        theta_doc, kl_doc, _ = self._encode(bow_all)

        # 2) decide mode
        if self.use_subdoc_phrase and x_sub is not None:
            # subdoc + phrase mode
            B,S,Vw = x_sub.shape
            sf = x_sub.view(B*S, Vw)
            h = F.softplus(self.sub_fc1(sf))
            h = self.drop(F.softplus(self.sub_fc2(h)))
            mu_e   = self.enc_mu(h)
            logv_e = self.enc_logv(h)
            e_ds   = reparameterise(mu_e,logv_e).view(B,S,self.n_topics)
            kl_e   = (0.5*((logv_e.exp()/self.alpha2)+((mu_e-1)**2/self.alpha2)+np.log(self.alpha2)-logv_e-1)
                      .sum(1).view(B,S).mean(1))
            theta_ds = F.softmax(e_ds*theta_doc.unsqueeze(1), dim=-1)
            # reconstruct words + phrases
            beta = F.softmax(-pairwise_euclidean(self.topic_embeddings,self.word_embeddings)/self.beta_temp, dim=1)
            # recon = theta_ds @ beta
            print(theta_ds.shape)
            print(beta.shape)
            recon = F.softmax(self.decoder_bn(theta_ds @ beta), dim=-1)
            log_recon = torch.log(recon + 1e-10)
            nll_word = -(x_sub * log_recon).sum(-1).mean()
            # idx1,idx2 = self.phrase_idx.t()
            # phrase_logp = log_recon[:,:,idx1] + log_recon[:,:,idx2]
            # nll_phrase = -(phrase_counts.unsqueeze(1) * phrase_logp).sum(-1).mean()
            nll = nll_word
        else:
            # doc-only mode
            recon = theta_doc.unsqueeze(1) @ F.softmax(-pairwise_euclidean(self.topic_embeddings,self.word_embeddings)/self.beta_temp, dim=1)
            log_recon = torch.log(recon + 1e-10)
            nll = -(x_word.unsqueeze(1) * log_recon).sum(-1).mean()
            kl_e = torch.zeros_like(kl_doc)

        # 3) OT regularizer
        loss_ot = self.ot_loss(pairwise_euclidean(self.topic_embeddings,self.word_embeddings))

        total = nll + kl_doc.mean() + kl_e.mean() + loss_ot
        return {"loss": total, "loss_recon": nll, "loss_kl_doc": kl_doc.mean(), "loss_kl_adapter": kl_e.mean(), "loss_ot": loss_ot, "theta": theta_doc}

    def get_theta(self, x_word: torch.Tensor, phrase_counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        bow_all = x_word
        if self.use_subdoc_phrase:
            bow_all = torch.cat([x_word, phrase_counts], dim=1)
        theta,_,_ = self._encode(bow_all)
        return theta

    def get_top_words(self, vocab: List[str], k: int = 15) -> List[List[str]]:
        beta = F.softmax(-pairwise_euclidean(self.topic_embeddings,self.word_embeddings)/self.beta_temp, dim=1)
        top  = beta.topk(k, dim=1).indices
        return [[vocab[i] for i in row.tolist()] for row in top]
