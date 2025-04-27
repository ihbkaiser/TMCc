import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornOTLoss(nn.Module):
    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=1000, stopThr=5e-3):
        super(SinkhornOTLoss, self).__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.eps = 1e-16

    def forward(self, cost: torch.Tensor) -> torch.Tensor:
        T, V = cost.shape
        device = cost.device
        a = torch.full((T, 1), 1.0 / T, device=device)
        b = torch.full((V, 1), 1.0 / V, device=device)
        u = a.clone()
        K = torch.exp(-self.sinkhorn_alpha * cost)
        err = 1.0
        cnt = 0
        while err > self.stopThr and cnt < self.OT_max_iter:
            v = b / (K.t() @ u + self.eps)
            u = a / (K @ v + self.eps)
            cnt += 1
            if cnt % 50 == 0:
                err = (v * (K.t() @ u) - b).abs().sum()
        pi = u * (K * v.t())
        return (pi * cost).sum() * self.weight_loss_ECR


def pairwise_euclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.pow(2).sum(1, keepdim=True) + y.pow(2).sum(1) - 2.0 * (x @ y.t())

class TMSD(nn.Module):
    @staticmethod
    def _logistic_normal_prior(n_topics: int):
        a = np.ones((1, n_topics), dtype=np.float32)
        mu2 = (np.log(a) - np.mean(np.log(a), keepdims=True)).reshape(-1)
        var2 = (((1.0 / a) * (1 - 2.0 / n_topics)) + (1.0 / n_topics**2) * np.sum(1.0 / a, keepdims=True)).reshape(-1)
        return mu2, var2

    def __init__(self, args) -> None:
        super(TMSD, self).__init__()
        self.n_topics = args.num_topic
        self.beta_temp = args.beta_temp
        self.alpha2 = args.adapter_alpha ** 2
        self.use_subdoc_phrase = True

        # phrases buffer
        phrase_pairs = getattr(args, 'phrase_pairs', None)
        idx = torch.from_numpy(phrase_pairs).long() if (phrase_pairs is not None and self.use_subdoc_phrase) else torch.empty((0,2), dtype=torch.long)
        self.register_buffer('phrase_idx', idx)
        self.Vp = idx.size(0)

        # encoder layers
        vocab_size = args.vocab_size
        enc_hidden = args.enc_hidden
        self.enc_fc1 = nn.Linear(vocab_size, enc_hidden)
        self.enc_fc2 = nn.Linear(enc_hidden, enc_hidden)
        self.drop = nn.Dropout(args.dropout)
        self.enc_mu = nn.Linear(enc_hidden, self.n_topics)
        self.enc_logv = nn.Linear(enc_hidden, self.n_topics)
        self.bn_mu = nn.BatchNorm1d(self.n_topics, affine=False)
        self.bn_lv = nn.BatchNorm1d(self.n_topics, affine=False)

        # decoder batchnorm
        self.decoder_bn = nn.BatchNorm1d(vocab_size)
        self.decoder_bn.weight.requires_grad = False

        # subdoc adapter
        self.sub_fc1 = nn.Linear(vocab_size, enc_hidden)
        self.sub_fc2 = nn.Linear(enc_hidden, enc_hidden)

        # prior
        mu0, var0 = TMSD._logistic_normal_prior(self.n_topics)
        self.prior_mu = nn.Parameter(torch.from_numpy(mu0), requires_grad=False)
        self.prior_var = nn.Parameter(torch.from_numpy(var0), requires_grad=False)

        # embeddings
        w_emb = torch.from_numpy(args.word_embeddings).float()
        self.word_embeddings  = nn.Parameter(F.normalize(w_emb), requires_grad=False)
        self.topic_embeddings = nn.Parameter(F.normalize(torch.randn(self.n_topics, w_emb.size(1))))
        self.cluster_centers  = nn.Parameter(torch.randn(args.num_cluster, self.n_topics))

        # OT loss
        self.ot_loss = SinkhornOTLoss(
            weight_loss_ECR=args.ot_weight,
            sinkhorn_alpha=args.ot_alpha,
            OT_max_iter=getattr(args, 'OT_max_iter', 1000),
            stopThr=getattr(args, 'stopThr', 5e-3)
        )

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def _encode(self, bow: torch.Tensor):
        h = F.softplus(self.enc_fc1(bow))
        h = F.softplus(self.enc_fc2(h))
        h = self.drop(h)
        mu = self.bn_mu(self.enc_mu(h))
        logv = self.bn_lv(self.enc_logv(h))
        z = self.reparameterise(mu, logv)
        theta = F.softmax(z, dim=1)
        var = logv.exp()
        kl = 0.5 * ((var / self.prior_var) + ((mu - self.prior_mu)**2 / self.prior_var)
                   + self.prior_var.log() - logv - 1).sum(1)
        return theta, kl, z

    def forward(self, x_word: torch.Tensor, phrase_counts: torch.Tensor = None, x_sub: torch.Tensor = None):
        # document encoding
        theta_doc, kl_doc, z = self._encode(x_word)

        # doc + phrase only
        if self.use_subdoc_phrase and phrase_counts is not None and x_sub is None:
            beta = F.softmax(-pairwise_euclidean(self.topic_embeddings, self.word_embeddings) / self.beta_temp, dim=1)
            recon = theta_doc.unsqueeze(1) @ beta  # [B,1,Vw]
            log_recon = torch.log(recon + 1e-10)
            # word reconstruction NLL
            nll_word = -(x_word.unsqueeze(1) * log_recon).sum(-1).mean()
            # phrase reconstruction NLL
            idx1, idx2 = self.phrase_idx.t()
            phrase_logp = log_recon[:, :, idx1] + log_recon[:, :, idx2]  # [B,1,Vp]
            nll_phrase = -(phrase_counts.unsqueeze(1) * phrase_logp).sum(-1).mean()
            nll = nll_word + nll_phrase
            kl_e = torch.zeros_like(kl_doc)

        # subdoc + phrase
        elif self.use_subdoc_phrase and x_sub is not None:
            B, S, Vw = x_sub.shape
            sf = x_sub.view(B * S, Vw)
            h = F.softplus(self.sub_fc1(sf))
            h = self.drop(F.softplus(self.sub_fc2(h)))
            mu_e = self.enc_mu(h)
            logv_e = self.enc_logv(h)
            e_ds = self.reparameterise(mu_e, logv_e).view(B, S, self.n_topics)
            kl_e = (0.5 * ((logv_e.exp() / self.alpha2) + ((mu_e - 1)**2 / self.alpha2)
                          + np.log(self.alpha2) - logv_e - 1)
                    .sum(1).view(B, S).mean(1))
            theta_ds = F.softmax(e_ds * theta_doc.unsqueeze(1), dim=-1)
            beta = F.softmax(-pairwise_euclidean(self.topic_embeddings, self.word_embeddings) / self.beta_temp, dim=1)
            recon = F.softmax(self.decoder_bn(theta_ds @ beta), dim=-1)
            log_recon = torch.log(recon + 1e-10)
            # word-level NLL over subdocs
            nll_word = -(x_sub * log_recon).sum(-1).mean()
            # phrase-level NLL
            idx1, idx2 = self.phrase_idx.t()
            phrase_logp = log_recon[:, :, idx1] + log_recon[:, :, idx2]  # [B,S,Vp]
            nll_phrase = -(phrase_counts.unsqueeze(1) * phrase_logp).sum(-1).mean()
            nll = nll_word + nll_phrase

        # doc-only
        else:
            beta = F.softmax(-pairwise_euclidean(self.topic_embeddings, self.word_embeddings) / self.beta_temp, dim=1)
            recon = theta_doc.unsqueeze(1) @ beta
            log_recon = torch.log(recon + 1e-10)
            nll = -(x_word.unsqueeze(1) * log_recon).sum(-1).mean()
            kl_e = torch.zeros_like(kl_doc)

        # OT loss
        loss_ot = self.ot_loss(pairwise_euclidean(self.topic_embeddings, self.word_embeddings))
        total_loss = nll + kl_doc.mean() + kl_e.mean() + loss_ot
        return {
            'loss': total_loss,
            'loss_recon': nll,
            'loss_kl_doc': kl_doc.mean(),
            'loss_kl_adapter': kl_e.mean(),
            'loss_ot': loss_ot,
            'theta': theta_doc
        }

    def get_theta(self, x_word: torch.Tensor, phrase_counts: torch.Tensor = None) -> torch.Tensor:
        theta, _, _ = self._encode(x_word)
        return theta

    def get_beta(self) -> torch.Tensor:
        cost = pairwise_euclidean(self.topic_embeddings, self.word_embeddings)
        return F.softmax(-cost / self.beta_temp, dim=1)

    def get_top_words(self, vocab: list[str], k: int = 15) -> list[list[str]]:
        beta = self.get_beta()
        top = beta.topk(k, dim=1).indices
        return [[vocab[i] for i in row.tolist()] for row in top]