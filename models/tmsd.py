import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
torch.autograd.set_detect_anomaly(True)

class SinkhornOTLoss(nn.Module):
    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=1000, stopThr=0.5e-2):
        super(SinkhornOTLoss, self).__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        device = M.device
        T, V = M.shape
        a = torch.full((T, 1), 1.0/T, device=device)
        b = torch.full((V, 1), 1.0/V, device=device)
        u = a.clone()
        K = torch.exp(-M * self.sinkhorn_alpha)
        err = 1.0
        cpt = 0
        while err > self.stopThr and cpt < self.OT_max_iter:
            v = b / (K.t().matmul(u) + self.epsilon)
            u = a / (K.matmul(v) + self.epsilon)
            cpt += 1
            if cpt % 50 == 1:
                bb = v * K.t().matmul(u)
                err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))
        transp = u * (K * v.t())
        loss_ECR = torch.sum(transp * M) * self.weight_loss_ECR
        return loss_ECR

class TMSD(nn.Module):
    def __init__(self, args):
        super(TMSD, self).__init__()
        # store settings
        self.beta_temp = args.beta_temp
        self.tau = args.tau
        self.num_cluster = args.num_cluster
        # logistic-normal prior params
        self.a = np.ones((1, args.num_topic), dtype=np.float32)
        self.mu2 = nn.Parameter(torch.tensor((np.log(self.a).T - np.mean(np.log(self.a), axis=1)).T))
        self.var2 = nn.Parameter(torch.tensor((((1.0 / self.a) * (1 - (2.0 / args.num_topic))).T +
                                               (1.0 / (args.num_topic * args.num_topic)) * np.sum(1.0 / self.a, axis=1)).T))
        # document encoder layers
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
        # subdoc flag and adapter layers
        self.use_subdoc = getattr(args, 'use_subdoc', False)
        if self.use_subdoc:
            self.sub_fc1 = nn.Linear(args.vocab_size, args.en1_units)
            self.sub_fc2 = nn.Linear(args.en1_units, args.en1_units)
            self.sub_fc21 = nn.Linear(args.en1_units, args.num_topic)
            self.sub_fc22 = nn.Linear(args.en1_units, args.num_topic)
            self.adapter_alpha2 = args.adapter_alpha ** 2
        # embeddings and clustering
        w_emb = torch.from_numpy(args.word_embeddings).float()
        self.word_embeddings = nn.Parameter(F.normalize(w_emb))
        self.topic_embeddings = torch.empty((args.num_topic, self.word_embeddings.shape[1]))
        nn.init.trunc_normal_(self.topic_embeddings, std=0.1)
        self.topic_embeddings = nn.Parameter(F.normalize(self.topic_embeddings))
        self.cluster_centers = nn.Parameter(torch.randn(args.num_cluster, args.num_topic))
        # OT loss
        self.ot_loss = SinkhornOTLoss(
            weight_loss_ECR=args.weight_loss_ECR,
            sinkhorn_alpha=args.sinkhorn_alpha,
            OT_max_iter=getattr(args, 'OT_max_iter', 5000),
            stopThr=getattr(args, 'stopThr', 0.5e-2)
        )
        
        # new hyperparam for the contextual Top2Vec loss
        self.use_ctx_loss    = getattr(args, "use_ctx_loss", False)
        self.ctx_sigma       = getattr(args, "ctx_sigma", 1.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def compute_loss_KL(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        var = logvar.exp()
        diff = mu - self.mu2
        return 0.5 * ((var / self.var2) + (diff * diff / self.var2) + self.var2.log() - logvar - 1).sum(1).mean()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_dropout(e1)
        mu = self.mean_bn(self.fc21(e1))
        logvar = self.logvar_bn(self.fc22(e1))
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        loss_KL = self.compute_loss_KL(mu, logvar)
        return theta, loss_KL, z

    def pairwise_dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_norm = x.pow(2).sum(1, keepdim=True)
        y_norm = y.pow(2).sum(1)
        return x_norm + y_norm - 2 * (x @ y.t())

    def get_beta(self) -> torch.Tensor:
        dist = self.pairwise_dist(self.topic_embeddings, self.word_embeddings)
        return F.softmax(-dist / self.beta_temp, dim=1)

    def forward(self, x: torch.Tensor, x_sub: torch.Tensor = None, m_sub: torch.Tensor = None) -> dict:
        # document-level encoding
        theta, kl_doc, z = self.encode(x)
        beta = self.get_beta()
        # choose branch
        if self.use_subdoc and x_sub is not None:
            B, S, V = x_sub.shape
            flat = x_sub.view(B*S, V)
            h = F.softplus(self.sub_fc1(flat))
            h = F.softplus(self.sub_fc2(h))
            mu_e = self.sub_fc21(h)
            logv_e = self.sub_fc22(h)
            z_e = self.reparameterize(mu_e, logv_e).view(B, S, -1)
            # adapter KL
            var_e = logv_e.exp()
            diff_e = mu_e - 1
            kl_ad = 0.5 * ((var_e / self.adapter_alpha2) + (diff_e * diff_e / self.adapter_alpha2)
                          + np.log(self.adapter_alpha2) - logv_e - 1)
            kl_ad = kl_ad.sum(1).view(B, S).mean(1)
            # reconstruction per subdoc
            theta_ds = F.softmax(z_e * theta.unsqueeze(1), dim=-1)
            recon = F.softmax(self.decoder_bn((theta_ds @ beta).view(-1, beta.size(1))).view(B, S, -1), dim=-1)
            nll = -(x_sub * torch.log(recon + 1e-10)).sum(-1).mean()
            kl_adapter = kl_ad.mean()
            recon_doc = F.softmax(self.decoder_bn(theta @ beta), dim=-1)
            nll_doc = -(x * torch.log(recon_doc + 1e-10)).sum(1).mean()
            nll = nll + nll_doc
        else:
            recon = F.softmax(self.decoder_bn(theta @ beta), dim=-1)
            nll = -(x * torch.log(recon + 1e-10)).sum(1).mean()
            kl_adapter = torch.tensor(0.0, device=x.device)
            theta_ds = theta.unsqueeze(1)  # treat the whole doc as one “subdoc”
            
        # ---------- new contextual Top2Vec regularizer ------------
        reg_loss = torch.tensor(0.0, device=x.device)
        if self.use_ctx_loss and m_sub is not None:
            B, S, K = theta_ds.shape
            # flatten:
            θ_flat = theta_ds.reshape(B*S, K)        # (N, K), N=B*S
            m_flat = m_sub.reshape(B*S, -1)          # (N, E)
            # normalize m to unit‐length for cosine
            m_norm = F.normalize(m_flat, dim=1)      # (N, E)
            # cosine similarity matrix
            W = torch.exp((m_norm @ m_norm.t()) / self.ctx_sigma)  # (N, N)
            # pairwise ‖θ_i - θ_j‖ 
            θ_norm2 = (θ_flat**2).sum(1, keepdim=True)             # (N,1)
            pair_d2 = θ_norm2 + θ_norm2.t() - 2*(θ_flat @ θ_flat.t())  # squared dist
            pair_d = torch.sqrt(pair_d2 + 1e-8)                     # (N,N)
            # half‐sum
            reg_loss = 0.5 * (W * pair_d).sum()
        
        cost = self.pairwise_dist(self.topic_embeddings, self.word_embeddings)
        loss_ot = self.ot_loss(cost)
        dist_z = self.pairwise_dist(z, self.cluster_centers)
        p = F.softmax(-dist_z / self.tau, dim=1)
        loss_cluster = (p * torch.sqrt(dist_z + 1e-8)).sum(1).mean()
        total = nll + kl_doc + kl_adapter + loss_ot + reg_loss
        return {
            'loss': total,
            'loss_recon': nll,
            'loss_kl_doc': kl_doc,
            'loss_kl_adapter': kl_adapter,
            'loss_ot': loss_ot,
            'loss_cluster': loss_cluster,
            'loss_ctx': reg_loss,
        }

    def get_theta(self, x: torch.Tensor) -> torch.Tensor:
        theta, _, _ = self.encode(x)
        return theta

    def get_top_words(self, vocab, num_top_words=15):
        dist = self.pairwise_euclidean_distance(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=1)
        top_values, top_indices = torch.topk(beta, num_top_words, dim=1)
        top_words = []
        for indices in top_indices:
            topic_words = [vocab[idx] for idx in indices.cpu().numpy()]
            top_words.append(topic_words)
        return top_words