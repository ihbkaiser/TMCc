import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornOTLoss(nn.Module):
    def __init__(self, weight_loss_ECR, sinkhorn_alpha, OT_max_iter=1000, stopThr=0.5e-2):
        super(SinkhornOTLoss, self).__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.weight_loss_ECR = weight_loss_ECR
        self.stopThr = stopThr
        self.epsilon = 1e-16

    def forward(self, M):
        device = M.device
        a = (torch.ones(M.size(0)) / M.size(0)).unsqueeze(1).to(device)
        b = (torch.ones(M.size(1)) / M.size(1)).unsqueeze(1).to(device)
        u = torch.ones_like(a) / a.size(0)

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
        loss_ECR = torch.sum(transp * M)
        return loss_ECR * self.weight_loss_ECR

class VAE_OT(nn.Module):
    def __init__(self, args):
        super(VAE_OT, self).__init__()
        self.args = args
        self.num_topic = args.num_topic
        self.beta_temp = args.beta_temp
        self.tau = args.tau
        self.num_cluster = args.num_cluster
        # priors
        a = np.ones((1, args.num_topic), dtype=np.float32)
        mu2 = (np.log(a).T - np.mean(np.log(a), axis=1)).T
        var2 = (((1.0 / a) * (1 - 2.0/args.num_topic)).T +
                (1.0/(args.num_topic**2))*np.sum(1.0/a,axis=1)).T
        self.mu2 = nn.Parameter(torch.tensor(mu2), requires_grad=False)
        self.var2 = nn.Parameter(torch.tensor(var2), requires_grad=False)

        # document encoder
        self.fc11 = nn.Linear(args.vocab_size, args.en1_units)
        self.fc12 = nn.Linear(args.en1_units, args.en1_units)
        self.fc21 = nn.Linear(args.en1_units, args.num_topic)
        self.fc22 = nn.Linear(args.en1_units, args.num_topic)
        self.dropout = nn.Dropout(args.dropout)
        self.mean_bn = nn.BatchNorm1d(args.num_topic, affine=False)
        self.logvar_bn = nn.BatchNorm1d(args.num_topic, affine=False)

        # decoders
        self.decoder_bn = nn.BatchNorm1d(args.vocab_size, affine=False)
        self.phrase_decoder_bn = nn.BatchNorm1d(args.phrase_vocab_size, affine=False)

        # sub-document encoder
        self.sub_fc1 = nn.Linear(args.vocab_size, args.en1_units)
        self.sub_fc2 = nn.Linear(args.en1_units, args.en1_units)
        self.sub_fc21 = nn.Linear(args.en1_units, args.num_topic)
        self.sub_fc22 = nn.Linear(args.en1_units, args.num_topic)
        self.register_buffer('adapter_alpha2', torch.tensor(args.adapter_alpha ** 2))


        # embeddings
        self.word_embeddings = nn.Parameter(
            F.normalize(torch.from_numpy(args.word_embeddings).float())
        )
        self.phrase_embeddings = nn.Parameter(
            F.normalize(torch.from_numpy(args.phrase_embeddings).float())
        )
        self.topic_embeddings = nn.Parameter(
            F.normalize(torch.randn(args.num_topic, self.word_embeddings.size(1)))
        )
        self.cluster_centers = nn.Parameter(
            torch.randn(args.num_cluster, args.num_topic)
        )
        self.ot_loss = SinkhornOTLoss(
            weight_loss_ECR=args.weight_loss_ECR,
            sinkhorn_alpha=args.sinkhorn_alpha,
            OT_max_iter=args.OT_max_iter,
            stopThr=args.stopThr
        )

    def reparam(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        return mu

    def encode(self, x):
        h = F.softplus(self.fc11(x))
        h = F.softplus(self.fc12(h))
        h = self.dropout(h)
        mu = self.mean_bn(self.fc21(h))
        logvar = self.logvar_bn(self.fc22(h))
        z = self.reparam(mu, logvar)
        theta = F.softmax(z, dim=1)
        var = torch.exp(logvar)
        kld = 0.5*((var/self.var2 + (mu-self.mu2)**2/self.var2 + torch.log(self.var2) - logvar)
                   .sum(1) - self.num_topic).mean()
        return theta, kld, z

    def pairwise_dist(self, x, y):
        x_norm = x.pow(2).sum(1, keepdim=True)
        y_norm = y.pow(2).sum(1)
        return x_norm + y_norm - 2*x.matmul(y.t())

    def get_beta(self):
        M = self.pairwise_dist(self.topic_embeddings, self.word_embeddings)
        return F.softmax(-M/self.beta_temp,dim=0)

    def get_beta_phrase(self):
        M_p = self.pairwise_dist(self.topic_embeddings, self.phrase_embeddings)
        return F.softmax(-M_p/self.beta_temp,dim=0)

    def forward(self, x, x_sub=None, x_phrase=None):
        theta, kld, z = self.encode(x)
        beta = self.get_beta()
        recon_doc = F.softmax(self.decoder_bn(theta.matmul(beta)),dim=-1)
        loss_recon = -(x*recon_doc.log()).sum(1).mean()
        loss_TM = loss_recon + kld
        if x_sub is not None:
            B,S,V = x_sub.shape
            flat = x_sub.view(B*S,V)
            h = F.softplus(self.sub_fc1(flat))
            h = F.softplus(self.sub_fc2(h))
            mu_e = self.sub_fc21(h)
            logv_e = self.sub_fc22(h)
            z_e = self.reparam(mu_e, logv_e).view(B,S,-1)
            var_e = torch.exp(logv_e)
            diff_e = mu_e -1
            kl_ad = 0.5*((var_e/self.adapter_alpha2)+(diff_e**2/self.adapter_alpha2)
                         +torch.log(self.adapter_alpha2)-logv_e-1)
            kl_ad = kl_ad.sum(1).view(B,S).mean(1).mean()
            theta_ds = F.softmax(z_e*theta.unsqueeze(1),dim=-1)
            recon_sub = F.softmax(self.decoder_bn((theta_ds@beta).view(-1,beta.size(1))).view(B,S,-1),dim=-1)
            nll = -(x_sub*recon_sub.log()).sum(-1).mean()
            loss_TM += nll+kl_ad
        phrase_loss = torch.tensor(0.0, device=x.device)
        if x_phrase is not None:
            beta_p = self.get_beta_phrase()
            recon_phrase = F.softmax(self.phrase_decoder_bn(theta.matmul(beta_p)),dim=-1)
            phrase_loss = -(x_phrase*recon_phrase.log()).sum(1).mean()
            loss_TM += phrase_loss
        M = self.pairwise_dist(self.topic_embeddings, self.word_embeddings)
        loss_ot = self.ot_loss(M)
        dist_z = self.pairwise_dist(z, self.cluster_centers)
        p = F.softmax(-dist_z/self.tau,dim=1)
        cluster2 = (p*torch.sqrt(dist_z+1e-8)).sum(1).mean()
        loss = loss_TM + loss_ot + cluster2
        return {'loss':loss, 'loss_TM':loss_TM, 'loss_cluster':loss_ot, 'phrase_loss':phrase_loss}
    def get_theta(self, x):
        theta, _, _ = self.encode(x)
        return theta
    def get_top_words(self, vocab, num_top_words=15):
        dist = self.pairwise_dist(self.topic_embeddings, self.word_embeddings)
        beta = F.softmax(-dist / self.beta_temp, dim=1)
        top_values, top_indices = torch.topk(beta, num_top_words, dim=1)
        top_words = []
        for indices in top_indices:
            topic_words = [vocab[idx] for idx in indices.cpu().numpy()]
            top_words.append(topic_words)
        return top_words
    