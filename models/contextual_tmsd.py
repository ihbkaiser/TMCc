import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tmsd2 import VAE_OT, SinkhornOTLoss
class ContextualTMSD(VAE_OT):
    def __init__(self, args):
        super().__init__(args)
        V = args.vocab_size

        self.sub_embed_mlp = nn.Sequential(
            nn.Linear(args.sub_embed_dim, args.sub_proj_hidden),
            nn.ReLU(),
            nn.Linear(args.sub_proj_hidden, V),
        )

        self.sub_fc1   = nn.Linear(2 * V,         args.en1_units)
        self.sub_fc2   = nn.Linear(args.en1_units, args.en1_units)
        self.sub_fc21  = nn.Linear(args.en1_units, args.num_topic)
        self.sub_fc22  = nn.Linear(args.en1_units, args.num_topic)

        self.adapter_alpha2 = args.adapter_alpha ** 2

    def forward(self, x_doc, x_sub_bow, x_sub_embed):
        B, Q, V = x_sub_bow.shape
        sub_proj = (
            self.sub_embed_mlp(
                x_sub_embed.view(B*Q, -1)       # (B·Q, C)
            )
            .view(B, Q, V)                     # (B, Q, V)
        )

        sub_comb = torch.cat([x_sub_bow, sub_proj], dim=-1)  # (B, Q, 2V)
        flat_sub = sub_comb.view(B*Q, -1)                    # (B·Q, 2V)

        h = F.softplus(self.sub_fc1(flat_sub))
        h = F.softplus(self.sub_fc2(h))
        mu_e    = self.sub_fc21(h)
        logv_e  = self.sub_fc22(h)
        z_e     = self.reparameterize(mu_e, logv_e).view(B, Q, -1)

        var_e   = logv_e.exp()
        diff_e  = mu_e - 1
        kl_terms = 0.5 * (
            (var_e / self.adapter_alpha2)
          + (diff_e * diff_e / self.adapter_alpha2)
          + np.log(self.adapter_alpha2)
          - logv_e
          - 1
        )

        kl_ad    = kl_terms.sum(dim=1).view(B, Q).mean(dim=1)
        kl_adapter = kl_ad.mean()


        theta_doc, loss_kl_doc, _ = self.encode(x_doc)         

        beta       = self.get_beta()                           

        recon_doc  = F.softmax(self.decoder_bn(theta_doc @ beta), dim=-1)
        recon_loss_doc = -(x_doc * torch.log(recon_doc + 1e-10)).sum(dim=1).mean()


        theta_sub = F.softmax(z_e * theta_doc.unsqueeze(1), dim=-1)

        recon_sub = (
            F.softmax(
                self.decoder_bn((theta_sub.view(B*Q, -1) @ beta)), dim=-1
            )
            .view(B, Q, V)
        )

        recon_loss_sub = -(x_sub_bow * torch.log(recon_sub + 1e-10)) \
                            .sum(dim=-1).mean()


        loss_TM = (
            recon_loss_doc
          + loss_kl_doc
          + recon_loss_sub
          + kl_adapter
        )

        cost_matrix   = self.pairwise_euclidean_distance(
                            self.topic_embeddings,
                            self.word_embeddings
                        )
        cluster_loss  = self.ot_loss(cost_matrix)

        dist          = self.pairwise_euclidean_distance(
                            z_e.view(B*Q, -1),  
                            self.cluster_centers
                        )
        p             = F.softmax(-dist / self.tau, dim=1)
        distances     = torch.sqrt(dist + 1e-8)
        cluster_loss2 = (p * distances).sum(dim=1).mean()

        total_loss = loss_TM + cluster_loss + cluster_loss2

        return {
            'loss': total_loss,
            'loss_TM': loss_TM,
            'loss_cluster': cluster_loss,
            'loss_cluster2': cluster_loss2,
        }
