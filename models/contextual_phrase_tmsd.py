import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict, Any
from models.tmsd2 import VAE_OT, SinkhornOTLoss


class ContextualTMSD(VAE_OT):
    def __init__(self, args: Any) -> None:
        super().__init__(args)
        V = args.vocab_size

        # MLP to project sub-document embeddings into BoW space
        self.sub_embed_mlp = nn.Sequential(
            nn.Linear(args.sub_embed_dim, args.sub_proj_hidden),
            nn.ReLU(),
            nn.Linear(args.sub_proj_hidden, V),
        )

        # Encoder MLP layers for contextual sub-documents
        self.sub_fc1 = nn.Linear(2 * V, args.en1_units)
        self.sub_fc2 = nn.Linear(args.en1_units, args.en1_units)
        self.sub_fc21 = nn.Linear(args.en1_units, args.num_topic)
        self.sub_fc22 = nn.Linear(args.en1_units, args.num_topic)

        # Precompute squared adapter alpha and temperature
        self.adapter_alpha2 = args.adapter_alpha**2
        self.tau = args.tau

    def forward(
        self,
        x_doc: torch.Tensor,  # (B, V)
        x_sub_bow: torch.Tensor,  # (B, Q, V)
        x_sub_embed: torch.Tensor,  # (B, Q, D_sub)
        phrases: Optional[List[List[List[Tuple[int, int]]]]] = None,
        # phrases[b][q] = list of (w1,w2) pairs for each sub-doc
        phrase_bow: Optional[torch.Tensor] = None,  # (B, Q, P)
    ) -> Dict[str, torch.Tensor]:
        B, Q, V = x_sub_bow.shape

        # --- contextual sub-document encoding ---
        sub_proj = self.sub_embed_mlp(x_sub_embed.view(B * Q, -1)).view(B, Q, V)
        sub_comb = torch.cat([x_sub_bow, sub_proj], dim=-1)  # (B, Q, 2V)
        flat_sub = sub_comb.view(B * Q, -1)

        # MLP to Gaussian parameters for sub-doc encoder
        h = F.softplus(self.sub_fc1(flat_sub))
        h = F.softplus(self.sub_fc2(h))
        mu_e = self.sub_fc21(h)
        logv_e = self.sub_fc22(h)
        z_e = self.reparameterize(mu_e, logv_e).view(B, Q, -1)

        # KL divergence (adapter) for sub-doc encoder
        var_e = logv_e.exp()
        diff_e = mu_e - 1.0
        kl_terms = 0.5 * (
            var_e / self.adapter_alpha2
            + diff_e * diff_e / self.adapter_alpha2
            + np.log(self.adapter_alpha2)
            - logv_e
            - 1.0
        )
        kl_ad    = kl_terms.sum(dim=1).view(B, Q).mean(dim=1)
        kl_adapter = kl_ad.mean()

        # --- document-level encoding and reconstruction ---
        theta_doc, loss_kl_doc, _ = self.encode(x_doc)
        beta = self.get_beta()  # (K, V)
        recon_doc = F.softmax(self.decoder_bn(theta_doc @ beta), dim=-1)
        recon_loss_doc = -(x_doc * torch.log(recon_doc + 1e-10)).sum(dim=1).mean()

        # --- compute sub-document topic weights ---
        theta_sub = F.softmax(z_e * theta_doc.unsqueeze(1), dim=-1)  # (B, Q, K)

        recon_sub = (
            F.softmax(
                self.decoder_bn((theta_sub.view(B*Q, -1) @ beta)), dim=-1
            )
            .view(B, Q, V)
        )

        recon_loss_sub = -(x_sub_bow * torch.log(recon_sub + 1e-10)) \
                            .sum(dim=-1).mean()
                            
        # --- phrase reconstruction & loss (replacing recon_loss_sub) ---
        phrase_loss = 0.0
        if phrases is not None and phrase_bow is not None:
            # Build phrase embeddings by averaging word embeddings
            # phrase_embeds: (B, Q, P, D)
            embed_list = []
            for b in range(B):
                q_list = []
                for q in range(Q):
                    p_embs = []
                    for (w1, w2) in phrases[b][q]:
                        v1 = self.word_embeddings[w1]  # (D,)
                        v2 = self.word_embeddings[w2]
                        p_embs.append((v1 + v2) * 0.5)
                    q_list.append(torch.stack(p_embs, dim=0))
                embed_list.append(torch.stack(q_list, dim=0))
            phrase_embeds = torch.stack(embed_list, dim=0)
            B_, Q_, P_, D = phrase_embeds.shape

            # Flatten to (B*Q*P, D) for distance computation
            flat_phr = phrase_embeds.view(B_ * Q_ * P_, D)
            # Topic embeddings: (K, D)
            t_emb = self.topic_embeddings
            # Pairwise Euclidean distances: (B*Q*P, K)
            dist = torch.cdist(flat_phr, t_emb, p=2)
            # Convert to logits & reshape: (B, Q, P, K)
            phrase_logits = -dist.view(B_, Q_, P_, -1) / self.tau
            phrase_beta = F.softmax(phrase_logits, dim=-1)  # (B, Q, P, K)

            # Reconstruct phrase counts: hat_dp[b,q,p] = sum_k theta_sub[b,q,k] * phrase_beta[b,q,p,k]
            hat_dp = (theta_sub.unsqueeze(2) * phrase_beta).sum(dim=-1)  # (B, Q, P)

            # Negative log-likelihood over phrases
            phrase_loss = -(phrase_bow * torch.log(hat_dp + 1e-10)).sum(dim=-1).mean()

        # --- clustering and OT losses ---
        cost_matrix = self.pairwise_euclidean_distance(
            self.topic_embeddings, self.word_embeddings
        )
        cluster_loss = self.ot_loss(cost_matrix)

        dist2 = self.pairwise_euclidean_distance(
            z_e.view(B * Q, -1), self.cluster_centers
        )
        p2 = F.softmax(-dist2 / self.tau, dim=1)
        distances2 = torch.sqrt(dist2 + 1e-8)
        cluster_loss2 = (p2 * distances2).sum(dim=1).mean()

        # --- aggregate final losses ---
        loss_TM = recon_loss_doc + loss_kl_doc + recon_loss_sub + kl_adapter
        total_loss = loss_TM + cluster_loss + cluster_loss2 + phrase_loss

        return {
            "loss": total_loss,
            "loss_TM": loss_TM,
            "loss_cluster": cluster_loss,
            "loss_cluster2": cluster_loss2,
            "loss_phrase": phrase_loss,
        }
