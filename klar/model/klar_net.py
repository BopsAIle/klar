import math
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import pytorch_lightning as pl
    _BaseLightningModule = pl.LightningModule
except ImportError:
    try:
        import lightning.pytorch as pl
        _BaseLightningModule = pl.LightningModule
    except ImportError:
        pl = None
        _BaseLightningModule = nn.Module


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def segment_softmax(logits: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Computes softmax grouped by segment_ids (e.g. unique (target_node u, relation r) pairs).
    Ensures Relation-Aware Additive Attention normalizes strictly within N_r(u).
    """
    max_vals = torch.full((num_segments,), -1e9, dtype=logits.dtype, device=logits.device)
    max_vals.scatter_reduce_(0, segment_ids, logits, reduce="amax", include_self=True)
    shifted = logits - max_vals[segment_ids]
    exp_vals = torch.exp(shifted)
    sum_exp = torch.zeros((num_segments,), dtype=logits.dtype, device=logits.device)
    sum_exp.scatter_add_(0, segment_ids, exp_vals)
    return exp_vals / (sum_exp[segment_ids] + 1e-12)


class RelationAwareRGNNLayer(nn.Module):
    """
    Single RGNN layer with Relation-Aware Additive Attention (Section 3.2 of KLaR spec):
      h_u^{(l+1)} = sigma( W_0^{(l)} h_u^{(l)} + sum_{r in R} sum_{w in N_r(u)} alpha_{u,r,w} W_r^{(l)} h_w^{(l)} )
      alpha_{u,r,w} = softmax_{w in N_r(u)}( LeakyReLU( a_r^T [W_r^{(l)} h_u^{(l)} || W_r^{(l)} h_w^{(l)}] ) )
    """

    def __init__(self, num_relations: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_relations = num_relations
        self.dim = dim
        self.W_0 = nn.Linear(dim, dim, bias=True)
        # Relation-specific transformation matrices W_r^{(l)} in R^{|R| x d x d}
        self.W_r = nn.Parameter(torch.empty(num_relations, dim, dim))
        # Relation-specific additive attention vectors a_r in R^{|R| x 2d}
        self.att_src = nn.Parameter(torch.empty(num_relations, dim))
        self.att_dst = nn.Parameter(torch.empty(num_relations, dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_0.weight)
        nn.init.zeros_(self.W_0.bias)
        nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        # Self-transformation W_0 h_u
        out = self.W_0(h)

        # Memory-efficient computation of W_r h_u and W_r h_w:
        # Instead of materializing W_r[edge_rel] of shape [E, d, d],
        # compute h @ W_r of shape [|R|, |V|, d] (or iterate over relations) and gather!
        E = edge_src.size(0)
        trans_src = torch.empty((E, self.dim), dtype=h.dtype, device=h.device)
        trans_dst = torch.empty((E, self.dim), dtype=h.dtype, device=h.device)
        for r in range(self.num_relations):
            mask = edge_rel == r
            if not mask.any():
                continue
            W_r_mat = self.W_r[r]  # [d, d]
            trans_src[mask] = torch.matmul(h[edge_src[mask]], W_r_mat)
            trans_dst[mask] = torch.matmul(h[edge_dst[mask]], W_r_mat)

        # Additive attention logit: a_r^T [W_r h_u || W_r h_w]
        e_logits = (trans_dst * self.att_dst[edge_rel]).sum(dim=-1) + (
            trans_src * self.att_src[edge_rel]
        ).sum(dim=-1)
        e_logits = self.leaky_relu(e_logits)

        # Softmax strictly within N_r(u) (grouped by segment_ids = u * |R| + r)
        alpha = segment_softmax(e_logits, segment_ids, num_segments)
        alpha = self.dropout(alpha)

        messages = trans_src * alpha.unsqueeze(-1)  # [E, d]
        agg = torch.zeros_like(out)
        agg.index_add_(0, edge_dst, messages)

        return F.elu(self.norm(out + agg))


class RelationAwareRGNN(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_entities = num_entities
        # Include inverse relations so message passing flows in both directions
        self.total_relations = num_relations * 2
        self.entity_init = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_init.weight)

        self.layers = nn.ModuleList(
            [RelationAwareRGNNLayer(self.total_relations, dim, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_rel: torch.Tensor,
        segment_ids: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        h = self.entity_init.weight
        for layer in self.layers:
            h = layer(h, edge_src, edge_dst, edge_rel, segment_ids, num_segments)
        return h


class GatedStructureTextFusion(nn.Module):
    """
    Section 3.3: Gated Integration & InfoNCE Contrastive Alignment
      gamma_v = sigma(W_gamma [s_v; t_v] + b_gamma) in (0, 1)^d
      z_v = gamma_v * phi_s(s_v) + (1 - gamma_v) * phi_t(t_v)
    """

    def __init__(self, d_s: int, d_t: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.phi_s = nn.Sequential(
            nn.Linear(d_s, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.phi_t = nn.Sequential(
            nn.Linear(d_t, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )
        self.gate = nn.Linear(d_s + d_t, d, bias=True)

    def forward(self, s_v: torch.Tensor, t_v: torch.Tensor):
        proj_s = self.phi_s(s_v)
        proj_t = self.phi_t(t_v)
        gamma = torch.sigmoid(self.gate(torch.cat([s_v, t_v], dim=-1)))
        z_v = gamma * proj_s + (1.0 - gamma) * proj_t
        return z_v, proj_s, proj_t

    @staticmethod
    def compute_infonce_loss(
        proj_s: torch.Tensor, proj_t: torch.Tensor, batch_entities: torch.Tensor, temperature: float = 0.07
    ) -> torch.Tensor:
        s_b = F.normalize(proj_s[batch_entities], p=2, dim=-1)
        t_b = F.normalize(proj_t[batch_entities], p=2, dim=-1)
        logits = torch.matmul(s_b, t_b.transpose(0, 1)) / temperature
        labels = torch.arange(s_b.size(0), device=s_b.device)
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_s2t + loss_t2s)


class DistMultSparseMoEDecoder(nn.Module):
    """
    Section 3.4: Mechanism-Specialised Sparse Mixture-of-Experts Decoder
      - N experts {E_k}_{k=1}^N (default N=64)
      - RMSNorm on entity representations
      - Router logits: a(h, r) = MLP_g([z_h; r_r]) in R^N (2 linear layers, hidden dim d/2, ReLU)
      - Top-K expert selection and softmax normalization pi_k(h, r)
      - Expert k scoring: E_k(z_h, z_t, r_r) = <z_h, r_r^{(k)} * z_t>
      - Output: f(h, r, t) = sum_{k in Top-K} pi_k(h, r) E_k(z_h, z_t, r_r)
    """

    def __init__(self, num_relations: int, dim: int, num_experts: int = 64, top_k: int = 4):
        super().__init__()
        self.num_relations = num_relations
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.in_norm = RMSNorm(dim)
        self.out_norm = RMSNorm(dim)

        # Shared relation embedding for router r_r in R^{|R| x d}
        self.router_rel_emb = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.router_rel_emb.weight)

        hidden_dim = max(dim // 2, 32)
        self.router_mlp = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

        # Expert-specific relation embeddings r_r^{(k)} in R^{N x |R| x d}
        self.expert_rel_emb = nn.Parameter(torch.empty(num_experts, num_relations, dim))
        nn.init.xavier_uniform_(self.expert_rel_emb)

    def route(self, z_h: torch.Tensor, r_ids: torch.Tensor):
        """
        Args:
            z_h: [B, d]
            r_ids: [B]
        Returns:
            topk_idx: [B, K]
            pi_k: [B, K]
            moe_reg_loss: tuple (L_load, L_imp)
        """
        r_vec = self.router_rel_emb(r_ids)
        router_in = torch.cat([z_h, r_vec], dim=-1)
        logits = self.router_mlp(router_in)  # [B, N]

        topk_logits, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # [B, K]
        pi_k = F.softmax(topk_logits, dim=-1)  # [B, K]

        # Compute load balancing & importance regularization (Section 4)
        full_probs = F.softmax(logits, dim=-1)  # [B, N]
        importance = full_probs.sum(dim=0)  # [N]
        L_imp = self._cv_squared(importance)

        # Smooth differentiable load + hard assignment frequency
        mask = torch.zeros_like(logits).scatter_(1, topk_idx, 1.0)
        load = mask.sum(dim=0)
        L_load = self._cv_squared(load)

        return topk_idx, pi_k, L_load, L_imp

    @staticmethod
    def _cv_squared(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        if x.numel() <= 1:
            return torch.tensor(0.0, device=x.device)
        return x.float().var() / (x.float().mean().pow(2) + eps)

    def EffectiveRelation(self, z_h_norm: torch.Tensor, r_ids: torch.Tensor):
        """
        Computes the effective MoE-weighted relation vector for each (h, r) query:
        Since f(h, r, t) = sum_{k in Top-K} pi_k(h, r) <z_h, r_r^{(k)} * z_t>
                         = <z_h * (sum_{k in Top-K} pi_k(h, r) r_r^{(k)}), z_t>,
        we can compute the exact MoE output across 128 negatives or all |V| entities cleanly!
        """
        topk_idx, pi_k, L_load, L_imp = self.route(z_h_norm, r_ids)  # topk_idx: [B, K], pi_k: [B, K]
        # Gather expert relation embeddings: [B, K, d]
        r_expanded = r_ids.unsqueeze(1).expand(-1, self.top_k)  # [B, K]
        expert_rels = self.expert_rel_emb[topk_idx, r_expanded]  # [B, K, d]
        weighted_rel = torch.sum(pi_k.unsqueeze(-1) * expert_rels, dim=1)  # [B, d]
        hr_vec = self.out_norm(z_h_norm * weighted_rel)  # [B, d]
        return hr_vec, L_load, L_imp

    def score_triples(self, z_all: torch.Tensor, h_ids: torch.Tensor, r_ids: torch.Tensor, t_ids: torch.Tensor):
        """
        Scores positive triples [B] and optional negative triples [B, M_neg].
        """
        z_norm = self.in_norm(z_all)
        z_h = z_norm[h_ids]  # [B, d]
        hr_vec, L_load, L_imp = self.EffectiveRelation(z_h, r_ids)  # [B, d]

        if t_ids.dim() == 1:
            z_t = z_norm[t_ids]  # [B, d]
            scores = (hr_vec * z_t).sum(dim=-1)  # [B]
        else:
            z_t = z_norm[t_ids]  # [B, M_neg, d]
            scores = torch.bmm(z_t, hr_vec.unsqueeze(-1)).squeeze(-1)  # [B, M_neg]
        return scores, L_load, L_imp

    def score_all_tails(self, z_all: torch.Tensor, h_ids: torch.Tensor, r_ids: torch.Tensor) -> torch.Tensor:
        z_norm = self.in_norm(z_all)
        z_h = z_norm[h_ids]  # [B, d]
        hr_vec, _, _ = self.EffectiveRelation(z_h, r_ids)  # [B, d]
        return torch.matmul(hr_vec, z_norm.transpose(0, 1))  # [B, |V|]

    def score_all_heads(self, z_all: torch.Tensor, r_ids: torch.Tensor, t_ids: torch.Tensor) -> torch.Tensor:
        """
        Scores (?, r, t) for all candidate heads h' in V using the target entity context z_t for routing.
        """
        z_norm = self.in_norm(z_all)
        z_t = z_norm[t_ids]  # [B, d]
        tr_vec, _, _ = self.EffectiveRelation(z_t, r_ids)  # [B, d]
        return torch.matmul(tr_vec, z_norm.transpose(0, 1))  # [B, |V|]


class KLaRLightningModule(_BaseLightningModule):
    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        text_embeddings: torch.Tensor,
        train_triples: torch.Tensor,
        all_true_triples: torch.Tensor,
        d_s: int = 256,
        d: int = 256,
        num_rgnn_layers: int = 2,
        num_experts: int = 64,
        top_k: int = 4,
        num_negatives: int = 128,
        margin: float = 1.0,
        temperature: float = 0.07,
        lambda_align: float = 0.1,
        lambda_moe: float = 0.01,
        lambda_load: float = 1.0,
        lambda_imp: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        wo_kge: bool = False,
        wo_texte: bool = False,
        wo_align: bool = False,
        wo_moe: bool = False,
    ):
        super().__init__()
        if hasattr(self, "save_hyperparameters") and pl is not None:
            self.save_hyperparameters(ignore=["text_embeddings", "train_triples", "all_true_triples"])

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_negatives = num_negatives
        self.margin = margin
        self.temperature = temperature
        self.lambda_align = lambda_align
        self.lambda_moe = lambda_moe
        self.lambda_load = lambda_load
        self.lambda_imp = lambda_imp
        self.lr = lr
        self.weight_decay = weight_decay
        self.wo_kge = wo_kge
        self.wo_texte = wo_texte
        self.wo_align = wo_align
        self.wo_moe = wo_moe

        # Frozen pre-computed textual embeddings t_v in R^{|V| x d_t}
        self.register_buffer("text_emb", text_embeddings.float())
        d_t = text_embeddings.size(1)

        # Build Train Graph edges (strictly from train_triples) with inverse edges
        src_f, rel_f, dst_f = train_triples[:, 0], train_triples[:, 1], train_triples[:, 2]
        src_b, rel_b, dst_b = dst_f, rel_f + num_relations, src_f

        edge_src = torch.cat([src_f, src_b], dim=0)
        edge_dst = torch.cat([dst_f, dst_b], dim=0)
        edge_rel = torch.cat([rel_f, rel_b], dim=0)

        # Precompute compact segment_ids for Relation-Aware Softmax over N_r(u)
        raw_seg = edge_dst * (num_relations * 2) + edge_rel
        unique_segs, segment_ids = torch.unique(raw_seg, return_inverse=True)
        self.num_segments = int(unique_segs.numel())

        self.register_buffer("edge_src", edge_src)
        self.register_buffer("edge_dst", edge_dst)
        self.register_buffer("edge_rel", edge_rel)
        self.register_buffer("segment_ids", segment_ids)

        # Build filtered evaluation lookup tables: (h, r) -> set of true t, (r, t) -> set of true h
        self.hr_to_tails = {}
        self.rt_to_heads = {}
        for h_i, r_i, t_i in all_true_triples.tolist():
            self.hr_to_tails.setdefault((h_i, r_i), []).append(t_i)
            self.rt_to_heads.setdefault((r_i, t_i), []).append(h_i)

        # Modules
        self.rgnn = RelationAwareRGNN(
            num_entities=num_entities,
            num_relations=num_relations,
            dim=d_s,
            num_layers=num_rgnn_layers,
            dropout=dropout,
        )
        self.fusion = GatedStructureTextFusion(d_s=d_s, d_t=d_t, d=d, dropout=dropout)

        effective_experts = 1 if wo_moe else num_experts
        effective_topk = 1 if wo_moe else top_k
        self.decoder = DistMultSparseMoEDecoder(
            num_relations=num_relations,
            dim=d,
            num_experts=effective_experts,
            top_k=effective_topk,
        )

        self.val_ranks = []
        self.test_ranks = []

    @property
    def device(self):
        return self.text_emb.device

    def compute_entity_representations(self):
        s_all = self.rgnn(self.edge_src, self.edge_dst, self.edge_rel, self.segment_ids, self.num_segments)
        t_all = self.text_emb
        if self.wo_kge:
            s_all = torch.zeros_like(s_all)
        if self.wo_texte:
            t_all = torch.zeros_like(t_all)
        z_all, proj_s, proj_t = self.fusion(s_all, t_all)
        return z_all, proj_s, proj_t

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        # batch: [B, 3] -> (h, r, t)
        h_ids, r_ids, t_ids = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h_ids.size(0)

        z_all, proj_s, proj_t = self.compute_entity_representations()

        # Positive scores
        pos_scores, L_load, L_imp = self.decoder.score_triples(z_all, h_ids, r_ids, t_ids)  # [B]

        # Negative Sampling: 128 negatives per positive (50% head corruption, 50% tail corruption)
        half_neg = self.num_negatives // 2
        rem_neg = self.num_negatives - half_neg

        # Tail-corrupted negatives (h, r, t')
        rand_tails = torch.randint(0, self.num_entities, (B, half_neg), device=self.device)
        neg_tail_scores, _, _ = self.decoder.score_triples(z_all, h_ids, r_ids, rand_tails)  # [B, half_neg]

        # Head-corrupted negatives (h', r, t) scored via target context
        rand_heads = torch.randint(0, self.num_entities, (B, rem_neg), device=self.device)
        neg_head_scores, _, _ = self.decoder.score_triples(z_all, t_ids, r_ids, rand_heads)  # [B, rem_neg]

        neg_scores = torch.cat([neg_tail_scores, neg_head_scores], dim=1)  # [B, M_neg]

        # 1. Max-Margin Ranking Loss L_rank
        L_rank = F.relu(self.margin + neg_scores - pos_scores.unsqueeze(1)).mean()

        # 2. InfoNCE Contrastive Alignment Loss L_align
        if self.wo_align or self.wo_kge or self.wo_texte:
            L_align = torch.tensor(0.0, device=self.device)
        else:
            unique_batch_ents = torch.unique(torch.cat([h_ids, t_ids], dim=0))
            L_align = self.fusion.compute_infonce_loss(proj_s, proj_t, unique_batch_ents, self.temperature)

        # 3. MoE Regularization Loss L_moe
        if self.wo_moe or self.decoder.num_experts <= 1:
            L_moe = torch.tensor(0.0, device=self.device)
        else:
            L_moe = self.lambda_load * L_load + self.lambda_imp * L_imp

        total_loss = L_rank + self.lambda_align * L_align + self.lambda_moe * L_moe

        if hasattr(self, "log") and pl is not None and getattr(self, "_trainer", None) is not None:
            self.log("train_loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("L_rank", L_rank, prog_bar=True, on_step=False, on_epoch=True)
            self.log("L_align", L_align, prog_bar=False, on_step=False, on_epoch=True)
            self.log("L_moe", L_moe, prog_bar=False, on_step=False, on_epoch=True)
        return total_loss

    @torch.no_grad()
    def _evaluate_filtered_batch(self, batch: torch.Tensor, z_all: torch.Tensor):
        h_ids, r_ids, t_ids = batch[:, 0], batch[:, 1], batch[:, 2]
        B = h_ids.size(0)

        # Tail prediction: (h, r, ?)
        tail_scores = self.decoder.score_all_tails(z_all, h_ids, r_ids)  # [B, |V|]
        h_cpu, r_cpu, t_cpu = h_ids.tolist(), r_ids.tolist(), t_ids.tolist()

        ranks = []
        for i in range(B):
            h_i, r_i, t_i = h_cpu[i], r_cpu[i], t_cpu[i]
            target_score = tail_scores[i, t_i].item()
            true_tails = self.hr_to_tails.get((h_i, r_i), [])
            if true_tails:
                tail_scores[i, true_tails] = -1e9
            tail_scores[i, t_i] = target_score
            # Optimistic/strict filtered rank: 1 + count of candidates strictly greater than target_score
            rank_t = 1 + int((tail_scores[i] > target_score).sum().item())
            ranks.append(rank_t)

        return ranks

    def on_validation_epoch_start(self):
        self.val_ranks.clear()
        self._cached_eval_z, _, _ = self.compute_entity_representations()

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        ranks = self._evaluate_filtered_batch(batch, self._cached_eval_z)
        self.val_ranks.extend(ranks)

    def on_validation_epoch_end(self):
        self._cached_eval_z = None
        ranks_t = torch.tensor(self.val_ranks, dtype=torch.float32)
        mrr = (1.0 / ranks_t).mean().item()
        h1 = (ranks_t <= 1).float().mean().item()
        h3 = (ranks_t <= 3).float().mean().item()
        h10 = (ranks_t <= 10).float().mean().item()

        self.log("Eval-mrr", mrr, prog_bar=True, sync_dist=True)
        self.log("Eval-hits1", h1, prog_bar=True, sync_dist=True)
        self.log("Eval-hits3", h3, prog_bar=False, sync_dist=True)
        self.log("Eval-hits10", h10, prog_bar=True, sync_dist=True)

    def on_test_epoch_start(self):
        self.test_ranks.clear()
        self._cached_eval_z, _, _ = self.compute_entity_representations()

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        ranks = self._evaluate_filtered_batch(batch, self._cached_eval_z)
        self.test_ranks.extend(ranks)

    def on_test_epoch_end(self):
        self._cached_eval_z = None
        ranks_t = torch.tensor(self.test_ranks, dtype=torch.float32)
        mrr = (1.0 / ranks_t).mean().item()
        h1 = (ranks_t <= 1).float().mean().item()
        h3 = (ranks_t <= 3).float().mean().item()
        h10 = (ranks_t <= 10).float().mean().item()

        self.log("Test-mrr", mrr, prog_bar=True, sync_dist=True)
        self.log("Test-hits1", h1, prog_bar=True, sync_dist=True)
        self.log("Test-hits3", h3, prog_bar=True, sync_dist=True)
        self.log("Test-hits10", h10, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
