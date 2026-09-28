import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import GetPeriodicTable

class GlobalLoss(nn.Module):
    """
    Computes the total loss for the SAMG Hybrid System.
    L_total = lambda_1 * L_token + lambda_2 * L_geo + lambda_3 * L_pocket + lambda_4 * L_int
    """
    def __init__(self, lambda_token=1.0, lambda_geo=1.0, lambda_pocket=1.0, lambda_int=1.0, d_threshold=2.5):
        super().__init__()
        self.lambda_token = lambda_token
        self.lambda_geo = lambda_geo
        self.lambda_pocket = lambda_pocket
        self.lambda_int = lambda_int
        self.d_threshold = d_threshold

    def forward(
        self, 
        # Stream 1 (Semantic)
        logits_vocab, target_ids,
        # Stream 2 (Geometric từ Autoregressive Flow)
        loss_geo_raw, # Kích thước: [batch_size, seq_len]
        # Pocket Dynamics
        pred_tr, target_tr,
        pred_q, target_q,
        pred_chi, target_chi, chi_mask,
        # Interaction (Steric Clash)
        ligand_pos_3d, protein_pos_3d
    ):
        # ---------------------------------------------------------
        # A. L_Token: Semantic Cross-Entropy Loss
        # ---------------------------------------------------------
        vocab_size = logits_vocab.size(-1)
        # Bỏ qua loss ở các token đệm ([PAD] thường có ID = 0)
        loss_token = F.cross_entropy(logits_vocab.view(-1, vocab_size), target_ids.view(-1), ignore_index=0)

        # ---------------------------------------------------------
        # B. L_Geo: Geometric Flow NLL Loss (Masked)
        # ---------------------------------------------------------
        # Chỉ tính loss ở những vị trí có token hợp lệ, bỏ qua padding
        pad_mask = (target_ids != 0).float()
        loss_geo_unclamped = (loss_geo_raw * pad_mask).sum() / (pad_mask.sum() + 1e-8)
        loss_geo = 20.0 * torch.log1p(torch.abs(loss_geo_unclamped) / 20.0) * torch.sign(loss_geo_unclamped)

        # ---------------------------------------------------------
        # C. L_Pocket: Protein Dynamics Loss
        # ---------------------------------------------------------
        loss_tr = F.smooth_l1_loss(pred_tr, target_tr)
        
        pred_q_norm = pred_q / pred_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        target_q_norm = target_q / target_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot_product = torch.sum(pred_q_norm * target_q_norm, dim=-1)
        loss_q = 1.0 - torch.abs(dot_product).mean()
        
        chi_diff = pred_chi - target_chi
        loss_chi_raw = 1.0 - torch.cos(chi_diff)
        loss_chi = (loss_chi_raw * chi_mask).sum() / (chi_mask.sum() + 1e-8)
        
        loss_pocket = loss_tr + loss_q + loss_chi

        # ---------------------------------------------------------
        # D. L_Int: Steric Clash Penalty
        # ---------------------------------------------------------
        dist_matrix = torch.cdist(ligand_pos_3d, protein_pos_3d, p=2)
        clash_penalty = torch.relu(self.d_threshold - dist_matrix) ** 2
        # Đổi thành .mean() để tránh loss bị bùng nổ khi lượng nguyên tử quá lớn
        loss_int = clash_penalty.mean()

        # ---------------------------------------------------------
        # Total Loss Accumulation
        # ---------------------------------------------------------
        total_loss = (
            self.lambda_token * loss_token +
            self.lambda_geo * loss_geo +
            self.lambda_pocket * loss_pocket +
            self.lambda_int * loss_int
        )

        loss_dict = {
            "loss_total": total_loss,
            "loss_token": loss_token,
            "loss_geo": loss_geo,
            "loss_pocket": loss_pocket,
            "loss_int": loss_int,
            "loss_geo_unclamped": loss_geo_unclamped
        }
        
        return total_loss, loss_dict