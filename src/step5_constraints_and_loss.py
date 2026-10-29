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
    def __init__(self, lambda_token=1.0, lambda_geo=1.0, lambda_pocket=1.0, lambda_int=1.0, d_threshold=2.5,
                 lambda_link=1.0):
        super().__init__()
        # S7 (Rev 21): trong so cua ba loss kenh lien ket. 1,0 la QUY UOC DU AN -- khong co can cu de
        # dat khac (R16 muc 3). In rieng tung loss de phat hien neu mot cai troi.
        self.lambda_link = lambda_link
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
        ligand_pos_3d, protein_pos_3d,
        # S7: kenh lien ket (Rev 21). link_logits/link_targets = None => bo qua (tuong thich nguoc).
        link_logits=None, link_targets=None
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
        # ---------------------------------------------------------
        # E. L_Link: kenh lien ket (S7, Rev 21) -- ba cross-entropy.
        # `ignore_index=-100` bo qua ca padding lan vi tri GOC (khong co cha/diem gan).
        # ---------------------------------------------------------
        link_losses = {}
        loss_link = torch.zeros((), device=loss_token.device)
        if link_logits is not None and link_targets is not None:
            for key in ("parent", "attach_self", "attach_parent"):
                lg, tg = link_logits[key], link_targets[key]
                li = F.cross_entropy(lg.reshape(-1, lg.size(-1)), tg.reshape(-1), ignore_index=-100)
                if torch.isnan(li):          # ca batch toan -100 (hiem): bo qua, khong lam hong loss
                    li = torch.zeros((), device=loss_token.device)
                link_losses[f"loss_{key}"] = li
                loss_link = loss_link + li

        total_loss = (
            self.lambda_token * loss_token +
            self.lambda_geo * loss_geo +
            self.lambda_pocket * loss_pocket +
            self.lambda_int * loss_int +
            self.lambda_link * loss_link
        )

        loss_dict = {
            "loss_total": total_loss,
            "loss_token": loss_token,
            "loss_geo": loss_geo,
            "loss_pocket": loss_pocket,
            "loss_int": loss_int,
            "loss_geo_unclamped": loss_geo_unclamped,
            "loss_link": loss_link,
            **link_losses,
        }
        
        return total_loss, loss_dict