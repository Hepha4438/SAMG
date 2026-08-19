import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import GetPeriodicTable

# Import MDN NLL Loss from Step 2
from step2_mdn_module import mdn_nll_loss

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
        # Stream 2 (Geometric)
        pi, mu, sigma, target_7d,
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
        loss_token = F.cross_entropy(logits_vocab.view(-1, vocab_size), target_ids.view(-1))

        # ---------------------------------------------------------
        # B. L_Geo: Geometric MDN NLL Loss
        # ---------------------------------------------------------
        num_gaussians = pi.size(-1)
        pi_flat = pi.view(-1, num_gaussians)
        mu_flat = mu.view(-1, num_gaussians, 7)
        sigma_flat = sigma.view(-1, num_gaussians, 7)
        target_7d_flat = target_7d.view(-1, 7)
        
        loss_geo = mdn_nll_loss(pi_flat, mu_flat, sigma_flat, target_7d_flat)

        # ---------------------------------------------------------
        # C. L_Pocket: Protein Dynamics Loss
        # ---------------------------------------------------------
        # 1. Translation: MSE
        loss_tr = F.mse_loss(pred_tr, target_tr)
        
        # 2. Rotation: Dot Product Loss to handle Quaternion double cover (q and -q represent same rotation)
        pred_q_norm = pred_q / pred_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        target_q_norm = target_q / target_q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        dot_product = torch.sum(pred_q_norm * target_q_norm, dim=-1)
        loss_q = 1.0 - torch.abs(dot_product).mean()
        
        # 3. Chi Angles: Wrap-around MSE using Cosine (1 - cos(delta))
        chi_diff = pred_chi - target_chi
        loss_chi_raw = 1.0 - torch.cos(chi_diff)
        loss_chi = (loss_chi_raw * chi_mask).sum() / (chi_mask.sum() + 1e-8)
        
        loss_pocket = loss_tr + loss_q + loss_chi

        # ---------------------------------------------------------
        # D. L_Int: Steric Clash Penalty (In-training Alignment)
        # ---------------------------------------------------------
        # Calculate pairwise distances: [N_ligand, N_protein]
        dist_matrix = torch.cdist(ligand_pos_3d, protein_pos_3d, p=2)
        # Penalize if distance is strictly less than D_threshold
        clash_penalty = torch.relu(self.d_threshold - dist_matrix) ** 2
        loss_int = clash_penalty.sum()

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
            "loss_int": loss_int
        }
        
        return total_loss, loss_dict


class PoseBustersFilter:
    """
    Hard constraints filter based on PoseBusters steric clash logic.
    Used during inference (Step 4 Autoregressive Loop) for Logit Masking.
    """
    def __init__(self, clash_cutoff=0.75, radius_scale=1.0):
        self.clash_cutoff = clash_cutoff
        self.radius_scale = radius_scale
        self.periodic_table = GetPeriodicTable()
        
        # Cache standard VdW radii to avoid repeated C++ API calls
        self.vdw_radii = {
            "C": self.periodic_table.GetRvdw("C"),
            "N": self.periodic_table.GetRvdw("N"),
            "O": self.periodic_table.GetRvdw("O"),
            "S": self.periodic_table.GetRvdw("S"),
            "P": self.periodic_table.GetRvdw("P"),
            "F": self.periodic_table.GetRvdw("F"),
            "Cl": self.periodic_table.GetRvdw("Cl")
        }

    def spherical_to_cartesian(self, d, theta, phi, ref_pos):
        """
        Converts (d, theta, phi) back to (x, y, z) based on a reference origin.
        """
        x = d * torch.sin(theta) * torch.cos(phi)
        y = d * torch.sin(theta) * torch.sin(phi)
        z = d * torch.cos(theta)
        cartesian_offset = torch.stack([x, y, z], dim=-1)
        return ref_pos + cartesian_offset

    def posebusters_masking(self, pi_logits, mu, ref_pos, anti_target_pos, element_ligand="C"):
        """
        Applies a $-\infty$ mask to the logits of Gaussian peaks that result in steric clashes.
        Args:
            pi_logits: [K] Mixing coefficients logits from MDN
            mu: [K, 7] Means from MDN (Contains d, theta, phi at indices 0,1,2)
            ref_pos: [3] Current molecule local frame origin
            anti_target_pos: [N_atoms, 3] Coordinates of Anti-target atoms
            element_ligand: String of the atom/fragment type to fetch VdW radius
        Returns:
            masked_pi_logits: [K] Logits with clashed peaks set to -1e9
        """
        K = mu.size(0)
        masked_pi_logits = pi_logits.clone()
        
        # Assume average Protein VdW radius if exact element is unknown
        protein_vdw_avg = self.vdw_radii["C"] 
        ligand_vdw = self.vdw_radii.get(element_ligand, self.vdw_radii["C"])
        
        # PoseBusters Clash Threshold logic
        sum_radii = (ligand_vdw + protein_vdw_avg) * self.radius_scale
        min_allowed_dist = sum_radii * self.clash_cutoff
        
        for k in range(K):
            d, theta, phi = mu[k, 0], mu[k, 1], mu[k, 2]
            
            # Convert peak k's prediction to Cartesian space
            pred_xyz = self.spherical_to_cartesian(d, theta, phi, ref_pos)
            
            # Measure distances against all Anti-target atoms
            distances = torch.norm(anti_target_pos - pred_xyz.unsqueeze(0), dim=-1)
            min_dist = distances.min().item()
            
            # If the closest atom violates the threshold, apply hard mask
            if min_dist < min_allowed_dist:
                masked_pi_logits[k] = -1e9 # Mask out this Gaussian peak
                
        return masked_pi_logits


# ==============================================================================
# SCRIPT TEST: LOSS FUNCTION & POSEBUSTERS LOGIT MASKING
# ==============================================================================
if __name__ == "__main__":
    print("[*] Starting Step 5: Constraints & Global Loss Test...")
    device = torch.device("cpu")
    
    # 1. Test GlobalLoss
    print("\n[+] Testing GlobalLoss Module...")
    loss_fn = GlobalLoss(lambda_token=1.0, lambda_geo=1.0, lambda_pocket=0.5, lambda_int=2.0)
    
    # Mock Tensors for Loss
    logits_vocab = torch.randn(2, 5, 50)
    target_ids = torch.randint(0, 50, (2, 5))
    
    pi = torch.randn(2, 5, 10)
    mu = torch.randn(2, 5, 10, 7)
    sigma = torch.ones(2, 5, 10, 7) + 0.1
    target_7d = torch.randn(2, 5, 7)
    
    pred_tr = torch.randn(2, 20, 3)
    target_tr = torch.randn(2, 20, 3)
    pred_q = torch.randn(2, 20, 4)
    target_q = torch.randn(2, 20, 4)
    pred_chi = torch.randn(2, 20, 5)
    target_chi = torch.randn(2, 20, 5)
    chi_mask = torch.ones(2, 20, 5)
    
    lig_pos = torch.randn(10, 3)
    prot_pos = torch.randn(100, 3)
    
    total_loss, loss_dict = loss_fn(
        logits_vocab, target_ids, pi, mu, sigma, target_7d,
        pred_tr, target_tr, pred_q, target_q, pred_chi, target_chi, chi_mask,
        lig_pos, prot_pos
    )
    
    print("    -> Global Loss Evaluation Successful.")
    for k, v in loss_dict.items():
        print(f"       - {k}: {v.item():.4f}")

    # 2. Test PoseBusters Filter (Hard Constraints)
    print("\n[+] Testing PoseBusters Filter (Logit Masking)...")
    pb_filter = PoseBustersFilter(clash_cutoff=0.75, radius_scale=1.0)
    
    K_peaks = 5
    pi_logits_mock = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Mock Mu: [d, theta, phi, qw, qx, qy, qz]
    # Peak 0 predicts coordinates far away (safe)
    # Peak 1 predicts coordinates intentionally close to origin (clash)
    mu_mock = torch.randn(K_peaks, 7)
    mu_mock[0, 0:3] = torch.tensor([10.0, 0.0, 0.0]) # d=10 (Safe)
    mu_mock[1, 0:3] = torch.tensor([0.1, 0.0, 0.0])  # d=0.1 (Dangerous clash)
    
    ref_pos_mock = torch.tensor([0.0, 0.0, 0.0])
    anti_target_mock = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]) # Anti-target atoms at origin
    
    print(f"    -> Original PI Logits: {pi_logits_mock.tolist()}")
    masked_logits = pb_filter.posebusters_masking(pi_logits_mock, mu_mock, ref_pos_mock, anti_target_mock, element_ligand="C")
    print(f"    -> Masked PI Logits:   {masked_logits.tolist()}")
    
    if masked_logits[1].item() < -1e5:
        print("    [v] SUCCESS: PoseBusters successfully detected steric clash and masked the logit to -inf.")
    
    print("\n=== STEP 5 CONSTRAINTS & LOSS TEST COMPLETED ===")