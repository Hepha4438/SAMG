import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

import torch_geometric.nn as pyg_nn
from torch_cluster import knn_graph as cluster_knn_graph

# --- RUNTIME MONKEY PATCH ---
def custom_knn_graph(x, k=32, batch=None, loop=False, flow='source_to_target', cosine=False, num_workers=1):
    return cluster_knn_graph(x=x, k=k, batch=batch, loop=loop, flow=flow)
pyg_nn.knn_graph = custom_knn_graph
# ----------------------------

class MDNLayer(nn.Module):
    """
    Mixture Density Network (MDN) Layer for fragment-based ligand generation.
    Takes context vector H and outputs K Gaussian distributions for a 7D vector.
    7D Target: [d, theta, phi, q_w, q_x, q_y, q_z]
    """
    def __init__(self, hidden_dim, num_gaussians=10, out_dim=7):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.out_dim = out_dim
        
        # Using standard Linear layers instead of GVP for speed and SE(3) stability
        self.pi_net = nn.Linear(hidden_dim, num_gaussians)
        self.mu_net = nn.Linear(hidden_dim, num_gaussians * out_dim)
        self.sigma_net = nn.Linear(hidden_dim, num_gaussians * out_dim)

    def forward(self, h):
        """
        Args:
            h: Hidden state vector [batch_size, hidden_dim]
        Returns:
            pi_logits: Mixing coefficients [batch_size, K]
            mu: Gaussian means [batch_size, K, 7]
            sigma: Gaussian variances [batch_size, K, 7]
        """
        batch_size = h.size(0)
        
        # 1. Mixing Coefficients
        pi_logits = self.pi_net(h)
        
        # 2. Means
        mu = self.mu_net(h).view(batch_size, self.num_gaussians, self.out_dim)
        
        # Split spatial coordinates (0:3) and Quaternions (3:7)
        spatial_mu = mu[..., :3]
        quat_mu = mu[..., 3:7]
        
        # L2 Normalization for Quaternions to prevent Gimbal Lock
        quat_mu = quat_mu / quat_mu.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        mu = torch.cat([spatial_mu, quat_mu], dim=-1)
        
        # 3. Variances
        sigma_raw = self.sigma_net(h).view(batch_size, self.num_gaussians, self.out_dim)
        
        # ELU + 1 + 1e-6 guarantees strictly positive sigma, preventing NaN errors
        sigma = F.elu(sigma_raw) + 1.0 + 1e-6
        
        return pi_logits, mu, sigma


def mdn_nll_loss(pi_logits, mu, sigma, target):
    """
    Stable Negative Log-Likelihood Loss using Log-Sum-Exp trick.
    """
    # Expand target for broadcasting -> [batch_size, 1, 7]
    target = target.unsqueeze(1)
    
    var = sigma ** 2
    log_scale = torch.log(sigma)
    
    # Independent Log Normal PDF for 7 dimensions -> [batch_size, K, 7]
    log_normal = -0.5 * (math.log(2 * math.pi) + 2 * log_scale + ((target - mu) ** 2) / var)
    
    # Sum log probabilities across the 7 dimensions -> [batch_size, K]
    log_normal_sum = log_normal.sum(dim=-1)
    
    # Log probabilities of mixing coefficients -> [batch_size, K]
    log_pi = F.log_softmax(pi_logits, dim=-1)
    
    # Stable Log-Sum-Exp -> [batch_size]
    log_prob = torch.logsumexp(log_pi + log_normal_sum, dim=-1)
    
    return -log_prob.mean()


# ==============================================================================
# SCRIPT TEST: OVERFIT ON A SINGLE REAL BATCH
# ==============================================================================
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from datasets.pl_pair_dataset import PocketLigandPairDataset
    from datasets.pl_data import ProteinLigandDataLoader
    from step1_protein_encoder import DynamicEGNN

    print("[*] Starting MDN Module Overfitting Test using REAL dataset...")
    
    # 1. Load real LMDB data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/data_folder"))
    index_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/split_druglike_dict.pkl"))

    dataset = PocketLigandPairDataset(raw_path=raw_path, index_path=index_path, pocket_type="pocket")
    dataloader = ProteinLigandDataLoader(dataset, batch_size=2, shuffle=False)
    real_batch = next(iter(dataloader))
    
    device = torch.device("cpu")
    real_batch = real_batch.to(device)
    hidden_dim = 256
    
    # 2. Extract context vectors using Step 1 (DynamicEGNN)
    print("\n[*] Running Step 1 (DynamicEGNN) to extract real graph features...")
    config = OmegaConf.create({
        "num_blocks": 1, "num_layers": 2, "hidden_dim": hidden_dim,
        "n_heads": 4, "knn": 4, "edge_feat_dim": 0, "num_r_gaussian": 20, "num_node_types": 8
    })
    dynamic_egnn = DynamicEGNN(config).to(device)
    
    h_protein = real_batch.protein_atom_feature.float() if hasattr(real_batch, 'protein_atom_feature') else torch.randn(real_batch.protein_pos.size(0), hidden_dim, device=device)
    h_ligand = real_batch.ligand_atom_feature_full.float() if hasattr(real_batch, 'ligand_atom_feature_full') else torch.randn(real_batch.ligand_pos.size(0), hidden_dim, device=device)

    with torch.no_grad():
        step1_outputs = dynamic_egnn(
            h_protein=h_protein, h_ligand=h_ligand,
            protein_pos=real_batch.protein_pos, ligand_pos=real_batch.ligand_pos,
            batch_protein=real_batch.protein_element_batch, batch_ligand=real_batch.ligand_element_batch,
            data=real_batch
        )
    
    # Pool residue features to create graph representations -> [batch_size, 128]
    num_graphs = real_batch.protein_element_batch.max().item() + 1
    h_graph_real = scatter_mean(step1_outputs['residue_h'], real_batch.protein_translations_batch, dim=0, dim_size=num_graphs)
    
    # Retain only the 128 scalar dimensions to preserve SE(3) invariance
    h_graph_real = h_graph_real[:, :hidden_dim]

    # 3. Mock 7D Target: [d, theta, phi, q_w, q_x, q_y, q_z]
    target_7d = torch.randn(num_graphs, 7, device=device)
    target_7d[:, 3:7] = target_7d[:, 3:7] / target_7d[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # 4. Initialize MDN layer and Optimizer
    mdn_layer = MDNLayer(hidden_dim, num_gaussians=10).to(device)
    optimizer = torch.optim.Adam(mdn_layer.parameters(), lr=5e-3)

    print(f"\n[+] Extracted real graph features shape: {h_graph_real.shape}")
    print("[+] Training loop over 1000 epochs to force overfitting on 1 batch:")
    
    for epoch in range(1, 2001):
        optimizer.zero_grad()
        
        pi_logits, mu, sigma = mdn_layer(h_graph_real)
        loss = mdn_nll_loss(pi_logits, mu, sigma, target_7d)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdn_layer.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d} | NLL Loss: {loss.item():.4f} | Min Sigma: {sigma.min().item():.6f}")
            
    print("\n[v] Test Evaluation:")
    if loss.item() < -3.0:
        print("    -> SUCCESS: NLL Loss decreased significantly into negative space.")
    if sigma.min().item() > 0:
        print("    -> SUCCESS: Sigma strictly positive, no NaN errors detected.")
        
    print("=== STEP 2 MDN MODULE REAL DATA TEST COMPLETED ===")