import os
import pickle
import torch
import torch.nn.functional as F

from step4_ligand_generator import DualStreamLigandGenerator
from step2_mdn_module import mdn_nll_loss

def test_step4_real_data():
    print("[*] Starting Step 4 Real Data Integration Test...")
    
    # 1. Resolve path to the generated 7D sequence from Step 0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.abspath(os.path.join(current_dir, "../dataset/processed/3txj__1__1.A__1.K_sequence_7d.pkl"))
    
    if not os.path.exists(pkl_path):
        print(f"[!] File not found: {pkl_path}")
        print("    Please ensure Step 0 was run and the file name matches exactly.")
        return
        
    with open(pkl_path, "rb") as f:
        sequence_7d = pickle.load(f)
        
    print(f"[v] Loaded real sequence with {len(sequence_7d)} fragments.")
    
    # 2. Build a dynamic Vocabulary mapping SMILES to Token IDs
    vocab = {"[SOS]": 0}
    input_ids_list = [0] 
    target_7d_list = [[0.0] * 7] 
    
    for item in sequence_7d:
        smiles = item["smiles"]
        if smiles not in vocab:
            vocab[smiles] = len(vocab)
            
        input_ids_list.append(vocab[smiles])
        target_7d_list.append(item["spatial_tokens"])
        
    vocab_size = len(vocab)
    print(f"    -> Built dynamic vocabulary (Size: {vocab_size}): {vocab}")
    
    # 3. Prepare Tensors for Teacher Forcing Training
    device = torch.device("cpu")
    
    # Autoregressive shifting: Predict step t+1 given step 1..t
    # Input : [SOS, Frag1, Frag2, Frag3]
    # Target: [Frag1, Frag2, Frag3, Frag4]
    input_ids = torch.tensor([input_ids_list[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([input_ids_list[1:]], dtype=torch.long, device=device)
    target_7d = torch.tensor([target_7d_list[1:]], dtype=torch.float, device=device)
    
    print(f"    -> Teacher Forcing Input IDs shape: {input_ids.shape}")
    print(f"    -> Teacher Forcing Target 7D shape: {target_7d.shape}")
    
    # 4. Mock Protein Contexts (Outputs from Step 1)
    hidden_dim = 256
    h_target = torch.randn(1, 20, hidden_dim, device=device)
    h_anti_1 = torch.randn(1, 15, hidden_dim, device=device)
    list_h_anti = [h_anti_1]
    
    # 5. Initialize Dual-Stream Generator
    generator = DualStreamLigandGenerator(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    
    # 6. Forward Pass
    print("\n[*] Executing Forward Pass (Dual-Stream AR)...")
    logits, pi, mu, sigma, attn = generator(input_ids, h_target, list_h_anti)
    
    # 7. Calculate Real Losses
    print("[*] Calculating Real Multi-Objective Losses...")
    
    # Stream 1: Semantic Loss (Token Classification)
    # Reshape logits to [batch * seq_len, vocab_size], targets to [batch * seq_len]
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), target_ids.view(-1))
    
    # Stream 2: Geometric Loss (7D MDN Regression)
    # Reshape MDN distributions and target coordinates
    num_gaussians = pi.size(2)
    pi_flat = pi.view(-1, num_gaussians)
    mu_flat = mu.view(-1, num_gaussians, 7)
    sigma_flat = sigma.view(-1, num_gaussians, 7)
    target_7d_flat = target_7d.view(-1, 7)
    
    mdn_loss = mdn_nll_loss(pi_flat, mu_flat, sigma_flat, target_7d_flat)
    
    print("\n[v] Loss Calculation Successful:")
    print(f"    -> Stream 1 (Semantic) Cross-Entropy Loss: {ce_loss.item():.4f}")
    print(f"    -> Stream 2 (Geometric) MDN NLL Loss:      {mdn_loss.item():.4f}")
    print("=== STEP 4 REAL DATA INTEGRATION TEST COMPLETED ===")

if __name__ == "__main__":
    test_step4_real_data()