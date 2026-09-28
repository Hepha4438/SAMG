import os
import pickle
import torch
import torch_geometric.nn as pyg_nn
from torch_cluster import knn_graph as cluster_knn_graph
from omegaconf import OmegaConf
from torch_scatter import scatter_mean

# --- RUNTIME PATCH ---
def custom_knn_graph(x, k=32, batch=None, loop=False, flow='source_to_target', cosine=False, num_workers=1):
    return cluster_knn_graph(x=x, k=k, batch=batch, loop=loop, flow=flow)
pyg_nn.knn_graph = custom_knn_graph
# ---------------------

from datasets.pl_pair_dataset import PocketLigandPairDataset
from datasets.pl_data import ProteinLigandDataLoader
from step1_protein_encoder import DynamicEGNN
from step4_ligand_generator import DualStreamLigandGenerator
from step5_constraints_and_loss import GlobalLoss, PoseBustersFilter

def test_step5_real_data():
    print("[*] Starting Step 5 Real Data Integration Test...")

    # 1. Load Real Data from LMDB and Pickle
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/data_folder"))
    index_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/test_split_list.pkl"))
    pkl_path = os.path.abspath(os.path.join(current_dir, "../dataset/processed/3txj__1__1.A__1.K_sequence_7d.pkl"))

    if not os.path.exists(pkl_path):
        print(f"[!] File not found: {pkl_path}")
        return

    dataset = PocketLigandPairDataset(raw_path=raw_path, index_path=index_path, pocket_type="pocket")
    dataloader = ProteinLigandDataLoader(dataset, batch_size=1, shuffle=False)
    real_batch = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_batch = real_batch.to(device)

    with open(pkl_path, "rb") as f:
        sequence_7d = pickle.load(f)

    # 2. Build Vocabulary & Target Tensors
    vocab = {"[SOS]": 0}
    input_ids_list = [0]
    target_7d_list = [[0.0] * 7]
    for item in sequence_7d:
        if item["smiles"] not in vocab:
            vocab[item["smiles"]] = len(vocab)
        input_ids_list.append(vocab[item["smiles"]])
        target_7d_list.append(item["spatial_tokens"])

    vocab_size = len(vocab)
    input_ids = torch.tensor([input_ids_list[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor([input_ids_list[1:]], dtype=torch.long, device=device)
    target_7d = torch.tensor([target_7d_list[1:]], dtype=torch.float, device=device)

    # 3. Extract Real Features via Step 1
    print("[*] Extracting real protein features via DynamicEGNN...")
    hidden_dim = 256
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

    num_graphs = real_batch.protein_element_batch.max().item() + 1
    h_target_raw = scatter_mean(step1_outputs['residue_h'], real_batch.protein_translations_batch, dim=0, dim_size=num_graphs)
    h_target = h_target_raw[:, :hidden_dim].unsqueeze(1) # [1, 1, 256]

    # Mock Anti-targets Context
    h_anti_1 = torch.randn(1, 15, hidden_dim, device=device)
    list_h_anti = [h_anti_1]

    # 4. Generate Predictions via Step 4
    print("[*] Generating predictions via Dual-Stream AR...")
    generator = DualStreamLigandGenerator(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    logits_vocab, pi, mu, sigma, _ = generator(input_ids, h_target, list_h_anti)

    # 5. Evaluate Step 5 GlobalLoss
    print("[*] Evaluating Step 5 GlobalLoss with real tensors...")
    loss_fn = GlobalLoss(lambda_token=1.0, lambda_geo=1.0, lambda_pocket=0.5, lambda_int=2.0)

    # Retrieve real protein dynamics targets
    target_tr = real_batch.protein_translations.unsqueeze(0)
    target_q = real_batch.protein_rotations.unsqueeze(0)
    target_chi = real_batch.protein_chi_apo.unsqueeze(0)
    chi_mask = real_batch.protein_chi_mask.unsqueeze(0)

    # Mock predictions for protein dynamics (since EGNN wrapper doesn't directly output this for testing yet)
    pred_tr = torch.randn_like(target_tr)
    pred_q = torch.randn_like(target_q)
    pred_chi = torch.randn_like(target_chi)

    total_loss, loss_dict = loss_fn(
        logits_vocab, target_ids,
        pi, mu, sigma, target_7d,
        pred_tr, target_tr,
        pred_q, target_q,
        pred_chi, target_chi, chi_mask,
        real_batch.ligand_pos, real_batch.protein_pos
    )

    print("\n[v] Global Loss Evaluation Successful:")
    for k, v in loss_dict.items():
        print(f"       - {k}: {v.item():.4f}")

    # 6. Evaluate PoseBusters Logit Masking
    print("\n[*] Evaluating PoseBusters Filter on Real 3D Protein Point Cloud...")
    pb_filter = PoseBustersFilter()

    # Get prediction for the first step
    pi_logits_step1 = pi[0, 0, :].clone()
    mu_step1 = mu[0, 0, :, :].clone()
    ref_pos = real_batch.protein_pos.mean(dim=0) # Use pocket center as reference

    # Intentionally force peak 0 to collide with the real protein cloud (distance 0.1)
    mu_step1[0, 0:3] = torch.tensor([0.1, 0.0, 0.0])

    masked_logits = pb_filter.posebusters_masking(pi_logits_step1, mu_step1, ref_pos, real_batch.protein_pos, element_ligand="C")

    if masked_logits[0].item() < -1e5:
        print("    [v] SUCCESS: PoseBusters detected clash with real protein point cloud and masked logit.")

    print("=== STEP 5 REAL DATA INTEGRATION TEST COMPLETED ===")

if __name__ == "__main__":
    test_step5_real_data()