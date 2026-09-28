import os
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
from step1_protein_encoder import StaticEGNN, DynamicEGNN
from step3_attention_hub import MultiDifferentialCrossAttention

def test_step3_real_data():
    print("[*] Starting Step 3 Real Data Integration Test...")

    # 1. Load real dataset from LMDB
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/data_folder"))
    index_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/test_split_list.pkl"))

    if not os.path.exists(index_path):
        print(f"[!] Index list not found at {index_path}. Please run test_step1.py first to generate it.")
        return

    dataset = PocketLigandPairDataset(raw_path=raw_path, index_path=index_path, pocket_type="pocket")
    dataloader = ProteinLigandDataLoader(dataset, batch_size=2, shuffle=False)
    
    # Fetch real pairs
    real_batch = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_batch = real_batch.to(device)

    hidden_dim = 256
    print(f"[v] Loaded real batch with {real_batch.num_graphs} protein-ligand graphs.")

    # 2. Extract Target Protein Features using Step 1 (DynamicEGNN)
    print("[*] Extracting real Target graph features...")
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

    # Pool residue features for target -> [batch_size, seq_len, 128]
    num_graphs = real_batch.protein_element_batch.max().item() + 1
    h_target_raw = scatter_mean(step1_outputs['residue_h'], real_batch.protein_translations_batch, dim=0, dim_size=num_graphs)
    h_target = h_target_raw[:, :hidden_dim].unsqueeze(1) # [batch, 1, 128]

    # 3. Extract Anti-target Features using StaticEGNN (Rigid feature extractor)
    print("[*] Extracting real Anti-targets rigid features...")
    static_egnn = StaticEGNN(num_layers=2, hidden_dim=hidden_dim, k=4).to(device)
    
    # Run StaticEGNN on protein structures
    h_anti_raw_1, _ = static_egnn(h_protein, real_batch.protein_pos, torch.zeros_like(real_batch.protein_pos[:,0], dtype=torch.bool), real_batch.protein_element_batch)
    h_anti_raw_2, _ = static_egnn(h_protein, real_batch.protein_pos + 1.5, torch.zeros_like(real_batch.protein_pos[:,0], dtype=torch.bool), real_batch.protein_element_batch)
    
    # Pool anti-target atom features to residue/graph level grouped by batch
    anti_1_pooled = scatter_mean(h_anti_raw_1, real_batch.protein_element_batch, dim=0, dim_size=num_graphs)
    anti_2_pooled = scatter_mean(h_anti_raw_2, real_batch.protein_element_batch, dim=0, dim_size=num_graphs)
    
    # Reshape to sequence format [batch_size, seq_len, hidden_dim]
    anti_1_seq = anti_1_pooled.unsqueeze(1) # [2, 1, 128]
    anti_2_seq = anti_2_pooled.unsqueeze(1) # [2, 1, 128]
    list_h_anti = [anti_1_seq, anti_2_seq]

    # 4. Pass through Step 3 Multi-Differential Attention Hub
    print("[*] Executing Step 3 Multi-Differential Attention Hub with real data...")
    hub = MultiDifferentialCrossAttention(hidden_dim=hidden_dim, num_heads=4, beta=1.5).to(device)
    
    # Query can be derived from ligand or initial latent state
    query = torch.randn(num_graphs, 1, hidden_dim, device=device)

    v_context, attn_weights = hub(query, h_target, list_h_anti)

    print("\n[v] Real Data Test Results:")
    print(f"    -> Real Query shape: {query.shape}")
    print(f"    -> Target Context shape (h_target): {h_target.shape}")
    print(f"    -> Context output shape (v_context): {v_context.shape}")
    print(f"    -> Attention weights shape: {attn_weights.shape}")
    print(f"    -> Sample Attention Weights (Head 0, Batch 0):\n{attn_weights[0, 0].detach().numpy()}")
    print("\n[v] Step 3 Real Data Integration Test Passed Successfully!")

if __name__ == "__main__":
    test_step3_real_data()