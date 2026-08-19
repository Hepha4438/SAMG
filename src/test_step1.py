import os
import pickle
import torch
import torch_geometric.nn as pyg_nn
from torch_cluster import knn_graph as cluster_knn_graph
from omegaconf import OmegaConf

# =====================================================================
# --- RUNTIME MONKEY PATCH ---
def custom_knn_graph(x, k=32, batch=None, loop=False, flow='source_to_target', cosine=False, num_workers=1):
    return cluster_knn_graph(x=x, k=k, batch=batch, loop=loop, flow=flow)

pyg_nn.knn_graph = custom_knn_graph
# =====================================================================

from datasets.pl_pair_dataset import PocketLigandPairDataset
from datasets.pl_data import ProteinLigandDataLoader
from step1_protein_encoder import StaticEGNN, DynamicEGNN

def test_step1_forward_real_data():
    print("[*] Starting Step 1 GNN forward pass test using REAL dataset...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/data_folder"))
    dict_index_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/split_druglike_dict.pkl"))
    flat_index_path = os.path.abspath(os.path.join(current_dir, "../dataset/Apo2Mol_Dataset/test_split_list.pkl"))

    # --- ĐỌC DICTIONARY THÀNH LIST ---
    if os.path.exists(dict_index_path):
        with open(dict_index_path, 'rb') as f:
            data_dict = pickle.load(f)
        # Lấy riêng danh sách của tập 'test'
        test_list = data_dict.get('test', [])
        with open(flat_index_path, 'wb') as f:
            pickle.dump(test_list, f)
    else:
        print(f"[!] Cannot find index file at: {dict_index_path}")
        return
    
    # --- FIX: XÓA ĐÚNG FILE TĨNH CỦA LMDB CHỨ KHÔNG PHẢI FOLDER ---
    lmdb_path = raw_path + "_pocket_apo2mol_final.lmdb"
    lock_path = lmdb_path + "-lock"
    
    deleted_old = False
    if os.path.exists(lmdb_path):
        os.remove(lmdb_path)
        deleted_old = True
    if os.path.exists(lock_path):
        os.remove(lock_path)
        
    if deleted_old:
        print("[*] Deleted old corrupted LMDB database files successfully!")

    print(f"[*] Connecting to LMDB dataset at: {lmdb_path}")
    
    # Vì file LMDB hỏng đã bị xóa thật, class này BẮT BUỘC phải chạy lại hàm _process()
    dataset = PocketLigandPairDataset(raw_path=raw_path, index_path=flat_index_path, pocket_type="pocket")
    dataloader = ProteinLigandDataLoader(dataset, batch_size=2, shuffle=False)
    
    try:
        real_batch = next(iter(dataloader))
        print(f"[v] Successfully loaded a real batch with {real_batch.num_graphs} protein-ligand pairs.")
    except StopIteration:
        print("[!] Error: Dataloader is still empty. Please check your physical data_folder files!")
        return

    device = torch.device("cpu")
    real_batch = real_batch.to(device)

    # Trích xuất Tensor gốc từ Dataloader
    h_protein = real_batch.protein_atom_feature.float() if hasattr(real_batch, 'protein_atom_feature') else torch.randn(real_batch.protein_pos.size(0), 128, device=device)
    protein_pos = real_batch.protein_pos
    batch_protein = real_batch.protein_element_batch
    mask_ligand = torch.zeros(protein_pos.size(0), dtype=torch.bool, device=device)

    # 3. Test StaticEGNN
    print("\n[*] Testing StaticEGNN with real Anti-target features...")
    hidden_dim = h_protein.size(-1)
    static_egnn = StaticEGNN(num_layers=2, hidden_dim=hidden_dim, k=4).to(device)
    h_static, x_static = static_egnn(h_protein, protein_pos, mask_ligand, batch_protein)
    
    print(f"    -> Static output features shape: {h_static.shape}")
    print(f"    -> Static output coordinates shape: {x_static.shape}")
    assert torch.allclose(protein_pos, x_static), "[!] Error: StaticEGNN modified coordinates!"
    print("    [v] StaticEGNN real data test passed successfully.")

    # 4. Test DynamicEGNN
    print("\n[*] Testing DynamicEGNN with real Target pocket data...")
    config = OmegaConf.create({
        "num_blocks": 1,
        "num_layers": 2,
        "hidden_dim": hidden_dim,
        "n_heads": 4,
        "knn": 4,
        "edge_feat_dim": 0,
        "num_r_gaussian": 20,
        "num_node_types": 8
    })

    dynamic_egnn = DynamicEGNN(config).to(device)

    h_ligand = real_batch.ligand_atom_feature_full.float() if hasattr(real_batch, 'ligand_atom_feature_full') else torch.randn(real_batch.ligand_pos.size(0), hidden_dim, device=device)
    ligand_pos = real_batch.ligand_pos
    batch_ligand = real_batch.ligand_element_batch

    outputs = dynamic_egnn(
        h_protein=h_protein,
        h_ligand=h_ligand,
        protein_pos=protein_pos,
        ligand_pos=ligand_pos,
        batch_protein=batch_protein,
        batch_ligand=batch_ligand,
        data=real_batch
    )

    print(f"    -> Updated real protein positions shape: {outputs['updated_protein_pos'].shape}")
    print(f"    -> Predicted translation vector shape: {outputs['pred_res_tr'].shape}")
    print(f"    -> Predicted rotation quaternion shape: {outputs['pred_res_rot'].shape}")
    print(f"    -> Predicted side-chain chi angles shape: {outputs['pred_res_chi'].shape}")
    print("    [v] DynamicEGNN real data test passed successfully.")

    print("\n=== STEP 1 REAL DATA FORWARD PASS TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_step1_forward_real_data()