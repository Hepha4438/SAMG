"""
test_encoder_no_ligand.py -- BUOC 0 (P2b): UniTransformer/DynamicEGNN co chay duoc voi
KHONG nguyen tu ligand nao hay khong, so voi phuong an dung 1 dummy atom tai pocket_t.

KHONG chay trong sandbox phat trien dung de viet script nay (thieu torch_geometric/
torch_scatter/torch_cluster/pytorch_lightning/pyro-ppl -- da xac nhan bang
`pip show torch-geometric torch-scatter torch-cluster pytorch-lightning pyro-ppl`).
Chay tren may GPU co du dependency:
    python src/test_encoder_no_ligand.py

Ket luan cua script nay quyet dinh nhanh nao duoc dung MAC DINH cho P2b (co
`ligand_mode` trong step6_trainer.py/step7_evaluation.py): neu nhanh "LIGAND RONG"
bao OK thi co the doi config sang "empty"; neu FAIL thi giu nguyen "dummy" (mac dinh
an toan hien tai, xem docstring cua ligand_mode).

Yeu cau du lieu: dataset/processed da qua P1c (pkl la dict {"sequence": [...]}) VA
da chay add_pocket_frame_to_pkl.py (co "pocket_R"/"pocket_t") -- neu chua, nhanh
DUMMY ATOM se bao loi ro rang khi doc pocket_t thay vi am tham dung sai vi tri.
"""
import os
import sys
import pickle
import traceback

import torch
from omegaconf import OmegaConf

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import step6_trainer as st  # side-effect: áp dụng monkeypatch pyg_nn.knn_graph trước khi models/* import nó


def build_one_batch(batch_size=2):
    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(st.VOCAB_PATH):
        with open(st.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

    scaler_path = os.path.join(st.PROCESSED_DIR, "scaler_7d.pt")
    pos_scale = None
    if os.path.exists(scaler_path):
        scaler_dict = torch.load(scaler_path, map_location="cpu")
        pos_scale = scaler_dict["scale"].flatten()[0].item()

    dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE,
        split_mode="train", pos_scale=pos_scale,
    )
    from datasets.pl_data import ProteinLigandDataLoader
    loader = ProteinLigandDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return next(iter(loader)), len(vocab)


def try_forward(model, batch, h_ligand, ligand_pos, batch_ligand, label):
    print(f"\n[*] --- Nhánh: {label} ---")
    try:
        h_protein = (
            batch.protein_atom_feature.float()
            if hasattr(batch, "protein_atom_feature")
            else model.prot_emb(batch.protein_element.long())
        )
        out = model.dynamic_encoder(
            h_protein=h_protein,
            h_ligand=h_ligand,
            protein_pos=batch.protein_pos,
            ligand_pos=ligand_pos,
            batch_protein=batch.protein_element_batch,
            batch_ligand=batch_ligand,
            data=batch,
        )
        residue_h = out["residue_h"]
        print(f"    OK -- residue_h.shape = {tuple(residue_h.shape)}")
        return True, None
    except Exception as e:
        traceback.print_exc()
        print(f"    FAIL -- {type(e).__name__}: {e}")
        return False, repr(e)


def main():
    config = OmegaConf.create({
        "hidden_dim": 256, "num_heads": 4,
        "protein_encoder": {
            "num_blocks": 3, "num_layers": 3, "hidden_dim": 256, "n_heads": 4,
            "knn": 16, "edge_feat_dim": 5, "num_r_gaussian": 20, "num_node_types": 8,
        },
        "loss_weights": {"token": 1.0, "geo": 1.0, "pocket": 1.0, "int": 0.0, "d_threshold": 2.5},
        "lr": 1e-4,
    })

    batch, vocab_size = build_one_batch(batch_size=2)
    model = st.SAMGLightningModule(config, vocab_size=vocab_size)
    model.eval()

    hidden_dim = config.hidden_dim
    device = batch.protein_pos.device
    num_graphs = batch.protein_element_batch.max().item() + 1

    # Nhánh 1: LIGAND RỖNG
    h_ligand_empty = torch.zeros(0, hidden_dim, device=device)
    ligand_pos_empty = torch.zeros(0, 3, device=device)
    batch_ligand_empty = torch.zeros(0, dtype=torch.long, device=device)
    ok_empty, err_empty = try_forward(
        model, batch, h_ligand_empty, ligand_pos_empty, batch_ligand_empty, "LIGAND RỖNG"
    )

    # Nhánh 2: DUMMY ATOM tại pocket_t, element id riêng (100). Vị trí lấy từ pocket_t
    # đọc trong pkl (P2a-2 migration) -- KHÔNG tính lại bằng code khác. pocket_t được
    # PyG batch bằng cat dim0 (mỗi sample góp 3 số) nên reshape lại thành [B, 3].
    pocket_t = batch.pocket_t.view(num_graphs, 3).to(device)
    # Đặc trưng dummy: vector ngẫu nhiên đại diện cho id embedding riêng (việc mở rộng
    # nn.Embedding(100,...) -> nn.Embedding(101,...) là thay đổi riêng cho P2b, không
    # phải mục tiêu của phép thử cấu trúc này).
    h_ligand_dummy = torch.randn(num_graphs, hidden_dim, device=device)
    ligand_pos_dummy = pocket_t
    batch_ligand_dummy = torch.arange(num_graphs, device=device)
    ok_dummy, err_dummy = try_forward(
        model, batch, h_ligand_dummy, ligand_pos_dummy, batch_ligand_dummy, "DUMMY ATOM"
    )

    print("\n" + "=" * 60)
    print("[BƯỚC 0] KẾT LUẬN:")
    print(f"    ligand rỗng : {'OK' if ok_empty else 'FAIL'}" + (f"  ({err_empty})" if err_empty else ""))
    print(f"    dummy atom  : {'OK' if ok_dummy else 'FAIL'}" + (f"  ({err_dummy})" if err_dummy else ""))
    print("=" * 60)


if __name__ == "__main__":
    main()
