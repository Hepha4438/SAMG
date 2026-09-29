import os
import glob
import pickle
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint 
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from torch_scatter import scatter_mean

import torch_geometric.nn as pyg_nn
from torch_cluster import knn_graph as cluster_knn_graph

# Tận dụng Tensor Cores trên RTX A5000 và bật chế độ TF32 tăng tốc toán học
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# --- RUNTIME PATCH ---
def custom_knn_graph(x, k=32, batch=None, loop=False, flow='source_to_target', cosine=False, num_workers=1):
    return cluster_knn_graph(x=x, k=k, batch=batch, loop=loop, flow=flow)
pyg_nn.knn_graph = custom_knn_graph

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
SAMG_ROOT = os.path.dirname(CURRENT_DIR)                 
DATASET_DIR = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "data_folder")
SPLIT_FILE = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "split_druglike_dict.pkl")
PROCESSED_DIR = os.path.join(SAMG_ROOT, "dataset", "processed")
VOCAB_PATH = os.path.join(PROCESSED_DIR, "global_vocab.pkl")

import sys
sys.path.append(os.path.join(SAMG_ROOT, "src", "utils"))
from system_logger import SAMGLoggingCallback

from utils.data import PDBProtein, parse_sdf_file, compute_residue_transforms
from datasets.pl_data import ApoHoloLigandData, torchify_dict

from step1_protein_encoder import DynamicEGNN, StaticEGNN
from step4_ligand_generator import DualStreamLigandGenerator
from step5_constraints_and_loss import GlobalLoss


class SAMGOptimizedDataset(Dataset):
    def __init__(self, dataset_dir, processed_dir, vocab, split_file, split_mode="train"):
        self.dataset_dir = dataset_dir
        self.processed_dir = processed_dir
        self.vocab = vocab
        
        with open(split_file, "rb") as f:
            splits = pickle.load(f)
            
        if split_mode == "val" and "valid" in splits:
            split_mode = "valid"
            
        self.target_entries = splits.get(split_mode, [])
        available_pkls = sorted(glob.glob(os.path.join(processed_dir, "*_sequence_7d.pkl")))
        self.pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p for p in available_pkls}

        n_has_pkl = 0
        n_excluded_unstable = 0
        n_excluded_frag_unstable = 0
        self.valid_indices = []
        for i, entry in enumerate(self.target_entries):
            pli_id = entry[2].split("/")[0]
            pkl_path = self.pkl_by_id.get(pli_id)
            if pkl_path is None:
                continue
            n_has_pkl += 1
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            if not pkl_data.get("frame_stable", False):
                n_excluded_unstable += 1
                continue
            if any(not item.get("frag_frame_stable", True) for item in pkl_data["sequence"]):
                n_excluded_frag_unstable += 1
                continue
            self.valid_indices.append(i)

        print(f"[*] [{split_mode}] tổng entry: {len(self.target_entries)}  "
              f"có pkl: {n_has_pkl}  bị loại bởi frame_stable: {n_excluded_unstable}  "
              f"bị loại bởi frag_frame_stable: {n_excluded_frag_unstable}  "
              f"valid: {len(self.valid_indices)}")
        assert len(self.valid_indices) > 0, f"split {split_mode}: 0 mẫu hợp lệ"

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # --- 1. GHÉP ENTRY <-> PKL THEO pli_id (KHÔNG theo index) ---
        entry = self.target_entries[self.valid_indices[idx]]
        pkl_path = self.pkl_by_id[entry[2].split("/")[0]]

        holo_pocket_fn, apo_pocket_fn, ligand_fn = entry[0], entry[1], entry[2]

        ligand_path = os.path.join(self.dataset_dir, ligand_fn)
        ligand_dict = parse_sdf_file(ligand_path)

        if ligand_dict is None:
            raise RuntimeError(f"Không đọc được ligand: {ligand_path}")

        # --- 2. NẠP DỮ LIỆU PROTEIN VÀ KHỞI TẠO BATCH ---
        apo_pocket_dict = PDBProtein(os.path.join(self.dataset_dir, apo_pocket_fn)).to_dict_atom()
        holo_pocket_dict = PDBProtein(os.path.join(self.dataset_dir, holo_pocket_fn)).to_dict_atom()
        
        data = ApoHoloLigandData.from_apo_holo_ligand_dicts(
            apo_dict=torchify_dict(apo_pocket_dict),
            holo_dict=torchify_dict(holo_pocket_dict),
            ligand_dict=torchify_dict(ligand_dict), 
        )
        
        # --- 3. AUTO-CACHING CHỈ SỐ GÓC (TĂNG TỐC EPOCH 1 TRỞ ĐI) ---
        cache_path = pkl_path.replace("_sequence_7d.pkl", "_protein_transforms.pt")
        
        if os.path.exists(cache_path):
            try:
                # Nạp thẳng từ ổ cứng nếu đã tính ở Epoch 0
                cached_data = torch.load(cache_path, weights_only=True)
                rotations = cached_data['rotations']
                translations = cached_data['translations']
                chi_apo = cached_data['chi_apo']
                chi_holo = cached_data['chi_holo']
                chi_mask = cached_data['chi_mask']
            except Exception:
                # Nếu file hỏng, tính lại
                rotations, _, translations, chi_apo, chi_holo, chi_mask = compute_residue_transforms(
                    protein_pos_apo=data.protein_pos, protein_pos_holo=data.protein_pos_holo,
                    protein_atom_name=data.protein_atom_name, protein_atom_to_aa_name=data.protein_atom_to_aa_name,
                    protein_atom_to_aa_group=data.protein_atom_to_aa_group,
                )
        else:
            # Lần chạy đầu tiên: Tính toán nặng
            rotations, _, translations, chi_apo, chi_holo, chi_mask = compute_residue_transforms(
                protein_pos_apo=data.protein_pos, protein_pos_holo=data.protein_pos_holo,
                protein_atom_name=data.protein_atom_name, protein_atom_to_aa_name=data.protein_atom_to_aa_name,
                protein_atom_to_aa_group=data.protein_atom_to_aa_group,
            )
            # Lưu file cache kèm Process ID (tránh lỗi xung đột khi chạy đa luồng num_workers > 0)
            try:
                tmp_path = f"{cache_path}.tmp.{os.getpid()}"
                torch.save({
                    'rotations': rotations, 'translations': translations,
                    'chi_apo': chi_apo, 'chi_holo': chi_holo, 'chi_mask': chi_mask
                }, tmp_path)
                os.rename(tmp_path, cache_path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # Gán thuộc tính (Step 7 vẫn đọc bình thường)
        data.protein_rotations = rotations
        data.protein_translations = translations
        data.protein_chi_apo = chi_apo
        data.protein_chi_holo = chi_holo
        data.protein_chi_mask = chi_mask
        
        # --- 4. KHỞI TẠO TENSOR BATCH & SEQUENCE 7D ---
        num_residues = max(data.protein_atom_to_aa_group).item() + 1
        data.protein_translations_batch = torch.zeros(num_residues, dtype=torch.long)
        data.protein_element_batch = torch.zeros(data.protein_element.size(0), dtype=torch.long)
        data.ligand_element_batch = torch.zeros(data.ligand_element.size(0), dtype=torch.long)

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        sequence_7d = pkl_data["sequence"]

        input_ids_list = [self.vocab.get("[SOS]", 0)]
        target_7d_list = [[0.0] * 7]
        for item in sequence_7d:
            smiles_val = item.get("smiles")
            input_ids_list.append(self.vocab.get(smiles_val, self.vocab.get("[UNK]", 1)))
            target_7d_list.append(item["spatial_tokens"])

        data.input_ids = torch.tensor(input_ids_list[:-1], dtype=torch.long)
        data.target_ids = torch.tensor(input_ids_list[1:], dtype=torch.long)
        data.target_7d = torch.tensor(target_7d_list[1:], dtype=torch.float)
        data.seq_len = len(input_ids_list) - 1

        # Dọn rác
        if hasattr(data, 'keys'):
            keys_to_delete = [k for k in data.keys() if isinstance(data[k], dict)]
            for k in keys_to_delete:
                del data[k]

        return data


class SAMGLightningModule(pl.LightningModule):
    def __init__(self, config, vocab_size, shift_factors=None, scale_factors=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.prot_emb = nn.Embedding(100, config.hidden_dim)
        self.lig_emb = nn.Embedding(100, config.hidden_dim)
        
        self.dynamic_encoder = DynamicEGNN(config.protein_encoder)
        # self.static_encoder = StaticEGNN(
        #     num_layers=config.protein_encoder.num_layers, 
        #     hidden_dim=config.hidden_dim, 
        #     k=config.protein_encoder.knn, 
        #     edge_feat_dim=config.protein_encoder.edge_feat_dim,
        #     num_r_gaussian=config.protein_encoder.num_r_gaussian
        # )
        if scale_factors is None:
            import warnings
            warnings.warn("scale_factors is None: standardization sẽ là no-op trong DualStreamLigandGenerator.")

        self.generator = DualStreamLigandGenerator(
            vocab_size=vocab_size,
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            shift_factors=shift_factors,
            scale_factors=scale_factors,
        )
        self.global_loss = GlobalLoss(
            lambda_token=config.loss_weights.token, 
            lambda_geo=config.loss_weights.geo, 
            lambda_pocket=config.loss_weights.pocket, 
            lambda_int=config.loss_weights.int, 
            d_threshold=config.loss_weights.d_threshold
        )

    def forward(self, data_batch):
        device = data_batch.protein_pos.device
        hidden_dim = self.config.hidden_dim
        
        h_protein = data_batch.protein_atom_feature.float() if hasattr(data_batch, 'protein_atom_feature') else self.prot_emb(data_batch.protein_element.long())
        h_ligand = data_batch.ligand_atom_feature_full.float() if hasattr(data_batch, 'ligand_atom_feature_full') else self.lig_emb(data_batch.ligand_element.long())

        step1_outputs = self.dynamic_encoder(
            h_protein=h_protein, h_ligand=h_ligand, protein_pos=data_batch.protein_pos, ligand_pos=data_batch.ligand_pos,
            batch_protein=data_batch.protein_element_batch, batch_ligand=data_batch.ligand_element_batch, data=data_batch
        )
        
        num_graphs = data_batch.protein_element_batch.max().item() + 1
        prot_trans_batch = data_batch.protein_translations_batch
        
        residue_h = step1_outputs['residue_h']
        h_target = scatter_mean(residue_h, prot_trans_batch, dim=0, dim_size=num_graphs)[:, :hidden_dim].unsqueeze(1) 

        # h_anti_raw, _ = self.static_encoder(h_protein, data_batch.protein_pos + 1.5, torch.zeros_like(data_batch.protein_pos[:,0], dtype=torch.bool), data_batch.protein_element_batch)
        # list_h_anti = [scatter_mean(h_anti_raw, data_batch.protein_element_batch, dim=0, dim_size=num_graphs).unsqueeze(1)] 
        list_h_anti = []

        lens = data_batch.seq_len.tolist()
        if isinstance(lens, int): lens = [lens]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(list(torch.split(data_batch.input_ids, lens)), batch_first=True, padding_value=0).to(device)
        target_ids = torch.nn.utils.rnn.pad_sequence(list(torch.split(data_batch.target_ids, lens)), batch_first=True, padding_value=0).to(device)
        target_7d = torch.nn.utils.rnn.pad_sequence(list(torch.split(data_batch.target_7d, lens)), batch_first=True, padding_value=0.0).to(device)

        logits_vocab, loss_geo_raw, sampled_7d, _ = self.generator(input_ids, h_target, list_h_anti, target_7d=target_7d)

        # --- ĐỒNG BỘ KÍCH THƯỚC ĐỘNG GIỮA PREDICTION VÀ TARGET (ĐẶC BIỆT TÁCH BẠCH PROTEIN VS LIGAND) ---
        num_res_pred = step1_outputs['pred_res_tr'].size(0)
        num_res_chi_pred = step1_outputs['pred_res_chi'].size(0)

        def align_tensor(target_tensor, target_len, default_val=0.0):
            if target_tensor is None:
                return torch.full((target_len,), default_val, device=device)
            current_len = target_tensor.size(0)
            if current_len == target_len:
                return target_tensor
            elif current_len > target_len:
                return target_tensor[:target_len]
            else:
                pad_shape = (target_len - current_len, *target_tensor.shape[1:])
                padding = torch.full(pad_shape, default_val, dtype=target_tensor.dtype, device=target_tensor.device)
                return torch.cat([target_tensor, padding], dim=0)

        target_tr = align_tensor(data_batch.protein_translations if hasattr(data_batch, 'protein_translations') else None, num_res_pred).unsqueeze(0)
        target_q = align_tensor(data_batch.protein_rotations if hasattr(data_batch, 'protein_rotations') else None, num_res_pred).unsqueeze(0)
        target_chi = align_tensor(data_batch.protein_chi_apo if hasattr(data_batch, 'protein_chi_apo') else None, num_res_chi_pred).unsqueeze(0)
        chi_mask = align_tensor(data_batch.protein_chi_mask if hasattr(data_batch, 'protein_chi_mask') else None, num_res_chi_pred, default_val=0).unsqueeze(0)

        total_loss, loss_dict = self.global_loss(
            logits_vocab, target_ids, 
            loss_geo_raw, 
            step1_outputs['pred_res_tr'].unsqueeze(0), target_tr, 
            step1_outputs['pred_res_rot'].unsqueeze(0), target_q,
            step1_outputs['pred_res_chi'].unsqueeze(0), target_chi, 
            chi_mask, data_batch.ligand_pos, data_batch.protein_pos
        )
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch)

        log_dict = {k: v.detach() for k, v in loss_dict.items()}

        self.log_dict(log_dict, batch_size=batch.num_graphs, sync_dist=True,
                       on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch)
        self.log("val/total_loss", loss, batch_size=batch.num_graphs, sync_dist=True)
        self.log("val_total_loss", loss, batch_size=batch.num_graphs, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss"}}


if __name__ == "__main__":
    config = OmegaConf.create({
        "hidden_dim": 256,
        "num_heads": 4,
        "num_gaussians": 10,
        "lr": 1e-4,
        "batch_size": 8,  
        "protein_encoder": {
            "num_blocks": 3,
            "num_layers": 3,
            "hidden_dim": 256,
            "n_heads": 4,
            "knn": 16,
            "edge_feat_dim": 5,
            "num_r_gaussian": 20,
            "num_node_types": 8
        },
        "loss_weights": {
            # "int": 0.0 vì L_Int hiện dùng ligand_pos ground-truth -> gradient
            # = 0. Bật lại khi nối 3D Assembler vào training.
            "token": 1.0, "geo": 1.0, "pocket": 1.0, "int": 0.0, "d_threshold": 2.5
        }
    })

    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

    train_dataset = SAMGOptimizedDataset(DATASET_DIR, PROCESSED_DIR, vocab, SPLIT_FILE, split_mode="train")
    val_dataset = SAMGOptimizedDataset(DATASET_DIR, PROCESSED_DIR, vocab, SPLIT_FILE, split_mode="valid")
    
    print(f"[*] Total valid Train complexes: {len(train_dataset)}")
    print(f"[*] Total valid Val complexes: {len(val_dataset)}")

    from datasets.pl_data import ProteinLigandDataLoader
    train_loader = ProteinLigandDataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )
    val_loader = ProteinLigandDataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=6, pin_memory=True, persistent_workers=True, prefetch_factor=4
    )

    SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler_7d.pt")
    if os.path.exists(SCALER_PATH):
        scaler_dict = torch.load(SCALER_PATH, map_location="cpu")
        shift_factors = scaler_dict["shift"]
        scale_factors = scaler_dict["scale"]
        print("[*] Đã Load Exact Standard Scaler từ Dataset.")
    else:
        shift_factors, scale_factors = None, None
        print("[!] Không tìm thấy scaler_7d.pt. Dùng cấu hình chuẩn hóa gốc.")

    # -- Khởi tạo Model --
    model = SAMGLightningModule(
        config, 
        vocab_size=len(vocab), 
        shift_factors=shift_factors, 
        scale_factors=scale_factors
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(SAMG_ROOT, "saved_checkpoints_flow"),
        filename="samg-opt-{epoch:03d}-{val_total_loss:.4f}",
        monitor="val_total_loss", mode="min", save_top_k=3, save_last=True
    )
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(SAMG_ROOT, "saved_checkpoints_flow"),
        filename="periodic-{epoch:03d}",
        every_n_epochs=5, save_top_k=-1, save_last=False
    )

    logger = WandbLogger(project="SAMG-Drug-Design", name="sequence_generator_optimized")
    logging_callback = SAMGLoggingCallback(log_dir=os.path.join(SAMG_ROOT, "logs"))
    trainer = pl.Trainer(
        max_epochs=25,
        accelerator="gpu",           
        devices=1,                   
        precision="32-true",
        logger=logger,
        callbacks=[checkpoint_callback, periodic_checkpoint_callback, logging_callback],
        log_every_n_steps=10,
        gradient_clip_val=1.0,          
        accumulate_grad_batches=2     
    )

    ckpt_path = None
    ckpt_dir = os.path.join(SAMG_ROOT, "saved_checkpoints_flow")
    if os.path.exists(ckpt_dir):
        ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        if ckpts:
            ckpt_path = max(ckpts, key=os.path.getmtime)
            print(f"\n[*] Đang khôi phục quá trình huấn luyện từ: {os.path.basename(ckpt_path)}")

    print("\n[*] Starting Optimized SAMG Training...")
    
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)