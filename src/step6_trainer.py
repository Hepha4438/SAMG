import os
import glob
import pickle
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from torch_scatter import scatter_mean

import torch_geometric.nn as pyg_nn
from torch_cluster import knn_graph as cluster_knn_graph

# --- RUNTIME PATCH ---
def custom_knn_graph(x, k=32, batch=None, loop=False, flow='source_to_target', cosine=False, num_workers=1):
    return cluster_knn_graph(x=x, k=k, batch=batch, loop=loop, flow=flow)
pyg_nn.knn_graph = custom_knn_graph
# ---------------------

from datasets.pl_pair_dataset import PocketLigandPairDataset
from datasets.pl_data import ProteinLigandDataLoader
from step1_protein_encoder import DynamicEGNN, StaticEGNN
from step4_ligand_generator import DualStreamLigandGenerator
from step5_constraints_and_loss import GlobalLoss

class SAMGDatasetWrapper(Dataset):
    def __init__(self, base_dataset, processed_dir, vocab, split_file):
        self.base_dataset = base_dataset
        self.processed_dir = processed_dir
        self.vocab = vocab
        
        with open(split_file, "rb") as f:
            self.train_data_list = pickle.load(f)["train"]
            
        pkl_files = glob.glob(os.path.join(processed_dir, "*_sequence_7d.pkl"))
        self.valid_pli_ids = {os.path.basename(f).replace("_sequence_7d.pkl", "") for f in pkl_files}
        self.valid_indices = self._filter_valid_entries()

    def _filter_valid_entries(self):
        valid_indices = []
        for idx in range(len(self.base_dataset)):
            if idx >= len(self.train_data_list): break
            entry = self.train_data_list[idx]
            pli_id = entry[1].split("/")[0] 
            if pli_id in self.valid_pli_ids:
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        data = self.base_dataset[real_idx]
        
        entry = self.train_data_list[real_idx]
        pli_id = entry[1].split("/")[0] 
        pkl_path = os.path.join(self.processed_dir, f"{pli_id}_sequence_7d.pkl")
        
        input_ids_list = [self.vocab["[SOS]"]]
        target_7d_list = [[0.0] * 7]
        
        with open(pkl_path, "rb") as f:
            sequence_7d = pickle.load(f)
            
        for item in sequence_7d:
            token_id = self.vocab.get(item["smiles"], self.vocab["[UNK]"])
            input_ids_list.append(token_id)
            target_7d_list.append(item["spatial_tokens"])

        data.input_ids = torch.tensor(input_ids_list[:-1], dtype=torch.long)
        data.target_ids = torch.tensor(input_ids_list[1:], dtype=torch.long)
        data.target_7d = torch.tensor(target_7d_list[1:], dtype=torch.float)
        
        data.seq_len = len(input_ids_list) - 1

        # ---> BẢN VÁ TỐI THƯỢNG: Tiêu diệt các Dictionary ngầm gây Crash DataLoader <---
        keys_to_delete = [k for k in data.keys() if isinstance(data[k], dict)]
        for k in keys_to_delete:
            del data[k]
        # --------------------------------------------------------------------------------

        return data


class SAMGLightningModule(pl.LightningModule):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.prot_emb = nn.Embedding(100, config.hidden_dim)
        self.lig_emb = nn.Embedding(100, config.hidden_dim)
        
        self.dynamic_encoder = DynamicEGNN(config.protein_encoder)
        self.static_encoder = StaticEGNN(
            num_layers=config.protein_encoder.num_layers, 
            hidden_dim=config.hidden_dim, 
            k=config.protein_encoder.knn,
            edge_feat_dim=config.protein_encoder.edge_feat_dim,
            num_r_gaussian=config.protein_encoder.num_r_gaussian
        )
        
        self.generator = DualStreamLigandGenerator(
            vocab_size=vocab_size, hidden_dim=config.hidden_dim, 
            num_heads=config.num_heads, num_gaussians=config.num_gaussians
        )
        
        self.global_loss = GlobalLoss(
            lambda_token=config.loss_weights.token, lambda_geo=config.loss_weights.geo, 
            lambda_pocket=config.loss_weights.pocket, lambda_int=config.loss_weights.int, 
            d_threshold=config.loss_weights.d_threshold
        )

    def forward(self, data_batch):
        device = data_batch.protein_pos.device
        hidden_dim = self.config.hidden_dim
        
        h_protein = data_batch.protein_atom_feature.float() if hasattr(data_batch, 'protein_atom_feature') else self.prot_emb(data_batch.protein_element.long())
        h_ligand = data_batch.ligand_atom_feature_full.float() if hasattr(data_batch, 'ligand_atom_feature_full') else self.lig_emb(data_batch.ligand_element.long())

        step1_outputs = self.dynamic_encoder(
            h_protein=h_protein, h_ligand=h_ligand,
            protein_pos=data_batch.protein_pos, ligand_pos=data_batch.ligand_pos,
            batch_protein=data_batch.protein_element_batch, batch_ligand=data_batch.ligand_element_batch, data=data_batch
        )
        
        num_graphs = data_batch.protein_element_batch.max().item() + 1
        prot_trans_batch = data_batch.protein_translations_batch if hasattr(data_batch, 'protein_translations_batch') else data_batch.protein_element_batch
        h_target = scatter_mean(step1_outputs['residue_h'], prot_trans_batch, dim=0, dim_size=num_graphs)[:, :hidden_dim].unsqueeze(1) 

        h_anti_raw, _ = self.static_encoder(h_protein, data_batch.protein_pos + 1.5, torch.zeros_like(data_batch.protein_pos[:,0], dtype=torch.bool), data_batch.protein_element_batch)
        list_h_anti = [scatter_mean(h_anti_raw, data_batch.protein_element_batch, dim=0, dim_size=num_graphs).unsqueeze(1)] 

        # ---> BẢN VÁ: Tách Tensor phẳng thành List các Sequences nhờ seq_len <---
        lens = data_batch.seq_len.tolist()
        if isinstance(lens, int): lens = [lens]
        
        in_ids_split = list(torch.split(data_batch.input_ids, lens))
        tg_ids_split = list(torch.split(data_batch.target_ids, lens))
        tg_7d_split = list(torch.split(data_batch.target_7d, lens))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(in_ids_split, batch_first=True, padding_value=0).to(device)
        target_ids = torch.nn.utils.rnn.pad_sequence(tg_ids_split, batch_first=True, padding_value=0).to(device)
        target_7d = torch.nn.utils.rnn.pad_sequence(tg_7d_split, batch_first=True, padding_value=0.0).to(device)
        # --------------------------------------------------------------------------

        logits_vocab, pi, mu, sigma, _ = self.generator(input_ids, h_target, list_h_anti)

        target_tr = data_batch.protein_translations.unsqueeze(0) if hasattr(data_batch, 'protein_translations') else torch.zeros_like(step1_outputs['pred_res_tr']).unsqueeze(0)
        target_q = data_batch.protein_rotations.unsqueeze(0) if hasattr(data_batch, 'protein_rotations') else torch.zeros_like(step1_outputs['pred_res_rot']).unsqueeze(0)
        target_chi = data_batch.protein_chi_apo.unsqueeze(0) if hasattr(data_batch, 'protein_chi_apo') else (data_batch.protein_chi_angles.unsqueeze(0) if hasattr(data_batch, 'protein_chi_angles') else torch.zeros_like(step1_outputs['pred_res_chi']).unsqueeze(0))
        chi_mask = data_batch.protein_chi_mask.unsqueeze(0) if hasattr(data_batch, 'protein_chi_mask') else torch.ones_like(step1_outputs['pred_res_chi']).unsqueeze(0)

        total_loss, loss_dict = self.global_loss(
            logits_vocab, target_ids, pi, mu, sigma, target_7d,
            step1_outputs['pred_res_tr'].unsqueeze(0), target_tr, step1_outputs['pred_res_rot'].unsqueeze(0), target_q,
            step1_outputs['pred_res_chi'].unsqueeze(0), target_chi, chi_mask, data_batch.ligand_pos, data_batch.protein_pos
        )
        return total_loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch)
        self.log("train/total_loss", loss, batch_size=batch.num_graphs, sync_dist=True)
        for k, v in loss_dict.items(): self.log(f"train/{k}", v, batch_size=batch.num_graphs, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.forward(batch)
        self.log("val/total_loss", loss, batch_size=batch.num_graphs, sync_dist=True)
        for k, v in loss_dict.items(): self.log(f"val/{k}", v, batch_size=batch.num_graphs, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss"}}