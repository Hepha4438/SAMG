import torch
import torch.nn as nn
import esm

# Importing Apo2Mol core modules (assuming they are placed in SAMG/src/models and SAMG/src/utils)
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral
from utils.data import apply_transforms_tensor_batch

class ESM2FeatureExtractor(nn.Module):
    """
    Extracts deep evolutionary features for protein residues using ESM-2.
    The weights are frozen to save VRAM and computation time.
    """
    def __init__(self, model_name="esm2_t33_650M_UR50D", freeze=True):
        super().__init__()
        print(f"[*] Loading ESM-2 model: {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    @torch.no_grad()
    def forward(self, raw_sequences):
        """
        Args:
            raw_sequences: List of tuples, e.g., [("protein1", "MKTVRQ..."), ...]
        Returns:
            token_representations: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_labels, batch_strs, batch_tokens = self.batch_converter(raw_sequences)
        batch_tokens = batch_tokens.to(next(self.model.parameters()).device)
        
        results = self.model(batch_tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        token_representations = results["representations"][self.model.num_layers]
        
        # Remove the <cls> and <eos> tokens (first and last elements)
        return token_representations[:, 1:-1, :]


class StaticEGNN(nn.Module):
    """
    Static Equivariant Graph Neural Network for Anti-targets.
    Extracts geometric features but freezes coordinate updates (update_x=False) 
    to drastically reduce VRAM usage.
    """
    def __init__(self, num_layers=4, hidden_dim=256, edge_feat_dim=5, num_r_gaussian=20, k=16, cutoff=10.0):
        super().__init__()
        # EGNN initialized with update_x=False to act as a rigid feature extractor
        self.egnn = EGNN(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            edge_feat_dim=edge_feat_dim,
            num_r_gaussian=num_r_gaussian,
            k=k,
            cutoff=cutoff,
            cutoff_mode='knn',
            update_x=False,  # CRITICAL: Do not update coordinates for Anti-targets
            act_fn='silu',
            norm=True
        )

    @torch.no_grad() # Disconnect from the computation graph to save VRAM
    def forward(self, h, x, mask_ligand, batch):
        """
        Returns updated features `h` while `x` (coordinates) remains strictly unchanged.
        """
        outputs = self.egnn(h, x, mask_ligand, batch)
        # Only return the extracted keys (features) and original coordinates
        return outputs['h'], outputs['x']

class DynamicEGNN(nn.Module):
    """
    Dynamic Equivariant Graph Neural Network for the Target pocket.
    Predicts tr, q, chi to simulate Apo-to-Holo conformational changes.
    """
    def __init__(self, config):
        super().__init__()
        # Powerful UniTransformer equipped with GVPLayer and SAGPoolNet
        self.transformer = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn='relu',
            norm=True,
            cutoff_mode='knn',
            ew_net_type='r',
            num_x2h=1, num_h2x=1, r_max=10.0,
            x2h_out_fc=True, sync_twoup=False
        )

        # Inference heads for structural deformation
        self.res_inference = nn.Sequential(
            nn.Linear(config.hidden_dim + 3, config.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_dim, 3 + 4 + 5) # translation(3), quaternion(4), chi_angles(5)
        )

    def forward(self, h_protein, h_ligand, protein_pos, ligand_pos, batch_protein, batch_ligand, data):
            # 1. Forward pass through UniTransformer
            outputs = self.transformer(
                h_protein=h_protein,
                h_ligand=h_ligand,
                protein_pos=protein_pos,
                ligand_pos=ligand_pos,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                protein_atom_to_aa_group=data.protein_atom_to_aa_group,
                fix_x=False  # Allow coordinate updates internally
            )

            residue_h_joint = outputs['residue_h']
            
            # --- ĐIỂM SỬA LỖI Ở ĐÂY ---
            # UniTransformer gộp chung cả Protein và Ligand vào residue_h_joint.
            # Chúng ta cần cắt lấy đúng phần Protein Residue (nằm ở phần đầu của tensor).
            num_residues = data.protein_translations_batch.size(0)
            protein_residue_h = residue_h_joint[:num_residues]
            
            # 2. Predict conformational changes (tr, rot, chi) chỉ cho Protein
            final_res_out = self.res_inference(protein_residue_h)
            pred_res_tr = final_res_out[:, :3]
            
            # Extract and normalize quaternion to avoid Gimbal Lock
            pred_res_rot = final_res_out[:, 3:7]
            pred_res_rot = pred_res_rot / pred_res_rot.norm(dim=-1, keepdim=True).clamp_min(1e-8) 
            
            pred_res_chi = final_res_out[:, 7:]

            # 3. Apply geometric transformations to 3D coordinates
            updated_protein_pos = apply_transforms_tensor_batch(
                protein_pos=protein_pos,
                protein_atom_name=data.protein_atom_name,
                protein_atom_to_aa_name=data.protein_atom_to_aa_name,
                protein_atom_to_aa_group=data.protein_atom_to_aa_group,
                protein_element_batch=batch_protein,
                rotations=pred_res_rot,
                translations=pred_res_tr,
                chi_update=pred_res_chi,
                chi_mask=data.protein_chi_mask,
                protein_translations_batch=data.protein_translations_batch
            )

            return {
                'updated_protein_pos': updated_protein_pos,
                'pred_res_tr': pred_res_tr,
                'pred_res_rot': pred_res_rot,
                'pred_res_chi': pred_res_chi,
                'ligand_h': outputs['ligand_h'],
                'residue_h': protein_residue_h
            }   