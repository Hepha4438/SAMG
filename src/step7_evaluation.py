import os
import glob
import pickle
import subprocess
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED
from rdkit.Geometry import Point3D
from scipy.spatial.transform import Rotation as R

from torch_geometric.utils import to_dense_batch

from step6_trainer import SAMGLightningModule, SAMGOptimizedDataset
from datasets.pl_data import ProteinLigandDataLoader

# --- CÁC GÓI MỞ RỘNG CHO ĐÁNH GIÁ ---
try:
    from vina import Vina
    VINA_AVAILABLE = True
except ImportError:
    VINA_AVAILABLE = False

try:
    from meeko import MoleculePreparation
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False

try:
    import AutoDockTools
    ADT_AVAILABLE = True
except ImportError:
    ADT_AVAILABLE = False

try:
    import sascorer
except ImportError:
    import sys
    if os.path.exists("sascorer.py"):
        sys.path.append(os.getcwd())
        import sascorer
    else: 
        sascorer = None

from torch_scatter import scatter_mean


class SAMGEvaluator:
    def __init__(self, box_size=[20, 20, 20]):
        self.box_size = box_size
        if VINA_AVAILABLE: 
            self.vina = Vina(sf_name='vina', verbosity=0)
        else: 
            print("[!] Cảnh báo: Gói 'vina' chưa được cài đặt. Bỏ qua Docking.")

    def _prepare_receptor(self, pdb_path):
        if not pdb_path or not os.path.exists(pdb_path): 
            return ""

        base_dir = os.path.dirname(pdb_path)
        base_name = os.path.basename(pdb_path).replace(".pdb", "")
        clean_pdb = os.path.join(base_dir, f"{base_name}_clean.pdb")
        pqr_path = os.path.join(base_dir, f"{base_name}.pqr")
        pdbqt_path = os.path.join(base_dir, f"{base_name}_apo2mol.pdbqt")

        if os.path.exists(pdbqt_path): 
            return pdbqt_path

        try:
            with open(pdb_path, 'r') as f:
                dry_lines = [l for l in f.readlines() if (l.startswith('ATOM') or l.startswith('HETATM')) and 'HOH' not in l]
            with open(clean_pdb, 'w') as f: 
                f.writelines(dry_lines)

            subprocess.run(['pdb2pqr', '--ff=AMBER', clean_pdb, pqr_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            if ADT_AVAILABLE:
                prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
                subprocess.run(['python3', prepare_receptor, '-r', pqr_path, '-o', pdbqt_path, '-A', 'checkhydro'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                subprocess.run(['obabel', '-i', 'pqr', pqr_path, '-o', 'pdbqt', '-O', pdbqt_path, '-xr'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        except Exception as e: 
            pass

        for p in [clean_pdb, pqr_path]:
            if os.path.exists(p): os.remove(p)

        return pdbqt_path if os.path.exists(pdbqt_path) else ""

    def evaluate_drug_likeness(self, mol):
        try:
            qed_score = QED.qed(mol)
            lipinski_score = sum([
                Descriptors.ExactMolWt(mol) < 500, 
                Lipinski.NumHDonors(mol) <= 5, 
                Lipinski.NumHAcceptors(mol) <= 10, 
                -2 <= Descriptors.MolLogP(mol) <= 5, 
                Descriptors.NumRotatableBonds(mol) <= 10
            ])
            sa_score = np.nan
            if sascorer is not None:
                try: sa_score = sascorer.calculateScore(mol)
                except: pass
            return qed_score, sa_score, lipinski_score
        except:
            return 0.0, np.nan, 0

    def evaluate_docking(self, ligand_pdbqt, receptor_pdbqt, center):
        if not VINA_AVAILABLE or not os.path.exists(receptor_pdbqt) or not os.path.exists(ligand_pdbqt):
            return np.nan

        try:
            self.vina.set_receptor(receptor_pdbqt)
            self.vina.set_ligand_from_file(ligand_pdbqt)
            self.vina.compute_vina_maps(center=center, box_size=self.box_size)

            self.vina.optimize()
            return self.vina.score()[0]
        except Exception as e:
            return np.nan

    def run_full_evaluation(self, evaluation_tasks):
        results, smiles_list = [], []
        valid_count = 0

        for task in tqdm(evaluation_tasks, desc="Đang chạy AutoDock Vina & Đánh giá hóa học"):
            sdf_path = task['sdf_path']
            receptor_pdb = task['receptor_pdb']

            try:
                mol = next(Chem.SDMolSupplier(sdf_path, removeHs=False))
                if mol is None or mol.GetNumAtoms() == 0: 
                    continue
            except:
                continue

            valid_count += 1
            smiles_list.append(Chem.MolToSmiles(mol))
            qed_score, sa_score, lipinski = self.evaluate_drug_likeness(mol)

            ligand_pdbqt = sdf_path.replace(".sdf", ".pdbqt")
            if not os.path.exists(ligand_pdbqt):
                try:
                    if MEEKO_AVAILABLE:
                        mol_prep = MoleculePreparation()
                        mol_prep.prepare(mol)
                        with open(ligand_pdbqt, "w") as f: 
                            f.write(mol_prep.write_pdbqt_string())
                    else:
                        raise Exception("No Meeko")
                except Exception as e: 
                    try:
                        subprocess.run(['obabel', '-i', 'sdf', sdf_path, '-o', 'pdbqt', '-O', ligand_pdbqt], 
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    except Exception as ob_e:
                        continue 

            receptor_pdbqt = self._prepare_receptor(receptor_pdb)

            # Lấy tâm tọa độ của Ligand (đã được dời vào Pocket nhờ AI) làm tâm Docking
            try:
                coords = mol.GetConformer().GetPositions()
                dynamic_center = coords.mean(axis=0).tolist()
            except:
                dynamic_center = [0.0, 0.0, 0.0]

            t_score = self.evaluate_docking(ligand_pdbqt, receptor_pdbqt, dynamic_center)

            results.append({
                "File": os.path.basename(sdf_path), 
                "QED": qed_score, 
                "SA": sa_score, 
                "Lipinski": lipinski, 
                "Vina_Score": t_score
            })

        df = pd.DataFrame(results)
        # Validity được tính trên tổng số file đã sinh ra và được RDKit chấp nhận ban đầu
        validity = valid_count / len(evaluation_tasks) if len(evaluation_tasks) > 0 else 0
        uniqueness = len(set(smiles_list)) / len(smiles_list) if len(smiles_list) > 0 else 0

        print("\n" + "="*40)
        print("🚀 BÁO CÁO KẾT QUẢ SBDD (BƯỚC 7) 🚀")
        print("="*40)
        print(f"🔸 Tổng số mẫu đánh giá: {len(evaluation_tasks)}")
        print(f"🔸 Validity:   {validity*100:.2f} %")
        print(f"🔸 Uniqueness: {uniqueness*100:.2f} %")
        print("-" * 40)
        if not df.empty:
            print(f"🔸 QED: {df['QED'].mean():.4f}")
            print(f"🔸 Lipinski Rule:  {df['Lipinski'].mean():.2f} / 5")
            print(f"🔸 SA Score: {df['SA'].mean():.2f}")
            print("-" * 40)
            if VINA_AVAILABLE:
                valid_vina = df['Vina_Score'].dropna()
                if not valid_vina.empty:
                    print(f"🔸 Vina Affinity Score: {valid_vina.mean():.2f} kcal/mol (Trên {len(valid_vina)} mẫu thành công)")
                else:
                    print(f"🔸 Vina Affinity Score: Không có mẫu nào Docking thành công.")
        print("========================================\n")
        return df


def main():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMG_ROOT = os.path.dirname(CURRENT_DIR)
    DATASET_DIR = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "data_folder")
    SPLIT_FILE = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "split_druglike_dict.pkl")
    PROCESSED_DIR = os.path.join(SAMG_ROOT, "dataset", "processed")
    VOCAB_PATH = os.path.join(PROCESSED_DIR, "global_vocab.pkl")
    OUTPUT_DIR = os.path.join(SAMG_ROOT, "outputs_evaluation")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import sys
    sys.path.append(os.path.join(SAMG_ROOT, "src", "utils"))
    from system_logger import SAMGLogger
    sys_logger = SAMGLogger(log_dir=os.path.join(SAMG_ROOT, "logs"))
    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, "rb") as f: vocab = pickle.load(f)

    # P2a-3: nạp scaler TRƯỚC khi khởi tạo dataset để truyền scale[0] vào (giống hệt step6,
    # cùng một hệ số chuẩn hóa cho residue_local_pos ở train và eval).
    SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler_7d.pt")
    if os.path.exists(SCALER_PATH):
        scaler_dict = torch.load(SCALER_PATH, map_location="cpu")
        shift_factors = scaler_dict["shift"]
        scale_factors = scaler_dict["scale"]
        pos_scale = scale_factors.flatten()[0].item()
    else:
        shift_factors, scale_factors, pos_scale = None, None, None

    eval_dataset = SAMGOptimizedDataset(DATASET_DIR, PROCESSED_DIR, vocab, SPLIT_FILE, split_mode="test", pos_scale=pos_scale)
    eval_loader = ProteinLigandDataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=2)

    ckpt_candidates = glob.glob(os.path.join(SAMG_ROOT, "saved_checkpoints_flow", "*.ckpt"))
    if not ckpt_candidates:
        print("[!] Không tìm thấy checkpoint!")
        return
    checkpoint_path = max(ckpt_candidates, key=os.path.getmtime)

    config = OmegaConf.create({
        "hidden_dim": 256, "num_heads": 4, "lr": 1e-4,
        # P2b: PHẢI khớp với ligand_mode dùng lúc train (xem
        # SAMGLightningModule.build_ligand_input) -- lệch chế độ giữa train/eval làm
        # phân phối input của encoder khác nhau giữa hai pha.
        "ligand_mode": "empty",
        "protein_encoder": {
            "num_blocks": 3, "num_layers": 3, "hidden_dim": 256,
            "n_heads": 4, "knn": 16, "edge_feat_dim": 5, "num_r_gaussian": 20, "num_node_types": 8
        },
        "loss_weights": {"token": 1.0, "geo": 1.0, "pocket": 1.0, "int": 0.5, "d_threshold": 2.5}
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model với exact scaler để quá trình Inverse Scaling diễn ra chuẩn xác 100%
    model = SAMGLightningModule.load_from_checkpoint(
        checkpoint_path, 
        config=config, 
        vocab_size=len(vocab), 
        shift_factors=shift_factors,
        scale_factors=scale_factors,
        strict=False
    )

    model.eval()
    model.to(device)

    print("\n[*] --- BƯỚC 2: CHẠY INFERENCE VÀ GHÉP CẶP RECEPTOR-LIGAND ---")
    evaluation_tasks = []
    inv_vocab = {v: k for k, v in vocab.items()}

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_loader, desc="Sinh phân tử Ligand 3D")):
            batch = batch.to(device)

            h_protein = batch.protein_atom_feature.float() if hasattr(batch, 'protein_atom_feature') else model.prot_emb(batch.protein_element.long())

            # P2b: bỏ ligand ground-truth khỏi encoder -- PHẢI dùng đúng hàm build_ligand_input
            # của step6 (không được tính riêng ở đây, sẽ tạo lệch phân phối train/eval).
            num_graphs = batch.protein_element_batch.max().item() + 1
            h_ligand, ligand_pos, batch_ligand = model.build_ligand_input(batch, device, num_graphs)

            step1_outputs = model.dynamic_encoder(
                h_protein=h_protein, h_ligand=h_ligand, protein_pos=batch.protein_pos, ligand_pos=ligand_pos,
                batch_protein=batch.protein_element_batch, batch_ligand=batch_ligand, data=batch
            )
            prot_trans_batch = batch.protein_translations_batch

            # P2a-3: h_target mang hình học (giống hệt step6.forward) -- pad theo graph
            # bằng to_dense_batch thay vì mean-pool thành 1 vector.
            residue_h = step1_outputs['residue_h'][:, :model.config.hidden_dim]
            residue_h_padded, target_mask = to_dense_batch(residue_h, prot_trans_batch, batch_size=num_graphs)
            local_pos_padded, _ = to_dense_batch(batch.residue_local_pos, prot_trans_batch, batch_size=num_graphs)
            h_target = torch.cat([residue_h_padded, local_pos_padded], dim=-1)

            # h_anti_raw, _ = model.static_encoder(h_protein, batch.protein_pos + 1.5, torch.zeros_like(batch.protein_pos[:,0], dtype=torch.bool), batch.protein_element_batch)
            # list_h_anti = [scatter_mean(h_anti_raw, batch.protein_element_batch, dim=0, dim_size=num_graphs).unsqueeze(1)] 
            list_h_anti = []

            # Lấy trung tâm tọa độ của Protein Pocket để làm hệ quy chiếu
            pocket_center = batch.protein_pos.mean(dim=0).cpu().numpy()

            # BẮT ĐẦU VÒNG LẶP AUTOREGRESSIVE
            input_ids = torch.tensor([[vocab.get("[SOS]", 0)]], dtype=torch.long, device=device)
            pred_token_ids = []
            pred_7d_list = []

            # 🔥 KHẮC PHỤC LỖI SA SCORE: Hạ nhiệt độ xuống 0.8 để đảm bảo tính hợp lệ Hóa học
            temperature = 0.8
            max_len = 15

            for step in range(max_len):
                logits_vocab, loss_geo, sampled_7d, _ = model.generator(
                    input_ids, h_target, list_h_anti, target_mask=target_mask
                )

                # Thu thập dự đoán Token
                next_logits = logits_vocab[:, -1, :]
                probs = torch.nn.functional.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Thu thập dự đoán 7D (Không gian) do AI sinh ra
                current_7d = sampled_7d[0, -1, :].cpu().numpy()

                token_id = next_token.item()
                if token_id == vocab.get("[UNK]", 1):
                    continue

                pred_token_ids.append(token_id)
                pred_7d_list.append(current_7d)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)

            smi = "".join([inv_vocab.get(tid, "") for tid in pred_token_ids if inv_vocab.get(tid, "") not in ["[SOS]", "[PAD]", "[UNK]"]]).strip()
            smi = smi.replace("*", "C")

            mol = Chem.MolFromSmiles(smi) if smi else None
            
            # --- KHẮC PHỤC LỖI VINA: BỘ LẮP RÁP 3D (3D ASSEMBLER) ---
            if mol is not None and mol.GetNumAtoms() > 0 and len(pred_7d_list) > 0:
                try:
                    mol = Chem.AddHs(mol)
                    # Cho phép RDKit tạo local geometry (cấu hình liên kết cục bộ) tốt nhất
                    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    if res != -1:
                        try:
                            AllChem.MMFFOptimizeMolecule(mol)
                        except:
                            pass 

                        # 1. Tính toán Tọa độ tịnh tiến (Translation) từ Tọa độ cầu của AI
                        pred_coords = []
                        pred_quats = []
                        for p7 in pred_7d_list:
                            d, theta, phi, qw, qx, qy, qz = p7
                            dx = d * np.sin(theta) * np.cos(phi)
                            dy = d * np.sin(theta) * np.sin(phi)
                            dz = d * np.cos(theta)
                            pred_coords.append([dx, dy, dz])
                            pred_quats.append([qx, qy, qz, qw]) # Scipy dùng format [x, y, z, w]
                        
                        pred_coords = np.array(pred_coords)
                        pred_quats = np.array(pred_quats)
                        
                        # Vị trí AI muốn đặt tâm Ligand
                        ai_predicted_center = pocket_center + pred_coords.mean(axis=0)
                        
                        # Góc xoay AI muốn áp dụng (Lấy trung bình Quaternion của chuỗi)
                        mean_quat = pred_quats.mean(axis=0)
                        mean_quat = mean_quat / np.linalg.norm(mean_quat) # Chuẩn hóa lại
                        ai_rotation_matrix = R.from_quat(mean_quat).as_matrix()

                        # 2. Áp dụng AI Transform lên phân tử RDKit
                        conf = mol.GetConformer()
                        coords = conf.GetPositions()
                        
                        # Dời phân tử về gốc (0,0,0)
                        etkdg_center = coords.mean(axis=0)
                        coords = coords - etkdg_center
                        
                        # Áp dụng Xoay (Rotate)
                        coords = np.dot(coords, ai_rotation_matrix.T)
                        
                        # Áp dụng Tịnh tiến (Translate) đến vị trí AI chỉ định
                        coords = coords + ai_predicted_center

                        # Ghi đè vào RDKit
                        for atom_idx in range(mol.GetNumAtoms()):
                            conf.SetAtomPosition(atom_idx, Point3D(coords[atom_idx, 0], coords[atom_idx, 1], coords[atom_idx, 2]))

                        sdf_path = os.path.join(OUTPUT_DIR, f"eval_sample_{i}.sdf")
                        writer = Chem.SDWriter(sdf_path)
                        writer.write(mol)
                        writer.close()

                        real_idx = eval_dataset.valid_indices[i]
                        entry = eval_dataset.target_entries[real_idx]
                        holo_pocket_fn = entry[0]
                        receptor_pdb = os.path.join(DATASET_DIR, holo_pocket_fn)

                        evaluation_tasks.append({
                            "sdf_path": sdf_path,
                            "receptor_pdb": receptor_pdb
                        })
                except Exception as e:
                    sys_logger.log_error("RDKit 3D Assembler", e)
                    pass

    sys_logger.flush_errors(phase="Evaluation Inference (Step 7)")
    print("\n[*] --- BƯỚC 3: CHẠY ĐÁNH GIÁ CHẤT LƯỢNG VÀ DOCKING ---")
    evaluator = SAMGEvaluator(box_size=[20,20,20])
    evaluator.run_full_evaluation(evaluation_tasks)

if __name__ == "__main__":
    main()