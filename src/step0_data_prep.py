import os
import glob
import json
import pickle
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R
from Bio.PDB import PDBParser, Superimposer, PDBIO
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class SAMGDataPreprocessor:
    def __init__(self, dataset_dir, anti_target_dir, split_dict_path, config_path, out_dir):
        self.dataset_dir = dataset_dir
        self.anti_target_dir = anti_target_dir
        self.split_dict_path = split_dict_path
        self.config_path = config_path
        self.out_dir = out_dir
        
        self.parser = PDBParser(QUIET=True)
        self.io = PDBIO()
        
        with open(self.config_path, "r") as f:
            self.system_config = json.load(f)
            
        os.makedirs(self.out_dir, exist_ok=True)

    def add_explicit_hydrogens(self, pdb_file, out_file):
        import shutil
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        try:
            cmd = ["pdb2pqr", "--ff=AMBER", "--titration-state-method=propka", 
                   "--with-ph=7.4", pdb_file, out_file]
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except Exception:
            try:
                shutil.copy2(pdb_file, out_file)
                return False
            except Exception as copy_err:
                raise copy_err

    def align_anti_targets(self, target_apo_path, anti_target_paths, out_prefix):
        target_structure = self.parser.get_structure("target_apo", target_apo_path)
        ref_atoms = [atom for atom in target_structure.get_atoms() if atom.get_name() == "CA"]
        
        aligned_files = []
        super_imposer = Superimposer()
        
        for i, anti_path in enumerate(anti_target_paths):
            anti_structure = self.parser.get_structure(f"anti_{i}", anti_path)
            alt_atoms = [atom for atom in anti_structure.get_atoms() if atom.get_name() == "CA"]
            
            min_len = min(len(ref_atoms), len(alt_atoms))
            if min_len == 0: continue
                
            super_imposer.set_atoms(ref_atoms[:min_len], alt_atoms[:min_len])
            super_imposer.apply(anti_structure.get_models())
            
            aligned_file = os.path.join(self.out_dir, f"{out_prefix}_anti_{i}_aligned.pdb")
            self.io.set_structure(anti_structure)
            self.io.save(aligned_file)
            aligned_files.append(aligned_file)
            
        return aligned_files

    def _get_local_frame(self, points):
        if len(points) < 3: return np.eye(3), points[0] if len(points) > 0 else np.zeros(3)
        p1 = points[0]
        for i in range(1, len(points)):
            for j in range(i + 1, len(points)):
                p2, p3 = points[i], points[j]
                v1, v2 = p2 - p1, p3 - p1
                cross_prod = np.cross(v1, v2)
                if np.linalg.norm(cross_prod) > 1e-4:
                    x_axis = v1 / np.linalg.norm(v1)
                    y_axis = cross_prod / np.linalg.norm(cross_prod)
                    z_axis = np.cross(x_axis, y_axis)
                    return np.column_stack((x_axis, y_axis, z_axis)), p1
        return np.eye(3), points[0]

    def _get_spherical_coords(self, vector):
        d = np.linalg.norm(vector)
        if d < 1e-6: return 0.0, 0.0, 0.0
        theta = np.arccos(np.clip(vector[2] / d, -1.0, 1.0))
        phi = np.arctan2(vector[1], vector[0])
        return float(d), float(theta), float(phi)

    def fragment_ligand_frag2seq(self, ligand_path):
        supplier = Chem.SDMolSupplier(ligand_path)
        if len(supplier) == 0: return None
        mol = supplier[0]
        if mol is None: return None
            
        bonds_to_break = [bond.GetIdx() for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing() and bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1]
        if not bonds_to_break: return None
            
        fragmented_mol = Chem.FragmentOnBonds(mol, bonds_to_break)
        frags = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False)
        
        frag_data = []
        for frag in frags:
            canonical_smiles = Chem.MolToSmiles(frag, isomericSmiles=True, canonical=True)
            conf = frag.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(frag.GetNumAtoms())])
            frag_data.append({"smiles": canonical_smiles, "coords": coords, "center": np.mean(coords, axis=0)})
            
        frag_data.sort(key=lambda x: x["smiles"])
        centers = np.array([f["center"] for f in frag_data])
        R_m_to_w, t_m_to_w = self._get_local_frame(centers)
        
        sequence_7d = []
        for f in frag_data:
            rel_center = np.dot(R_m_to_w.T, (f["center"] - t_m_to_w))
            d, theta, phi = self._get_spherical_coords(rel_center)
            R_g_to_w, t_g_to_w = self._get_local_frame(f["coords"])
            R_g_to_m = np.dot(R_m_to_w.T, R_g_to_w)
            qx, qy, qz, qw = R.from_matrix(R_g_to_m).as_quat() 
            sequence_7d.append({"smiles": f["smiles"], "spatial_tokens": [d, theta, phi, float(qw), float(qx), float(qy), float(qz)]})
        return sequence_7d

    def process_system(self, system_name, scenario_type, entry, verbose=False):
        holo_pocket_rel, apo_pocket_rel, ligand_rel, _, _, _, _ = entry
        pli_id = apo_pocket_rel.split("/")[0]
        
        target_apo_path = os.path.join(self.dataset_dir, apo_pocket_rel)
        target_holo_path = os.path.join(self.dataset_dir, holo_pocket_rel)
        ligand_path = os.path.join(self.dataset_dir, ligand_rel)
        
        scenario_config = self.system_config.get(system_name, {}).get("scenarios", {}).get(scenario_type, {})
        anti_target_paths = [os.path.join(self.anti_target_dir, f"{at['pdb_id']}.pdb") for at in scenario_config.get("anti_targets", []) if os.path.exists(os.path.join(self.anti_target_dir, f"{at['pdb_id']}.pdb"))]
                
        apo_h_path = os.path.join(self.out_dir, f"{pli_id}_apo_H.pdb")
        holo_h_path = os.path.join(self.out_dir, f"{pli_id}_holo_H.pdb")
        
        self.add_explicit_hydrogens(target_apo_path, apo_h_path)
        self.add_explicit_hydrogens(target_holo_path, holo_h_path)
        
        if anti_target_paths:
            self.align_anti_targets(apo_h_path, anti_target_paths, out_prefix=pli_id)
        
        sequence_7d = self.fragment_ligand_frag2seq(ligand_path)
        if sequence_7d:
            with open(os.path.join(self.out_dir, f"{pli_id}_sequence_7d.pkl"), 'wb') as f:
                pickle.dump(sequence_7d, f)

    def build_global_vocab(self):
        print("\n[*] Building Global Vocabulary from all processed 7D sequences...")
        vocab = {"[SOS]": 0, "[UNK]": 1}
        sequence_files = glob.glob(os.path.join(self.out_dir, "*_sequence_7d.pkl"))
        if not sequence_files: return None
            
        for file_path in tqdm(sequence_files, desc="Building Vocab", unit="file"):
            with open(file_path, "rb") as f:
                for item in pickle.load(f):
                    if item["smiles"] not in vocab: vocab[item["smiles"]] = len(vocab)
                        
        with open(os.path.join(self.out_dir, "global_vocab.pkl"), "wb") as f: pickle.dump(vocab, f)
        print(f"[v] Global vocabulary saved. Total Unique Tokens: {len(vocab)}")
        return vocab


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/Apo2Mol_Dataset/data_folder"))
    ANTI_TARGET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/AntiTargets_PDB"))
    SPLIT_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/Apo2Mol_Dataset/split_druglike_dict.pkl"))
    CONFIG_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/system_config.json"))
    OUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/processed"))
    
    processor = SAMGDataPreprocessor(DATASET_DIR, ANTI_TARGET_DIR, SPLIT_FILE, CONFIG_FILE, OUT_DIR)
    
    with open(SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
        
    num_workers = min(multiprocessing.cpu_count(), 8)
    
    def worker_process(entry):
        try:
            if "HSP90_System" in processor.system_config:
                processor.process_system("HSP90_System", "easy", entry, verbose=False)
            elif "JNK_System" in processor.system_config:
                processor.process_system("JNK_System", "hard", entry, verbose=False)
            else:
                processor.process_system("HSP90_System", "easy", entry, verbose=False)
        except Exception: pass

    for split_name, subset in splits.items():
        print(f"\n[*] ===========================================")
        print(f"[*] TIỀN XỬ LÝ TẬP DỮ LIỆU: {split_name.upper()}")
        print(f"[*] Tìm thấy {len(subset)} mẫu trong tập này.")
        print(f"[*] ===========================================")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(worker_process, subset), total=len(subset), desc=f"Processing {split_name}"))
            
    processor.build_global_vocab()