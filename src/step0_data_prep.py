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

    def _compute_pocket_axes(self, pocket_coords, m3_thresh, ref_thresh, ref_vec):
        """Mot lan tinh frame (khong jitter): tra ve (R, t, decisive)."""
        t = pocket_coords.mean(0)
        C = pocket_coords - t

        cov = (C.T @ C) / len(C)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        decisive = True
        axes = []
        for k in range(3):
            v_k = eigvecs[:, k]
            proj = C @ v_k
            std = proj.std()
            m3 = (proj ** 3).sum()
            m3n = m3 / (len(C) * (std ** 3) + 1e-12)
            ref_ok = ref_vec is not None and abs(np.dot(v_k, ref_vec)) >= ref_thresh
            if abs(m3n) >= m3_thresh:
                if m3n < 0:
                    v_k = -v_k
            elif ref_ok:
                if np.dot(v_k, ref_vec) < 0:
                    v_k = -v_k
            else:
                decisive = False
            axes.append(v_k)

        v1, v2 = axes[0], axes[1]
        v3 = np.cross(v1, v2)
        R_mat = np.column_stack((v1, v2, v3))
        return R_mat, t, decisive

    def _get_pocket_frame(self, pocket_coords, m3_thresh=0.05, ref_thresh=0.05,
                          ref_vec=None, n_jitter=5, jitter_sigma=0.3, jitter_max_deg=5.0):
        """
        Tra ve (R, t, frame_stable, n_jitter_fail). R: cot la 3 truc canonical, t: tam hoc.

        frame_stable -- DINH NGHIA REV 2 (plan.md, PHASE 1 / P1b):
          a) dau dut khoat: voi TUNG truc, |m3n| >= m3_thresh HOAC |v_k . ref| >= ref_thresh.
             Neu mot truc khong dat ca hai dieu kien nay thi khong on dinh.
          b) phep thu jitter: n_jitter lan cong nhieu Gaussian sigma=jitter_sigma A vao
             pocket_coords, tinh lai frame voi cung quy uoc dau; goc lech toi da giua frame
             goc va cac frame nhieu phai < jitter_max_deg.
          frame_stable = a) AND b). "Dung tang 2" (fallback bang ref_vec hoa hoc) KHONG con
          la ly do de coi la bat on dinh -- day la fallback tat dinh, bat bien SE(3); jitter
          da bao phu truc tiep tinh lien tuc cua frame nen KHONG con dieu kien khe tri rieng.
        """
        pocket_coords = np.asarray(pocket_coords, dtype=np.float64)

        if ref_vec is not None:
            ref_norm = np.linalg.norm(ref_vec)
            ref_vec = ref_vec / ref_norm if ref_norm > 1e-8 else None

        R_mat, t, decisive = self._compute_pocket_axes(pocket_coords, m3_thresh, ref_thresh, ref_vec)

        seed = int(pocket_coords.shape[0])
        rng = np.random.RandomState(seed)
        n_jitter_fail = 0
        for _ in range(n_jitter):
            noisy_coords = pocket_coords + rng.normal(scale=jitter_sigma, size=pocket_coords.shape)
            R_noisy, _, _ = self._compute_pocket_axes(noisy_coords, m3_thresh, ref_thresh, ref_vec)
            dev = float(np.degrees(np.arccos(np.clip((np.trace(R_mat.T @ R_noisy) - 1) / 2, -1.0, 1.0))))
            if dev >= jitter_max_deg:
                n_jitter_fail += 1

        frame_stable = decisive and (n_jitter_fail == 0)

        return R_mat, t, frame_stable, n_jitter_fail

    def _load_pocket_geometry(self, pocket_path):
        """Doc pocket PDB (bo HOH). Tra ve (pocket_coords, ref_vec) voi
        ref_vec = centroid(N khung) - centroid(O khung), da chuan hoa; None neu khong tinh duoc."""
        structure = self.parser.get_structure("pocket", pocket_path)
        coords, n_coords, o_coords = [], [], []
        for residue in structure.get_residues():
            if residue.get_resname() == "HOH":
                continue
            for atom in residue:
                coords.append(atom.get_coord())
                if atom.get_name() == "N":
                    n_coords.append(atom.get_coord())
                elif atom.get_name() == "O":
                    o_coords.append(atom.get_coord())

        pocket_coords = np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))
        if n_coords and o_coords:
            ref_vec = np.mean(n_coords, axis=0) - np.mean(o_coords, axis=0)
            norm = np.linalg.norm(ref_vec)
            ref_vec = ref_vec / norm if norm > 1e-8 else None
        else:
            ref_vec = None
        return pocket_coords, ref_vec

    def _get_spherical_coords(self, vector):
        d = np.linalg.norm(vector)
        if d < 1e-6: return 0.0, 0.0, 0.0
        theta = np.arccos(np.clip(vector[2] / d, -1.0, 1.0))
        phi = np.arctan2(vector[1], vector[0])
        return float(d), float(theta), float(phi)

    def fragment_ligand_frag2seq(self, ligand_path, apo_pocket_path):
        """
        Tra ve dict {"frame_stable": bool, "n_jitter_fail": int, "sequence": [...]}.
        LUU Y: dinh dang pkl da doi tu Rev 4 (P1c) -- truoc day ham nay tra ve TRUC TIEP
        list "sequence"; gio bao thanh dict de mang theo do on dinh cua frame hoc (frame_stable,
        n_jitter_fail tu `_get_pocket_frame`) o cap FILE, khong phai cap token.
        """
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
            try:
                ranks = list(Chem.CanonicalRankAtoms(frag, breakTies=True))
                order = np.argsort(ranks)
                coords_canon = coords[order]
                frag_frame_stable = True
            except Exception:
                coords_canon = coords
                frag_frame_stable = False
            frag_data.append({"smiles": canonical_smiles, "coords": coords, "coords_canon": coords_canon, "center": np.mean(coords, axis=0), "frag_frame_stable": frag_frame_stable})

        frag_data.sort(key=lambda x: x["smiles"])

        pocket_coords, ref_vec = self._load_pocket_geometry(apo_pocket_path)
        if ref_vec is not None and len(pocket_coords) >= 3:
            R_m_to_w, t_m_to_w, frame_stable, n_jitter_fail = self._get_pocket_frame(pocket_coords, ref_vec=ref_vec)
        else:
            R_m_to_w = np.eye(3)
            t_m_to_w = pocket_coords.mean(0) if len(pocket_coords) else np.zeros(3)
            frame_stable, n_jitter_fail = False, 0

        sequence = []
        for f in frag_data:
            rel_center = np.dot(R_m_to_w.T, (f["center"] - t_m_to_w))
            d, theta, phi = self._get_spherical_coords(rel_center)
            R_g_to_w, t_g_to_w = self._get_local_frame(f["coords_canon"])
            R_g_to_m = np.dot(R_m_to_w.T, R_g_to_w)
            qx, qy, qz, qw = R.from_matrix(R_g_to_m).as_quat()
            sequence.append({"smiles": f["smiles"], "spatial_tokens": [d, theta, phi, float(qw), float(qx), float(qy), float(qz)], "frag_frame_stable": f["frag_frame_stable"]})
        return {"frame_stable": frame_stable, "n_jitter_fail": n_jitter_fail, "sequence": sequence}

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
        
        sequence_7d = self.fragment_ligand_frag2seq(ligand_path, target_apo_path)
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
                # Dinh dang pkl da doi (P1c): dict {"frame_stable", "n_jitter_fail", "sequence"}
                for item in pickle.load(f)["sequence"]:
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