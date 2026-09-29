import os
import glob
import json
import pickle
import subprocess
import contextlib
import numpy as np
from scipy.spatial.transform import Rotation as R
from Bio.PDB import PDBParser, Superimposer, PDBIO
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

RDLogger.DisableLog('rdApp.*')


@contextlib.contextmanager
def silence_stderr():
    saved = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(dn)
        os.close(saved)


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

    def _get_pocket_frame(self, pocket_coords, atom_names, tau=0.30):
        """Frame canonical cua hoc. Tra ve (R, t, stable, diag).
        R: cot la 3 truc. t: tam hoc. Khong PCA, khong tri rieng, khong quy uoc dau.

        BAN CUOI (plan.md, PHASE 1 / P1b) -- bo hoan toan PCA + momen bac ba + tang 2 +
        jitter cua cac ban sua truoc: co che hai tang tao ra mot diem gian doan o nguong,
        khien cau truc gan nguong bi lat 180 deg giua hai lan chay. Gram-Schmidt tu hai
        huong hoa hoc (N->O va sidechain->backbone) khong co nhanh re, nen tat dinh va
        lien tuc theo toa do nguyen tu (tru dung mot cho ref1 || ref2, buoc 4 kiem truc tiep).

        1. BB = {N, CA, C, O}; bb = nguyen tu co ten trong BB; sc = cac nguyen tu con lai.
        2. ref1 = centroid(ten==N) - centroid(ten==O); n1 = |ref1|.
           Neu thieu N hoac O, hoac n1 < tau => stable=False.
        3. e1 = ref1 / n1.
        4. ref2 = centroid(sc) - centroid(bb); ref2p = ref2 - (ref2.e1)e1; n2 = |ref2p|.
           Neu thieu sc/bb, hoac n2 < tau (hai huong gan song song) => stable=False.
        5. e2 = ref2p / n2; e3 = cross(e1, e2); R = stack([e1, e2, e3], axis=1).
        6. t = pocket_coords.mean(0).
        7. diag = {"n1", "n2", "missing"} de theo doi ty le loai theo nguyen nhan.
        """
        pocket_coords = np.asarray(pocket_coords, dtype=np.float64)
        atom_names = np.asarray(atom_names)
        t = pocket_coords.mean(0)

        BB = {"N", "CA", "C", "O"}
        bb_mask = np.isin(atom_names, list(BB))
        sc_mask = ~bb_mask
        n_mask = atom_names == "N"
        o_mask = atom_names == "O"

        stable = True
        missing = []
        n1 = n2 = None

        if n_mask.any() and o_mask.any():
            ref1 = pocket_coords[n_mask].mean(0) - pocket_coords[o_mask].mean(0)
            n1 = float(np.linalg.norm(ref1))
            if n1 < tau:
                stable = False
            e1 = ref1 / n1 if n1 > 1e-12 else np.array([1.0, 0.0, 0.0])
        else:
            missing.append("N/O")
            stable = False
            e1 = np.array([1.0, 0.0, 0.0])

        if sc_mask.any() and bb_mask.any():
            ref2 = pocket_coords[sc_mask].mean(0) - pocket_coords[bb_mask].mean(0)
            ref2p = ref2 - np.dot(ref2, e1) * e1
            n2 = float(np.linalg.norm(ref2p))
            if n2 < tau:
                stable = False
            e2 = ref2p / n2 if n2 > 1e-12 else np.array([0.0, 1.0, 0.0])
        else:
            missing.append("sidechain/backbone")
            stable = False
            e2 = np.array([0.0, 1.0, 0.0])

        e3 = np.cross(e1, e2)
        R_mat = np.column_stack((e1, e2, e3))

        diag = {"n1": n1, "n2": n2, "missing": ", ".join(missing)}
        return R_mat, t, stable, diag

    def _load_pocket_geometry(self, pocket_path):
        """Doc pocket PDB (bo HOH). Tra ve (pocket_coords, atom_names) cho _get_pocket_frame."""
        structure = self.parser.get_structure("pocket", pocket_path)
        coords, names = [], []
        for residue in structure.get_residues():
            if residue.get_resname() == "HOH":
                continue
            for atom in residue:
                coords.append(atom.get_coord())
                names.append(atom.get_name())

        pocket_coords = np.array(coords, dtype=np.float64) if coords else np.zeros((0, 3))
        atom_names = np.array(names) if names else np.array([], dtype=object)
        return pocket_coords, atom_names

    def _get_spherical_coords(self, vector):
        d = np.linalg.norm(vector)
        if d < 1e-6: return 0.0, 0.0, 0.0
        theta = np.arccos(np.clip(vector[2] / d, -1.0, 1.0))
        phi = np.arctan2(vector[1], vector[0])
        return float(d), float(theta), float(phi)

    def fragment_ligand_frag2seq(self, ligand_path, apo_pocket_path):
        """
        Tra ve dict {"frame_stable": bool, "frame_diag": dict, "sequence": [...]}.
        LUU Y: dinh dang pkl da doi tu Rev 4 (P1c) -- truoc day ham nay tra ve TRUC TIEP
        list "sequence"; gio bao thanh dict de mang theo do on dinh cua frame hoc (frame_stable,
        frame_diag tu `_get_pocket_frame`) o cap FILE, khong phai cap token.
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

        pocket_coords, atom_names = self._load_pocket_geometry(apo_pocket_path)
        R_m_to_w, t_m_to_w, frame_stable, frame_diag = self._get_pocket_frame(pocket_coords, atom_names)

        sequence = []
        for f in frag_data:
            rel_center = np.dot(R_m_to_w.T, (f["center"] - t_m_to_w))
            d, theta, phi = self._get_spherical_coords(rel_center)
            R_g_to_w, t_g_to_w = self._get_local_frame(f["coords_canon"])
            R_g_to_m = np.dot(R_m_to_w.T, R_g_to_w)
            qx, qy, qz, qw = R.from_matrix(R_g_to_m).as_quat()
            sequence.append({"smiles": f["smiles"], "spatial_tokens": [d, theta, phi, float(qw), float(qx), float(qy), float(qz)], "frag_frame_stable": f["frag_frame_stable"]})
        return {"frame_stable": frame_stable, "frame_diag": frame_diag, "sequence": sequence}

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
            return sequence_7d["frame_stable"], sequence_7d["frame_diag"]
        return None, None

    def build_global_vocab(self):
        print("\n[*] Building Global Vocabulary from all processed 7D sequences...")
        vocab = {"[SOS]": 0, "[UNK]": 1}
        sequence_files = glob.glob(os.path.join(self.out_dir, "*_sequence_7d.pkl"))
        if not sequence_files: return None
            
        for file_path in tqdm(sequence_files, desc="Building Vocab", unit="file"):
            with open(file_path, "rb") as f:
                # Dinh dang pkl da doi (P1c): dict {"frame_stable", "frame_diag", "sequence"}
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
                return processor.process_system("HSP90_System", "easy", entry, verbose=False)
            elif "JNK_System" in processor.system_config:
                return processor.process_system("JNK_System", "hard", entry, verbose=False)
            else:
                return processor.process_system("HSP90_System", "easy", entry, verbose=False)
        except Exception:
            return None, None

    all_frame_results = []
    for split_name, subset in splits.items():
        print(f"\n[*] ===========================================")
        print(f"[*] TIỀN XỬ LÝ TẬP DỮ LIỆU: {split_name.upper()}")
        print(f"[*] Tìm thấy {len(subset)} mẫu trong tập này.")
        print(f"[*] ===========================================")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(worker_process, subset), total=len(subset), desc=f"Processing {split_name}"))
        all_frame_results.extend(results)

    processor.build_global_vocab()

    # P1c: ty le complex co frame_stable=False, tach theo nguyen nhan (n1<tau / n2<tau / thieu nhom nguyen tu)
    valid_results = [(fs, diag) for fs, diag in all_frame_results if fs is not None]
    n_total = len(valid_results)
    if n_total > 0:
        tau = 0.30
        n_unstable = sum(1 for fs, _ in valid_results if not fs)
        n_n1 = sum(1 for fs, diag in valid_results if not fs and diag["n1"] is not None and diag["n1"] < tau)
        n_n2 = sum(1 for fs, diag in valid_results if not fs and diag["n2"] is not None and diag["n2"] < tau)
        n_missing = sum(1 for fs, diag in valid_results if not fs and diag["missing"])
        print(f"\n[*] frame_stable=False: {n_unstable}/{n_total} ({n_unstable / n_total * 100:.2f}%)")
        print(f"    n1 < tau             : {n_n1}")
        print(f"    n2 < tau             : {n_n2}")
        print(f"    thieu nhom nguyen tu : {n_missing}")