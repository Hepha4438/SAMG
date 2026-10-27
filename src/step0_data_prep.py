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

    @staticmethod
    def _get_pocket_frame(pocket_coords, atom_names, tau=0.30):
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

    @staticmethod
    def _load_pocket_geometry(pocket_path):
        """Doc pocket PDB (bo HOH). Tra ve (pocket_coords, atom_names) cho _get_pocket_frame.

        @staticmethod (P2a-1): de import/goi duoc tu ngoai (vd. add_pocket_frame_to_pkl.py)
        ma khong can khoi tao ca SAMGDataPreprocessor. Dung PDBParser rieng thay vi self.parser
        -- cung cau hinh QUIET=True nhu __init__, khong doi hanh vi/gia tri tra ve."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pocket", pocket_path)
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
            
        # Muc 37.1/37.4: KHONG truyen dummyLabels thi RDKit gan isotope cua nguyen tu dummy =
        # CHI SO NGUYEN TU cua dau kia lien ket bi cat ⇒ cung mot fragment hoa hoc o hai phan tu
        # khac nhau cho HAI token khac nhau. Do duoc: U/V = 0,2803 (19.031 token cho 5.334 fragment
        # hoa hoc; rieng '[*]C[*]' co 672 bien the). dummyLabels=(0,0) cho moi lien ket ⇒ moi dummy
        # thanh [*] ⇒ token la SMILES HOA HOC thuan. Lien ket giua cac fragment KHONG can nhan nay:
        # Step 7 suy lai tu hinh hoc qua reconstruct_from_generated (Apo2Mol utils/reconstruct.py,
        # Muc 37.3), dung chuan cua nganh.
        fragmented_mol = Chem.FragmentOnBonds(mol, bonds_to_break,
                                               dummyLabels=[(0, 0)] * len(bonds_to_break))
        frags = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False)

        # N1 (Muc 37-pre.6, 37.4): thu tu fragment theo Frag2Seq muc 3.2/3.3.1 -- "sort the k fragments
        # based on the canonical order L(M), specifically by the order of a fragment's first-ranked atom",
        # tuc theo THU TU XUAT HIEN TRONG CANONICAL SMILES CUA CA PHAN TU. Truoc Rev 20 la
        # frag_data.sort(key=smiles) (bang chu cai) -- khong co nguon, va khong phan anh lien ket hoa hoc
        # doc chuoi. FragmentOnBonds noi them cac nguyen tu dummy VAO CUOI, nen atom idx < n_orig anh xa
        # truc tiep sang chi so nguyen tu cua phan tu goc.
        n_orig = mol.GetNumAtoms()
        frag_atom_idx = Chem.GetMolFrags(fragmented_mol, asMols=False)
        try:
            orig_ranks = list(Chem.CanonicalRankAtoms(mol))
            frag_sort_keys = []
            for idxs in frag_atom_idx:
                real = [orig_ranks[i] for i in idxs if i < n_orig]
                frag_sort_keys.append(min(real) if real else 10 ** 9)
            frag_order_mode = "frag2seq_canonical_appearance"
        except Exception:
            frag_sort_keys = [None] * len(frags)
            frag_order_mode = "fallback_smiles_alphabetical"

        frag_data = []
        for frag_i, frag in enumerate(frags):
            canonical_smiles = Chem.MolToSmiles(frag, isomericSmiles=True, canonical=True)
            conf = frag.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(frag.GetNumAtoms())])
            g = Chem.Mol(frag)
            with silence_stderr():
                try:
                    g.UpdatePropertyCache(strict=False)
                    Chem.FastFindRings(g)
                    ranks = list(Chem.CanonicalRankAtoms(g, breakTies=True))
                    order = np.argsort(ranks)
                    coords_canon = coords[order]
                    frag_frame_stable = True
                except Exception:
                    order = np.arange(frag.GetNumAtoms())
                    coords_canon = coords
                    frag_frame_stable = False
            # Muc 37-pre.3: so hieu nguyen tu theo CUNG hoan vi canonical voi coords_canon. Dummy co
            # atomic_num = 0 -- giu lai de chi so khop coords_canon; Step 7 bo chung truoc khi
            # reconstruct_from_generated.
            atomic_nums = np.array([frag.GetAtomWithIdx(int(i)).GetAtomicNum() for i in order], dtype=np.int16)
            frag_data.append({"smiles": canonical_smiles, "coords": coords, "coords_canon": coords_canon,
                              "atomic_nums": atomic_nums, "center": np.mean(coords, axis=0),
                              "frag_frame_stable": frag_frame_stable,
                              "sort_key": frag_sort_keys[frag_i]})

        # N1: sap theo thu tu xuat hien trong canonical SMILES cua ca phan tu; smiles la khoa phu de
        # tat dinh khi hai fragment co cung first-ranked atom (khong xay ra voi CanonicalRankAtoms
        # breakTies mac dinh, nhung giu cho chac).
        if frag_order_mode == "frag2seq_canonical_appearance":
            frag_data.sort(key=lambda x: (x["sort_key"], x["smiles"]))
        else:
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
            # Muc 37-pre.3: toa do nguyen tu trong FRAME FRAGMENT. Day la mat xich con thieu de tai
            # dung phan tu 3D ma KHONG can ETKDG (Frag2Seq muc 3.3.1). Truoc Rev 20 bi tinh roi vut.
            # Nghich dao o Step 7: atom_world = R_m_to_w @ R_g_to_m @ (local - local.mean(0)) + center_world
            local_coords = np.dot(R_g_to_w.T, (f["coords_canon"] - t_g_to_w).T).T
            sequence.append({"smiles": f["smiles"],
                             "spatial_tokens": [d, theta, phi, float(qw), float(qx), float(qy), float(qz)],
                             "frag_frame_stable": f["frag_frame_stable"],
                             "local_coords": local_coords.astype(np.float32),
                             "atomic_nums": f["atomic_nums"]})
        return {"frame_stable": frame_stable, "frame_diag": frame_diag, "sequence": sequence,
                "frag_order_mode": frag_order_mode}

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
            frag_flags = [item["frag_frame_stable"] for item in sequence_7d["sequence"]]
            return sequence_7d["frame_stable"], sequence_7d["frame_diag"], frag_flags
        return None, None, None

    def build_global_vocab(self, train_pli_ids=None):
        """Rev 20, hai sua doi co can cu:
        (1) `sorted(glob)` -- truoc day `glob.glob` tra ve theo thu tu he thong tep ⇒ token id KHAC
            nhau giua cac lan chay/may, tuc tu vung KHONG tai lap duoc. Cung nguyen tac voi P3a-2
            (seed) va R2: hai run khong so sanh duoc neu dinh danh token khong tat dinh.
        (2) `train_pli_ids` -- chi dung fragment cua tap TRAIN. Nguon: Frag2Seq muc 3.3.1, "we save
            each splited fragment FROM THE TRAINING SET into a dictionary". Truoc day tu vung dung ca
            valid/test ⇒ khong gian dau ra cua mo hinh duoc dinh hinh boi du lieu danh gia.
            Fragment chi co o valid/test se roi vao [UNK] -- dung nhu khi trien khai that.
        """
        print("\n[*] Building Global Vocabulary (sorted, train-only) ...")
        vocab = {"[SOS]": 0, "[UNK]": 1}
        sequence_files = sorted(glob.glob(os.path.join(self.out_dir, "*_sequence_7d.pkl")))
        if train_pli_ids is not None:
            keep = set(train_pli_ids)
            sequence_files = [f for f in sequence_files
                              if os.path.basename(f)[: -len("_sequence_7d.pkl")] in keep]
            print(f"[*] chi dung {len(sequence_files)} pkl cua tap TRAIN de dung tu vung.")
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
            return None, None, None

    all_frame_results = []
    for split_name, subset in splits.items():
        print(f"\n[*] ===========================================")
        print(f"[*] TIỀN XỬ LÝ TẬP DỮ LIỆU: {split_name.upper()}")
        print(f"[*] Tìm thấy {len(subset)} mẫu trong tập này.")
        print(f"[*] ===========================================")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(worker_process, subset), total=len(subset), desc=f"Processing {split_name}"))
        all_frame_results.extend(results)

    # Rev 20: tu vung chi tu tap TRAIN, va duyet tep theo thu tu da sap (xem docstring build_global_vocab).
    train_pli_ids = [e[2].split("/")[0] for e in splits.get("train", [])]
    processor.build_global_vocab(train_pli_ids=train_pli_ids)

    # P1c: ty le complex co frame_stable=False, tach theo nguyen nhan (n1<tau / n2<tau / thieu nhom nguyen tu)
    valid_results = [(fs, diag, frags) for fs, diag, frags in all_frame_results if fs is not None]
    n_total = len(valid_results)
    if n_total > 0:
        tau = 0.30
        n_unstable = sum(1 for fs, _, _ in valid_results if not fs)
        n_n1 = sum(1 for fs, diag, _ in valid_results if not fs and diag["n1"] is not None and diag["n1"] < tau)
        n_n2 = sum(1 for fs, diag, _ in valid_results if not fs and diag["n2"] is not None and diag["n2"] < tau)
        n_missing = sum(1 for fs, diag, _ in valid_results if not fs and diag["missing"])
        print(f"\n[*] frame_stable=False: {n_unstable}/{n_total} ({n_unstable / n_total * 100:.2f}%)")
        print(f"    n1 < tau             : {n_n1}")
        print(f"    n2 < tau             : {n_n2}")
        print(f"    thieu nhom nguyen tu : {n_missing}")

        # P1a: ty le fragment/complex co frag_frame_stable=False
        all_frag_flags = [flag for _, _, frags in valid_results for flag in frags]
        n_frag_total = len(all_frag_flags)
        n_frag_unstable = sum(1 for flag in all_frag_flags if not flag)
        n_complex_with_unstable_frag = sum(1 for _, _, frags in valid_results if any(not f for f in frags))
        if n_frag_total > 0:
            print(f"\n[*] frag_frame_stable=False: {n_frag_unstable}/{n_frag_total} fragment "
                  f"({n_frag_unstable / n_frag_total * 100:.2f}%)")
            print(f"    complex co it nhat 1 fragment nhu vay: {n_complex_with_unstable_frag}/{n_total} "
                  f"({n_complex_with_unstable_frag / n_total * 100:.2f}%)")