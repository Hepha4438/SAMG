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

class SAMGDataPreprocessor:
    def __init__(self, dataset_dir, anti_target_dir, split_dict_path, config_path, out_dir):
        """
        Initialize the Step 0 Preprocessing Pipeline for SAMG.
        """
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

    def load_split(self, split_name="train"):
        print(f"[*] Loading '{split_name}' split from pickle file...")
        with open(self.split_dict_path, "rb") as f:
            splits = pickle.load(f)
        return splits[split_name]

    def add_explicit_hydrogens(self, pdb_file, out_file):
        """
        Add explicit hydrogens at pH 7.4 using PDB2PQR via subprocess.
        """
        import shutil
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        try:
            # Hide PDB2PQR output to keep tqdm clean
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
        """
        Align M Anti-target pockets onto the Target Apo reference frame.
        """
        target_structure = self.parser.get_structure("target_apo", target_apo_path)
        ref_atoms = [atom for atom in target_structure.get_atoms() if atom.get_name() == "CA"]
        
        aligned_files = []
        super_imposer = Superimposer()
        
        for i, anti_path in enumerate(anti_target_paths):
            anti_structure = self.parser.get_structure(f"anti_{i}", anti_path)
            alt_atoms = [atom for atom in anti_structure.get_atoms() if atom.get_name() == "CA"]
            
            min_len = min(len(ref_atoms), len(alt_atoms))
            if min_len == 0:
                continue
                
            super_imposer.set_atoms(ref_atoms[:min_len], alt_atoms[:min_len])
            super_imposer.apply(anti_structure.get_models())
            
            aligned_file = os.path.join(self.out_dir, f"{out_prefix}_anti_{i}_aligned.pdb")
            self.io.set_structure(anti_structure)
            self.io.save(aligned_file)
            aligned_files.append(aligned_file)
            
        return aligned_files

    def _get_local_frame(self, points):
        """
        Construct an SE(3)-equivariant local frame (x, y, z) given a set of 3D points.
        Uses the first three non-collinear points.
        Returns the Rotation matrix (3x3) and Translation vector (3).
        """
        if len(points) < 3:
            return np.eye(3), points[0] if len(points) > 0 else np.zeros(3)

        p1 = points[0]
        for i in range(1, len(points)):
            for j in range(i + 1, len(points)):
                p2, p3 = points[i], points[j]
                v1 = p2 - p1
                v2 = p3 - p1
                cross_prod = np.cross(v1, v2)
                
                # Check for non-collinearity
                if np.linalg.norm(cross_prod) > 1e-4:
                    x_axis = v1 / np.linalg.norm(v1)
                    y_axis = cross_prod / np.linalg.norm(cross_prod)
                    z_axis = np.cross(x_axis, y_axis)
                    
                    rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
                    translation_vector = p1
                    return rotation_matrix, translation_vector
                    
        return np.eye(3), points[0]

    def _get_spherical_coords(self, vector):
        """
        Convert a 3D Cartesian vector into Spherical coordinates (d, theta, phi).
        """
        d = np.linalg.norm(vector)
        if d < 1e-6:
            return 0.0, 0.0, 0.0
            
        theta = np.arccos(np.clip(vector[2] / d, -1.0, 1.0))
        phi = np.arctan2(vector[1], vector[0])
        return float(d), float(theta), float(phi)

    def fragment_ligand_frag2seq(self, ligand_path):
        """
        Fragment the ground truth ligand and extract SE(3)-invariant 7D sequences.
        Generates spherical coordinates and quaternions relative to molecule and fragment frames.
        """
        supplier = Chem.SDMolSupplier(ligand_path)
        mol = supplier[0]
        
        if mol is None:
            return None
            
        bonds_to_break = []
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE and not bond.IsInRing():
                if bond.GetBeginAtom().GetDegree() > 1 and bond.GetEndAtom().GetDegree() > 1:
                    bonds_to_break.append(bond.GetIdx())
        
        if not bonds_to_break:
            return None
            
        fragmented_mol = Chem.FragmentOnBonds(mol, bonds_to_break)
        frags = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=False)
        
        frag_data = []
        for frag in frags:
            canonical_smiles = Chem.MolToSmiles(frag, isomericSmiles=True, canonical=True)
            
            conf = frag.GetConformer()
            coords = np.array([conf.GetAtomPosition(i) for i in range(frag.GetNumAtoms())])
            center = np.mean(coords, axis=0)
            
            frag_data.append({
                "smiles": canonical_smiles,
                "coords": coords,
                "center": center,
                "mol_obj": frag
            })
            
        frag_data.sort(key=lambda x: x["smiles"])
        
        centers = np.array([f["center"] for f in frag_data])
        R_m_to_w, t_m_to_w = self._get_local_frame(centers)
        
        sequence_7d = []
        
        for f in frag_data:
            rel_center = np.dot(R_m_to_w.T, (f["center"] - t_m_to_w))
            d, theta, phi = self._get_spherical_coords(rel_center)
            
            R_g_to_w, t_g_to_w = self._get_local_frame(f["coords"])
            R_g_to_m = np.dot(R_m_to_w.T, R_g_to_w)
            
            r = R.from_matrix(R_g_to_m)
            qx, qy, qz, qw = r.as_quat() 
            
            sequence_7d.append({
                "smiles": f["smiles"],
                "spatial_tokens": [d, theta, phi, float(qw), float(qx), float(qy), float(qz)]
            })
            
        return sequence_7d

    def process_system(self, system_name, scenario_type, entry, verbose=False):
        holo_pocket_rel, apo_pocket_rel, ligand_rel, _, _, _, _ = entry
        pli_id = apo_pocket_rel.split("/")[0]
        
        if verbose:
            print(f"\n=== PROCESSING: {pli_id} | SYSTEM: {system_name} ({scenario_type}) ===")
        
        target_apo_path = os.path.join(self.dataset_dir, apo_pocket_rel)
        target_holo_path = os.path.join(self.dataset_dir, holo_pocket_rel)
        ligand_path = os.path.join(self.dataset_dir, ligand_rel)
        
        scenario_config = self.system_config.get(system_name, {}).get("scenarios", {}).get(scenario_type, {})
        anti_target_paths = []
        for at in scenario_config.get("anti_targets", []):
            at_path = os.path.join(self.anti_target_dir, f"{at['pdb_id']}.pdb")
            if os.path.exists(at_path):
                anti_target_paths.append(at_path)
                
        apo_h_path = os.path.join(self.out_dir, f"{pli_id}_apo_H.pdb")
        holo_h_path = os.path.join(self.out_dir, f"{pli_id}_holo_H.pdb")
        
        self.add_explicit_hydrogens(target_apo_path, apo_h_path)
        self.add_explicit_hydrogens(target_holo_path, holo_h_path)
        
        if anti_target_paths:
            self.align_anti_targets(apo_h_path, anti_target_paths, out_prefix=pli_id)
        
        sequence_7d = self.fragment_ligand_frag2seq(ligand_path)
        
        if sequence_7d:
            if verbose:
                print(f"[v] Extracted {len(sequence_7d)} canonical 3D fragments with 7D spatial tokens.")
            out_pkl = os.path.join(self.out_dir, f"{pli_id}_sequence_7d.pkl")
            with open(out_pkl, 'wb') as f:
                pickle.dump(sequence_7d, f)

    def build_global_vocab(self):
        """
        Scans all generated 7D sequence files to build a comprehensive global vocabulary.
        Saves the vocabulary to a pickle file for consistent usage in the trainer.
        """
        print("\n[*] Building Global Vocabulary from all processed 7D sequences...")
        vocab = {"[SOS]": 0, "[UNK]": 1}
        
        search_pattern = os.path.join(self.out_dir, "*_sequence_7d.pkl")
        sequence_files = glob.glob(search_pattern)
        
        if not sequence_files:
            print("[!] No 7D sequence files found. Please process the dataset first.")
            return None
            
        # Add tqdm for vocabulary building
        for file_path in tqdm(sequence_files, desc="Building Vocab", unit="file"):
            with open(file_path, "rb") as f:
                sequence_7d = pickle.load(f)
                for item in sequence_7d:
                    smiles = item["smiles"]
                    if smiles not in vocab:
                        vocab[smiles] = len(vocab)
                        
        vocab_path = os.path.join(self.out_dir, "global_vocab.pkl")
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
            
        print(f"[v] Global vocabulary successfully built and saved to {vocab_path}")
        print(f"    -> Total Unique Tokens: {len(vocab)}")
        return vocab


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/Apo2Mol_Dataset/data_folder"))
    ANTI_TARGET_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/AntiTargets_PDB"))
    SPLIT_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/Apo2Mol_Dataset/split_druglike_dict.pkl"))
    CONFIG_FILE = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/system_config.json"))
    OUT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../dataset/processed"))
    
    processor = SAMGDataPreprocessor(
        dataset_dir=DATASET_DIR,
        anti_target_dir=ANTI_TARGET_DIR,
        split_dict_path=SPLIT_FILE,
        config_path=CONFIG_FILE,
        out_dir=OUT_DIR
    )
    
    train_data = processor.load_split("train")
    if train_data:
        
        # NOTE: Modify slicing (e.g. [:100]) for quick testing, or remove slicing for full dataset processing
        test_subset = train_data[:1000] 
        
        print(f"\n[*] Starting preprocessing for {len(test_subset)} complexes...")
        
        # Wrap processing loop with tqdm and set verbose=False to keep terminal clean
        for entry in tqdm(test_subset, desc="Processing Complexes", unit="complex"):
            if "HSP90_System" in processor.system_config:
                processor.process_system("HSP90_System", "easy", entry, verbose=False)
            elif "JNK_System" in processor.system_config:
                processor.process_system("JNK_System", "hard", entry, verbose=False)
                
    # Build global vocabulary after all files are processed
    processor.build_global_vocab()