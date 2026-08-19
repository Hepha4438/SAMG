import os
import glob
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs

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

class SAMGEvaluator:
    def __init__(self, target_pdb, anti_target_pdbs=[], center=[0,0,0], box_size=[20, 20, 20]):
        self.target_pdb = target_pdb
        self.anti_target_pdbs = anti_target_pdbs
        self.center = center
        self.box_size = box_size
        
        self.target_pdbqt = self._prepare_receptor(target_pdb)
        self.anti_target_pdbqts = [self._prepare_receptor(at) for at in anti_target_pdbs]
        
        if VINA_AVAILABLE:
            self.vina = Vina(sf_name='vina', verbosity=0)
        else:
            print("[!] Warning: 'vina' package not found. Docking skipped.")

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
                lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')]
                dry_lines = [l for l in lines if 'HOH' not in l]
            with open(clean_pdb, 'w') as f:
                f.writelines(dry_lines)
                
            subprocess.run(['pdb2pqr30', '--ff=AMBER', clean_pdb, pqr_path], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                           
            if ADT_AVAILABLE:
                prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
                subprocess.run(['python3', prepare_receptor, '-r', pqr_path, '-o', pdbqt_path, '-A', 'checkhydro'], 
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                subprocess.run(['obabel', '-i', 'pqr', pqr_path, '-o', 'pdbqt', '-O', pdbqt_path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception as e:
            print(f"[!] Receptor preparation error for {pdb_path}: {e}")
            
        for p in [clean_pdb, pqr_path]:
            if os.path.exists(p): os.remove(p)
                
        return pdbqt_path if os.path.exists(pdbqt_path) else ""

    def evaluate_drug_likeness(self, mol):
        qed_score = QED.qed(mol)
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        logp = Descriptors.MolLogP(mol)
        rule_4 = -2 <= logp <= 5
        rule_5 = Descriptors.NumRotatableBonds(mol) <= 10
        lipinski_score = sum([rule_1, rule_2, rule_3, rule_4, rule_5])
        
        sa_score = np.nan
        if sascorer is not None:
            try:
                sa_score = sascorer.calculateScore(mol)
            except: pass
                
        return qed_score, sa_score, lipinski_score

    def evaluate_docking_and_selectivity(self, ligand_pdbqt):
        if not VINA_AVAILABLE or not self.target_pdbqt or not os.path.exists(self.target_pdbqt):
            return np.nan, np.nan, np.nan, False

        try:
            self.vina.set_receptor(self.target_pdbqt)
            self.vina.set_ligand_from_file(ligand_pdbqt)
            self.vina.compute_vina_maps(center=self.center, box_size=self.box_size)
            target_score = self.vina.score()[0]

            anti_target_scores = []
            for at_pdbqt in self.anti_target_pdbqts:
                if at_pdbqt and os.path.exists(at_pdbqt):
                    self.vina.set_receptor(at_pdbqt)
                    self.vina.set_ligand_from_file(ligand_pdbqt)
                    self.vina.compute_vina_maps(center=self.center, box_size=self.box_size)
                    anti_target_scores.append(self.vina.score()[0])

            min_at_score = min(anti_target_scores) if anti_target_scores else 0.0
            selectivity = min_at_score - target_score
            is_hit = (target_score <= -8.0) and (selectivity >= 1.4)
            
            return target_score, min_at_score, selectivity, is_hit
        except Exception as e:
            return np.nan, np.nan, np.nan, False

    def evaluate_diversity_and_series(self, mols, is_hit_list):
        valid_mols = [m for m in mols if m is not None]
        n_mols = len(valid_mols)
        if n_mols < 2: return 0.0, 0
            
        fps = [Chem.RDKFingerprint(m) for m in valid_mols]
        sims = []
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
        avg_diversity = 1.0 - np.mean(sims)
        
        scaffolds = defaultdict(list)
        for idx, mol in enumerate(valid_mols):
            try:
                core = MurckoScaffold.GetScaffoldForMol(mol)
                core_smiles = Chem.MolToSmiles(core)
                scaffolds[core_smiles].append(is_hit_list[idx])
            except: continue
                
        num_chemical_series = 0
        for core_smiles, hit_flags in scaffolds.items():
            if len(hit_flags) >= 3 and any(hit_flags):
                num_chemical_series += 1
                
        # ĐÃ SỬA LỖI: Trả về đúng biến num_chemical_series
        return avg_diversity, num_chemical_series

    def run_full_evaluation(self, generated_sdfs):
        results, mols, is_hit_list, valid_count, smiles_list = [], [], [], 0, []
        
        for sdf in tqdm(generated_sdfs, desc="Evaluating Pipeline"):
            mol = next(Chem.SDMolSupplier(sdf))
            if mol is None or mol.GetNumAtoms() == 0: continue
                
            valid_count += 1
            mols.append(mol)
            smiles_list.append(Chem.MolToSmiles(mol))
            
            qed_score, sa_score, lipinski = self.evaluate_drug_likeness(mol)
            
            pdbqt_path = sdf.replace(".sdf", ".pdbqt")
            if MEEKO_AVAILABLE and not os.path.exists(pdbqt_path):
                try:
                    mol_h = Chem.AddHs(mol, addCoords=True)
                    mol_prep = MoleculePreparation()
                    mol_prep.prepare(mol_h)
                    with open(pdbqt_path, "w") as f:
                        f.write(mol_prep.write_pdbqt_string())
                except: pass
                    
            if os.path.exists(pdbqt_path):
                t_score, at_score, sel, is_hit = self.evaluate_docking_and_selectivity(pdbqt_path)
            else:
                t_score, at_score, sel, is_hit = (np.nan, np.nan, np.nan, False)
                
            is_hit_list.append(is_hit)
            results.append({"File": os.path.basename(sdf), "QED": qed_score, "SA": sa_score, "Lipinski": lipinski, "Vina_Target": t_score, "Selectivity": sel, "Is_Hit": is_hit})
            
        df = pd.DataFrame(results)
        validity = valid_count / len(generated_sdfs) if len(generated_sdfs) > 0 else 0
        uniqueness = len(set(smiles_list)) / len(smiles_list) if len(smiles_list) > 0 else 0
        diversity, num_series = self.evaluate_diversity_and_series(mols, is_hit_list)
        
        print("\n" + "="*30)
        print("🚀 FINAL SBDD METRICS SUMMARY 🚀")
        print("="*30)
        print(f"🔸 Total Evaluated: {len(generated_sdfs)}")
        print(f"🔸 Validity:   {validity*100:.2f} %")
        print(f"🔸 Uniqueness: {uniqueness*100:.2f} %")
        print(f"🔸 Diversity (Tanimoto): {diversity:.4f}")
        print("-" * 30)
        if not df.empty:
            print(f"🔸 Avg QED: {df['QED'].mean():.4f}")
            print(f"🔸 Avg Lipinski Pass: {df['Lipinski'].mean():.2f} / 5")
            print(f"🔸 Avg SA Score: {df['SA'].mean():.2f}")
            print("-" * 30)
            if VINA_AVAILABLE:
                print(f"🔸 Avg Target Affinity: {df['Vina_Target'].mean():.2f} kcal/mol")
                print(f"🔸 Avg Selectivity: {df['Selectivity'].mean():.2f} kcal/mol")
        print("================================")
        return df, {"Validity": validity, "Uniqueness": uniqueness, "Diversity": diversity}