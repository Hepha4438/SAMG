"""
verify_frame_stability.py -- KIEM FRAME HOC GRAM-SCHMIDT (READ-ONLY, KHONG SUA GI)

Muc dich: kiem `_get_pocket_frame` (step0_data_prep.py, P1b BAN CUOI -- Gram-Schmidt tu
N->O va sidechain->backbone) TRUOC khi tai sinh du lieu. Voi cac cap pli_id cung PDB id,
frame hoc tinh o hai ben phai khop nhau sau khi ap phep quay Kabsch giua hai pocket --
neu frame tat dinh va lien tuc thi goc lech phai nho.

Cach chay:
    python src/verify_frame_stability.py
"""
import os
import glob
from collections import defaultdict

import numpy as np
from Bio.PDB import PDBParser

from step0_data_prep import SAMGDataPreprocessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(SAMG_ROOT, "dataset", "processed")
DATA_FOLDER = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "data_folder")

_parser = PDBParser(QUIET=True)


def pdb_id_of(pli_id):
    return pli_id.split("__")[0]


def find_pairs():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_sequence_7d.pkl")))
    pli_ids = [os.path.basename(f)[: -len("_sequence_7d.pkl")] for f in files]
    groups = defaultdict(list)
    for pli_id in pli_ids:
        groups[pdb_id_of(pli_id)].append(pli_id)

    pairs = []
    for ids in groups.values():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
    return pairs


def load_pocket_atoms(pli_id):
    """Tra ve atom_map {(resseq, icode, atom_name): coord}, bo HOH."""
    pdb_path = os.path.join(DATA_FOLDER, pli_id, "receptor_apo_pocket10.pdb")
    if not os.path.exists(pdb_path):
        return None

    structure = _parser.get_structure(pli_id, pdb_path)
    atom_map = {}
    for residue in structure.get_residues():
        if residue.get_resname() == "HOH":
            continue
        _, resseq, icode = residue.id
        for atom in residue:
            atom_map[(resseq, icode, atom.get_name())] = atom.get_coord()
    return atom_map


def kabsch(P, Q):
    """R (A->B) sao cho R @ p_i xap xi q_i, voi P, Q da tru centroid."""
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def angle_deg(M):
    tr = np.trace(M)
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1.0, 1.0))))


def summarize(name, arr):
    if not arr:
        print(f"    {name}: KHONG CO DU LIEU")
        return
    arr = np.array(arr)
    print(f"    {name}: n={len(arr)}  median={np.median(arr):.3f} deg  p90={np.percentile(arr, 90):.3f} deg  "
          f"ty le <2deg={np.mean(arr < 2.0) * 100:.2f}%  ty le >30deg={np.mean(arr > 30.0) * 100:.2f}%")


def main():
    # _get_pocket_frame khong dung self.xxx nao ca -> khong can __init__ day du (config, dataset_dir...)
    preproc = SAMGDataPreprocessor.__new__(SAMGDataPreprocessor)

    pairs = find_pairs()
    print(f"[*] So cap ung vien cung PDB id: {len(pairs)}")

    tau = 0.30
    rmsds = []
    devs_all = []
    devs_stable = []
    n_excluded_unstable = 0
    n_excluded_n1 = 0
    n_excluded_n2 = 0
    n_excluded_missing = 0

    for pid_a, pid_b in pairs:
        atoms_a = load_pocket_atoms(pid_a)
        atoms_b = load_pocket_atoms(pid_b)
        if not atoms_a or not atoms_b:
            continue

        common_keys = sorted(set(atoms_a.keys()) & set(atoms_b.keys()))
        if len(common_keys) < 3:
            continue

        P = np.array([atoms_a[k] for k in common_keys], dtype=np.float64)
        Q = np.array([atoms_b[k] for k in common_keys], dtype=np.float64)
        atom_names = np.array([k[2] for k in common_keys])

        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        R_ab = kabsch(Pc, Qc)
        rmsd = float(np.sqrt(np.mean(np.sum((Qc - Pc @ R_ab.T) ** 2, axis=1))))
        if rmsd > 1.0:
            continue
        rmsds.append(rmsd)

        Fa, _, stable_a, diag_a = preproc._get_pocket_frame(P, atom_names, tau=tau)
        Fb, _, stable_b, diag_b = preproc._get_pocket_frame(Q, atom_names, tau=tau)

        dev = angle_deg(Fb.T @ (R_ab @ Fa))
        devs_all.append(dev)
        if stable_a and stable_b:
            devs_stable.append(dev)
        else:
            n_excluded_unstable += 1
            if (diag_a["n1"] is not None and diag_a["n1"] < tau) or (diag_b["n1"] is not None and diag_b["n1"] < tau):
                n_excluded_n1 += 1
            if (diag_a["n2"] is not None and diag_a["n2"] < tau) or (diag_b["n2"] is not None and diag_b["n2"] < tau):
                n_excluded_n2 += 1
            if diag_a["missing"] or diag_b["missing"]:
                n_excluded_missing += 1

    print(f"\n[*] n_cap (sau loc RMSD<=1.0A): {len(rmsds)}")
    if rmsds:
        print(f"[*] RMSD median: {np.median(rmsds):.4f} A")
    else:
        print("[*] RMSD median: KHONG CO CAP HOP LE")

    print("\n[*] Goc lech frame hoc -- TAT CA cap:")
    summarize("TAT CA", devs_all)
    print("\n[*] Goc lech frame hoc -- chi cap co stable=True ca hai ben:")
    summarize("STABLE", devs_stable)

    if devs_all:
        n = len(devs_all)
        print(f"\n[*] Ty le cap bi loai boi stable=False: {n_excluded_unstable / n * 100:.2f}%  "
              f"({n_excluded_unstable}/{n})")
        print(f"    trong do -- n1 < tau              : {n_excluded_n1 / n * 100:.2f}%  ({n_excluded_n1}/{n})")
        print(f"    trong do -- n2 < tau              : {n_excluded_n2 / n * 100:.2f}%  ({n_excluded_n2}/{n})")
        print(f"    trong do -- thieu nhom nguyen tu  : {n_excluded_missing / n * 100:.2f}%  ({n_excluded_missing}/{n})")


if __name__ == "__main__":
    main()
