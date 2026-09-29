"""
verify_label_quality.py -- KIEM CHAT LUONG NHAN 7D (READ-ONLY, KHONG SUA DU LIEU)

Gop hai phep do thanh mot script:
[A] Do tat dinh cua nhan theo vi tri token (entropy 0 o pos 0/1/2 khi frame con dung
    3 tam fragment dau -- moc tham chieu TRUOC Phase 1 la ~48%).
[B] Phan phoi bien d (moc tham chieu: intramolecular cu mean 3.94, max 27.13).
[C] G1/G3 do tren cap complex CUNG pdb_id VA CUNG tuple SMILES fragment (nen token
    tuong ung 1-1 theo vi tri), CO PHAN TANG theo do giong nhau cua hoc -- BAT BUOC
    phan tang, khong phan tang thi con so vo nghia (xem RESEARCH_CONTEXT 17.2).

CONG CHAN Phase 1 (dong "frac>=0.99" trong bang [C]): G1 < 2 deg VA G3 < 0.05 A.

Cach chay:
    python src/verify_label_quality.py
"""
import os
import glob
import pickle
from collections import defaultdict

import numpy as np
from Bio.PDB import PDBParser

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
PROCESSED_DIR = os.path.join(SAMG_ROOT, "dataset", "processed")
DATA_FOLDER = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "data_folder")

_parser = PDBParser(QUIET=True)


def load_all_sequences():
    """Tra ve dict {pli_id: sequence (list cac token)}. Bo qua file dinh dang CU (truoc P1c
    -- pkl la list truc tiep, khong phai dict {"sequence": [...]})va dem so luong de bao cao."""
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_sequence_7d.pkl")))
    out = {}
    n_legacy = 0
    for f in files:
        pli_id = os.path.basename(f)[: -len("_sequence_7d.pkl")]
        with open(f, "rb") as fh:
            data = pickle.load(fh)
        if isinstance(data, dict) and "sequence" in data:
            out[pli_id] = data["sequence"]
        else:
            n_legacy += 1
    if n_legacy > 0:
        print(f"[!] Bo qua {n_legacy} file dinh dang CU (truoc P1c, khong phai dict) -- "
              f"can chay lai step0 (P1e) de co dinh dang moi.")
    return out


def is_deterministic(spatial_tokens):
    d, theta, phi = spatial_tokens[0], spatial_tokens[1], spatial_tokens[2]
    return (abs(d) < 1e-4 or abs(theta - np.pi / 2) < 1e-4
            or abs(phi) < 1e-4 or abs(abs(phi) - np.pi) < 1e-4)


def block_a(sequences):
    print("\n" + "=" * 78)
    print("[A] DO TAT DINH CUA NHAN THEO VI TRI TOKEN (moc tham chieu TRUOC Phase 1: ~48%)")
    print("=" * 78)
    buckets = defaultdict(lambda: [0, 0])  # pos -> [n_deterministic, n_total]
    for seq in sequences.values():
        for i, item in enumerate(seq):
            pos = i if i < 3 else 3
            buckets[pos][1] += 1
            if is_deterministic(item["spatial_tokens"]):
                buckets[pos][0] += 1

    labels = {0: "pos 0", 1: "pos 1", 2: "pos 2", 3: "pos >=3"}
    n_det_total, n_total = 0, 0
    for pos in sorted(buckets):
        n_det, n = buckets[pos]
        n_det_total += n_det
        n_total += n
        rate = (n_det / n * 100) if n > 0 else 0.0
        print(f"    {labels[pos]:8}: {n_det}/{n}  ({rate:.2f}%)")
    if n_total > 0:
        print(f"    {'TONG':8}: {n_det_total}/{n_total}  ({n_det_total / n_total * 100:.2f}%)")
    else:
        print("    KHONG CO DU LIEU")


def block_b(sequences):
    print("\n" + "=" * 78)
    print("[B] PHAN PHOI d (moc tham chieu: intramolecular cu mean 3.94, max 27.13)")
    print("=" * 78)
    all_d = np.array([item["spatial_tokens"][0] for seq in sequences.values() for item in seq])
    if len(all_d) == 0:
        print("    KHONG CO DU LIEU")
        return
    print(f"    n={len(all_d)}  mean={all_d.mean():.4f}  std={all_d.std():.4f}  max={all_d.max():.4f}")


def pdb_id_of(pli_id):
    return pli_id.split("__")[0]


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


def find_label_pairs(sequences):
    """Gom cap (pli_id_a, pli_id_b) cung pdb_id VA cung tuple SMILES fragment
    (nen token tuong ung 1-1 theo vi tri, vi frag_data.sort(key=smiles) tat dinh)."""
    groups = defaultdict(list)
    for pli_id, seq in sequences.items():
        smiles_tuple = tuple(item["smiles"] for item in seq)
        if not smiles_tuple:
            continue
        groups[(pdb_id_of(pli_id), smiles_tuple)].append(pli_id)

    pairs = []
    for ids in groups.values():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.append((ids[i], ids[j]))
    return pairs


def quat_angle_deg(qa, qb):
    """qa, qb = (qw,qx,qy,qz). 2*degrees(arccos(|dot|)) de gap double cover q vs -q."""
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return 2.0 * np.degrees(np.arccos(dot))


def block_c(sequences):
    print("\n" + "=" * 78)
    print("[C] G1/G3 CO PHAN TANG THEO DO GIONG NHAU CUA HOC")
    print("    Cong chan Phase 1: dong 'frac>=0.99' phai dat G1 < 2 deg VA G3 < 0.05 A.")
    print("    Tham chieu nhan CU: G1 9.157 deg  G3 0.0239 A")
    print("=" * 78)

    pairs = find_label_pairs(sequences)
    print(f"[*] So cap complex cung pdb_id + cung tuple SMILES fragment: {len(pairs)}")

    conditions = ["TAT CA", "frac>=0.99", "frac>=0.99 & RMSD<=0.3", "frac>=0.99 & RMSD<=0.05"]
    rows = {name: [] for name in conditions}

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
        frac = len(common_keys) / max(len(atoms_a), len(atoms_b))

        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        R_ab = kabsch(Pc, Qc)
        rmsd = float(np.sqrt(np.mean(np.sum((Qc - Pc @ R_ab.T) ** 2, axis=1))))

        for tok_a, tok_b in zip(sequences[pid_a], sequences[pid_b]):
            sa, sb = tok_a["spatial_tokens"], tok_b["spatial_tokens"]
            d_diff = abs(sa[0] - sb[0])
            ang = quat_angle_deg(np.array(sa[3:7]), np.array(sb[3:7]))

            rows["TAT CA"].append((ang, d_diff))
            if frac >= 0.99:
                rows["frac>=0.99"].append((ang, d_diff))
                if rmsd <= 0.3:
                    rows["frac>=0.99 & RMSD<=0.3"].append((ang, d_diff))
                if rmsd <= 0.05:
                    rows["frac>=0.99 & RMSD<=0.05"].append((ang, d_diff))

    print(f"\n    {'dieu kien':28}{'n token':>10}{'G1 median':>14}{'G3 median':>14}")
    for name in conditions:
        vals = rows[name]
        if not vals:
            print(f"    {name:28}{0:>10}{'--':>14}{'--':>14}")
            continue
        angs = np.array([v[0] for v in vals])
        ds = np.array([v[1] for v in vals])
        print(f"    {name:28}{len(vals):>10}{np.median(angs):>14.4f}{np.median(ds):>14.4f}")


def main():
    sequences = load_all_sequences()
    print(f"[*] So pli_id doc duoc tu dataset/processed: {len(sequences)}")
    block_a(sequences)
    block_b(sequences)
    block_c(sequences)


if __name__ == "__main__":
    main()
