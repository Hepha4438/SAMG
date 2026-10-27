"""
check_valid_adjacency.py -- plan.md "PLAN Rev 18" (a); RESEARCH_CONTEXT Muc 34.3.

Cau hoi: vi sao M10 (loader shuffle=False, derangement TRONG batch 16) cho delta R2(d) ~0,08 trong khi
xao loader (M12, K=20) cho 0,216? Gia thuyet: cac complex LIEN KE trong valid hay chung mot hoc, nen
derangement trong batch thuong doi hoc lay mot hoc gan nhu giong het.

KHONG GPU, KHONG model. Dung SAMGOptimizedDataset (cung loc frame_stable) va DUNG thu tu valid_indices
(= thu tu loader shuffle=False). PDB = pdb_id_of (import tu check_split_leakage, R14). Tam hoc = pocket_t
trong pkl. "Cung hoc" = cung PDB VA ||pocket_t_i - pocket_t_j|| < r (r = 2,0 A chinh; bao cao 1,0 / 4,0).

s(bs) = trung binh_i [ so thanh vien KHAC cung batch (batch lien tiep co kich co bs) thoa dieu kien /
(n_batch - 1) ] = xac suat cap derangement trong batch la "cung hoc". s_rand: doi tac la moi complex
khac trong toan valid.

Tieu chi (plan Rev 18 (a), ghi cung truoc khi chay): du doan delta_M10 = (1 - s_pocket(16)) * 0,2160.
  s_pocket(16) >= 0,50 VA du doan trong [0,0451 ; 0,1124]  => XAC NHAN
  s_pocket(16) <  0,20                                     => BAC
  con lai                                                  => KHONG KET LUAN
"""
import os
import sys
import shutil
import pickle
import socket
import argparse
import tempfile
import subprocess
from collections import Counter

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

SEED = 1234
DELTA_PIPE_ALL = 0.2160            # M12, Muc 34.3
BAND_LO, BAND_HI = 0.0751, 0.0824  # ho M10
RADII = [1.0, 2.0, 4.0]
R_MAIN = 2.0
BATCH_SIZES = [16, 8]


def batch_share(same, bs):
    """same: ma tran bool [n, n] (duong cheo False). Tra ve s(bs) theo batch lien tiep."""
    n = same.shape[0]
    vals = []
    for start in range(0, n, bs):
        idx = np.arange(start, min(start + bs, n))
        if len(idx) < 2:
            continue
        sub = same[np.ix_(idx, idx)]
        vals.extend((sub.sum(axis=1) / (len(idx) - 1)).tolist())
    return float(np.mean(vals))


# ---------------------------------------------------------------- Rev 19 (a'): cung protein -------
# "Cung protein" = cung cluster MMseqs2 o 30% do trung chuoi (chuan split CrossDocked2020, dung trong
# Frag2Seq va 3DMolFormer -- WORKING_RULES G4). Chuoi = chain dong gop nhieu residue nhat vao hoc
# receptor_holo_pocket10.pdb (hoc 10 A, dinh nghia Apo2Mol), lay tu receptor_holo_prot.pdb.
# Nguong (plan Rev 19, suy tu du lieu Muc 35): s trong [0,446 ; 0,823] => TUONG THICH.
DELTA_PIPE_REV18 = 0.2156
S_LO, S_HI = 0.446, 0.823
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
       "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
       "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


def pocket_main_chain(pocket_pdb):
    res = {}
    with open(pocket_pdb) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                res.setdefault(line[21], set()).add(line[22:27])
    return max(res, key=lambda c: len(res[c])) if res else None


def chain_sequence(prot_pdb, chain):
    seq, seen = [], set()
    with open(prot_pdb) as f:
        for line in f:
            if line.startswith("ATOM") and line[21] == chain and line[12:16].strip() == "CA":
                key = line[22:27]
                if key in seen:
                    continue
                seen.add(key)
                seq.append(AA3.get(line[17:20].strip(), "X"))
    return "".join(seq)


def mmseqs_labels(mmseqs, seqs, min_seq_id):
    tmp = tempfile.mkdtemp(prefix="samg_mmseqs_")
    try:
        fa = os.path.join(tmp, "in.fasta")
        with open(fa, "w") as f:
            for k, sq in enumerate(seqs):
                f.write(f">c{k}\n{sq}\n")
        prefix = os.path.join(tmp, "clu")
        subprocess.run([mmseqs, "easy-cluster", fa, prefix, os.path.join(tmp, "work"),
                        "--min-seq-id", str(min_seq_id)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lab = {}
        with open(prefix + "_cluster.tsv") as f:
            for line in f:
                rep_, mem = line.split()
                lab[int(mem[1:])] = rep_
        return np.array([lab[k] for k in range(len(seqs))])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_seqid(ds, dataset_dir, mmseqs):
    ver = subprocess.run([mmseqs, "version"], capture_output=True, text=True).stdout.strip()
    print(f"\n[*] Rev 19 (a'): mmseqs version = {ver}  ({mmseqs})", flush=True)
    seqs, n_short = [], 0
    for k, i in enumerate(ds.valid_indices):
        e = ds.target_entries[i]
        ch = pocket_main_chain(os.path.join(dataset_dir, e[0]))
        sq = chain_sequence(os.path.join(dataset_dir, e[3]), ch) if ch else ""
        n_short += int(len(sq) < 20)
        seqs.append(sq if sq else "X")
        if (k + 1) % 200 == 0:
            print(f"  doc chuoi {k + 1}/{len(ds.valid_indices)}", flush=True)
    lens = np.array([len(x) for x in seqs])
    print(f"[*] do dai chuoi: min={lens.min()}  median={int(np.median(lens))}  max={lens.max()}  "
          f"so chuoi < 20 aa = {n_short}")
    n = len(seqs)
    print("\n" + "=" * 90)
    print("Rev 19 (a'): XAC SUAT CAP DERANGEMENT 'CUNG CLUSTER MMseqs2'")
    print("=" * 90)
    print(f"{'min-seq-id':14}{'n_cluster':>11}" + "".join(f"{'s(bs=' + str(b) + ')':>12}" for b in BATCH_SIZES)
          + f"{'s_rand':>12}")
    s30 = None
    for t in (0.3, 0.95):
        lab = mmseqs_labels(mmseqs, seqs, t)
        same = lab[:, None] == lab[None, :]
        np.fill_diagonal(same, False)
        row = [batch_share(same, b) for b in BATCH_SIZES]
        s_rand = float((same.sum(axis=1) / (n - 1)).mean())
        tag = "" if t == 0.3 else "  (quy uoc du an, chi tham khao)"
        print(f"{t:<14}{len(set(lab)):>11}" + "".join(f"{v:12.4f}" for v in row) + f"{s_rand:12.4f}{tag}")
        if t == 0.3:
            s30 = row[0]
    pred = (1 - s30) * DELTA_PIPE_REV18
    print("\n" + "=" * 90)
    print("TIEU CHI (plan Rev 19 (a'))")
    print("=" * 90)
    print(f"    s_seq30(16) = {s30:.4f}  =>  du doan delta_M10 = (1 - s) x {DELTA_PIPE_REV18} = {pred:.4f}")
    if S_LO <= s30 <= S_HI:
        print(f"    => TUONG THICH (s trong [{S_LO}, {S_HI}]): ghep cung protein du giai thich M10 do thap. "
              f"Chua phai bang chung nhan qua.")
    elif s30 < S_LO:
        print(f"    => BAC (s < {S_LO}): cung protein khong du giai thich.")
    else:
        print(f"    => KHONG KHOP (s > {S_HI}): gia dinh 'cung protein => delta ~ 0' sai mot phan; khong ket luan.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmseqs", default=None, help="Rev 19 (a'): duong dan binary mmseqs")
    args = ap.parse_args()
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(CURRENT_DIR),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")
    print(f"SEED: {SEED}  (khong dung ngau nhien)")
    print(f"device: khong dung (script khong GPU)  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    from check_split_leakage import pdb_id_of

    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(st.VOCAB_PATH):
        with open(st.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)
    ds = st.SAMGOptimizedDataset(st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode="valid")

    pli_ids, pdbs, centers = [], [], []
    for k, i in enumerate(ds.valid_indices):
        pli_id = ds.target_entries[i][2].split("/")[0]
        with open(ds.pkl_by_id[pli_id], "rb") as f:
            pkl = pickle.load(f)
        pli_ids.append(pli_id)
        pdbs.append(pdb_id_of(pli_id))
        centers.append(np.asarray(pkl["pocket_t"], dtype=np.float64).reshape(3))
        if (k + 1) % 200 == 0:
            print(f"  doc pkl {k + 1}/{len(ds.valid_indices)}", flush=True)
    n = len(pli_ids)
    centers = np.stack(centers)
    pdbs_arr = np.array(pdbs)

    cnt = Counter(pdbs)
    sizes = Counter(cnt.values())
    print(f"\n[*] valid: n_complex = {n}  so PDB duy nhat = {len(cnt)}")
    print(f"[*] phan phoi so complex / PDB: {dict(sorted(sizes.items()))}")
    print(f"[*] 10 PDB nhieu complex nhat: {cnt.most_common(10)}")
    print(f"[*] 20 pli_id dau tien theo thu tu loader: {pli_ids[:20]}")

    same_pdb = pdbs_arr[:, None] == pdbs_arr[None, :]
    np.fill_diagonal(same_pdb, False)
    dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    adj_same = float(np.mean([pdbs[i] == pdbs[i + 1] for i in range(n - 1)]))
    print(f"[*] ty le cap LIEN KE (i, i+1) cung PDB = {adj_same:.4f}")

    s_rand_pdb = float((same_pdb.sum(axis=1) / (n - 1)).mean())
    print("\n" + "=" * 90)
    print("XAC SUAT CAP DERANGEMENT 'CUNG PDB' / 'CUNG HOC'")
    print("=" * 90)
    print(f"{'dieu kien':28}" + "".join(f"{'s(bs=' + str(b) + ')':>12}" for b in BATCH_SIZES) + f"{'s_rand':>12}")
    print(f"{'cung PDB':28}" + "".join(f"{batch_share(same_pdb, b):12.4f}" for b in BATCH_SIZES) + f"{s_rand_pdb:12.4f}")
    s_main = None
    for r in RADII:
        same_pk = same_pdb & (dist < r)
        row = [batch_share(same_pk, b) for b in BATCH_SIZES]
        s_rand = float((same_pk.sum(axis=1) / (n - 1)).mean())
        print(f"{'cung hoc (PDB & <' + str(r) + ' A)':28}" + "".join(f"{v:12.4f}" for v in row) + f"{s_rand:12.4f}")
        if r == R_MAIN:
            s_main = row[0]

    pred = (1 - s_main) * DELTA_PIPE_ALL
    print("\n" + "=" * 90)
    print("TIEU CHI (plan Rev 18 (a))")
    print("=" * 90)
    print(f"    s_pocket(16) = {s_main:.4f}  =>  du doan delta_M10 = (1 - s) x {DELTA_PIPE_ALL} = {pred:.4f}")
    print(f"    khoang chap nhan [{BAND_LO - 0.03:.4f} ; {BAND_HI + 0.03:.4f}]")
    if s_main >= 0.50 and BAND_LO - 0.03 <= pred <= BAND_HI + 0.03:
        print("    => XAC NHAN: thu tu loader giai thich su do thap cua M10.")
    elif s_main < 0.20:
        print("    => BAC: thu tu loader KHONG phai nguyen nhan chinh.")
    else:
        print("    => KHONG KET LUAN (mot phan).")
    print("    Gioi han: PDB chi la can duoi cua 'cung protein'.")

    if args.mmseqs:
        run_seqid(ds, st.DATASET_DIR, args.mmseqs)


if __name__ == "__main__":
    main()
