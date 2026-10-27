"""
build_fragment_dict.py -- PLAN Rev 20 A1; RESEARCH_CONTEXT 37-pre.3, 37.3, 37.4.

Tu dien {canonical SMILES cua fragment -> atom types + toa do cuc bo trong frame fragment}.
Nguon: Frag2Seq muc 3.3.1 ("save each splited fragment ... with the key to be the canonical SMILES
of each fragment and the value to be the atom types and atom local coordinates in each fragment
under the associated fragment local frame").

Doc tu cac pkl DA CO (`*_sequence_7d.pkl` co truong `local_coords`/`atomic_nums` tu Rev 20) -- KHONG
parse lai SDF, KHONG chay lai step0. CHI dung split TRAIN (tu dien la tham so hoc tu du lieu).

Xap xi (cua ca Frag2Seq, khong phai rieng du an): mot SMILES co the co nhieu conformer; tu dien giu
conformer DAU TIEN gap. Script do `rmsd_spread` = RMSD (sau Kabsch) giua cac lan xuat hien khac va
conformer da giu, de biet xap xi nay dat gia bao nhieu.

Tieu chi doc (ghi cung, plan Rev 20 A1):
  median(rmsd_spread) > 0,5 A  => xap xi la nguon loi dang ke; ghi vao bao cao, KHONG quy ket sai
                                  lech Vina cho mo hinh. (0,5 A = nua nguong RMSD 1,0 A cua nganh
                                  docking/PoseBusters -- quy uoc du an, R16 muc 3.)
  do phu token valid < 95%     => bao cao; so Vina chi tinh tren phan tu co MOI fragment trong tu dien.
"""
import os
import sys
import glob
import pickle
import socket
import argparse
import subprocess

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

RMSD_MEDIAN_THRESHOLD = 0.5   # A
COVERAGE_THRESHOLD = 0.95


def kabsch_rmsd(P, Q):
    """RMSD sau khi can chinh toi uu (Kabsch 1976). P, Q: [n, 3] da cung thu tu nguyen tu."""
    if P.shape != Q.shape or P.shape[0] == 0:
        return float("nan")
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R_opt = V @ D @ Wt
    diff = Pc @ R_opt - Qc
    return float(np.sqrt((diff ** 2).sum() / P.shape[0]))


def pli_ids_of_split(st, split_mode):
    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    if split_mode == "val" and "valid" in splits:
        split_mode = "valid"
    return [e[2].split("/")[0] for e in splits.get(split_mode, [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Mac dinh dataset/processed/fragment_dict.pkl")
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(CURRENT_DIR),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")
    print(f"SEED: 1234  (khong dung ngau nhien)")
    print(f"device: khong dung (script khong GPU)  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    out_path = args.out or os.path.join(st.PROCESSED_DIR, "fragment_dict.pkl")

    pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p
                 for p in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))}
    train_ids = pli_ids_of_split(st, "train")
    print(f"[*] train entry: {len(train_ids)}  co pkl: {sum(1 for i in train_ids if i in pkl_by_id)}")

    frag_dict, spreads, n_items, n_missing_field = {}, [], 0, 0
    for k, pid in enumerate(train_ids):
        path = pkl_by_id.get(pid)
        if path is None:
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not d.get("frame_stable", False):
            continue
        for item in d["sequence"]:
            if "local_coords" not in item or "atomic_nums" not in item:
                n_missing_field += 1
                continue
            n_items += 1
            smi = item["smiles"]
            local = np.asarray(item["local_coords"], dtype=np.float64)
            nums = np.asarray(item["atomic_nums"])
            if smi not in frag_dict:
                frag_dict[smi] = {"atomic_nums": nums, "local": local, "n_seen": 1}
            else:
                e = frag_dict[smi]
                e["n_seen"] += 1
                if e["local"].shape == local.shape:
                    spreads.append((smi, kabsch_rmsd(local, e["local"])))
        if (k + 1) % 2000 == 0:
            print(f"  doc {k + 1}/{len(train_ids)}  (|dict| = {len(frag_dict)})", flush=True)

    if n_missing_field and not frag_dict:
        raise SystemExit("[!] pkl KHONG co truong local_coords/atomic_nums. Phai chay lai step0 "
                         "(ban sua Muc 37.4) truoc khi dung script nay.")

    print(f"\n[*] token fragment doc duoc: {n_items}  (thieu truong: {n_missing_field})")
    print(f"[*] so SMILES duy nhat trong tu dien: {len(frag_dict)}")

    if spreads:
        vals = np.array([v for _, v in spreads], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        q = np.percentile(vals, [50, 90, 99]) if vals.size else [float("nan")] * 3
        print(f"[*] rmsd_spread (A) tren {vals.size} lan lap: median={q[0]:.4f}  p90={q[1]:.4f}  p99={q[2]:.4f}")
        worst = sorted(spreads, key=lambda x: -(x[1] if x[1] == x[1] else -1))[:5]
        print(f"    5 fragment lech nhat: {[(s[:30], round(v, 3)) for s, v in worst]}")
        med = float(q[0])
    else:
        med = float("nan")
        print("[*] khong co SMILES nao lap lai -- khong do duoc rmsd_spread")

    # do phu tren valid
    valid_ids = pli_ids_of_split(st, "valid")
    n_tok, n_cov, n_mol, n_mol_full = 0, 0, 0, 0
    for pid in valid_ids:
        path = pkl_by_id.get(pid)
        if path is None:
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not d.get("frame_stable", False):
            continue
        n_mol += 1
        ok_all = True
        for item in d["sequence"]:
            n_tok += 1
            if item["smiles"] in frag_dict:
                n_cov += 1
            else:
                ok_all = False
        n_mol_full += int(ok_all)
    cov = n_cov / max(n_tok, 1)
    cov_mol = n_mol_full / max(n_mol, 1)
    print(f"[*] do phu tren VALID: token {n_cov}/{n_tok} = {cov:.4f}  |  "
          f"phan tu co DU moi fragment: {n_mol_full}/{n_mol} = {cov_mol:.4f}")

    with open(out_path, "wb") as f:
        pickle.dump(frag_dict, f)
    print(f"[*] da luu {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print("\n" + "=" * 90)
    print("TIEU CHI (plan Rev 20 A1)")
    print("=" * 90)
    print(f"    median(rmsd_spread) = {med:.4f} A  (nguong {RMSD_MEDIAN_THRESHOLD})")
    if med == med and med > RMSD_MEDIAN_THRESHOLD:
        print("    => VUOT: xap xi 'mot conformer moi SMILES' la nguon loi dang ke. Ghi vao bao cao;")
        print("       KHONG quy ket sai lech Vina cho mo hinh.")
    else:
        print("    => DUOI nguong: xap xi chap nhan duoc.")
    print(f"    do phu token valid = {cov:.4f}  (nguong {COVERAGE_THRESHOLD})")
    if cov < COVERAGE_THRESHOLD:
        print("    => DUOI 95%: so Vina chi tinh tren phan tu co MOI fragment trong tu dien; bao cao ty le.")
    else:
        print("    => DAT.")


if __name__ == "__main__":
    main()
