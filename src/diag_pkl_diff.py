"""
diag_pkl_diff.py -- K6 that bai (Muc 45): sau S5, `d/theta/phi` BIT-IDENTICAL nhung bon chieu
quaternion lech 1-3%. Chi co the la gia tri quaternion cua tung token doi, chu khong phai doi thu tu
(trung binh bat bien voi thu tu). Script nay so TUNG TRUONG giua pkl CU (ban sao luu) va pkl MOI de
chi dung cho, thay vi doan.

Ghep fragment giua hai ban bang (smiles, tam) -- bat bien voi thu tu.
"""
import os
import sys
import glob
import pickle
import socket
import argparse
import subprocess
from collections import Counter

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)


def centroid_key(it):
    return tuple(np.round(np.asarray(it["local_coords"], np.float64).mean(0), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True, help="dataset/processed_backup_prechannel_*")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMG_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}\ndevice: khong dung  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    news = sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))[: args.n]

    n_file, n_frag, verdict = 0, 0, Counter()
    dq_max, dlocal_max, dcent_max = 0.0, 0.0, 0.0
    sign_flip, examples = 0, []
    for npath in news:
        opath = os.path.join(args.old_dir, os.path.basename(npath))
        if not os.path.exists(opath):
            continue
        with open(npath, "rb") as f:
            dn = pickle.load(f)
        with open(opath, "rb") as f:
            do = pickle.load(f)
        if len(dn["sequence"]) != len(do["sequence"]):
            verdict["so fragment KHAC"] += 1
            continue
        n_file += 1
        # ghep theo (smiles, tam cua local_coords) -- bat bien voi thu tu
        old_by = {}
        for it in do["sequence"]:
            old_by.setdefault((it["smiles"], centroid_key(it)), []).append(it)
        for itn in dn["sequence"]:
            k = (itn["smiles"], centroid_key(itn))
            cand = old_by.get(k)
            if not cand:
                verdict["khong ghep duoc fragment"] += 1
                continue
            ito = cand.pop(0)
            n_frag += 1
            a = np.asarray(itn["spatial_tokens"], np.float64)
            b = np.asarray(ito["spatial_tokens"], np.float64)
            dcent_max = max(dcent_max, float(np.abs(a[:3] - b[:3]).max()))
            dq = float(np.abs(a[3:] - b[3:]).max())
            dq_sum = float(np.abs(a[3:] + b[3:]).max())     # q vs -q
            if dq > 1e-6 and dq_sum < 1e-6:
                sign_flip += 1
                verdict["quaternion DOI DAU (q -> -q)"] += 1
                if len(examples) < 3:
                    examples.append((os.path.basename(npath), itn["smiles"],
                                     np.round(b[3:], 4).tolist(), np.round(a[3:], 4).tolist()))
            elif dq > 1e-6:
                verdict["quaternion KHAC (khong phai doi dau)"] += 1
                if len(examples) < 3:
                    examples.append((os.path.basename(npath), itn["smiles"],
                                     np.round(b[3:], 4).tolist(), np.round(a[3:], 4).tolist()))
            dq_max = max(dq_max, dq)
            lo, ln = np.asarray(ito["local_coords"], np.float64), np.asarray(itn["local_coords"], np.float64)
            if lo.shape == ln.shape:
                dlocal_max = max(dlocal_max, float(np.abs(lo - ln).max()))
            else:
                verdict["local_coords KHAC SHAPE"] += 1
            if list(np.asarray(ito["atomic_nums"]).reshape(-1)) != list(np.asarray(itn["atomic_nums"]).reshape(-1)):
                verdict["atomic_nums KHAC"] += 1

    print(f"\n[*] so sanh {n_file} file, {n_frag} fragment ghep duoc")
    print(f"    max|delta d/theta/phi|   = {dcent_max:.3e}")
    print(f"    max|delta quaternion|    = {dq_max:.3e}")
    print(f"    max|delta local_coords|  = {dlocal_max:.3e}")
    print(f"    quaternion doi dau       = {sign_flip}/{n_frag} = {sign_flip / max(n_frag, 1):.4f}")
    print("\n    phan loai:")
    for k, v in verdict.most_common():
        print(f"        {k:44} x{v}")
    for fn, smi, qo, qn in examples:
        print(f"\n    vi du {fn}  {smi[:40]}\n        cu : {qo}\n        moi: {qn}")
    print("\n    DOC: 'quaternion DOI DAU' troi  => double cover q ~ -q (Muc D): cung MOT phep quay,")
    print("         nhan khong doi ve y nghia, nhung scaler/mo hinh thi doi. Phai chuan hoa dau.")
    print("         'quaternion KHAC' troi      => frame fragment that su doi => S5 da cham vao hinh hoc.")


if __name__ == "__main__":
    main()
