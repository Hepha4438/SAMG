"""
check_step0_channel.py -- PLAN Rev 21, kiem khong-doi K5-K9 sau khi chay lai step0 voi S5
(thu tu DFS canonical + kenh lien ket). Khong GPU, khong model.

  K5 `parent[t] < t` o 100% vi tri t>=1, va `parent[0] == -1` o 100% phan tu  -> DFS dung
  K6 scaler khop ban cu trong 1e-4 (nhan 7D KHONG doi, chi doi thu tu)         -> S5 khong lam doi nhan
  K7 `len(vocab)` khong doi (3.983)                                            -> tap fragment khong doi
  K8 tai dung tu `parent`/`attach_*` (KHONG dung hinh hoc lien-fragment) cho do thi TRUNG KHOP
     phan tu goc o >= 99% complex                                              -> TIEU CHI QUAN TRONG NHAT
  K9 `frame_stable` giu 17.200 / 764 / 255
K6 hoac K9 hong => S5 da lam doi nhan, DUNG. K8 hong => kenh lien ket khong du, huong (b) sai.
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
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

REL_TOL = 1e-4
VOCAB_EXPECTED = 3983
K8_THRESHOLD = 0.99
EXPECTED_STABLE = {"train": 17200, "valid": 764, "test": 255}
DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]
OLD_SCALER = {"d": (5.161378, 2.478297), "theta": (1.531624, 0.664945), "phi": (0.391346, 1.759804),
              "qw": (0.200377, 0.456459), "qx": (0.204276, 0.458135), "qy": (0.201783, 0.457270),
              "qz": (0.185659, 0.464332)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-scaler", default=None)
    ap.add_argument("--k8-n", type=int, default=400, help="so complex kiem K8 (0 = het train+valid+test)")
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMG_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}\nSEED: 1234  (khong dung ngau nhien)")
    print(f"device: khong dung  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    import assemble as asm
    fails = []

    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p
                 for p in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))}

    # ---------------------------------------------------------------- K5
    print("\n" + "=" * 90 + "\nK5: parent[t] < t o moi t>=1, parent[0] == -1\n" + "=" * 90)
    n_mol, n_bad_root, n_pos, n_bad_pos, modes = 0, 0, 0, 0, Counter()
    for split in ["train", "valid", "test"]:
        for e in splits.get(split, []):
            p = pkl_by_id.get(e[2].split("/")[0])
            if p is None:
                continue
            with open(p, "rb") as f:
                d = pickle.load(f)
            seq = d["sequence"]
            if not seq or "parent" not in seq[0]:
                continue
            n_mol += 1
            modes[d.get("frag_order_mode", "?")] += 1
            if seq[0]["parent"] != -1:
                n_bad_root += 1
            for t, it in enumerate(seq[1:], start=1):
                n_pos += 1
                if not (0 <= it["parent"] < t):
                    n_bad_pos += 1
    print(f"    phan tu kiem: {n_mol}  |  frag_order_mode: {dict(modes)}")
    print(f"    goc co parent != -1: {n_bad_root}")
    print(f"    vi tri t>=1 co parent khong hop le: {n_bad_pos}/{n_pos} = {n_bad_pos / max(n_pos, 1):.6f}")
    k5 = (n_bad_root == 0 and n_bad_pos == 0 and n_mol > 0)
    print(f"    => K5 {'DAT' if k5 else 'KHONG DAT'}")
    if not k5:
        fails.append("K5")

    # ---------------------------------------------------------------- K6
    print("\n" + "=" * 90 + "\nK6: scaler khop ban cu (sai so tuong doi 1e-4)\n" + "=" * 90)
    new = torch.load(os.path.join(st.PROCESSED_DIR, "scaler_7d.pt"), map_location="cpu")
    n_shift, n_scale = new["shift"].view(7).double(), new["scale"].view(7).double()
    if args.old_scaler and os.path.exists(args.old_scaler):
        old = torch.load(args.old_scaler, map_location="cpu")
        o_shift, o_scale = old["shift"].view(7).double(), old["scale"].view(7).double()
    else:
        o_shift = torch.tensor([OLD_SCALER[n][0] for n in DIM_NAMES], dtype=torch.float64)
        o_scale = torch.tensor([OLD_SCALER[n][1] for n in DIM_NAMES], dtype=torch.float64)
    k6 = True
    for i, nm in enumerate(DIM_NAMES):
        r = max(abs(n_shift[i] - o_shift[i]).item() / max(abs(o_shift[i]).item(), 1e-12),
                abs(n_scale[i] - o_scale[i]).item() / max(abs(o_scale[i]).item(), 1e-12))
        if r > REL_TOL:
            k6 = False
            print(f"    {nm}: LECH  shift {o_shift[i]:.6f}->{n_shift[i]:.6f}  "
                  f"scale {o_scale[i]:.6f}->{n_scale[i]:.6f}  rel={r:.2e}")
    print(f"    => K6 {'DAT' if k6 else 'KHONG DAT'}")
    if not k6:
        fails.append("K6")

    # ---------------------------------------------------------------- K7
    print("\n" + "=" * 90 + "\nK7: vocab khong doi\n" + "=" * 90)
    with open(st.VOCAB_PATH, "rb") as f:
        V = len(pickle.load(f))
    k7 = (V == VOCAB_EXPECTED)
    print(f"    vocab = {V}  (ky vong {VOCAB_EXPECTED})  => K7 {'DAT' if k7 else 'KHONG DAT'}")
    if not k7:
        fails.append("K7")

    # ---------------------------------------------------------------- K9
    print("\n" + "=" * 90 + "\nK9: so complex frame_stable\n" + "=" * 90)
    k9 = True
    for split in ["train", "valid", "test"]:
        n = 0
        for e in splits.get(split, []):
            p = pkl_by_id.get(e[2].split("/")[0])
            if p is None:
                continue
            with open(p, "rb") as f:
                d = pickle.load(f)
            if d.get("frame_stable", False) and all(it.get("frag_frame_stable", True) for it in d["sequence"]):
                n += 1
        ok = n == EXPECTED_STABLE[split]
        k9 &= ok
        print(f"    {split:6}: {n:6d}  (ky vong {EXPECTED_STABLE[split]})" + ("" if ok else "  <-- LECH"))
    print(f"    => K9 {'DAT' if k9 else 'KHONG DAT'}")
    if not k9:
        fails.append("K9")

    # ---------------------------------------------------------------- K8 (quan trong nhat)
    print("\n" + "=" * 90)
    print("K8: tai dung tu KENH LIEN KET (khong dung hinh hoc lien-fragment) -> do thi trung phan tu goc")
    print("=" * 90)
    n_try, n_match, errs = 0, 0, []
    for split in ["train", "valid", "test"]:
        for e in splits.get(split, []):
            if args.k8_n and n_try >= args.k8_n:
                break
            p = pkl_by_id.get(e[2].split("/")[0])
            if p is None:
                continue
            with open(p, "rb") as f:
                d = pickle.load(f)
            if not d.get("frame_stable", False) or "parent" not in d["sequence"][0]:
                continue
            if any(not it.get("frag_frame_stable", True) for it in d["sequence"]):
                continue
            ref = Chem.SDMolSupplier(os.path.join(st.DATASET_DIR, e[2]))[0]
            if ref is None:
                continue
            n_try += 1
            try:
                items = [{"smiles": it["smiles"], "local": np.asarray(it["local_coords"], np.float64),
                          "z": np.asarray(it["atomic_nums"]).reshape(-1), "parent": it["parent"],
                          "attach_self": it["attach_self"], "attach_parent": it["attach_parent"]}
                         for it in d["sequence"]]
                mol, diag = asm.assemble_by_channel(items)
                a = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)) or mol)
                b = Chem.MolToSmiles(ref)
                if a == b:
                    n_match += 1
                else:
                    errs.append(f"do thi khac (n_manh={diag['n_manh']}, link_fail={diag['n_link_that_bai']})")
            except Exception as exc:
                errs.append(f"{type(exc).__name__}: {str(exc)[:60]}")
        if args.k8_n and n_try >= args.k8_n:
            break
    rate = n_match / max(n_try, 1)
    print(f"    trung khop: {n_match}/{n_try} = {rate:.4f}  (nguong {K8_THRESHOLD})")
    for k, v in Counter(errs).most_common(5):
        print(f"      {k[:76]:78} x{v}")
    k8 = rate >= K8_THRESHOLD
    print(f"    => K8 {'DAT' if k8 else 'KHONG DAT -- kenh lien ket KHONG du, huong (b) sai'}")
    if not k8:
        fails.append("K8")

    print("\n" + "=" * 90)
    print("TAT CA K5-K9 DAT => chay build_fragment_dict roi sang S6/S7 (head + loss)." if not fails
          else f"KHONG DAT: {', '.join(fails)}")
    if "K6" in fails or "K9" in fails:
        print("    K6/K9 hong => S5 DA LAM DOI NHAN. DUNG.")
    print("=" * 90)


if __name__ == "__main__":
    main()
