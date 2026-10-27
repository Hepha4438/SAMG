"""
check_step0_rerun.py -- PLAN Rev 20 A0; RESEARCH_CONTEXT 37.7.
Kiem khong-doi (R15) sau khi chay lai step0 voi S1-S4. Khong GPU, khong model.

S1-S4 KHONG duoc doi gia tri nhan 7D -- chi doi nhan dummy, thu tu fragment, va tu vung.
  K1 scaler : khop ban cu trong sai so TUONG DOI 1e-4 o ca 14 so (sai so != 0 vi S3 doi thu tu cong).
  K2 vocab  : 3.500 <= len(vocab) <= 5.336  (5.334 fragment hoa hoc do o 37.1, + [SOS] + [UNK]).
  K3 complex: frame_stable giu 17.200 / 764 / 255 (A1).
  K4 truong : moi item co local_coords [n,3] + atomic_nums [n] cung do dai; frag_order_mode dung.
K1 hoac K3 khong dat => DUNG, S1-S4 da lam doi nhan.
"""
import os
import sys
import glob
import pickle
import socket
import argparse
import subprocess

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

REL_TOL = 1e-4
VOCAB_LO, VOCAB_HI = 3500, 5336
EXPECTED_STABLE = {"train": 17200, "valid": 764, "test": 255}
DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]
# A1 (RESULTS_LEDGER): shift/scale cua scaler CU
OLD_SCALER = {
    "d": (5.161378, 2.478297), "theta": (1.531624, 0.664945), "phi": (0.391346, 1.759804),
    "qw": (0.200377, 0.456459), "qx": (0.204276, 0.458135), "qy": (0.201783, 0.457270),
    "qz": (0.185659, 0.464332),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-scaler", default=None, help="scaler_7d.pt cu (neu khong co, dung bang A1)")
    ap.add_argument("--old-vocab", default=None, help="global_vocab.pkl cu (chi de in doi chieu)")
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
    fails = []

    # ---------------------------------------------------------------- K1: scaler
    print("\n" + "=" * 90)
    print("K1: SCALER phai khop ban cu (sai so tuong doi 1e-4)")
    print("=" * 90)
    new = torch.load(os.path.join(st.PROCESSED_DIR, "scaler_7d.pt"), map_location="cpu")
    n_shift, n_scale = new["shift"].view(7).double(), new["scale"].view(7).double()
    if args.old_scaler and os.path.exists(args.old_scaler):
        old = torch.load(args.old_scaler, map_location="cpu")
        o_shift, o_scale = old["shift"].view(7).double(), old["scale"].view(7).double()
        src = args.old_scaler
    else:
        o_shift = torch.tensor([OLD_SCALER[n][0] for n in DIM_NAMES], dtype=torch.float64)
        o_scale = torch.tensor([OLD_SCALER[n][1] for n in DIM_NAMES], dtype=torch.float64)
        src = "bang A1 trong RESULTS_LEDGER"
    print(f"    doi chieu voi: {src}")
    print(f"    {'chieu':8}{'shift cu':>12}{'shift moi':>12}{'rel':>10}{'scale cu':>12}{'scale moi':>12}{'rel':>10}")
    k1_ok = True
    for i, nm in enumerate(DIM_NAMES):
        r_sh = abs(n_shift[i] - o_shift[i]).item() / max(abs(o_shift[i]).item(), 1e-12)
        r_sc = abs(n_scale[i] - o_scale[i]).item() / max(abs(o_scale[i]).item(), 1e-12)
        bad = (r_sh > REL_TOL) or (r_sc > REL_TOL)
        k1_ok &= not bad
        print(f"    {nm:8}{o_shift[i]:12.6f}{n_shift[i]:12.6f}{r_sh:10.2e}"
              f"{o_scale[i]:12.6f}{n_scale[i]:12.6f}{r_sc:10.2e}" + ("   <-- LECH" if bad else ""))
    print(f"    => K1 {'DAT' if k1_ok else 'KHONG DAT'}")
    if not k1_ok:
        fails.append("K1")

    # ---------------------------------------------------------------- K2: vocab
    print("\n" + "=" * 90)
    print(f"K2: VOCAB trong [{VOCAB_LO}, {VOCAB_HI}]")
    print("=" * 90)
    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    V = len(vocab)
    if args.old_vocab and os.path.exists(args.old_vocab):
        with open(args.old_vocab, "rb") as f:
            print(f"    vocab CU = {len(pickle.load(f))}")
    print(f"    vocab MOI = {V}")
    n_iso = sum(1 for t in vocab if "*]" in t and any(c.isdigit() for c in t.split("*]")[0].split("[")[-1]))
    print(f"    token con chi so trong dummy (phai ~0): {n_iso}")
    k2_ok = VOCAB_LO <= V <= VOCAB_HI
    print(f"    => K2 {'DAT' if k2_ok else 'KHONG DAT'}")
    if not k2_ok:
        fails.append("K2")

    # ---------------------------------------------------------------- K3 + K4
    print("\n" + "=" * 90)
    print("K3: so complex frame_stable  |  K4: truong moi")
    print("=" * 90)
    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p
                 for p in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))}

    k4_bad, n_items, n_order_ok, n_files = 0, 0, 0, 0
    k3_ok = True
    for split in ["train", "valid", "test"]:
        n_stable = 0
        for e in splits.get(split, []):
            pid = e[2].split("/")[0]
            path = pkl_by_id.get(pid)
            if path is None:
                continue
            with open(path, "rb") as f:
                d = pickle.load(f)
            if not d.get("frame_stable", False):
                continue
            if any(not it.get("frag_frame_stable", True) for it in d["sequence"]):
                continue
            n_stable += 1
            n_files += 1
            n_order_ok += int(d.get("frag_order_mode") == "frag2seq_canonical_appearance")
            for it in d["sequence"]:
                n_items += 1
                lc, an = it.get("local_coords"), it.get("atomic_nums")
                if lc is None or an is None or np.asarray(lc).ndim != 2 \
                        or np.asarray(lc).shape[1] != 3 or np.asarray(lc).shape[0] != len(np.asarray(an)):
                    k4_bad += 1
        exp = EXPECTED_STABLE[split]
        ok = n_stable == exp
        k3_ok &= ok
        print(f"    {split:6}: {n_stable:6d}  (ky vong {exp})" + ("" if ok else "   <-- LECH"))
    print(f"    => K3 {'DAT' if k3_ok else 'KHONG DAT'}")
    if not k3_ok:
        fails.append("K3")

    pct_order = n_order_ok / max(n_files, 1) * 100
    print(f"\n    item sequence kiem: {n_items}  |  thieu/sai truong moi: {k4_bad}")
    print(f"    frag_order_mode == 'frag2seq_canonical_appearance': {n_order_ok}/{n_files} ({pct_order:.2f}%)")
    k4_ok = (k4_bad == 0) and (pct_order > 99.0)
    print(f"    => K4 {'DAT' if k4_ok else 'KHONG DAT'}")
    if not k4_ok:
        fails.append("K4")

    print("\n" + "=" * 90)
    if not fails:
        print("TAT CA K1-K4 DAT => S1-S4 khong lam doi nhan. Chay tiep build_fragment_dict.py (A1).")
    else:
        print(f"KHONG DAT: {', '.join(fails)}")
        if "K1" in fails or "K3" in fails:
            print("    K1 hoac K3 hong => S1-S4 DA LAM DOI NHAN. DUNG, tim loi truoc khi di tiep (37.7).")
        else:
            print("    K2/K4 hong nhung K1+K3 dat => nhan khong doi; bao cao roi quyet dinh.")
    print("=" * 90)


if __name__ == "__main__":
    main()
