"""
diag_fragment_graph.py -- do dac tinh DO THI FRAGMENT truoc khi thiet ke kenh lien ket (huong (b),
Muc 43.7). Khong GPU, khong model, khong sua du lieu.

Ba cau hoi quyet dinh cau truc cua kenh lien ket (R16 muc 2: nguong suy tu dac tinh du lieu):
  Q1 Do thi fragment co luon la CAY khong? (`step0` chi cat lien ket **khong trong vong** nen ve ly
     thuyet phai la cay; phai DO, khong duoc gia dinh.)
  Q2 Trong thu tu canonical cua Frag2Seq, CHA cua moi fragment co luon xuat hien TRUOC no khong?
     Neu co, chi can du doan MOT chi so cha cho moi vi tri t >= 1 la du dinh nghia ca cay.
  Q3 Phan phoi so DIEM GAN (dummy) moi fragment -- quyet dinh kich co dau ra cua head du doan
     diem gan.
Kem: so fragment/phan tu, va so vi tri cha co the co (de biet head con tro phai rong bao nhieu).
"""
import os
import sys
import pickle
import socket
import argparse
import subprocess
from collections import Counter

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)


def analyse(mol, Chem, break_fn):
    """Tra ve dict dac tinh cho MOT phan tu, hoac None neu khong cat duoc."""
    bonds = break_fn(mol)
    if not bonds:
        return None
    n_orig = mol.GetNumAtoms()
    fm = Chem.FragmentOnBonds(mol, bonds, dummyLabels=[(0, 0)] * len(bonds))
    groups = Chem.GetMolFrags(fm, asMols=False)
    frags = Chem.GetMolFrags(fm, asMols=True, sanitizeFrags=False)
    k = len(groups)

    # chi so nguyen tu goc -> chi so fragment
    owner = {}
    for fi, idxs in enumerate(groups):
        for a in idxs:
            if a < n_orig:
                owner[a] = fi

    # thu tu canonical cua Frag2Seq (giong step0 sau N1): theo nguyen tu xep hang dau tien
    ranks = list(Chem.CanonicalRankAtoms(mol))
    key = []
    for idxs in groups:
        real = [ranks[a] for a in idxs if a < n_orig]
        key.append(min(real) if real else 10 ** 9)
    order = sorted(range(k), key=lambda i: (key[i], Chem.MolToSmiles(frags[i])))
    pos = {fi: p for p, fi in enumerate(order)}   # chi so fragment -> vi tri trong chuoi

    # canh cua do thi fragment, theo VI TRI trong chuoi
    edges = []
    for bi in bonds:
        b = mol.GetBondWithIdx(bi)
        a1, a2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if a1 in owner and a2 in owner and owner[a1] != owner[a2]:
            edges.append((pos[owner[a1]], pos[owner[a2]]))

    # Q1: cay? (lien thong va |E| = |V| - 1)
    adj = {i: set() for i in range(k)}
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v); stack.append(v)
    is_tree = (len(seen) == k) and (len(edges) == k - 1)

    # Q2: moi vi tri t >= 1 co hang xom o vi tri NHO HON khong?
    has_earlier = [any(v < t for v in adj[t]) for t in range(1, k)]

    # Q3: so dummy moi fragment
    n_dummy = [sum(1 for a in f.GetAtoms() if a.GetAtomicNum() == 0) for f in frags]

    return {"k": k, "is_tree": is_tree, "n_edge": len(edges),
            "all_parent_earlier": all(has_earlier) if has_earlier else True,
            "n_no_earlier": sum(1 for x in has_earlier if not x),
            "n_dummy": n_dummy, "deg": [len(adj[i]) for i in range(k)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMG_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}\nSEED: 1234  (khong dung ngau nhien)")
    print(f"device: khong dung  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    from step0_data_prep import SAMGDataPreprocessor
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    entries = splits.get(args.split, [])[: args.n]

    res, n_fail = [], 0
    for i, e in enumerate(entries):
        path = os.path.join(st.DATASET_DIR, e[2])
        if not os.path.exists(path):
            continue
        m = Chem.SDMolSupplier(path)[0]
        if m is None:
            n_fail += 1
            continue
        try:
            r = analyse(m, Chem, SAMGDataPreprocessor.bonds_to_break)
        except Exception:
            n_fail += 1
            continue
        if r:
            res.append(r)
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{len(entries)}", flush=True)

    n = len(res)
    print(f"\n[*] phan tich {n} phan tu (split={args.split}, doc loi/khong cat duoc: {n_fail})")

    print("\n" + "=" * 86)
    print("Q1: DO THI FRAGMENT CO LA CAY KHONG?")
    print("=" * 86)
    n_tree = sum(1 for r in res if r["is_tree"])
    print(f"    la cay: {n_tree}/{n} = {n_tree / max(n, 1):.4f}")
    print("    (ly thuyet: step0 chi cat lien ket KHONG TRONG VONG ⇒ phai la cay. Khac 1,0 = co gia dinh sai.)")

    print("\n" + "=" * 86)
    print("Q2: CHA CO LUON XUAT HIEN TRUOC TRONG THU TU CANONICAL KHONG?")
    print("=" * 86)
    n_ok = sum(1 for r in res if r["all_parent_earlier"])
    tot_bad = sum(r["n_no_earlier"] for r in res)
    tot_pos = sum(r["k"] - 1 for r in res)
    print(f"    phan tu ma MOI fragment t>=1 deu co hang xom o vi tri nho hon: {n_ok}/{n} = {n_ok / max(n, 1):.4f}")
    print(f"    so vi tri vi pham: {tot_bad}/{tot_pos} = {tot_bad / max(tot_pos, 1):.4f}")
    print("    (DAT ⇒ chi can du doan MOT chi so cha cho moi t>=1 la du dinh nghia ca cay.)")

    print("\n" + "=" * 86)
    print("Q3: KICH CO CAN THIET CHO CAC HEAD")
    print("=" * 86)
    dummies = [d for r in res for d in r["n_dummy"]]
    ks = [r["k"] for r in res]
    degs = [d for r in res for d in r["deg"]]
    cd = Counter(dummies)
    print(f"    so dummy / fragment: " + "  ".join(f"{v}:{cd[v]}" for v in sorted(cd)))
    print(f"        max={max(dummies)}  p99={int(np.percentile(dummies, 99))}  mean={np.mean(dummies):.2f}")
    print(f"    so fragment / phan tu: max={max(ks)}  p99={int(np.percentile(ks, 99))}  mean={np.mean(ks):.2f}")
    print(f"    bac trong do thi fragment: max={max(degs)}  p99={int(np.percentile(degs, 99))}")
    print(f"\n    => head 'diem gan' can {max(dummies)} lop (p99 = {int(np.percentile(dummies, 99))})")
    print(f"    => head 'cha' la con tro tren toi da {max(ks) - 1} vi tri truoc do")


if __name__ == "__main__":
    main()
