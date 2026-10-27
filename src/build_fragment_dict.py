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
K_SAMPLE = 50                 # Muc 39.4: so lan xuat hien lay mau de chon MEDOID


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

    # Muc 38: KHONG loc theo `frame_stable` (do on dinh cua frame HOC). Template fragment nam trong
    # FRAME FRAGMENT, doc lap hoan toan voi frame hoc ⇒ loc theo frame_stable chi lam mat do phu vo co
    # (`build_global_vocab` cung khong loc). Chi bo item co `frag_frame_stable=False`: voi chung
    # `_get_local_frame` roi ve eye(3) nen local_coords nam trong huong THE GIOI, khong phai template.
    frag_dict, spreads, n_items, n_missing_field, n_unstable_frag = {}, [], 0, 0, 0
    for k, pid in enumerate(train_ids):
        path = pkl_by_id.get(pid)
        if path is None:
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        for item in d["sequence"]:
            if not item.get("frag_frame_stable", True):
                n_unstable_frag += 1
                continue
            if "local_coords" not in item or "atomic_nums" not in item:
                n_missing_field += 1
                continue
            n_items += 1
            smi = item["smiles"]
            local = np.asarray(item["local_coords"], dtype=np.float64)
            nums = np.asarray(item["atomic_nums"])
            if smi not in frag_dict:
                frag_dict[smi] = {"atomic_nums": nums, "local": local, "n_seen": 1,
                                   "n_atoms": int(local.shape[0]), "_samples": [local]}
            else:
                e = frag_dict[smi]
                e["n_seen"] += 1
                if e["local"].shape == local.shape:
                    spreads.append((smi, kabsch_rmsd(local, e["local"])))
                    # Muc 39.4: giu toi da K_SAMPLE lan xuat hien de chon medoid o cuoi.
                    if len(e["_samples"]) < K_SAMPLE:
                        e["_samples"].append(local)
        if (k + 1) % 2000 == 0:
            print(f"  doc {k + 1}/{len(train_ids)}  (|dict| = {len(frag_dict)})", flush=True)

    if n_missing_field and not frag_dict:
        raise SystemExit("[!] pkl KHONG co truong local_coords/atomic_nums. Phai chay lai step0 "
                         "(ban sua Muc 37.4) truoc khi dung script nay.")

    print(f"\n[*] token fragment doc duoc: {n_items}  (thieu truong: {n_missing_field}; "
          f"bo vi frag_frame_stable=False: {n_unstable_frag})")

    # ---------------------------------------------------------------- Muc 39.4: chon MEDOID
    # Lua chon "conformer DAU TIEN gap" la tuy tien: neu lan dau gap roi dung vao conformer hong
    # (Muc 39.3 -- co truong hop span > 12 A tren fragment 3 nguyen tu) thi MOI phan tu sinh ra chua
    # fragment do deu co hinh hoc vo ly. Medoid = conformer cuc tieu hoa tong RMSD toi cac conformer
    # con lai: ben voi ngoai lai, va luon la mot conformer CO THAT nen hinh hoc hoa hoc hop le.
    print(f"[*] chon conformer MEDOID (toi da K={K_SAMPLE} mau moi SMILES) ...", flush=True)
    n_changed, medoid_spreads, spans = 0, [], []
    for smi, e in frag_dict.items():
        S = e.pop("_samples")
        m = len(S)
        if m > 1:
            D = np.zeros((m, m))
            for i in range(m):
                for j in range(i + 1, m):
                    D[i, j] = D[j, i] = kabsch_rmsd(S[i], S[j])
            D = np.nan_to_num(D, nan=0.0)
            best = int(np.argmin(D.sum(axis=1)))
            e["medoid_spread"] = float(D[best].mean())
            if best != 0:
                d0 = kabsch_rmsd(S[0], S[best])
                if d0 == d0 and d0 > 0.5:
                    n_changed += 1
                e["local"] = S[best]
            medoid_spreads.append(e["medoid_spread"])
        else:
            e["medoid_spread"] = 0.0
        e["n_sample"] = m
        loc = e["local"]
        e["span"] = float(np.linalg.norm(loc - loc.mean(0), axis=-1).max() * 2)
        spans.append((smi, e["span"], e["n_atoms"]))
    print(f"    so fragment ma medoid KHAC conformer dau > 0,5 A: {n_changed}/{len(frag_dict)} "
          f"({n_changed / max(len(frag_dict), 1) * 100:.2f}%)")
    if medoid_spreads:
        ms = np.array(medoid_spreads)
        print(f"    medoid_spread (A): median={np.median(ms):.4f}  p90={np.percentile(ms, 90):.4f}  "
              f"p99={np.percentile(ms, 99):.4f}")
    # ngoai lai hinh hoc: span lon bat thuong so voi so nguyen tu (~1,5 A moi lien ket, rat rong rai)
    out = [(sm, sp, na) for sm, sp, na in spans if sp > 1.8 * na]
    print(f"    ngoai lai hinh hoc trong TEMPLATE da chon (span > 1,8 x n_atom): {len(out)}")
    for sm, sp, na in sorted(out, key=lambda x: -x[1])[:5]:
        print(f"        {sm[:40]:42} span={sp:6.2f} A  n_atom={na}")
    print(f"[*] so SMILES duy nhat trong tu dien: {len(frag_dict)}")

    if spreads:
        vals = np.array([v for _, v in spreads], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        q = np.percentile(vals, [50, 90, 99]) if vals.size else [float("nan")] * 3
        print(f"[*] rmsd_spread (A) tren {vals.size} lan lap: median={q[0]:.4f}  p90={q[1]:.4f}  p99={q[2]:.4f}")
        worst = sorted(spreads, key=lambda x: -(x[1] if x[1] == x[1] else -1))[:8]
        print("    8 lan lech nhat (smiles, rmsd, n_atom) -- n_atom nho + rmsd lon => nhieu kha nang la")
        print("    hoan vi thu tu nguyen tu cua fragment DOI XUNG, khong phai khac conformer that:")
        for sm, v in worst:
            print(f"        {sm[:40]:42} rmsd={v:7.3f}  n_atom={frag_dict[sm]['n_atoms']}")
        big = [(sm, v) for sm, v in spreads if v == v and v > 1.0]
        small_big = [1 for sm, v in big if frag_dict[sm]["n_atoms"] <= 5]
        print(f"    so lan rmsd > 1,0 A: {len(big)}/{vals.size} ({len(big) / max(vals.size, 1) * 100:.2f}%); "
              f"trong do fragment <= 5 nguyen tu: {len(small_big)}")
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
        if any(not it.get("frag_frame_stable", True) for it in d["sequence"]):
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

    # Muc 38: do phu TU VUNG -- quan trong hon do phu valid, vi luc SINH mo hinh chi phat ra token
    # trong tu vung; token nao khong co template thi khong dat pose duoc.
    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    specials = {"[SOS]", "[UNK]", "[PAD]", "[EOS]"}
    vtoks = [t for t in vocab if t not in specials]
    missing = [t for t in vtoks if t not in frag_dict]
    print(f"[*] do phu TU VUNG: {len(vtoks) - len(missing)}/{len(vtoks)} = "
          f"{(len(vtoks) - len(missing)) / max(len(vtoks), 1):.4f}  (thieu {len(missing)} token)")
    if missing:
        print(f"    vi du token khong co template: {missing[:5]}")

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
