"""
check_assembly_ceiling.py -- PLAN Rev 20 A2b (buoc chan truoc khi sinh mau); RESEARCH_CONTEXT 39, 40.

CAU HOI: truoc khi do chat luong phan tu do MO HINH sinh ra, duong LAP RAP mat bao nhieu?
Ba hang, deu KHONG dung mo hinh, do tren cung tap complex:

  1. `reference`    -- doc thang SDF goc              -> moc that
  2. `gt_exact`     -- NHAN THAT + `local_coords` theo TUNG LAN XUAT HIEN -> reconstruct
                       => do RIENG buoc lap rap (suy lien ket tu hinh hoc). Toa do CHINH XAC
                          (Muc 39.2: tai dung RMSD = 0,0000), nen moi mat mat la cua lap rap.
  3. `gt_template`  -- NHAN THAT + template MEDOID tu `fragment_dict.pkl` -> reconstruct
                       => them chi phi cua xap xi mot-conformer (Muc 39.4). Day la tran tren
                          THAT SU cua mo hinh: khong mo hinh nao vuot qua duoc hang nay.

Dung lai code cua Apo2Mol (R14): `utils/reconstruct.py:455 reconstruct_from_generated` (OpenBabel bond
perception -- chuan cua nganh, `eval_split.py:122`), `utils/evaluation/scoring_func.py` (obey_lipinski,
get_logp), `utils/evaluation/sascorer.py` (compute_sa_score). Chi so theo Apo2Mol (WORKING_RULES G5).

TIEU CHI (ghi cung TRUOC khi chay, plan Rev 20 A2b):
  T1 `recon_success(gt_exact) >= 0,90`  => duong lap rap dung duoc, di tiep sinh mau.
     `< 0,70`                           => LAP RAP la nut that; sua truoc, KHONG sinh mau.
     giua                               => bao cao, quyet dinh.
     (Tu toa do CHINH XAC, suy lien ket la bai toan da giai; moi that bai la mat mat thuan cua phuong
      phap. 0,90 la quy uoc du an -- R16 muc 3 -- va so tho duoc bao cao du the nao.)
  T2 `|mean QED(gt_exact) - mean QED(reference)| <= 0,05` => lap rap giu duoc hoa hoc. (quy uoc du an)
  T3 Bao cao `gt_template` so voi `gt_exact`: chenh lech = gia cua xap xi mot-conformer. Khong nguong.
"""
import os
import sys
import glob
import math
import json
import pickle
import socket
import argparse
import subprocess

import types
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

# SAMG can `utils.data` (src/utils), Apo2Mol can `utils.reconstruct` + `utils.evaluation.*`
# (Apo2Mol/utils). Ca hai thu muc deu KHONG co __init__.py. Chi them Apo2Mol vao sys.path thi
# KHONG du -- da thu, bao `No module named 'utils.reconstruct'`: namespace package `utils` duoc
# phan giai tu mot muc sys.path va khong gop them. Nen dung goi `utils` TUONG MINH voi __path__
# tro toi CA HAI cay, dat TRUOC moi import khac de khong bi cache ban nao chen truoc.
_UTILS_PATHS = [os.path.join(SAMG_ROOT, "src", "utils"), os.path.join(SAMG_ROOT, "Apo2Mol", "utils")]
if "utils" not in sys.modules:
    _m = types.ModuleType("utils")
    _m.__path__ = _UTILS_PATHS
    sys.modules["utils"] = _m
else:
    for _p in _UTILS_PATHS:
        if _p not in list(getattr(sys.modules["utils"], "__path__", [])):
            sys.modules["utils"].__path__.append(_p)

RECON_OK_HI, RECON_OK_LO = 0.90, 0.70
QED_TOL = 0.05


def spherical_to_vec(d, theta, phi):
    return np.array([d * math.sin(theta) * math.cos(phi),
                     d * math.sin(theta) * math.sin(phi),
                     d * math.cos(theta)], dtype=np.float64)


def place_fragments(pkl_data, frag_dict=None):
    """Nghich dao 37-pre.3. frag_dict=None => dung local_coords theo tung lan xuat hien (gt_exact);
    khac None => dung template medoid theo SMILES (gt_template). Tra (xyz, z) hoac None neu thieu template."""
    from scipy.spatial.transform import Rotation as Rot
    R_m_to_w = np.asarray(pkl_data["pocket_R"], dtype=np.float64)
    t_m_to_w = np.asarray(pkl_data["pocket_t"], dtype=np.float64).reshape(3)
    xyz_all, z_all = [], []
    for item in pkl_data["sequence"]:
        if frag_dict is None:
            local = np.asarray(item["local_coords"], dtype=np.float64)
            z = np.asarray(item["atomic_nums"]).reshape(-1)
        else:
            e = frag_dict.get(item["smiles"])
            if e is None:
                return None
            local = np.asarray(e["local"], dtype=np.float64)
            z = np.asarray(e["atomic_nums"]).reshape(-1)
        d, theta, phi, qw, qx, qy, qz = item["spatial_tokens"]
        center_world = R_m_to_w @ spherical_to_vec(d, theta, phi) + t_m_to_w
        R_g_to_w = R_m_to_w @ Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        world = (R_g_to_w @ (local - local.mean(axis=0)).T).T + center_world
        keep = z != 0                                   # bo nguyen tu dummy
        xyz_all.append(world[keep])
        z_all.append(z[keep])
    if not xyz_all:
        return None
    return np.concatenate(xyz_all, 0), np.concatenate(z_all, 0)


def chem_metrics(mol):
    """QED, SA, logP, Lipinski -- dung ham cua Apo2Mol (R14)."""
    from rdkit.Chem.QED import qed
    from utils.evaluation.sascorer import compute_sa_score
    from utils.evaluation.scoring_func import obey_lipinski, get_logp
    return {"QED": float(qed(mol)), "SA": float(compute_sa_score(mol)),
            "logP": float(get_logp(mol)), "Lipinski": float(obey_lipinski(mol))}


def summarize(name, rows, n_attempt):
    ok = [r for r in rows if r is not None]
    rate = len(ok) / max(n_attempt, 1)
    print(f"\n  [{name}]  thanh cong {len(ok)}/{n_attempt} = {rate:.4f}")
    if not ok:
        return {"recon_success": rate, "n": 0}
    out = {"recon_success": rate, "n": len(ok)}
    print(f"      {'chi so':10}{'mean':>10}{'median':>10}")
    for k in ["QED", "SA", "logP", "Lipinski"]:
        v = np.array([r[k] for r in ok], dtype=np.float64)
        out[f"{k}_mean"], out[f"{k}_median"] = float(v.mean()), float(np.median(v))
        print(f"      {k:10}{v.mean():10.4f}{np.median(v):10.4f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid", "test"], default="test")
    ap.add_argument("--n", type=int, default=0, help="0 = het split")
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMG_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")
    print("SEED: 1234  (khong dung ngau nhien)")
    print(f"device: khong dung (script khong GPU)  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from utils.reconstruct import reconstruct_from_generated

    with open(os.path.join(st.PROCESSED_DIR, "fragment_dict.pkl"), "rb") as f:
        frag_dict = pickle.load(f)
    print(f"[*] fragment_dict: {len(frag_dict)} SMILES")

    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p
                 for p in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))}

    rows_ref, rows_exact, rows_tmpl = [], [], []
    n_attempt, n_no_template = 0, 0
    for e in splits.get(args.split, []):
        if args.n and n_attempt >= args.n:
            break
        pid = e[2].split("/")[0]
        path = pkl_by_id.get(pid)
        if path is None:
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not d.get("frame_stable", False) or "pocket_R" not in d:
            continue
        if any(not it.get("frag_frame_stable", True) for it in d["sequence"]):
            continue
        n_attempt += 1

        # 1. reference: doc thang SDF
        try:
            m = Chem.SDMolSupplier(os.path.join(st.DATASET_DIR, e[2]))[0]
            rows_ref.append(chem_metrics(m) if m is not None else None)
        except Exception:
            rows_ref.append(None)

        # 2. gt_exact  /  3. gt_template
        for frag_dict_arg, bucket in ((None, rows_exact), (frag_dict, rows_tmpl)):
            placed = place_fragments(d, frag_dict_arg)
            if placed is None:
                if frag_dict_arg is not None:
                    n_no_template += 1
                bucket.append(None)
                continue
            xyz, z = placed
            try:
                m = reconstruct_from_generated(xyz.tolist(), z.tolist())
                bucket.append(chem_metrics(m) if m is not None else None)
            except Exception:
                bucket.append(None)

        if n_attempt % 25 == 0:
            print(f"  ... {n_attempt} complex", flush=True)

    print("\n" + "=" * 90)
    print(f"TRAN LAP RAP -- split={args.split}, {n_attempt} complex")
    print("=" * 90)
    s_ref = summarize("reference  (SDF goc)", rows_ref, n_attempt)
    s_ex = summarize("gt_exact   (nhan that + local_coords tung lan)", rows_exact, n_attempt)
    s_tm = summarize("gt_template(nhan that + template medoid)", rows_tmpl, n_attempt)
    print(f"\n  [gt_template] so complex thieu template (fragment ngoai tu dien train): {n_no_template}")

    print("\n" + "=" * 90)
    print("TIEU CHI (plan Rev 20 A2b)")
    print("=" * 90)
    r = s_ex["recon_success"]
    print(f"    T1 recon_success(gt_exact) = {r:.4f}")
    if r >= RECON_OK_HI:
        print("       => DAT: duong lap rap dung duoc. Di tiep sinh mau.")
    elif r < RECON_OK_LO:
        print("       => NUT THAT: lap rap hong tu toa do CHINH XAC. Sua truoc, KHONG sinh mau.")
    else:
        print("       => TRUNG GIAN: bao cao, quyet dinh truoc khi sinh mau.")
    if s_ex.get("n") and s_ref.get("n"):
        dq = abs(s_ex["QED_mean"] - s_ref["QED_mean"])
        print(f"    T2 |QED(gt_exact) - QED(reference)| = {dq:.4f}  (nguong {QED_TOL})"
              f"  => {'DAT' if dq <= QED_TOL else 'KHONG DAT: lap rap lam doi hoa hoc'}")
    if s_tm.get("n") and s_ex.get("n"):
        print(f"    T3 gia cua xap xi mot-conformer: recon_success {s_ex['recon_success']:.4f} -> "
              f"{s_tm['recon_success']:.4f}; QED {s_ex['QED_mean']:.4f} -> {s_tm['QED_mean']:.4f}")
        print(f"       (gt_template la TRAN TREN that su cua mo hinh -- khong mo hinh nao vuot qua)")

    out = {"split": args.split, "n_attempt": n_attempt, "head": head,
           "reference": s_ref, "gt_exact": s_ex, "gt_template": s_tm}
    dst = os.path.join(SAMG_ROOT, "logs", f"assembly_ceiling_{args.split}.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] da luu {dst}")


if __name__ == "__main__":
    main()
