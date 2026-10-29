"""
check_assembly_ceiling.py -- PLAN Rev 20 A2b; RESEARCH_CONTEXT 39, 40.1.

CAU HOI: truoc khi do chat luong phan tu do MO HINH sinh ra, duong LAP RAP tu no mat bao nhieu?
Ba hang, deu KHONG dung mo hinh, tren cung tap:

  1. `reference`    -- doc thang SDF goc                                   -> moc that
  2. `gt_exact`     -- nhan that + `local_coords` theo TUNG LAN XUAT HIEN  -> do RIENG buoc lap rap
                       (toa do chinh xac: Muc 39.2 tai dung RMSD = 0,0000)
  3. `gt_template`  -- nhan that + template MEDOID cua fragment_dict       -> + xap xi mot-conformer

**So sanh phai tren CUNG TAP PHAN TU.** `gt_template` that bai o moi phan tu chua fragment ngoai tu
dien train; neu lay trung binh cua no tren tap con roi so voi `reference` tren tap day du thi chenh
lech co the chi la "hai tap phan tu khac nhau", khong phai gia cua template. Nen bao cao hai lan:
tren toan bo, va tren **TAP CHUNG** (complex ma ca ba hang deu thanh cong).

**Do phu la mot so RIENG, khong phai that bai cua lap rap.** Tu vung dung tu TRAIN, nen phan tu test
chua fragment la khong co template -- nhung mo hinh **cung khong bao gio sinh ra** fragment do
(do phu TU VUNG = 100%, Muc 38.3). Vi vay tran tren cua mo hinh la `gt_template` **tren tap duoc phu**,
con ty le khong duoc phu la mot gioi han rieng: phan test ma tu vung cua mo hinh khong voi toi.

Lap rap: `src/assemble.py` cua du an (khong dung Apo2Mol -- server khong co repo do; va ham cua ho
vut bo lien ket roi suy lai bang OpenBabel, trong khi ta DA BIET lien ket trong tung fragment).
Chi so hoa hoc: QED (rdkit), SA (sascorer chinh thuc di kem RDKit Contrib -- Ertl & Schuffenhauer
2009, cung nguon ma Apo2Mol chep lai), logP (Crippen), Lipinski (5 quy tac, dinh nghia nhu
Apo2Mol `utils/evaluation/scoring_func.py:obey_lipinski` -- WORKING_RULES G5).

TIEU CHI (ghi cung TRUOC khi chay, Muc 40.1):
  T1 `recon_success(gt_exact) >= 0,90` => duong lap rap dung duoc, di tiep sinh mau.
     `< 0,70`                          => LAP RAP la nut that; sua truoc, KHONG sinh mau.
  T2 `|mean QED(gt_exact) - mean QED(reference)| <= 0,05` => lap rap giu duoc hoa hoc.
  T3 [SUA, Muc 41.1] QED/SA/logP/Lipinski la **mo ta 2D**: chi phu thuoc DO THI phan tu, khong phu
     thuoc toa do 3D. gt_exact va gt_template cho CUNG do thi (lap rap thanh cong, cung lien ket)
     ⇒ bon chi so do **buoc phai** bang nhau; tieu chi cu khong co kha nang phat hien gi (vi pham R5).
     Thay bang **RMSD 3D tuyet doi** giua hai ban (cung thu tu nguyen tu, cung frame the gioi) --
     day moi la gia that cua xap xi mot-conformer, va la cai anh huong toi Vina.
"""
import os
import sys
import glob
import json
import math
import pickle
import socket
import argparse
import subprocess
from collections import Counter

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

RECON_OK_HI, RECON_OK_LO = 0.90, 0.70
QED_TOL = 0.05


def _sa_scorer():
    """sascorer di kem RDKit (Contrib/SA_Score) -- ban chinh thuc cua Ertl & Schuffenhauer (2009)."""
    from rdkit.Chem import RDConfig
    p = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if p not in sys.path:
        sys.path.append(p)
    import sascorer
    return sascorer.calculateScore


def chem_metrics(mol, sa_fn):
    """Lipinski theo dung 5 quy tac cua Apo2Mol `scoring_func.py:obey_lipinski` (MW<500, HBD<=5,
    HBA<=10, logP trong [-2,5], so lien ket xoay duoc <=10)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    from rdkit.Chem.QED import qed
    logp = float(Crippen.MolLogP(mol))
    lip = sum(int(b) for b in [
        Descriptors.ExactMolWt(mol) < 500,
        Lipinski.NumHDonors(mol) <= 5,
        Lipinski.NumHAcceptors(mol) <= 10,
        (-2 <= logp <= 5),
        rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10,
    ])
    return {"QED": float(qed(mol)), "SA": float(sa_fn(mol)), "logP": logp, "Lipinski": float(lip)}


def spherical_to_vec(d, theta, phi):
    return np.array([d * math.sin(theta) * math.cos(phi),
                     d * math.sin(theta) * math.sin(phi),
                     d * math.cos(theta)], dtype=np.float64)


def build_molecule(pkl_data, frag_dict=None):
    """Nghich dao 37-pre.3 roi lap rap. frag_dict=None => local_coords tung lan xuat hien (gt_exact);
    khac None => template medoid (gt_template). Tra (mol, diag) hoac nem AssemblyError/KeyError."""
    from scipy.spatial.transform import Rotation as Rot
    import assemble as asm
    R_m_to_w = np.asarray(pkl_data["pocket_R"], dtype=np.float64)
    t_m_to_w = np.asarray(pkl_data["pocket_t"], dtype=np.float64).reshape(3)
    placed = []
    for item in pkl_data["sequence"]:
        smi = item["smiles"]
        if frag_dict is None:
            local, z = np.asarray(item["local_coords"], np.float64), np.asarray(item["atomic_nums"])
        else:
            e = frag_dict.get(smi)
            if e is None:
                raise KeyError(smi)
            local, z = np.asarray(e["local"], np.float64), np.asarray(e["atomic_nums"])
        d, theta, phi, qw, qx, qy, qz = item["spatial_tokens"]
        center = R_m_to_w @ spherical_to_vec(d, theta, phi) + t_m_to_w
        R_g_to_w = R_m_to_w @ Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        placed.append(asm.place(asm.fragment_mol_with_coords(smi, local, z.reshape(-1)), R_g_to_w, center))
    return asm.assemble(placed)


def summarize(name, rows, n_attempt, errs=None, idx=None):
    """idx=None => tinh tren moi hang thanh cong. idx=tap chi so => chi tinh tren cac chi so do."""
    sel = range(len(rows)) if idx is None else sorted(idx)
    ok = [rows[i] for i in sel if rows[i] is not None]
    rate = len(ok) / max(n_attempt, 1)
    print(f"\n  [{name}]  thanh cong {len(ok)}/{n_attempt} = {rate:.4f}")
    if errs:
        for k, v in Counter(errs).most_common(4):
            print(f"      loi: {k[:72]:74} x{v}")
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
    import assemble as asm
    sa_fn = _sa_scorer()

    with open(os.path.join(st.PROCESSED_DIR, "fragment_dict.pkl"), "rb") as f:
        frag_dict = pickle.load(f)
    print(f"[*] fragment_dict: {len(frag_dict)} SMILES")

    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p
                 for p in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))}

    rows = {"reference": [], "gt_exact": [], "gt_template": []}
    errs = {"gt_exact": [], "gt_template": []}
    diags = {"gt_exact": [], "gt_template": []}
    covered = set()          # complex ma MOI fragment deu co trong tu dien train
    rmsd_tmpl = []           # T3: RMSD 3D gt_exact vs gt_template
    n_attempt = 0
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
        if all(it["smiles"] in frag_dict for it in d["sequence"]):
            covered.add(n_attempt - 1)

        try:
            m = Chem.SDMolSupplier(os.path.join(st.DATASET_DIR, e[2]))[0]
            rows["reference"].append(chem_metrics(m, sa_fn) if m is not None else None)
        except Exception:
            rows["reference"].append(None)

        mols = {}
        for key, fd in (("gt_exact", None), ("gt_template", frag_dict)):
            try:
                mol, diag = build_molecule(d, fd)
                mols[key] = mol
                rows[key].append(chem_metrics(mol, sa_fn))
                diags[key].append(diag)
            except KeyError as exc:
                rows[key].append(None)
                errs[key].append(f"thieu template cho {exc}")
            except Exception as exc:
                rows[key].append(None)
                errs[key].append(f"{type(exc).__name__}: {exc}")
        # T3 (Muc 41.1): RMSD 3D TUYET DOI giua hai ban -- cung thu tu nguyen tu (cung duong dung),
        # cung frame the gioi. KHONG can chinh Kabsch: phan tu phai nam DUNG CHO trong hoc, nen do
        # lech tuyet doi moi la dai luong co nghia.
        if "gt_exact" in mols and "gt_template" in mols:
            A, B = mols["gt_exact"], mols["gt_template"]
            if A.GetNumAtoms() == B.GetNumAtoms():
                PA, PB = A.GetConformer().GetPositions(), B.GetConformer().GetPositions()
                rmsd_tmpl.append(float(np.sqrt(((PA - PB) ** 2).sum(axis=1).mean())))

        if n_attempt % 25 == 0:
            print(f"  ... {n_attempt} complex", flush=True)

    print("\n" + "=" * 92)
    print(f"TRAN LAP RAP -- split={args.split}, {n_attempt} complex")
    print("=" * 92)
    s_ref = summarize("reference   (SDF goc)", rows["reference"], n_attempt)
    s_ex = summarize("gt_exact    (nhan that + local_coords tung lan)", rows["gt_exact"], n_attempt, errs["gt_exact"])
    s_tm = summarize("gt_template (nhan that + template medoid)", rows["gt_template"], n_attempt, errs["gt_template"])

    # --- DO PHU: mot so RIENG, khong phai that bai cua lap rap -------------------------------
    cov = len(covered) / max(n_attempt, 1)
    print("\n" + "-" * 92)
    print(f"  DO PHU TU DIEN tren {args.split}: {len(covered)}/{n_attempt} = {cov:.4f} phan tu co MOI "
          f"fragment trong tu dien train.")
    print(f"  Phan con lai ({n_attempt - len(covered)}) chua fragment chi xuat hien o {args.split}. Mo hinh")
    print(f"  **khong bao gio sinh ra** chung (do phu TU VUNG = 100%, Muc 38.3) ⇒ day la gioi han RIENG:")
    print(f"  phan cua {args.split} ma tu vung cua mo hinh khong voi toi, KHONG phai loi lap rap.")

    # --- TAP CHUNG: ca ba hang deu thanh cong -> so sanh moi hop le ---------------------------
    common = {i for i in range(n_attempt)
              if all(rows[k][i] is not None for k in ("reference", "gt_exact", "gt_template"))}
    print("\n" + "-" * 92)
    print(f"  SO SANH TREN TAP CHUNG ({len(common)}/{n_attempt} complex ca ba hang deu thanh cong) --")
    print("  day moi la phep so hop le: cung tap phan tu, chi khac nguon toa do.")
    c_ref = summarize("reference   [tap chung]", rows["reference"], len(common), idx=common)
    c_ex = summarize("gt_exact    [tap chung]", rows["gt_exact"], len(common), idx=common)
    c_tm = summarize("gt_template [tap chung]", rows["gt_template"], len(common), idx=common)

    for key in ("gt_exact", "gt_template"):
        if diags[key]:
            nm = np.array([x["n_manh"] for x in diags[key]])
            no = np.array([x["n_dummy_mo_coi"] for x in diags[key]])
            nb = np.array([x["n_bond_lien_frag"] for x in diags[key]])
            print(f"\n  [{key}] lap rap: lien ket lien-fragment trung binh {nb.mean():.2f}  |  "
                  f"dummy mo coi trung binh {no.mean():.2f}  |  phan tu lien thong (1 manh) "
                  f"{(nm == 1).sum()}/{len(nm)} = {(nm == 1).mean():.4f}")

    print("\n" + "=" * 92)
    print("TIEU CHI (Muc 40.1)")
    print("=" * 92)
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
    print(f"    T3 gia cua xap xi mot-conformer = RMSD 3D tuyet doi (gt_exact vs gt_template)")
    if rmsd_tmpl:
        v = np.array(rmsd_tmpl)
        print(f"       tren {v.size} complex: median={np.median(v):.4f} A  p90={np.percentile(v, 90):.4f}  "
              f"max={v.max():.4f}")
        print(f"       (QED/SA/logP/Lipinski la mo ta 2D nen BUOC PHAI bang nhau -- xem Muc 41.1)")
        out_rmsd = {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
                    "max": float(v.max()), "n": int(v.size)}
    else:
        print("       khong do duoc (khong complex nao thanh ca hai ban)")
        out_rmsd = {}
    print(f"       `gt_template [tap chung]` la TRAN TREN cua mo hinh tren phan test ma tu vung voi toi.")

    out = {"split": args.split, "n_attempt": n_attempt, "head": head,
           "coverage": cov, "n_covered": len(covered), "n_common": len(common),
           "rmsd_template_vs_exact": out_rmsd,
           "toan_bo": {"reference": s_ref, "gt_exact": s_ex, "gt_template": s_tm},
           "tap_chung": {"reference": c_ref, "gt_exact": c_ex, "gt_template": c_tm}}
    dst = os.path.join(SAMG_ROOT, "logs", f"assembly_ceiling_{args.split}.json")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] da luu {dst}")


if __name__ == "__main__":
    main()
