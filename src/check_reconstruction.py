"""
check_reconstruction.py -- PLAN Rev 20 A2; RESEARCH_CONTEXT 37-pre.3, 38.2.

MUC DICH 1 (chinh): kiem phep NGHICH DAO cua bieu dien 7D, dung NHAN THAT, KHONG dung mo hinh,
KHONG dung tu dien fragment (lay `local_coords` theo TUNG LAN XUAT HIEN trong pkl). Neu phep nghich
dao dung thi tai dung lai duoc dung phan tu goc trong SDF.

    rel          = [d sin(theta) cos(phi), d sin(theta) sin(phi), d cos(theta)]
    center_world = pocket_R @ rel + pocket_t
    R_g_to_w     = pocket_R @ R_g_to_m          (R_g_to_m tu quaternion [qw,qx,qy,qz])
    atom_world   = R_g_to_w @ (local - local.mean(0)) + center_world

So khop voi SDF goc bang gan ghep toi uu (Hungarian, scipy.optimize.linear_sum_assignment) co rang
buoc cung nguyen to -- vi thu tu nguyen tu giua ban tai dung va SDF khong nhat thiet trung.

TIEU CHI (ghi cung o plan Rev 20 A2, TRUOC khi chay): **RMSD < 0,5 A o >= 18/20 complex** => duong tai
dung DUNG. Khong dat => DUNG, phep nghich dao sai, khong doc so Vina nao.
(0,5 A = nua nguong RMSD 1,0 A cua nganh docking/PoseBusters -- quy uoc du an, R16 muc 3.)

MUC DICH 2 (chan doan): giai thich duoi `rmsd_spread` o Muc 38.2 (`*O*` 6,25 A tren fragment 3 nguyen
tu -- KHONG the giai thich bang hoan vi thu tu, vi 3 diem chi trai rong ~2,4 A). `--diag-smiles '*O*'`
in toa do cuc bo that cua nhieu lan xuat hien + **vector khoang cach doi mot da sap** (bat bien voi
thu tu nguyen tu):
    sorted-distance khop nhau nhung Kabsch lon  => dung la hoan vi thu tu.
    sorted-distance KHAC nhau                    => hinh hoc that su khac (hoac loi) -- phai truy tiep.
"""
import os
import sys
import glob
import math
import pickle
import socket
import argparse
import subprocess

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

RMSD_THRESHOLD = 0.5      # A
N_DEFAULT = 20
N_PASS_REQUIRED = 18


def spherical_to_vec(d, theta, phi):
    return np.array([d * math.sin(theta) * math.cos(phi),
                     d * math.sin(theta) * math.sin(phi),
                     d * math.cos(theta)], dtype=np.float64)


def reconstruct_complex(pkl_data):
    """Tra ve (xyz [N,3], atomic_nums [N]) cua MOI nguyen tu that (bo dummy, atomic_num == 0)."""
    from scipy.spatial.transform import Rotation as Rot
    R_m_to_w = np.asarray(pkl_data["pocket_R"], dtype=np.float64)
    t_m_to_w = np.asarray(pkl_data["pocket_t"], dtype=np.float64).reshape(3)
    xyz_all, z_all = [], []
    for item in pkl_data["sequence"]:
        d, theta, phi, qw, qx, qy, qz = item["spatial_tokens"]
        center_world = R_m_to_w @ spherical_to_vec(d, theta, phi) + t_m_to_w
        R_g_to_m = Rot.from_quat([qx, qy, qz, qw]).as_matrix()      # scipy: [x,y,z,w]
        R_g_to_w = R_m_to_w @ R_g_to_m
        local = np.asarray(item["local_coords"], dtype=np.float64)
        z = np.asarray(item["atomic_nums"]).reshape(-1)
        world = (R_g_to_w @ (local - local.mean(axis=0)).T).T + center_world
        keep = z != 0                                                # bo nguyen tu dummy
        xyz_all.append(world[keep])
        z_all.append(z[keep])
    if not xyz_all:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    return np.concatenate(xyz_all, axis=0), np.concatenate(z_all, axis=0)


def matched_rmsd(xyz_a, z_a, xyz_b, z_b):
    """RMSD sau gan ghep toi uu (Hungarian), cam ghep khac nguyen to. Tra (rmsd, max_dev, n_bad_elem)."""
    from scipy.optimize import linear_sum_assignment
    if xyz_a.shape[0] != xyz_b.shape[0] or xyz_a.shape[0] == 0:
        return float("nan"), float("nan"), -1
    cost = np.linalg.norm(xyz_a[:, None, :] - xyz_b[None, :, :], axis=-1)
    BIG = 1e6
    cost = cost + BIG * (z_a[:, None] != z_b[None, :])
    ri, ci = linear_sum_assignment(cost)
    dev = np.linalg.norm(xyz_a[ri] - xyz_b[ci], axis=-1)
    n_bad = int((cost[ri, ci] >= BIG / 2).sum())
    return float(np.sqrt((dev ** 2).mean())), float(dev.max()), n_bad


def kabsch_rmsd(P, Q):
    if P.shape != Q.shape or P.shape[0] == 0:
        return float("nan")
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(V @ Wt)))])
    diff = Pc @ (V @ D @ Wt) - Qc
    return float(np.sqrt((diff ** 2).sum() / P.shape[0]))


def sorted_pdist(X):
    """Vector khoang cach doi mot da sap -- BAT BIEN voi thu tu nguyen tu va voi phep quay."""
    n = X.shape[0]
    if n < 2:
        return np.zeros(0)
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    return np.sort(d[np.triu_indices(n, k=1)])


def run_diag(st, smiles, max_occ):
    print("\n" + "=" * 90)
    print(f"CHAN DOAN (Muc 38.2): fragment {smiles!r}")
    print("=" * 90)
    occ = []
    for path in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl"))):
        with open(path, "rb") as f:
            d = pickle.load(f)
        for it in d["sequence"]:
            if it["smiles"] == smiles and "local_coords" in it:
                occ.append((os.path.basename(path), np.asarray(it["local_coords"], dtype=np.float64),
                            np.asarray(it["atomic_nums"]).reshape(-1)))
        if len(occ) >= max_occ:
            break
    if len(occ) < 2:
        print(f"    chi tim thay {len(occ)} lan xuat hien -- khong chan doan duoc.")
        return
    print(f"    tim thay {len(occ)} lan xuat hien (lay toi da {max_occ})")
    ref_name, ref_local, ref_z = occ[0]
    ref_sd = sorted_pdist(ref_local)
    print(f"    moc: {ref_name}  n_atom={ref_local.shape[0]}  atomic_nums={ref_z.tolist()}")
    print(f"    sorted-distance cua moc: {np.round(ref_sd, 3).tolist()}")
    print(f"\n    {'file':34}{'kabsch':>9}{'max|d sorted|':>15}{'span':>8}")
    rows = []
    for name, loc, z in occ[1:]:
        if loc.shape != ref_local.shape:
            print(f"    {name[:32]:34}  SHAPE KHAC: {loc.shape} vs {ref_local.shape}")
            continue
        k = kabsch_rmsd(loc, ref_local)
        sd = sorted_pdist(loc)
        dsd = float(np.abs(sd - ref_sd).max()) if sd.size == ref_sd.size else float("nan")
        span = float(np.linalg.norm(loc - loc.mean(0), axis=-1).max() * 2)
        rows.append((name, k, dsd, span, loc, z))
    rows.sort(key=lambda r: -(r[1] if r[1] == r[1] else -1))
    for name, k, dsd, span, _, _ in rows[:8]:
        print(f"    {name[:32]:34}{k:9.3f}{dsd:15.3f}{span:8.2f}")
    if rows:
        name, k, dsd, span, loc, z = rows[0]
        print(f"\n    --- lan lech nhat: {name} (kabsch={k:.3f}, max|d sorted|={dsd:.3f}) ---")
        print(f"    local MOC :\n{np.round(ref_local, 3)}")
        print(f"    local LECH:\n{np.round(loc, 3)}  atomic_nums={z.tolist()}")
        print(f"    sorted-distance LECH: {np.round(sorted_pdist(loc), 3).tolist()}")
        print("\n    DOC: max|d sorted| ~ 0 nhung kabsch lon  => HOAN VI thu tu nguyen tu (vo hai cho")
        print("         viec dat pose, vi luc sinh ta dat nguyen template).")
        print("         max|d sorted| LON => hinh hoc THAT SU khac (hoac loi) -- phai truy tiep.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid", "test"], default="valid")
    ap.add_argument("--n", type=int, default=N_DEFAULT, help="so complex kiem (tieu chi dat cho n=20)")
    ap.add_argument("--diag-smiles", default=None, help="chan doan rmsd_spread cho mot SMILES")
    ap.add_argument("--diag-max-occ", type=int, default=300)
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(CURRENT_DIR),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")
    print("SEED: 1234  (khong dung ngau nhien)")
    print(f"device: khong dung (script khong GPU)  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    if args.diag_smiles:
        run_diag(st, args.diag_smiles, args.diag_max_occ)

    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    entries = splits.get(args.split, [])
    pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p
                 for p in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))}

    print("\n" + "=" * 90)
    print(f"KIEM TAI DUNG TU NHAN THAT -- split={args.split}, muc tieu {args.n} complex")
    print("=" * 90)
    print(f"    {'pli_id':26}{'n_atom':>8}{'RMSD':>9}{'max dev':>9}{'sai nguyen to':>15}")

    results, n_done, n_skip_nopocket = [], 0, 0
    for e in entries:
        if n_done >= args.n:
            break
        pid = e[2].split("/")[0]
        path = pkl_by_id.get(pid)
        if path is None:
            continue
        with open(path, "rb") as f:
            d = pickle.load(f)
        if not d.get("frame_stable", False):
            continue
        if "pocket_R" not in d or "pocket_t" not in d:
            n_skip_nopocket += 1
            continue
        if any("local_coords" not in it for it in d["sequence"]):
            continue

        xyz_r, z_r = reconstruct_complex(d)
        mol = Chem.SDMolSupplier(os.path.join(st.DATASET_DIR, e[2]))[0]
        if mol is None:
            continue
        conf = mol.GetConformer()
        xyz_o = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())], dtype=np.float64)
        z_o = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)

        rmsd, maxdev, n_bad = matched_rmsd(xyz_r, z_r, xyz_o, z_o)
        note = ""
        if xyz_r.shape[0] != xyz_o.shape[0]:
            note = f"  <-- so nguyen tu lech: tai dung {xyz_r.shape[0]} vs SDF {xyz_o.shape[0]}"
        print(f"    {pid[:24]:26}{xyz_r.shape[0]:8d}{rmsd:9.4f}{maxdev:9.4f}{n_bad:15d}{note}")
        results.append(rmsd)
        n_done += 1

    if n_skip_nopocket:
        print(f"\n    [!] bo qua {n_skip_nopocket} complex vi pkl THIEU pocket_R/pocket_t.")
        print("        => phai chay `python -u src/add_pocket_frame_to_pkl.py` truoc (Muc 38.6).")

    ok = [r for r in results if r == r and r < RMSD_THRESHOLD]
    print("\n" + "=" * 90)
    print("TIEU CHI (plan Rev 20 A2)")
    print("=" * 90)
    print(f"    so complex kiem duoc: {len(results)}  |  RMSD < {RMSD_THRESHOLD} A: {len(ok)}")
    if results:
        arr = np.array([r for r in results if r == r])
        print(f"    RMSD: median={np.median(arr):.4f}  max={arr.max():.4f}")
    if len(results) < args.n:
        print(f"    => KHONG DOC DUOC: chi kiem duoc {len(results)}/{args.n} complex.")
    elif len(ok) >= N_PASS_REQUIRED:
        print(f"    => DAT ({len(ok)}/{len(results)} >= {N_PASS_REQUIRED}): duong tai dung DUNG. "
              f"Di tiep A2 (viet lai Step 7).")
    else:
        print(f"    => KHONG DAT ({len(ok)}/{len(results)} < {N_PASS_REQUIRED}): phep nghich dao SAI. "
              f"DUNG, khong doc so Vina nao.")


if __name__ == "__main__":
    main()
