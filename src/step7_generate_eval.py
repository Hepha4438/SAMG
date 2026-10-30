"""
step7_generate_eval.py -- PLAN Rev 21, hang muc S8.

Duong di:
  model -> chuoi token fragment + 7D + KENH LIEN KET (parent / attach_self / attach_parent)
        -> template medoid (fragment_dict)
        -> nghich dao 37-pre.3 (RMSD = 0,0000, Muc 39.2)
        -> assemble_by_channel (Muc 43.7: do thi do TOKEN quyet dinh, khong do hinh hoc)
        -> chi so hoa hoc

Khac ban Rev 20: lap rap bang kenh lien ket thay vi ghep dummy theo khoang cach. Co `--assemble
geometry` de chay lai duong cu tren CUNG bo mau, nen chenh lech giua hai che do do dung bang
phan ma kenh lien ket mang lai.

====================================================================================
CHUAN DANH GIA -- doi chieu cac cong trinh truoc (R16)
====================================================================================
Bon bai trong thu muc du an dung cac bo chi so sau:

  Frag2Seq (muc 4.1)   : Vina Score, High Affinity, QED, SA, Lipinski, Diversity, Time.
                         Vina tinh bang QVina; tinh chat hoa hoc bang RDKit.
  Apo2Mol  (muc Eval)  : Vina min (AutoDock Vina, Eberhardt et al. 2021), QED, SA, logP,
                         Lipinski, High Affinity. "All evaluation metrics, except High
                         Affinity, are reported using both their mean and median values."
  3DMolFormer (muc 4.2): Vina Score, Vina Dock, QED, SA, Success Rate
                         (Vina Dock < -8.18 VA QED > 0.25 VA SA > 0.59); 100 phan tu/hoc.

Ba quy uoc phai theo cho dung, neu khong so khong so sanh duoc:
  1. SA. Ban goc (Ertl & Schuffenhauer 2009) nam trong [1, 10], THAP la tot. Cac bai SBDD
     bao cao ban bien doi tuyen tinh SA = (10 - SA_goc)/9 thuoc [0, 1], CAO la tot
     (3DMolFormer, Apx D, theo Guan et al. 2023b). Ham `chem_metrics` duoi day tra ve
     CA HAI: "SA_raw" (goc) va "SA" (da chuan hoa) -- de khong lan.
  2. Lipinski la SO QUY TAC thoa man trong nam quy tac, khong phai bien nhi phan. Nam quy
     tac theo Apo2Mol Apx A: MW < 500; HBD <= 5; HBA <= 10; logP trong [-2, 5];
     so lien ket quay duoc <= 10.
  3. Bao cao CA mean VA median (Apo2Mol).

  Diversity: Frag2Seq co bao cao (1 - Tanimoto trung binh tung cap trong mot hoc).
  3DMolFormer CO Y KHONG bao cao, lap luan rang moi chi so da dang hien co deu dua tren do
  thi 2D trong khi bai toan la 3D (Apx D). Ta bao cao nhung ghi kem bat dong nay.

**Tinh LIEN THONG khong co trong bai nao.** Voi phuong phap sinh theo nguyen tu hoac khuech
tan, phan tu lien thong theo cau truc nen khong ai do. Voi Frag2Seq, Dinh ly 3.4 GIA DINH
chuoi tai tao duoc phan tu -- bai khong do ty le that bai. Muc 43 cho thay o mo hinh da
huan luyen that, gia dinh do sai (0/29). Nen day la chi so RIENG cua du an, khong co tien
le de cite; can cu cua no la dac tinh du lieu, khong phai mot bai bao.

Hai dieu phai ghi ro khi doc ket qua:
  * QED/SA/logP/Lipinski la mo ta **2D** -> **mu voi chat luong pose** (Muc 41.2). Chung do
    "mo hinh chon dung fragment nao", KHONG do "dat dung cho khong".
  * Tu vung khong voi toi 42% phan tu test (Muc 41.4) -- bao cao kem moi bang ket qua.

Vina chay RIENG qua `--out-sdf` (can meeko + vina + AutoDockTools). Theo Apo2Mol dung
Vina min (`mode='minimize'`), theo Frag2Seq dung QVina cho Vina Score.

Do dai chuoi: tu vung **khong co `[EOS]`** (Muc 40.2) nen mo hinh khong the bao "da xong".
Tam thoi lay do dai tu phan phoi thuc nghiem so fragment/phan tu cua train. Ban sua that la
them `[EOS]` -> hoan sang Rev 22.
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
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)

SEED = 1234
NOVELTY_TANIMOTO = 0.4      # Apo2Mol Apx C.1: "novelty ... Tanimoto similarity <= 0.4" (Morgan)


def sa_scorer():
    from rdkit.Chem import RDConfig
    p = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if p not in sys.path:
        sys.path.append(p)
    import sascorer
    return sascorer.calculateScore


def chem_metrics(mol, sa_fn):
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
    sa_raw = float(sa_fn(mol))
    # 3DMolFormer Apx D (theo Guan et al. 2023b): SA = (10 - SA_goc)/9, CAO la tot.
    return {"QED": float(qed(mol)), "SA": (10.0 - sa_raw) / 9.0, "SA_raw": sa_raw,
            "logP": logp, "Lipinski": float(lip)}


def spherical_to_vec(d, theta, phi):
    return np.array([d * math.sin(theta) * math.cos(phi),
                     d * math.sin(theta) * math.sin(phi),
                     d * math.cos(theta)], dtype=np.float64)


def empirical_lengths(st, pkl_by_id):
    """So fragment/phan tu tren TRAIN -- nguon do dai chuoi khi sinh (Muc 40.2: vocab khong co EOS)."""
    with open(st.SPLIT_FILE, "rb") as f:
        splits = pickle.load(f)
    out = []
    for e in splits.get("train", []):
        p = pkl_by_id.get(e[2].split("/")[0])
        if p is None:
            continue
        with open(p, "rb") as f:
            d = pickle.load(f)
        if d.get("frame_stable", False):
            out.append(len(d["sequence"]))
    return np.array(out, dtype=int)


@torch.no_grad()
def generate_one(model, h_target, target_mask, vocab_size, length, temperature, device, banned):
    """Lay mau TOAN BO chuoi token truoc, roi MOT forward cuoi voi target_ids de sinh 7D dung tren
    token cua chinh vi tri do (Muc 42.2). Ton L+1 forward thay vi 2L."""
    sos = torch.zeros(1, 1, dtype=torch.long, device=device)      # [SOS] = id 0
    inp = sos
    toks = []
    for _ in range(length):
        logits, _, _, _, _lk = model.generator(inp, h_target, [], target_mask=target_mask)
        p = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        p[:, banned] = 0.0
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        nxt = torch.multinomial(p, 1)
        toks.append(int(nxt.item()))
        inp = torch.cat([inp, nxt], dim=1)
    tgt = torch.tensor([toks], dtype=torch.long, device=device)
    inp_final = torch.cat([sos, tgt[:, :-1]], dim=1)
    _, _, sampled_7d, _, link_logits = model.generator(inp_final, h_target, [],
                                                       target_mask=target_mask, target_ids=tgt)
    # S8: link_logits lay tu DUNG forward nay -- cac head lien ket deu doc `geo_in`/`frag_repr`
    # dung tren target_ids, giong het luc huan luyen (teacher forcing tren token).
    return toks, sampled_7d[0].double().cpu().numpy(), link_logits


def n_dummies(entry):
    z = np.asarray(entry["atomic_nums"]).reshape(-1)
    return int((z == 0).sum())


def canon_of_ordinal(entry, o):
    """Nghich dao cua `_ord_in` o step6: thu tu trong so dummy -> chi so CANONICAL."""
    if o is None:
        return None
    z = np.asarray(entry["atomic_nums"]).reshape(-1)
    dummies = [i for i, zz in enumerate(z) if int(zz) == 0]
    return int(dummies[o]) if 0 <= o < len(dummies) else None


def decode_links(link_logits, n_dum, greedy=True):
    """S8: giai ma parent / attach_self / attach_parent theo thu tu t, CO MASK.

    Mask ap dung -- deu la rang buoc **tinh dung dan**, khong phai de lam dep so lien thong:
      * t = 0 bat buoc la GOC.
      * cha phai la j < t va con it nhat mot dummy chua dung.
      * attach_self < so dummy cua fragment t.
      * attach_parent < so dummy cua cha, va dummy do chua bi chiem.

    GOC VAN DUOC PHEP o t > 0. Ligand nhieu thanh phan la co that (kiem K5 da thay), va quan
    trong hon: cam GOC se lam chi so "lien thong" dat 100% mot cach tam thuong, tuc bien no
    thanh mot tieu chi khong the that bai -- vi pham R5. Chi so chi co nghia khi mo hinh duoc
    phep tra loi "day la manh roi".
    """
    from step4_ligand_generator import MAX_ATTACH
    T = len(n_dum)
    lp = link_logits["parent"][0].double().cpu().numpy()
    ls = link_logits["attach_self"][0].double().cpu().numpy()
    la = link_logits["attach_parent"][0].double().cpu().numpy()
    used = [set() for _ in range(T)]
    par, a_s, a_p = [], [], []

    def pick(vec, allowed):
        if not allowed:
            return None
        v = np.full(vec.shape, -np.inf)
        idx = np.asarray(sorted(allowed), dtype=int)
        v[idx] = vec[idx]
        if greedy:
            return int(np.argmax(v))
        e = np.exp(v - np.max(v))
        e = e / e.sum()
        return int(np.random.choice(v.size, p=e))

    for t in range(T):
        free_self = set(range(min(n_dum[t], MAX_ATTACH))) - used[t]
        if t == 0 or not free_self:
            par.append(-1); a_s.append(None); a_p.append(None); continue
        cand = {0}
        for j in range(t):
            if set(range(min(n_dum[j], MAX_ATTACH))) - used[j]:
                cand.add(j + 1)
        c = pick(lp[t], cand)
        if c is None or c == 0:
            par.append(-1); a_s.append(None); a_p.append(None); continue
        pj = c - 1
        free_par = set(range(min(n_dum[pj], MAX_ATTACH))) - used[pj]
        os_ = pick(ls[t], free_self)
        op_ = pick(la[t], free_par)
        if os_ is None or op_ is None:
            par.append(-1); a_s.append(None); a_p.append(None); continue
        used[t].add(os_)
        used[pj].add(op_)
        par.append(pj); a_s.append(os_); a_p.append(op_)
    return par, a_s, a_p


def build_mol_channel(toks, poses, inv_vocab, frag_dict, pocket_R, pocket_t,
                      link_logits, greedy=True):
    """S8: do thi phan tu do KENH LIEN KET quyet dinh; hinh hoc chi dat toa do (Muc 43.7)."""
    from scipy.spatial.transform import Rotation as Rot
    import assemble as asm
    entries, place_info = [], []
    for tid, p7 in zip(toks, poses):
        smi = inv_vocab.get(tid)
        e = frag_dict.get(smi)
        if e is None:
            raise KeyError(f"token {tid} ({smi!r}) khong co template")
        d, theta, phi, qw, qx, qy, qz = [float(v) for v in p7]
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if n < 1e-8:
            raise ValueError("quaternion suy bien")
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        center = pocket_R @ spherical_to_vec(d, theta, phi) + pocket_t
        R_g_to_w = pocket_R @ Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        entries.append((smi, e))
        place_info.append((R_g_to_w, center))

    n_dum = [n_dummies(e) for _, e in entries]
    par, a_s, a_p = decode_links(link_logits, n_dum, greedy=greedy)
    items = []
    for k, (smi, e) in enumerate(entries):
        pj = par[k]
        items.append({
            "smiles": smi,
            "local": np.asarray(e["local"], dtype=np.float64),
            "z": np.asarray(e["atomic_nums"]).reshape(-1),
            "parent": pj,
            "attach_self": canon_of_ordinal(e, a_s[k]),
            "attach_parent": canon_of_ordinal(entries[pj][1], a_p[k]) if pj >= 0 else None,
        })
    mol, diag = asm.assemble_by_channel(items, place_info=place_info)
    diag["n_goc"] = int(sum(1 for x in par if x < 0))
    return mol, diag


def build_mol(toks, poses, inv_vocab, frag_dict, pocket_R, pocket_t):
    from scipy.spatial.transform import Rotation as Rot
    import assemble as asm
    placed = []
    for tid, p7 in zip(toks, poses):
        smi = inv_vocab.get(tid)
        e = frag_dict.get(smi)
        if e is None:
            raise KeyError(f"token {tid} ({smi!r}) khong co template")
        d, theta, phi, qw, qx, qy, qz = [float(v) for v in p7]
        n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        if n < 1e-8:
            raise ValueError("quaternion suy bien")
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        center = pocket_R @ spherical_to_vec(d, theta, phi) + pocket_t
        R_g_to_w = pocket_R @ Rot.from_quat([qx, qy, qz, qw]).as_matrix()
        local = np.asarray(e["local"], dtype=np.float64)
        z = np.asarray(e["atomic_nums"]).reshape(-1)
        placed.append(asm.place(asm.fragment_mol_with_coords(smi, local, z), R_g_to_w, center))
    return asm.assemble(placed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", choices=["valid", "test"], default="test")
    ap.add_argument("--num-samples", type=int, default=5, help="Apo2Mol sample_split.py: 5 moi hoc")
    ap.add_argument("--temperature", type=float, default=1.0,
                     help="1,0 = lay mau dung phan phoi da hoc. Ban cu hard-code 0,8 khong co can cu.")
    ap.add_argument("--n-pockets", type=int, default=0, help="0 = het split")
    ap.add_argument("--out-sdf", default=None, help="thu muc ghi SDF (de chay Vina rieng)")
    # S8. "channel" = do thi do kenh lien ket quyet dinh (Rev 21). "geometry" = ghep dummy theo
    # khoang cach (duong cu, Rev 20) -- giu lai de A/B tren CUNG mot bo mau sinh ra.
    ap.add_argument("--assemble", choices=["channel", "geometry"], default="channel")
    ap.add_argument("--link-mode", choices=["greedy", "sample"], default="greedy",
                     help="greedy = doan tot nhat cua mo hinh ve cau truc; sample = lay mau co mask")
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMG_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"git rev-parse HEAD: {head}")
    print(f"SEED: {SEED}")
    print(f"device: {device}  hostname: {socket.gethostname()}", flush=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    import step6_trainer as st
    from omegaconf import OmegaConf
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    import assemble as asm
    sa_fn = sa_scorer()

    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    inv_vocab = {v: k for k, v in vocab.items()}
    with open(os.path.join(st.PROCESSED_DIR, "fragment_dict.pkl"), "rb") as f:
        frag_dict = pickle.load(f)
    scaler = torch.load(os.path.join(st.PROCESSED_DIR, "scaler_7d.pt"), map_location="cpu")
    shift_factors, scale_factors = scaler["shift"], scaler["scale"]
    pos_scale = scale_factors.flatten()[0].item()
    print(f"[*] vocab {len(vocab)}  |  fragment_dict {len(frag_dict)} SMILES")

    ckpt = args.ckpt
    if ckpt is None:
        cands = glob.glob(os.path.join(SAMG_ROOT, "saved_checkpoints_flow", "*.ckpt"))
        if not cands:
            raise SystemExit("[!] Khong co checkpoint nao trong saved_checkpoints_flow/. Truyen --ckpt.")
        ckpt = max(cands, key=os.path.getmtime)
    print(f"[*] checkpoint: {ckpt}")

    config = OmegaConf.create({
        "hidden_dim": 256, "num_heads": 4, "lr": 1e-4, "ligand_mode": "empty", "geo_head": "maf",
        "protein_encoder": {"num_blocks": 3, "num_layers": 3, "hidden_dim": 256, "n_heads": 4,
                             "knn": 16, "edge_feat_dim": 5, "num_r_gaussian": 20, "num_node_types": 8},
        "loss_weights": {"token": 1.0, "geo": 1.0, "pocket": 1.0, "int": 0.0, "d_threshold": 2.5},
    })
    model = st.SAMGLightningModule.load_from_checkpoint(
        ckpt, map_location=device, config=config, vocab_size=len(vocab),
        shift_factors=shift_factors, scale_factors=scale_factors, strict=False)
    model.to(device).eval().requires_grad_(False)

    pkl_by_id = {os.path.basename(p)[: -len("_sequence_7d.pkl")]: p
                 for p in sorted(glob.glob(os.path.join(st.PROCESSED_DIR, "*_sequence_7d.pkl")))}
    lens = empirical_lengths(st, pkl_by_id)
    print(f"[*] do dai chuoi (train): n={lens.size}  mean={lens.mean():.2f}  median={int(np.median(lens))}  "
          f"min={lens.min()}  max={lens.max()}   <- nguon lay mau do dai (Muc 40.2, chua co [EOS])")

    dataset = st.SAMGOptimizedDataset(st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE,
                                       split_mode=args.split, pos_scale=pos_scale)
    from datasets.pl_data import ProteinLigandDataLoader
    loader = ProteinLigandDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    banned = torch.tensor([vocab.get("[SOS]", 0), vocab.get("[UNK]", 1)], dtype=torch.long, device=device)

    if args.out_sdf:
        os.makedirs(args.out_sdf, exist_ok=True)

    rows, errs, diags, novel_flags = [], [], [], []
    n_pocket, n_gen = 0, 0
    for bi, batch in enumerate(loader):
        if args.n_pockets and n_pocket >= args.n_pockets:
            break
        batch = batch.to(device)
        with torch.no_grad():
            h_target, target_mask, _, _ = model.build_h_target(batch)   # R14: dung ham cua step6
        pocket_R = batch.pocket_R.view(3, 3).double().cpu().numpy()
        pocket_t = batch.pocket_t.view(3).double().cpu().numpy()

        entry = dataset.target_entries[dataset.valid_indices[bi]]
        ref_mol = Chem.SDMolSupplier(os.path.join(st.DATASET_DIR, entry[2]))[0]
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048) if ref_mol else None
        n_pocket += 1

        for s in range(args.num_samples):
            n_gen += 1
            L = int(np.random.choice(lens))
            try:
                toks, poses, link_logits = generate_one(model, h_target, target_mask, len(vocab),
                                                         L, args.temperature, device, banned)
                if args.assemble == "channel":
                    mol, diag = build_mol_channel(toks, poses, inv_vocab, frag_dict,
                                                   pocket_R, pocket_t, link_logits,
                                                   greedy=(args.link_mode == "greedy"))
                else:
                    mol, diag = build_mol(toks, poses, inv_vocab, frag_dict, pocket_R, pocket_t)
                rows.append(chem_metrics(mol, sa_fn))
                diags.append(diag)
                if ref_fp is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                    novel_flags.append(DataStructs.TanimotoSimilarity(fp, ref_fp) <= NOVELTY_TANIMOTO)
                if args.out_sdf:
                    w = Chem.SDWriter(os.path.join(args.out_sdf, f"{entry[2].split('/')[0]}_{s}.sdf"))
                    w.write(mol)
                    w.close()
            except Exception as exc:
                rows.append(None)
                errs.append(f"{type(exc).__name__}: {str(exc)[:70]}")
        if n_pocket % 25 == 0:
            print(f"  ... {n_pocket} hoc, {n_gen} mau", flush=True)

    ok = [r for r in rows if r is not None]
    print("\n" + "=" * 92)
    print(f"SINH MAU -- split={args.split}  hoc={n_pocket}  mau={n_gen}  "
          f"({args.num_samples}/hoc, temperature={args.temperature}, "
          f"lap rap={args.assemble}, link={args.link_mode})")
    print("=" * 92)
    # Muc 43: "sanitize duoc" KHONG du de goi la phan tu -- mot dam fragment roi rac van sanitize
    # duoc. Validity that phai doi LIEN THONG (1 manh).
    n_conn = sum(1 for d_ in diags if d_["n_manh"] == 1)
    print(f"\n  Validity (chi sanitize duoc):        {len(ok)}/{n_gen} = {len(ok) / max(n_gen, 1):.4f}")
    print(f"  Validity (sanitize VA lien thong):   {n_conn}/{n_gen} = {n_conn / max(n_gen, 1):.4f}"
          f"   <- con so that")
    for k, v in Counter(errs).most_common(5):
        print(f"      loi: {k[:74]:76} x{v}")
    out = {"split": args.split, "head": head, "ckpt": os.path.basename(ckpt),
           "assemble": args.assemble, "link_mode": args.link_mode, "n_pocket": n_pocket,
           "n_gen": n_gen, "validity_sanitize": len(ok) / max(n_gen, 1),
           "validity_connected": n_conn / max(n_gen, 1), "temperature": args.temperature}
    if ok:
        print(f"\n      {'chi so':10}{'mean':>10}{'median':>10}")
        for k in ["QED", "SA", "SA_raw", "logP", "Lipinski"]:
            v = np.array([r[k] for r in ok])
            out[f"{k}_mean"], out[f"{k}_median"] = float(v.mean()), float(np.median(v))
            print(f"      {k:10}{v.mean():10.4f}{np.median(v):10.4f}")
        print("      (SA = ban chuan hoa (10-SA_goc)/9, CAO la tot; SA_raw = ban goc [1,10],"
              " THAP la tot)")
    if novel_flags:
        out["novelty"] = float(np.mean(novel_flags))
        print(f"\n  Novelty (Tanimoto <= {NOVELTY_TANIMOTO} so voi ligand tham chieu): {out['novelty']:.4f}")
    if diags:
        nm = np.array([x["n_manh"] for x in diags])
        out["lien_thong"] = float((nm == 1).mean())
        print(f"  Lien thong (1 manh): {(nm == 1).sum()}/{len(nm)} = {(nm == 1).mean():.4f}")
        if args.assemble == "channel":
            ng = np.array([x.get("n_goc", 0) for x in diags])
            nf = np.array([x.get("n_link_that_bai", 0) for x in diags])
            out["n_goc_mean"] = float(ng.mean())
            out["link_that_bai_mean"] = float(nf.mean())
            print(f"      so GOC trung binh: {ng.mean():.2f}  (1 = mot thanh phan; > 1 = mo hinh"
                  f" CHU DONG tra loi 'manh roi')")
            print(f"      link that bai trung binh: {nf.mean():.2f}  (dummy duoc chi khong ton tai"
                  f" hoac da bi chiem -- loi giai ma, khong phai loi pose)")
            print("      LUU Y: o che do channel, lien thong KHONG con phu thuoc pose (Muc 43.7)."
                  " Pose sai gio chi lam hinh hoc xau, khong lam dut phan tu.")
        else:
            no = np.array([x.get("n_dummy_mo_coi", 0) for x in diags])
            out["dummy_mo_coi_mean"] = float(no.mean())
            print(f"      dummy mo coi trung binh: {no.mean():.2f}  (cao = cac diem gan KHONG gap"
                  f" nhau => pose sai)")

    cpath = os.path.join(SAMG_ROOT, "logs", f"assembly_ceiling_{args.split}.json")
    if os.path.exists(cpath) and ok:
        with open(cpath) as f:
            ceil = json.load(f)
        c = ceil.get("tap_chung", {}).get("gt_template", {})
        if c:
            print("\n" + "-" * 92)
            print(f"  SO VOI TRAN TREN (gt_template tap chung, {ceil.get('n_common')} complex -- Muc 42.1):")
            print(f"      {'chi so':10}{'mo hinh':>10}{'tran tren':>12}{'lech':>10}")
            # check_assembly_ceiling.py ghi SA theo ban GOC. O day "SA" da duoc chuan hoa,
            # nen phai doi chieu bang "SA_raw" -- neu khong se so hai thang do nguoc chieu nhau.
            for k, kc in [("QED", "QED"), ("SA_raw", "SA"), ("logP", "logP"),
                          ("Lipinski", "Lipinski")]:
                if f"{k}_mean" in out and f"{kc}_mean" in c:
                    print(f"      {k:10}{out[k + '_mean']:10.4f}{c[kc + '_mean']:12.4f}"
                          f"{out[k + '_mean'] - c[kc + '_mean']:+10.4f}")
            print(f"      LUU Y (Muc 41.2): bon chi so nay la mo ta 2D -- MU voi chat luong pose.")
            print(f"      LUU Y (Muc 41.4): tu vung khong voi toi 42% phan tu test.")

    dst = os.path.join(SAMG_ROOT, "logs",
                        f"generate_eval_{args.split}_{args.assemble}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] da luu {dst}")
    if args.out_sdf:
        print(f"[*] SDF da ghi vao {args.out_sdf} (de chay Vina rieng)")


if __name__ == "__main__":
    main()
