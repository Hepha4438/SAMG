"""
step7_generate_eval.py -- PLAN Rev 20 A2c. THAY THE duong dat pose cua `step7_evaluation.py`
(hang muc B7; ba loi cua ban cu ghi o Muc 37-pre.2).

Duong di (moi khau da duoc xac nhan rieng truoc do):
  model -> chuoi token fragment + 7D  ->  template medoid (fragment_dict)
        -> nghich dao 37-pre.3 (RMSD = 0,0000, Muc 39.2)
        -> assemble.py (255/255, 0 dummy mo coi, 100% lien thong, Muc 42)
        -> chi so hoa hoc

Khac ban cu: KHONG ETKDG, KHONG trung binh quaternion, dung FRAME HOC (`pocket_R`/`pocket_t`),
va KHONG ghep chuoi ky tu SMILES.

Hai diem phai ghi ro khi doc ket qua:
  * QED/SA/logP/Lipinski la mo ta **2D** -> **mu voi chat luong pose** (Muc 41.2). Chung do "mo hinh
    chon dung fragment nao", KHONG do "dat dung cho khong". Pose chi doc duoc qua Vina / kiem hinh hoc.
  * Tu vung **khong voi toi 42% phan tu test** (Muc 41.4) -- bao cao kem moi bang ket qua.

Do dai chuoi: tu vung **khong co `[EOS]`** (Muc 40.2) nen mo hinh khong the bao "da xong". Tam thoi lay
do dai tu **phan phoi thuc nghiem so fragment/phan tu cua train** (quy uoc du an, R16 muc 3). Ban sua
that la them `[EOS]`, can chay lai step0 + huan luyen lai -> Rev 21.
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
    return {"QED": float(qed(mol)), "SA": float(sa_fn(mol)), "logP": logp, "Lipinski": float(lip)}


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
    _, _, sampled_7d, _, _lk = model.generator(inp_final, h_target, [], target_mask=target_mask, target_ids=tgt)
    return toks, sampled_7d[0].double().cpu().numpy()


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
                toks, poses = generate_one(model, h_target, target_mask, len(vocab), L,
                                            args.temperature, device, banned)
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
          f"({args.num_samples}/hoc, temperature={args.temperature})")
    print("=" * 92)
    # Muc 43: "sanitize duoc" KHONG du de goi la phan tu -- mot dam fragment roi rac van sanitize
    # duoc. Validity that phai doi LIEN THONG (1 manh).
    n_conn = sum(1 for d_ in diags if d_["n_manh"] == 1)
    print(f"\n  Validity (chi sanitize duoc):        {len(ok)}/{n_gen} = {len(ok) / max(n_gen, 1):.4f}")
    print(f"  Validity (sanitize VA lien thong):   {n_conn}/{n_gen} = {n_conn / max(n_gen, 1):.4f}"
          f"   <- con so that")
    for k, v in Counter(errs).most_common(5):
        print(f"      loi: {k[:74]:76} x{v}")
    out = {"split": args.split, "head": head, "ckpt": os.path.basename(ckpt), "n_pocket": n_pocket,
           "n_gen": n_gen, "validity_sanitize": len(ok) / max(n_gen, 1),
           "validity_connected": n_conn / max(n_gen, 1), "temperature": args.temperature}
    if ok:
        print(f"\n      {'chi so':10}{'mean':>10}{'median':>10}")
        for k in ["QED", "SA", "logP", "Lipinski"]:
            v = np.array([r[k] for r in ok])
            out[f"{k}_mean"], out[f"{k}_median"] = float(v.mean()), float(np.median(v))
            print(f"      {k:10}{v.mean():10.4f}{np.median(v):10.4f}")
    if novel_flags:
        out["novelty"] = float(np.mean(novel_flags))
        print(f"\n  Novelty (Tanimoto <= {NOVELTY_TANIMOTO} so voi ligand tham chieu): {out['novelty']:.4f}")
    if diags:
        nm = np.array([x["n_manh"] for x in diags])
        no = np.array([x["n_dummy_mo_coi"] for x in diags])
        out["lien_thong"] = float((nm == 1).mean())
        out["dummy_mo_coi_mean"] = float(no.mean())
        print(f"  Lien thong (1 manh): {(nm == 1).sum()}/{len(nm)} = {(nm == 1).mean():.4f}  |  "
              f"dummy mo coi trung binh: {no.mean():.2f}")
        print("      (dummy mo coi cao = cac diem gan KHONG gap nhau => pose sai. Day la chi bao pose"
              " 2D-free duy nhat ta co truoc khi chay Vina.)")

    cpath = os.path.join(SAMG_ROOT, "logs", f"assembly_ceiling_{args.split}.json")
    if os.path.exists(cpath) and ok:
        with open(cpath) as f:
            ceil = json.load(f)
        c = ceil.get("tap_chung", {}).get("gt_template", {})
        if c:
            print("\n" + "-" * 92)
            print(f"  SO VOI TRAN TREN (gt_template tap chung, {ceil.get('n_common')} complex -- Muc 42.1):")
            print(f"      {'chi so':10}{'mo hinh':>10}{'tran tren':>12}{'lech':>10}")
            for k in ["QED", "SA", "logP", "Lipinski"]:
                if f"{k}_mean" in out and f"{k}_mean" in c:
                    print(f"      {k:10}{out[k + '_mean']:10.4f}{c[k + '_mean']:12.4f}"
                          f"{out[k + '_mean'] - c[k + '_mean']:+10.4f}")
            print(f"      LUU Y (Muc 41.2): bon chi so nay la mo ta 2D -- MU voi chat luong pose.")
            print(f"      LUU Y (Muc 41.4): tu vung khong voi toi 42% phan tu test.")

    dst = os.path.join(SAMG_ROOT, "logs", f"generate_eval_{args.split}.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[*] da luu {dst}")
    if args.out_sdf:
        print(f"[*] SDF da ghi vao {args.out_sdf} (de chay Vina rieng)")


if __name__ == "__main__":
    main()
