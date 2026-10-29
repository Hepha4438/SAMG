"""
diag_pose_error.py -- chan doan sau Muc 43: vi sao 0/29 phan tu sinh ra lien thong?

Cau hoi phai tach lam ba, khong duoc gop:
  (A) Mo hinh chon SAI FRAGMENT hay dat SAI CHO?  -> teacher forcing: dung token THAT, chi lay pose
      cua mo hinh. Neu van hong thi loi o POSE.
  (B) Sai so la DO LECH (bias) hay DO TAN (variance cua phep lay mau)?  -> lay K mau cho cung mot vi
      tri; so sai so cua TUNG mau voi sai so cua TRUNG BINH K mau. Neu trung binh tot hon han thi van
      de la lay mau tu phan phoi qua rong, va nen sinh bang ky vong thay vi lay mau.
  (C) Sai so co du nho de LAP RAP duoc khong? -> nguong that: diem gan phai gap nhau trong 1,5 A
      (assemble.LINK_TOL). Do truc tiep sai so TAM FRAGMENT theo A, va goc quay theo do.

Doi chieu ly thuyet (Muc 35.3): R2 chuan cua `d` = 0,099, std(d) = 2,478 A
  => sai so du cua RIENG `d` ~ 2,478 * sqrt(1 - 0,099) = 2,35 A, da vuot nguong 1,5 A.
Neu do duoc dung co nay thi 0/29 KHONG phai loi moi, ma la he qua da duoc bao truoc.

Forward-only, khong train.
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
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, CURRENT_DIR)
SEED = 1234
LINK_TOL = 1.5          # assemble.LINK_TOL -- nguong ghep diem gan


def spherical_to_vec(d, theta, phi):
    return np.array([d * math.sin(theta) * math.cos(phi),
                     d * math.sin(theta) * math.sin(phi),
                     d * math.cos(theta)], dtype=np.float64)


def quat_angle_deg(q1, q2):
    """Goc quay giua hai quaternion [w,x,y,z] (tinh ca double cover q ~ -q)."""
    a = np.asarray(q1, dtype=np.float64); a /= max(np.linalg.norm(a), 1e-12)
    b = np.asarray(q2, dtype=np.float64); b /= max(np.linalg.norm(b), 1e-12)
    return float(2 * math.degrees(math.acos(min(1.0, abs(float(np.dot(a, b)))))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", choices=["valid", "test"], default="test")
    ap.add_argument("--n-pockets", type=int, default=40)
    ap.add_argument("-K", type=int, default=20, help="so mau pose moi vi tri (tach bias vs variance)")
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMG_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"git rev-parse HEAD: {head}\nSEED: {SEED}\ndevice: {device}  hostname: {socket.gethostname()}",
          flush=True)
    torch.manual_seed(SEED); np.random.seed(SEED)

    import step6_trainer as st
    from omegaconf import OmegaConf
    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    scaler = torch.load(os.path.join(st.PROCESSED_DIR, "scaler_7d.pt"), map_location="cpu")
    shift_factors, scale_factors = scaler["shift"], scaler["scale"]
    pos_scale = scale_factors.flatten()[0].item()

    ckpt = args.ckpt or max(glob.glob(os.path.join(SAMG_ROOT, "saved_checkpoints_flow", "*.ckpt")),
                             key=os.path.getmtime)
    print(f"[*] checkpoint: {os.path.basename(ckpt)}")
    config = OmegaConf.create({
        "hidden_dim": 256, "num_heads": 4, "lr": 1e-4, "ligand_mode": "empty", "geo_head": "maf",
        "protein_encoder": {"num_blocks": 3, "num_layers": 3, "hidden_dim": 256, "n_heads": 4,
                             "knn": 16, "edge_feat_dim": 5, "num_r_gaussian": 20, "num_node_types": 8},
        "loss_weights": {"token": 1.0, "geo": 1.0, "pocket": 1.0, "int": 0.0, "d_threshold": 2.5}})
    model = st.SAMGLightningModule.load_from_checkpoint(
        ckpt, map_location=device, config=config, vocab_size=len(vocab),
        shift_factors=shift_factors, scale_factors=scale_factors, strict=False)
    model.to(device).eval().requires_grad_(False)

    dataset = st.SAMGOptimizedDataset(st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE,
                                       split_mode=args.split, pos_scale=pos_scale)
    from datasets.pl_data import ProteinLigandDataLoader
    loader = ProteinLigandDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    err_sample, err_mean, rot_sample, rot_mean = [], [], [], []
    d_pred_all, d_true_all, spread = [], [], []
    n_pocket = 0
    for batch in loader:
        if n_pocket >= args.n_pockets:
            break
        batch = batch.to(device)
        with torch.no_grad():
            h_target, target_mask, _, _ = model.build_h_target(batch)
        L = int(batch.seq_len.item()) if hasattr(batch, "seq_len") else None
        tgt_ids = batch.target_ids.view(1, -1)
        inp_ids = batch.input_ids.view(1, -1)
        true7 = batch.target_7d.view(-1, 7).double().cpu().numpy()
        if tgt_ids.shape[1] != true7.shape[0]:
            continue
        n_pocket += 1

        samples = []
        with torch.no_grad():
            for _ in range(args.K):
                _, _, s7, _ = model.generator(inp_ids, h_target, [], target_mask=target_mask,
                                               target_ids=tgt_ids)
                samples.append(s7[0].double().cpu().numpy())
        S = np.stack(samples)                       # [K, L, 7]

        for t in range(true7.shape[0]):
            v_true = spherical_to_vec(*true7[t, :3])
            vs = np.stack([spherical_to_vec(*S[k, t, :3]) for k in range(args.K)])
            err_sample.extend(np.linalg.norm(vs - v_true, axis=1).tolist())
            err_mean.append(float(np.linalg.norm(vs.mean(axis=0) - v_true)))
            spread.append(float(np.linalg.norm(vs - vs.mean(axis=0), axis=1).mean()))
            rot_sample.extend([quat_angle_deg(S[k, t, 3:7], true7[t, 3:7]) for k in range(args.K)])
            qm = S[:, t, 3:7].mean(axis=0)
            rot_mean.append(quat_angle_deg(qm, true7[t, 3:7]))
            d_pred_all.extend(S[:, t, 0].tolist()); d_true_all.append(float(true7[t, 0]))

        if n_pocket % 10 == 0:
            print(f"  ... {n_pocket} hoc", flush=True)

    def q(name, v, unit):
        v = np.asarray(v)
        print(f"    {name:44}{np.median(v):9.3f}{np.percentile(v, 90):10.3f}{v.mean():10.3f}  {unit}")

    print("\n" + "=" * 92)
    print(f"SAI SO POSE (teacher forcing: token THAT, pose cua MO HINH) -- {n_pocket} hoc, K={args.K}")
    print("=" * 92)
    print(f"    {'dai luong':44}{'median':>9}{'p90':>10}{'mean':>10}")
    q("(A) sai so TAM fragment, tung mau", err_sample, "A")
    q("(B) sai so TAM fragment, TRUNG BINH K mau", err_mean, "A")
    q("    do tan cua K mau quanh trung binh", spread, "A")
    q("(A) sai so GOC QUAY, tung mau", rot_sample, "do")
    q("(B) sai so GOC QUAY, trung binh K mau", rot_mean, "do")
    dp, dt = np.array(d_pred_all), np.array(d_true_all)
    print(f"\n    phan phoi `d`: mo hinh mean={dp.mean():.3f} std={dp.std():.3f}  |  "
          f"that mean={dt.mean():.3f} std={dt.std():.3f}  |  scaler 5,161/2,478")

    print("\n" + "=" * 92)
    print("DOC (ghi cung truoc khi chay)")
    print("=" * 92)
    m_s, m_m = float(np.median(err_sample)), float(np.median(err_mean))
    print(f"    Nguong lap rap: diem gan phai gap nhau trong {LINK_TOL} A (assemble.LINK_TOL).")
    print(f"    (C) sai so tam fragment median = {m_s:.3f} A (tung mau) / {m_m:.3f} A (trung binh K).")
    if m_s > LINK_TOL:
        print(f"        => VUOT nguong: pose sai qua xa de lap rap duoc. Day la nguyen nhan 0/29 lien thong.")
    else:
        print(f"        => DUOI nguong: pose du tot; 0/29 lien thong phai co nguyen nhan KHAC -- truy tiep.")
    print(f"    (B) bias vs variance: trung binh K {'TOT HON HAN' if m_m < 0.6 * m_s else 'KHONG tot hon han'}"
          f" tung mau ({m_m:.3f} vs {m_s:.3f}).")
    if m_m < 0.6 * m_s:
        print(f"        => phan lon sai so la DO TAN cua phep lay mau. Sinh bang KY VONG (thay vi lay mau)")
        print(f"           se cai thien ngay, va day la thay doi o buoc SINH, khong can huan luyen lai.")
    else:
        print(f"        => sai so la DO LECH that: tam cua phan phoi da sai. Lay mau hay ky vong deu khong cuu.")
    print(f"    Doi chieu Muc 35.3: R2(d) = 0,099, std(d) = 2,478 ⇒ sai so du cua RIENG d ~ "
          f"{2.478 * math.sqrt(1 - 0.099):.2f} A.")


if __name__ == "__main__":
    main()
