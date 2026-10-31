"""check_eos.py -- kiem S9-S13 (Rev 22) TRUOC khi huan luyen lai. Khong train, mot batch.

  E1 vocab co [EOS], va no o CUOI (khong dich id fragment).
  E2 target_ids ket thuc bang [EOS]; input_ids bat dau bang [SOS]; seq_len = L + 1.
  E3 geo_mask = 1 o L vi tri fragment, 0 o vi tri EOS  =>  sum(geo_mask) = L.
  E4 ba muc tieu lien ket deu = -100 o vi tri EOS.
  E5 loss_geo tinh bang geo_mask PHAI KHAC loss_geo tinh theo cach cu (target_ids != 0).
     Day la phep kiem duy nhat co the that bai neu ban sua khong co tac dung: neu hai gia tri
     bang nhau thi geo_mask khong duoc ap, va vi tri EOS (pose RONG) van vao loss hinh hoc.
  E6 cosine annealing: LR phai giam don dieu qua cac epoch, khong phang.
"""
import os
import sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SAMG_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"git rev-parse HEAD: {head}\nSEED: 1234\ndevice: {device}  "
          f"hostname: {socket.gethostname()}", flush=True)
    torch.manual_seed(1234)

    import step6_trainer as st
    from step4_ligand_generator import MAX_ATTACH
    from omegaconf import OmegaConf
    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    sc = torch.load(os.path.join(st.PROCESSED_DIR, "scaler_7d.pt"), map_location="cpu")
    shift, scale = sc["shift"], sc["scale"]

    # ------------------------------------------------------------------ E1
    print("\n" + "=" * 86 + "\nE1: [EOS] trong tu vung\n" + "=" * 86)
    eos = vocab.get("[EOS]")
    print(f"    |vocab| = {len(vocab)}   [SOS]={vocab.get('[SOS]')}  [UNK]={vocab.get('[UNK]')}  "
          f"[EOS]={eos}")
    ok1 = eos is not None and eos == len(vocab) - 1
    if eos is None:
        print("    => E1 KHONG DAT: chua chay lai step0 sau S9.")
    elif eos != len(vocab) - 1:
        print(f"    => E1 KHONG DAT: [EOS] phai o cuoi ({len(vocab) - 1}), dang o {eos}.")
    else:
        print("    => E1 DAT")
    if not ok1:
        raise SystemExit("[!] Dung lai: khong kiem tiep duoc khi tu vung sai.")

    config = OmegaConf.create({
        "hidden_dim": 256, "num_heads": 4, "lr": 1e-4, "ligand_mode": "empty", "geo_head": "maf",
        "max_epochs": 25,
        "protein_encoder": {"num_blocks": 3, "num_layers": 3, "hidden_dim": 256, "n_heads": 4,
                             "knn": 16, "edge_feat_dim": 5, "num_r_gaussian": 20,
                             "num_node_types": 8},
        "loss_weights": {"token": 1.0, "geo": 1.0, "pocket": 1.0, "int": 0.0, "d_threshold": 2.5,
                          "link": 1.0}})
    model = st.SAMGLightningModule(config=config, vocab_size=len(vocab),
                                    shift_factors=shift, scale_factors=scale).to(device)
    model.eval()

    ds = st.SAMGOptimizedDataset(st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE,
                                  split_mode=args.split, pos_scale=scale.flatten()[0].item())
    from datasets.pl_data import ProteinLigandDataLoader
    batch = next(iter(ProteinLigandDataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                               num_workers=0))).to(device)
    lens = batch.seq_len.tolist()
    if isinstance(lens, int):
        lens = [lens]
    B = len(lens)

    def split(x):
        return list(torch.split(x, lens))

    tid = split(batch.target_ids)
    iid = split(batch.input_ids)
    gm = split(batch.geo_mask)
    lp, la, lb = split(batch.link_parent), split(batch.link_attach_self), \
        split(batch.link_attach_parent)

    # ------------------------------------------------------------------ E2, E3, E4
    print("\n" + "=" * 86 + "\nE2/E3/E4: chuoi muc tieu, geo_mask, muc tieu lien ket\n" + "=" * 86)
    print(f"    lens (= L+1) = {lens}")
    bad2 = bad3 = bad4 = 0
    for b in range(B):
        if int(tid[b][-1].item()) != eos or int(iid[b][0].item()) != vocab.get("[SOS]", 0) \
                or tid[b].numel() != lens[b]:
            bad2 += 1
        L = lens[b] - 1
        if abs(float(gm[b].sum().item()) - L) > 1e-6 or float(gm[b][-1].item()) != 0.0:
            bad3 += 1
        if not (int(lp[b][-1]) == -100 and int(la[b][-1]) == -100 and int(lb[b][-1]) == -100):
            bad4 += 1
    print(f"    E2 vi pham: {bad2}/{B}  (target_ids[-1]==[EOS], input_ids[0]==[SOS], do dai khop)")
    print(f"    E3 vi pham: {bad3}/{B}  (sum(geo_mask)==L va geo_mask[-1]==0)")
    print(f"    E4 vi pham: {bad4}/{B}  (ba muc tieu lien ket == -100 o vi tri EOS)")
    ok2, ok3, ok4 = bad2 == 0, bad3 == 0, bad4 == 0
    for nm, o in (("E2", ok2), ("E3", ok3), ("E4", ok4)):
        print(f"    => {nm} {'DAT' if o else 'KHONG DAT'}")

    # ------------------------------------------------------------------ E5
    print("\n" + "=" * 86 + "\nE5: geo_mask co THUC SU doi ket qua khong\n" + "=" * 86)
    with torch.no_grad():
        _, ld_new = model.forward(batch)
    pad = torch.nn.utils.rnn.pad_sequence
    t_ids = pad(split(batch.target_ids), batch_first=True, padding_value=0).to(device)
    g_m = pad(split(batch.geo_mask), batch_first=True, padding_value=0.0).to(device)
    cu = (t_ids != 0).float()
    n_khac = int((cu != g_m).sum().item())
    print(f"    so vi tri ma mask CU khac mask MOI: {n_khac}  (ky vong = so phan tu = {B},"
          f" moi phan tu dung mot vi tri EOS)")
    print(f"    sum(mask cu) = {float(cu.sum()):.1f}   sum(geo_mask) = {float(g_m.sum()):.1f}")
    ok5 = (n_khac == B)
    print(f"    => E5 {'DAT' if ok5 else 'KHONG DAT'}  "
          f"({'geo_mask dang loai dung vi tri EOS' if ok5 else 'mask khong duoc ap dung dung'})")
    print(f"    loss_geo (voi geo_mask) = {float(ld_new['loss_geo']):.4f}")
    print(f"    loss_token              = {float(ld_new['loss_token']):.4f}"
          f"   (ly thuyet khoi tao ~ ln({len(vocab)}) = {math.log(len(vocab)):.4f})")
    for k in ("loss_parent", "loss_attach_self", "loss_attach_parent"):
        if k in ld_new:
            print(f"    {k:20} = {float(ld_new[k]):.4f}")

    # ------------------------------------------------------------------ E6
    print("\n" + "=" * 86 + "\nE6: cosine annealing\n" + "=" * 86)
    cfg = model.configure_optimizers()
    sch = cfg["lr_scheduler"]["scheduler"]
    opt = cfg["optimizer"]
    lrs = []
    for _ in range(25):
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()
    print(f"    loai: {type(sch).__name__}")
    print(f"    LR epoch 0,6,12,18,24 = " + ", ".join(f"{lrs[i]:.3e}" for i in (0, 6, 12, 18, 24)))
    giam = all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))
    ok6 = (type(sch).__name__ == "CosineAnnealingLR" and giam and lrs[-1] < lrs[0] * 0.1)
    print(f"    giam don dieu: {giam}   LR cuoi/dau = {lrs[-1] / lrs[0]:.4f}")
    print(f"    => E6 {'DAT' if ok6 else 'KHONG DAT'}")

    print("\n" + "=" * 86)
    print("TAT CA E1-E6 DAT => chay lai step0 roi huan luyen."
          if all((ok1, ok2, ok3, ok4, ok5, ok6))
          else "CO TIEU CHI KHONG DAT -- khong huan luyen lai.")
    print("=" * 86)


if __name__ == "__main__":
    main()
