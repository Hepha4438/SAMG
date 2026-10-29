"""
check_link_heads.py -- kiem S6/S7 TRUOC khi huan luyen lai (Rev 21). Khong train, mot batch.

Kiem bon dieu, deu la dieu co the hong am tham:
  L1 Hinh dang: `logits_parent` [B,T,1+T]; `attach_*` [B,T,MAX_ATTACH].
  L2 CHE NHAN QUA: o vi tri t, moi cot cha j >= t phai la -inf (cha phai xuat hien TRUOC).
  L3 Muc tieu hop le: `parent` trong [0, t]; `attach_*` trong [0, MAX_ATTACH) hoac -100.
     Va ty le -100 phai khop ty le GOC (Muc 45.1: ~1 goc/phan tu + 0,0092% goc phu).
  L4 Loss ban dau xap xi ly thuyet: voi mo hinh chua hoc, cross-entropy ~ ln(so lua chon hop le).
     Lech xa => mask hoac muc tieu sai.
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
    print(f"git rev-parse HEAD: {head}\nSEED: 1234\ndevice: {device}  hostname: {socket.gethostname()}",
          flush=True)
    torch.manual_seed(1234)

    import step6_trainer as st
    from step4_ligand_generator import MAX_ATTACH
    from omegaconf import OmegaConf
    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    sc = torch.load(os.path.join(st.PROCESSED_DIR, "scaler_7d.pt"), map_location="cpu")
    shift, scale = sc["shift"], sc["scale"]

    config = OmegaConf.create({
        "hidden_dim": 256, "num_heads": 4, "lr": 1e-4, "ligand_mode": "empty", "geo_head": "maf",
        "protein_encoder": {"num_blocks": 3, "num_layers": 3, "hidden_dim": 256, "n_heads": 4,
                             "knn": 16, "edge_feat_dim": 5, "num_r_gaussian": 20, "num_node_types": 8},
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
    with torch.no_grad():
        total, ld = model.forward(batch)

    lens = batch.seq_len.tolist()
    lens = [lens] if isinstance(lens, int) else lens
    T = max(lens)
    B = len(lens)
    print(f"\n[*] batch: B={B}  T={T}  lens={lens}")

    # tai tao link_logits bang mot forward rieng (model.forward khong tra ra)
    with torch.no_grad():
        h_target, target_mask, _, num_graphs = model.build_h_target(batch)
        pad = lambda x, v: torch.nn.utils.rnn.pad_sequence(
            list(torch.split(x, lens)), batch_first=True, padding_value=v).to(device)
        inp = pad(batch.input_ids, 0)
        tgt = pad(batch.target_ids, 0)
        t7 = pad(batch.target_7d, 0.0)
        *_, link = model.generator(inp, h_target, [], target_7d=t7, target_mask=target_mask,
                                    target_ids=tgt)
    lp, las, lap = link["parent"], link["attach_self"], link["attach_parent"]

    print("\n" + "=" * 86 + "\nL1: hinh dang\n" + "=" * 86)
    ok1 = (tuple(lp.shape) == (B, T, 1 + T) and tuple(las.shape) == (B, T, MAX_ATTACH)
           and tuple(lap.shape) == (B, T, MAX_ATTACH))
    print(f"    parent {tuple(lp.shape)} (ky vong {(B, T, 1 + T)})")
    print(f"    attach_self {tuple(las.shape)}  attach_parent {tuple(lap.shape)}  "
          f"(ky vong {(B, T, MAX_ATTACH)})")
    print(f"    => L1 {'DAT' if ok1 else 'KHONG DAT'}")

    print("\n" + "=" * 86 + "\nL2: che nhan qua (cot cha j >= t phai la -inf)\n" + "=" * 86)
    bad = 0
    for t in range(T):
        for j in range(T):
            if j >= t and not torch.isinf(lp[:, t, 1 + j]).all():
                bad += 1
    print(f"    so (t, j) vi pham: {bad}  => L2 {'DAT' if bad == 0 else 'KHONG DAT'}")

    print("\n" + "=" * 86 + "\nL3: muc tieu hop le\n" + "=" * 86)
    tp = pad(batch.link_parent, -100)
    ta = pad(batch.link_attach_self, -100)
    tb = pad(batch.link_attach_parent, -100)
    n_real = sum(lens)
    bad_p = n_root = n_ign = 0
    for b in range(B):
        for t in range(lens[b]):
            v = int(tp[b, t])
            if v == 0:
                n_root += 1
            elif not (1 <= v <= t):
                bad_p += 1
            if int(ta[b, t]) == -100:
                n_ign += 1
            elif not (0 <= int(ta[b, t]) < MAX_ATTACH):
                bad_p += 1
    print(f"    token that: {n_real}  |  parent khong hop le: {bad_p}")
    print(f"    so GOC (parent=0): {n_root}  (ky vong ~ so phan tu = {B}, cong goc phu hiem)")
    print(f"    attach_self = -100: {n_ign}  (phai bang so GOC)")
    ok3 = (bad_p == 0 and n_ign == n_root)
    print(f"    => L3 {'DAT' if ok3 else 'KHONG DAT'}")

    print("\n" + "=" * 86 + "\nL4: loss ban dau vs ly thuyet (mo hinh CHUA hoc)\n" + "=" * 86)
    exp_par = float(np.mean([math.log(t + 1) for b in range(B) for t in range(lens[b])
                             if int(tp[b, t]) != -100 and t >= 1] or [0.0]))
    print(f"    loss_parent        = {float(ld.get('loss_parent', float('nan'))):.4f}   "
          f"ly thuyet ~ ln(so lua chon) trung binh = {exp_par:.4f}")
    print(f"    loss_attach_self   = {float(ld.get('loss_attach_self', float('nan'))):.4f}   "
          f"ly thuyet ~ ln({MAX_ATTACH}) = {math.log(MAX_ATTACH):.4f}")
    print(f"    loss_attach_parent = {float(ld.get('loss_attach_parent', float('nan'))):.4f}")
    print(f"    loss_link          = {float(ld.get('loss_link', float('nan'))):.4f}")
    print(f"    loss_token {float(ld['loss_token']):.4f}  loss_geo {float(ld['loss_geo']):.4f}  "
          f"loss_total {float(total):.4f}")
    print("    (lech xa ly thuyet => mask hoac muc tieu sai; gan => duong du lieu dung)")

    print("\n" + "=" * 86)
    print("TAT CA L1-L4 nhin DAT => chay huan luyen lai." if (ok1 and bad == 0 and ok3)
          else "CO TIEU CHI KHONG DAT -- khong huan luyen lai.")
    print("=" * 86)


if __name__ == "__main__":
    main()
