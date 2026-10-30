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
    # Muc 49: nap trong so tu checkpoint de TACH ba head khi khong co wandb.
    # Khong truyen --ckpt => mo hinh chua hoc, dung cho L1-L4 nhu truoc.
    ap.add_argument("--ckpt", default=None, help="duong dan .ckpt; 'auto' = moi nhat trong checkpoints/")
    ap.add_argument("--batches", type=int, default=1, help="so batch de lay trung binh")
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

    if args.ckpt:
        path = args.ckpt
        if path == "auto":
            import glob as _g
            # ModelCheckpoint ghi vao saved_checkpoints_flow/ (step6_trainer dong 577).
            cands = []
            for sub in ("saved_checkpoints_flow", "checkpoints"):
                cands += _g.glob(os.path.join(SAMG_ROOT, sub, "**", "*.ckpt"), recursive=True)
            if not cands:
                raise SystemExit("[!] khong tim thay .ckpt nao trong saved_checkpoints_flow/ hay checkpoints/")
            # Bo file dang duoc ghi do (sua doi trong 60 giay qua) de khong nap checkpoint cut.
            import time as _t
            fresh = [c for c in cands if _t.time() - os.path.getmtime(c) > 60]
            path = max(fresh or cands, key=os.path.getmtime)
        sd = torch.load(path, map_location=device)
        sd = sd.get("state_dict", sd)
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[*] nap checkpoint: {path}")
        print(f"    thieu khoa: {len(miss)}  thua khoa: {len(unexp)}")
        if miss:
            print(f"    vi du thieu: {list(miss)[:5]}")
        # R15b: neu khong khop khoa nao thi ta dang do mo hinh NGAU NHIEN va tuong la da hoc.
        n_load = sum(1 for k in model.state_dict() if k in sd)
        print(f"    khoa khop: {n_load}/{len(model.state_dict())}")
        if n_load == 0:
            raise SystemExit("[!] KHONG khoa nao khop -- checkpoint sai kien truc, dung lai.")

    ds = st.SAMGOptimizedDataset(st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE,
                                  split_mode=args.split, pos_scale=scale.flatten()[0].item())
    from datasets.pl_data import ProteinLigandDataLoader
    batch = next(iter(ProteinLigandDataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                               num_workers=0))).to(device)
    with torch.no_grad():
        total, ld = model.forward(batch)

    if args.batches > 1:
        import numpy as _np
        acc = {}
        dl = ProteinLigandDataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for i, b in enumerate(dl):
                if i >= args.batches:
                    break
                _, d_ = model.forward(b.to(device))
                for k, v in d_.items():
                    acc.setdefault(k, []).append(float(v))
        print(f"\n[*] trung binh tren {args.batches} batch (batch_size={args.batch_size}):")
        for k in ["loss_parent", "loss_attach_self", "loss_attach_parent", "loss_link",
                  "loss_token", "loss_geo", "loss_total"]:
            if k in acc:
                a = _np.asarray(acc[k])
                print(f"    {k:20} = {a.mean():8.4f}  (sd {a.std():.4f}, n={a.size})")

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
    # Vi tri t co (t+1) lua chon: cot 0 = GOC, cot 1..t = cac fragment truoc do.
    # CE duoc lay trung binh tren MOI vi tri hop le, KE CA t=0 (chi 1 lua chon, ln(1)=0).
    # Loai t=0 ra se thoi phong ky vong len -- do la loi cua ban truoc (xem Muc 47).
    exp_par = float(np.mean([math.log(t + 1) for b in range(B) for t in range(lens[b])
                             if int(tp[b, t]) != -100] or [0.0]))
    exp_as = math.log(MAX_ATTACH)
    l_par = float(ld.get("loss_parent", float("nan")))
    l_as = float(ld.get("loss_attach_self", float("nan")))
    l_ap = float(ld.get("loss_attach_parent", float("nan")))
    # Nguong: mau nho (vai chuc vi tri) nen sai so chuan cua uoc luong CE ~0.1 nat;
    # cho 0.25 nat tuyet doi. Lech hon the => mask hoac muc tieu sai, khong phai nhieu.
    TOL = 0.25
    print(f"    loss_parent        = {l_par:.4f}   ly thuyet (ke ca t=0) = {exp_par:.4f}"
          f"   |lech| = {abs(l_par - exp_par):.4f}")
    print(f"    loss_attach_self   = {l_as:.4f}   ly thuyet ln({MAX_ATTACH}) = {exp_as:.4f}"
          f"   |lech| = {abs(l_as - exp_as):.4f}")
    print(f"    loss_attach_parent = {l_ap:.4f}   ly thuyet ln({MAX_ATTACH}) = {exp_as:.4f}"
          f"   |lech| = {abs(l_ap - exp_as):.4f}")
    ok4 = (abs(l_par - exp_par) <= TOL and abs(l_as - exp_as) <= TOL
           and abs(l_ap - exp_as) <= TOL)
    print(f"    (nguong |lech| <= {TOL})  => L4 {'DAT' if ok4 else 'KHONG DAT'}")
    print(f"    loss_link          = {float(ld.get('loss_link', float('nan'))):.4f}")
    print(f"    loss_token {float(ld['loss_token']):.4f}  loss_geo {float(ld['loss_geo']):.4f}  "
          f"loss_total {float(total):.4f}")
    print("    (lech xa ly thuyet => mask hoac muc tieu sai; gan => duong du lieu dung)")

    print("\n" + "=" * 86)
    print("TAT CA L1-L4 DAT => chay huan luyen lai."
          if (ok1 and bad == 0 and ok3 and ok4)
          else "CO TIEU CHI KHONG DAT -- khong huan luyen lai.")
    print("=" * 86)


if __name__ == "__main__":
    main()
