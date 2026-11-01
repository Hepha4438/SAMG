"""check_pool.py -- kiem S16-S18 (Rev 23) TRUOC khi huan luyen lai. Khong train.

  P1 pool[0] == 0, do dai == L+1, va pool[L] == 0 VOI MOI phan tu.
     `pool[L] == 0` la dang thuc 69.2 (`sum(d) = 2 x so fragment co cha`). Day la phep kiem
     manh nhat: no kiem ca cach tinh pool, ca truong `parent` cua step0, ca `atomic_nums`.
  P2 pool[j] >= 1 o moi vi tri j ma fragment j CO CHA (phai co cho de gan).
  P3 khong vi tri nao cham tran POOL_MAX (neu cham thi tran dat qua thap).
  P4 pool co THUC SU doi logits_vocab khong. Day la phep kiem duy nhat co the that bai neu
     dac trung chua duoc noi vao: truyen hai pool khac nhau, logits phai khac nhau; va
     pool = None (lop "khong biet") phai khac ca hai.
  P5 pool KHONG duoc anh huong loss_geo va ba loss lien ket -- chung phai y nguyen nhu Rev 22,
     neu khong thi chenh lech do duoc se lan sang kenh hinh hoc (R5).
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
    ap.add_argument("--n-mol", type=int, default=2000, help="so phan tu kiem P1/P2/P3 (0 = het)")
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
    from step4_ligand_generator import POOL_MAX
    from omegaconf import OmegaConf
    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    sc = torch.load(os.path.join(st.PROCESSED_DIR, "scaler_7d.pt"), map_location="cpu")
    shift, scale = sc["shift"], sc["scale"]

    ds = st.SAMGOptimizedDataset(st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE,
                                  split_mode=args.split, pos_scale=scale.flatten()[0].item())

    # ------------------------------------------------------------ P1, P2, P3
    print("\n" + "=" * 86 + "\nP1/P2/P3: dang thuc pool, tinh kha thi, va tran\n" + "=" * 86)
    n = len(ds) if args.n_mol == 0 else min(args.n_mol, len(ds))
    bad1 = bad2 = bad3 = 0
    pool_max_seen = 0
    for i in range(n):
        d = ds[i]
        ps = d.pool_state.numpy()
        L = int(d.seq_len) - 1                       # seq_len = L + 1 (co [EOS])
        if ps.size != L + 1 or int(ps[0]) != 0 or int(ps[-1]) != 0:
            bad1 += 1
        par = d.link_parent.numpy()                  # lop 0 = GOC, j+1 = fragment j, -100 = EOS
        for j in range(L):
            if int(par[j]) != 0 and int(ps[j]) < 1:  # fragment j CO CHA nhung khong co cho gan
                bad2 += 1
                break
        pool_max_seen = max(pool_max_seen, int(ps.max()))
        if int(ps.max()) > POOL_MAX:
            bad3 += 1
    print(f"    kiem {n} phan tu  (POOL_MAX = {POOL_MAX})")
    print(f"    P1 vi pham: {bad1}/{n}   (pool[0]==0, do dai==L+1, pool[L]==0)")
    print(f"    P2 vi pham: {bad2}/{n}   (pool[j]>=1 o moi fragment CO CHA)")
    print(f"    P3 vi pham: {bad3}/{n}   (pool <= POOL_MAX);  pool lon nhat thay duoc = {pool_max_seen}")
    ok1, ok2, ok3 = bad1 == 0, bad2 == 0, bad3 == 0
    for nm, o in (("P1", ok1), ("P2", ok2), ("P3", ok3)):
        print(f"    => {nm} {'DAT' if o else 'KHONG DAT'}")

    # ------------------------------------------------------------ P4, P5
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
    from datasets.pl_data import ProteinLigandDataLoader
    batch = next(iter(ProteinLigandDataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                               num_workers=0))).to(device)
    lens = batch.seq_len.tolist()
    if isinstance(lens, int):
        lens = [lens]
    pad = torch.nn.utils.rnn.pad_sequence

    def P(x, v):
        return pad(list(torch.split(x, lens)), batch_first=True, padding_value=v).to(device)

    iid, tid = P(batch.input_ids, 0), P(batch.target_ids, 0)
    t7, gm = P(batch.target_7d, 0.0), P(batch.geo_mask, 0.0)
    ps = P(batch.pool_state, 0)
    # P5 doi hoi so sanh hai forward BIT-GIONG-NHAU tru dau vao pool. Hai nguon ngau nhien phai
    # xu ly, neu khong phep kiem khong the pass (da mac dung loi nay mot lan, xem Muc 70):
    #  (a) `step4` rut nhieu dequantization moi forward: `noise = randn_like(...) * DEQUANT_SIGMA`
    #      => phai reset seed TRUOC tung lan goi.
    #  (b) `logits_parent` co -inf o cac o bi mask; `(-inf) - (-inf) = nan` => chi so o o HUU HAN.
    def run(pool_arg):
        torch.manual_seed(20231102)
        with torch.no_grad():
            return model.generator(iid, h_t, [], target_7d=t7, target_mask=t_mask,
                                    target_ids=tid, geo_mask=gm, pool_state=pool_arg)

    with torch.no_grad():
        h_t, t_mask, _, _ = model.build_h_target(batch)
    out_a = run(ps)
    out_b = run(ps + 3)
    out_n = run(None)
    # Chung minh viec reset seed co tac dung: hai lan goi CUNG pool phai cho loss_geo Y NGUYEN.
    out_a2 = run(ps)
    d_rep = float((out_a[1] - out_a2[1]).abs().max())
    print(f"\n[*] kiem tien de: hai lan goi cung pool -> max|loss_geo lech| = {d_rep:.3e}"
          f"  ({'tat dinh, phep kiem co hieu luc' if d_rep == 0.0 else 'VAN NGAU NHIEN -- P5 vo nghia'})")

    print("\n" + "=" * 86 + "\nP4: pool co THUC SU doi logits_vocab khong\n" + "=" * 86)
    dab = float((out_a[0] - out_b[0]).abs().max())
    dan = float((out_a[0] - out_n[0]).abs().max())
    print(f"    max|logits(pool) - logits(pool+3)|      = {dab:.6f}")
    print(f"    max|logits(pool) - logits(khong biet)|  = {dan:.6f}")
    ok4 = dab > 1e-4 and dan > 1e-4
    print(f"    => P4 {'DAT' if ok4 else 'KHONG DAT: dac trung pool CHUA duoc noi vao'}")

    print("\n" + "=" * 86 + "\nP5: pool KHONG duoc anh huong kenh hinh hoc va lien ket\n" + "=" * 86)
    dgeo = float((out_a[1] - out_b[1]).abs().max())
    dlink = 0.0
    for k in ("parent", "attach_self", "attach_parent"):
        A, B = out_a[4][k], out_b[4][k]
        fin = torch.isfinite(A) & torch.isfinite(B)
        # cac o bi mask phai bi mask GIONG NHAU o ca hai; neu khac thi do la ro ri that
        if not torch.equal(torch.isfinite(A), torch.isfinite(B)):
            dlink = float("inf")
            print(f"    [!] {k}: vi tri bi mask KHAC nhau giua hai pool -- ro ri that")
            continue
        if fin.any():
            dlink = max(dlink, float((A[fin] - B[fin]).abs().max()))
    print(f"    max|loss_geo(pool) - loss_geo(pool+3)|   = {dgeo:.3e}   (phai = 0)")
    print(f"    max|link_logits(pool) - link(pool+3)|    = {dlink:.3e}   (phai = 0; chi o o huu han)")
    ok5 = (d_rep == 0.0) and dgeo == 0.0 and dlink == 0.0
    print(f"    => P5 {'DAT' if ok5 else 'KHONG DAT: pool dang ro sang duong khac, se lan nguyen nhan'}")

    with torch.no_grad():
        _, ld = model.forward(batch)
    print(f"\n    loss_token = {float(ld['loss_token']):.4f}   (ly thuyet khoi tao ~ "
          f"ln({len(vocab)}) = {math.log(len(vocab)):.4f})")
    for k in ("loss_geo", "loss_parent", "loss_attach_self", "loss_attach_parent"):
        if k in ld:
            print(f"    {k:20} = {float(ld[k]):.4f}")

    print("\n" + "=" * 86)
    print("TAT CA P1-P5 DAT => huan luyen lai."
          if all((ok1, ok2, ok3, ok4, ok5))
          else "CO TIEU CHI KHONG DAT -- khong huan luyen lai.")
    print("=" * 86)


if __name__ == "__main__":
    main()
