"""
measure_qy_signflip.py -- M4 (plan.md "PLAN Rev 5 -- bo sung sau M3"): kiem du doan dinh
luong cua gia thuyet (a) -- rang buoc dai so ||q||=1 lam `qy` "doan duoc" vi no dung SAU
ca ba `qw, qx, qz` trong permutation cua ARN (da phuc hoi tu checkpoint o M3,
RESEARCH_CONTEXT 22.1-22.2; `pinfrac(qy) = 0` o 22.5 mau thuan voi phien ban don gian cua
(a), nen can kiem truc tiep truoc khi hanh dong).

Du doan can kiem (RESEARCH_CONTEXT 21.4): neu (a) giai thich TRON VEN, `log_scale` cua
`qy` phai LUONG CUC theo su kien
    dom = |qy| > max(|qw|, |qx|, |qz|)   (tinh tu target_7d RAW, TRUOC chuan hoa)
-- nhom dom=True (qy la thanh phan troi nhat, dau cua no bi quy uoc canonicalize "thanh
phan lon nhat mang dau duong" buoc phai duong) phai co `log_scale` THAP hon han nhom
dom=False.

Dung checkpoint saved_checkpoints_flow/periodic-epoch=024.ckpt -- day la MAF (xem
RESEARCH_CONTEXT 20.9), PHAI load voi geo_head="maf" tuong minh (xem P3b: truyen config=
khi load_from_checkpoint se ghi de hparams da luu, khong duoc dua vao mac dinh). Mot batch
train that, KHONG train: forward boc torong torch.no_grad(), khong backward, khong
optimizer.step().

VE CO CHE LAY log_scale PER-SAMPLE (khong sua step2_mdn_module.py):
Hook `_log_scale_hook` co san trong AutoregressiveFlowLayer.__init__ (step2) CHI cong don
thanh _ls_sum/_ls_count (phuc vu trung binh toan epoch) va chi chay khi self.training=True
-- khong du de lay gia tri per-sample cho tung token. Script nay dang ky THEM mot forward
hook rieng tren chinh `self.arn`, chi GHI DE (khong cong don) mot bien capture moi lan
hook duoc goi; sau khi forward xong, bien do giu dung LAN GOI CUOI CUNG.

Day la gia tri DUNG, khong phai xap xi: da doc truc tiep tren source pyro-ppl==1.9.1
(pyro/distributions/transforms/affine_autoregressive.py, AffineAutoregressive._inverse(),
duoc ConditionalAffineAutoregressive.condition() goi toi qua mot functools.partial boc
quanh chinh self.arn -- nen van di qua nn.Module.__call__ va kich hoat hook binh thuong).
`_inverse()` lap qua permutation D=7 lan, moi lan goi self.arn(x) voi x dang duoc dien dan
tung chieu theo dung thu tu permutation. Do tinh chat MADE, log_scale[...,k] cho chieu k
CHI phu thuoc cac chieu dung TRUOC k trong permutation -- nen no da ON DINH/DUNG ngay tu
lan goi ma tat ca cac chieu dung truoc k da duoc dien vao x, va KHONG doi nua o cac lan goi
sau do (ke ca lan goi CUOI CUNG, vi luc do moi chieu da co chieu dung truoc no duoc dien
dung). Vi vay bat ky chieu nao, lay tu LAN GOI CUOI CUNG deu la gia tri da hoi tu/dung.

Hook log_scale CHI chay khi self.training=True (gate trong step2), nen can model.train();
toan bo phep do duoc boc trong torch.no_grad() nen khong co gradient/cap nhat trong so.

Cach chay:
    python src/measure_qy_signflip.py [--ckpt PATH] [--batch-size N]
"""
import os
import sys
import glob
import argparse
import subprocess
import pickle

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]
QY_IDX = DIM_NAMES.index("qy")


def print_git_head():
    samg_root = os.path.dirname(CURRENT_DIR)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=samg_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")


def default_ckpt(samg_root):
    preferred = os.path.join(samg_root, "saved_checkpoints_flow", "periodic-epoch=024.ckpt")
    if os.path.exists(preferred):
        return preferred
    ckpts = glob.glob(os.path.join(samg_root, "saved_checkpoints_flow", "*.ckpt"))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


def histogram_text(x, n_bins=40, width=50):
    x_np = np.asarray(x)
    counts, edges = np.histogram(x_np, bins=n_bins)
    total = counts.sum()
    max_c = counts.max() if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar_len = int(round((c / max_c) * width))
        pct = (c / total * 100) if total > 0 else 0.0
        lines.append(f"    [{edges[i]:+8.3f}, {edges[i+1]:+8.3f}) {'#' * bar_len:<{width}} {c:6d} ({pct:5.2f}%)")
    return "\n".join(lines)


def group_stats(name, ls, ls_floor):
    if ls.numel() == 0:
        print(f"    {name}: KHONG CO DU LIEU")
        return
    mean_v = ls.mean().item()
    median_v = ls.median().item()
    pinfrac = (ls <= ls_floor + 1e-3).float().mean().item()
    print(f"    {name}: n={ls.numel():6d}  mean={mean_v:.4f}  median={median_v:.4f}  pinfrac={pinfrac:.4f}")


def main():
    print_git_head()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Mac dinh: saved_checkpoints_flow/periodic-epoch=024.ckpt")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import step6_trainer as st
    from omegaconf import OmegaConf

    ckpt_path = args.ckpt or default_ckpt(st.SAMG_ROOT)
    if ckpt_path is None:
        raise SystemExit("[!] Khong tim thay checkpoint nao trong saved_checkpoints_flow/. Truyen --ckpt.")
    print(f"[*] Checkpoint: {ckpt_path}")

    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(st.VOCAB_PATH):
        with open(st.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

    scaler_path = os.path.join(st.PROCESSED_DIR, "scaler_7d.pt")
    if os.path.exists(scaler_path):
        scaler_dict = torch.load(scaler_path, map_location="cpu")
        shift_factors, scale_factors = scaler_dict["shift"], scaler_dict["scale"]
        pos_scale = scale_factors.flatten()[0].item()
    else:
        shift_factors, scale_factors, pos_scale = None, None, None

    # Checkpoint nay la MAF (RESEARCH_CONTEXT 20.9) -- geo_head TUONG MINH.
    config = OmegaConf.create({
        "hidden_dim": 256, "num_heads": 4, "lr": 1e-4,
        "ligand_mode": "empty", "geo_head": "maf",
        "protein_encoder": {
            "num_blocks": 3, "num_layers": 3, "hidden_dim": 256,
            "n_heads": 4, "knn": 16, "edge_feat_dim": 5, "num_r_gaussian": 20, "num_node_types": 8
        },
        "loss_weights": {"token": 1.0, "geo": 1.0, "pocket": 1.0, "int": 0.0, "d_threshold": 2.5}
    })

    model = st.SAMGLightningModule.load_from_checkpoint(
        ckpt_path, map_location="cpu", config=config, vocab_size=len(vocab),
        shift_factors=shift_factors, scale_factors=scale_factors, strict=False,
    )
    model.train()  # can cho hook co san trong step2 chay (gate self.training); xem docstring

    train_dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode="train", pos_scale=pos_scale
    )
    from datasets.pl_data import ProteinLigandDataLoader
    loader = ProteinLigandDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    # Hook RIENG tren self.arn (KHONG sua step2_mdn_module.py) -- xem docstring dau file
    # ve vi sao LAN GOI CUOI CUNG la gia tri dung cho MOI chieu.
    arn = model.generator.geometric_head.arn
    captured = {}
    hook_handle = arn.register_forward_hook(
        lambda m, i, o: captured.__setitem__("log_scale", o[1].detach().clone())
    )

    with torch.no_grad():
        model.forward(batch)
    hook_handle.remove()

    log_scale = captured["log_scale"]  # [N, 7], N = batch_size * seq_len (flatten theo hang)

    # pad_mask + target_7d RAW: lap lai DUNG phep pad_sequence cua SAMGLightningModule.forward()
    # tren cac truong co san cua batch -- khong co ham rieng de import cho doan reshape nay.
    lens = batch.seq_len.tolist()
    if isinstance(lens, int):
        lens = [lens]
    target_ids = torch.nn.utils.rnn.pad_sequence(
        list(torch.split(batch.target_ids, lens)), batch_first=True, padding_value=0
    )
    target_7d_raw = torch.nn.utils.rnn.pad_sequence(
        list(torch.split(batch.target_7d, lens)), batch_first=True, padding_value=0.0
    )

    pad_mask = (target_ids != 0).view(-1)
    target_7d_flat = target_7d_raw.view(-1, 7)

    if log_scale.shape[0] != pad_mask.shape[0]:
        raise RuntimeError(
            f"log_scale ({log_scale.shape[0]} hang) va pad_mask ({pad_mask.shape[0]} hang) "
            f"khong khop -- kiem tra lai batch_size*seq_len."
        )

    log_scale_qy = log_scale[:, QY_IDX][pad_mask]
    target_7d_valid = target_7d_flat[pad_mask]

    qw = target_7d_valid[:, DIM_NAMES.index("qw")]
    qx = target_7d_valid[:, DIM_NAMES.index("qx")]
    qy_raw = target_7d_valid[:, DIM_NAMES.index("qy")]
    qz = target_7d_valid[:, DIM_NAMES.index("qz")]

    others_max = torch.stack([qw.abs(), qx.abs(), qz.abs()], dim=-1).max(dim=-1).values
    dom = qy_raw.abs() > others_max

    n_total = dom.numel()
    n_dom = int(dom.sum().item())
    print(f"\n[*] So token THAT dung de do: {n_total}")
    print(f"[*] Ty le dom = |qy| > max(|qw|,|qx|,|qz|): {n_dom}/{n_total} = "
          f"{n_dom / n_total * 100:.2f}%  (du doan ~25%)")

    ls_floor = getattr(model.generator.geometric_head, "_ls_floor", -5.0)

    print("\n" + "=" * 78)
    print("log_scale_qy -- histogram 40 bin (TOAN BO token that)")
    print("=" * 78)
    print(histogram_text(log_scale_qy.numpy()))

    ls_dom_true = log_scale_qy[dom]
    ls_dom_false = log_scale_qy[~dom]

    print("\n" + "=" * 78)
    print("log_scale_qy -- histogram rieng cho dom=True")
    print("=" * 78)
    print(histogram_text(ls_dom_true.numpy()) if ls_dom_true.numel() > 0 else "    KHONG CO DU LIEU")

    print("\n" + "=" * 78)
    print("log_scale_qy -- histogram rieng cho dom=False")
    print("=" * 78)
    print(histogram_text(ls_dom_false.numpy()) if ls_dom_false.numel() > 0 else "    KHONG CO DU LIEU")

    print("\n" + "=" * 78)
    print("TONG KET mean/median/pinfrac theo nhom dom")
    print("=" * 78)
    group_stats("TOAN BO  ", log_scale_qy, ls_floor)
    group_stats("dom=True ", ls_dom_true, ls_floor)
    group_stats("dom=False", ls_dom_false, ls_floor)

    print("\n" + "=" * 78)
    print("TIEU CHI DOC (plan.md M4, ghi cung TRUOC khi chay)")
    print("=" * 78)
    if ls_dom_true.numel() > 0 and ls_dom_false.numel() > 0:
        diff = ls_dom_false.mean().item() - ls_dom_true.mean().item()
        print(f"    mean(dom=False) - mean(dom=True) = {diff:.4f} nat")
        if diff > 1.5:
            print("    => log_scale_qy LUONG CUC (chenh > 1.5 nat, dom=True thap hon):")
            print("       (a) GIAI THICH TRON VEN. Ket thuc tranh luan. Sang P3a -> diagonal")
            print("       head (10 epoch) -> G5.")
        else:
            print("    => log_scale_qy ON DINH/khong tach biet du (chenh <= 1.5 nat):")
            print("       (a) CHI LA MOT PHAN. Mo hinh co the co thong tin ve DAU cua qy tu")
            print("       nguon khac (token C3, d/theta/phi, hoac hoc that) -- phai tach nguon")
            print("       do truoc khi ket luan gi ve hoc.")
    else:
        print("    KHONG DU DU LIEU O MOT TRONG HAI NHOM DE SO SANH -- tang --batch-size.")

    if abs(n_dom / n_total - 0.25) > 0.10:
        print(f"\n    [!] Ty le dom ({n_dom / n_total * 100:.2f}%) lech xa 0.25 -- xem lai gia")
        print("        dinh doi xung o RESEARCH_CONTEXT 21.4.")


if __name__ == "__main__":
    main()
