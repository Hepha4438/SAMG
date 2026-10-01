"""
measure_logscale_realtokens.py -- M5 (plan.md "PLAN Rev 5 -- bo sung sau M4"): do lai
log_scale 7 chieu CHI tren token that, tren nhieu batch -- thay the bang phan ra Muc 20.4
(bi nhiem padding NANG, xem RESEARCH_CONTEXT 23.4: qy -1,357 toan hang vs -0,768 token
that, lech 43%. Bang 20.4 khong con dung lam can cu).

KHONG sua step2_mdn_module.py / step4_ligand_generator.py, KHONG train. Checkpoint
saved_checkpoints_flow/periodic-epoch=024.ckpt (MAF, RESEARCH_CONTEXT 20.9), geo_head=
"maf" tuong minh (P3b: config= khi load_from_checkpoint ghi de hparams da luu).

VE VIEC VO HIEU HOA NHIEU DEQUANTIZATION KHI DO (quan trong, doc truoc khi dung ket qua):
step4_ligand_generator.py cong them noise ngau nhien (torch.randn_like(...) * DEQUANT_SIGMA)
vao target_scaled TRUOC khi dua vao geometric_head. Voi MAF, `mu`/`log_scale` cua chieu k
(tai 1 lan goi arn) la ham CUA CAC CHIEU DUNG TRUOC k trong permutation (qua gia tri z da
giai nghich cua chung) -- nen de tinh z_k = (y_k - mu_k)/sigma_k cho DUNG, `y` phai la
CHINH XAC gia tri da duoc dua vao log_prob() luc do, khong phai mot ban noise-ngau-nhien-
khac-lan do TU minh ve lai. Vi script nay khong doc duoc noise noi bo cua mot lan forward
that, no PATCH tam thoi `torch.randn_like` thanh tra ve 0 (chi trong luc goi model.forward,
restore ngay sau) de buoc noise = 0 mot cach CO KIEM SOAT -- luc do `target_scaled` ma script
tu tinh lai (raw - shift)/scale KHOP CHINH XAC voi `y` thuc su duoc dua vao flow, va z tinh
duoc la DUNG, khong xap xi. Day la phep do trong CHE DO SACH (khong nhieu dequant), mot
chan doan tat dinh va tu nhat quan hon la co gang khop mot nguon ngau nhien khong quan sat
duoc. DEQUANT_SIGMA=0.1 nho so voi std=1 cua khong gian da chuan hoa nen anh huong con lai
toi ket luan ve calibration la khong dang ke.

Cong thuc NLL dung de doi chieu (xac minh truc tiep tren pyro-ppl==1.9.1,
AffineAutoregressive.log_abs_det_jacobian = log_scale.sum(-1), TransformedDistribution.
log_prob = base_log_prob(x) - log_abs_det_jacobian):
    NLL = 0.5*sum_i(z_i^2) + 3.5*ln(2*pi) + sum_i(log_scale_i)

Cach chay:
    python src/measure_logscale_realtokens.py [--ckpt PATH] [--batch-size N] [--n-batches N]
"""
import os
import sys
import glob
import math
import time
import argparse
import subprocess
import pickle

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]
QUAT_DIMS = ["qw", "qx", "qy", "qz"]


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


def main():
    print_git_head()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Mac dinh: saved_checkpoints_flow/periodic-epoch=024.ckpt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-batches", type=int, default=200, help="So batch toi da (>= 200 theo yeu cau M5)")
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
        ckpt_path, map_location=device, config=config, vocab_size=len(vocab),
        shift_factors=shift_factors, scale_factors=scale_factors, strict=False,
    )
    model.to(device)
    model.train()  # can cho hook co san trong step2 chay (gate self.training); forward boc no_grad ben duoi

    train_dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode="train", pos_scale=pos_scale
    )
    from datasets.pl_data import ProteinLigandDataLoader
    loader = ProteinLigandDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    geo_head = model.generator.geometric_head
    arn = geo_head.arn
    gen_shift = model.generator.shift_factors.view(1, 7).double()
    gen_scale = model.generator.scale_factors.view(1, 7).double()
    ls_floor = getattr(geo_head, "_ls_floor", -5.0)

    captured = {}
    hook_handle = arn.register_forward_hook(
        lambda m, i, o: captured.update(mu=o[0].detach().clone(), log_scale=o[1].detach().clone())
    )

    ls_all_chunks = []     # log_scale, MOI hang (ke ca padding)
    ls_real_chunks = []    # log_scale, CHI token that
    z_real_chunks = []     # z = (target_scaled_clean - mu) / exp(log_scale), CHI token that

    # Vo hieu hoa dequantization noise MOT CACH CO KIEM SOAT trong luc forward -- xem docstring.
    original_randn_like = torch.randn_like

    def _zero_randn_like(*a, **kw):
        return torch.zeros_like(*a, **kw)

    n_batches_done = 0
    n_real_so_far = 0
    t_start = time.time()
    try:
        torch.randn_like = _zero_randn_like
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            batch = batch.to(device)
            captured.clear()
            with torch.no_grad():
                model.forward(batch)
            n_batches_done += 1

            mu = captured["mu"].double()
            log_scale = captured["log_scale"].double()

            lens = batch.seq_len.tolist()
            if isinstance(lens, int):
                lens = [lens]
            target_ids = torch.nn.utils.rnn.pad_sequence(
                list(torch.split(batch.target_ids, lens)), batch_first=True, padding_value=0
            )
            target_7d_raw = torch.nn.utils.rnn.pad_sequence(
                list(torch.split(batch.target_7d, lens)), batch_first=True, padding_value=0.0
            ).double()

            pad_mask = (target_ids != 0).view(-1)
            target_7d_flat = target_7d_raw.view(-1, 7)
            # Noise da bi vo hieu hoa (patch o tren) nen day CHINH LA target_scaled thuc su
            # duoc dua vao geometric_head trong lan forward nay.
            target_scaled_clean = (target_7d_flat - gen_shift) / gen_scale

            if log_scale.shape[0] != pad_mask.shape[0]:
                raise RuntimeError(
                    f"log_scale ({log_scale.shape[0]} hang) va pad_mask ({pad_mask.shape[0]} hang) khong khop."
                )

            ls_all_chunks.append(log_scale)
            ls_real_chunks.append(log_scale[pad_mask])
            mu_real = mu[pad_mask]
            target_real = target_scaled_clean[pad_mask]
            z_real = (target_real - mu_real) / log_scale[pad_mask].exp()
            z_real_chunks.append(z_real)
            n_real_so_far += int(pad_mask.sum().item())

            if (i + 1) % 20 == 0:
                print(f"batch {i + 1}/{args.n_batches}, n_real={n_real_so_far} "
                      f"(elapsed {time.time() - t_start:.1f}s)", flush=True)
    finally:
        torch.randn_like = original_randn_like
        hook_handle.remove()

    print(f"[*] So batch da dung: {n_batches_done} (yeu cau toi thieu 200; dung het loader neu it hon)")

    ls_all = torch.cat(ls_all_chunks, dim=0).cpu()
    ls_real = torch.cat(ls_real_chunks, dim=0).cpu()
    z_real = torch.cat(z_real_chunks, dim=0).cpu()
    n_real = ls_real.shape[0]
    n_all = ls_all.shape[0]
    print(f"[*] n token THAT (sau pad_mask): {n_real}")
    print(f"[*] n hang TOAN BO (ke ca padding): {n_all}")

    print("\n" + "=" * 100)
    print(f"{'chieu':8}{'mean(real)':>14}{'median(real)':>14}{'std(real)':>12}{'pinfrac(real)':>15}"
          f"{'n(real)':>10}{'mean(ALL hang)':>16}")
    print("=" * 100)
    mean_ls_real_per_dim = {}
    ez2_per_dim = {}
    for i, name in enumerate(DIM_NAMES):
        col_real = ls_real[:, i]
        col_all = ls_all[:, i]
        mean_r = col_real.mean().item()
        median_r = col_real.median().item()
        std_r = col_real.std(unbiased=False).item()
        pin_r = (col_real <= ls_floor + 1e-3).float().mean().item()
        mean_a = col_all.mean().item()
        mean_ls_real_per_dim[name] = mean_r
        ez2_per_dim[name] = (z_real[:, i] ** 2).mean().item()
        print(f"{name:8}{mean_r:14.4f}{median_r:14.4f}{std_r:12.4f}{pin_r:15.4f}{n_real:10d}{mean_a:16.4f}")

    print("\n" + "=" * 100)
    print("DO NHIEM PADDING (chenh lech tuong doi giua mean(ALL hang) va mean(real), theo %)")
    print("=" * 100)
    for name in DIM_NAMES:
        mean_r = mean_ls_real_per_dim[name]
        mean_a = ls_all[:, DIM_NAMES.index(name)].mean().item()
        rel = abs(mean_a - mean_r) / (abs(mean_r) + 1e-12) * 100
        print(f"    {name:7}: mean(real)={mean_r:+.4f}  mean(ALL)={mean_a:+.4f}  lech={rel:5.2f}%")

    print("\n" + "=" * 100)
    print("E[z^2] per-dim tren token that (calibration: ky vong ~1.0 neu dung)")
    print("=" * 100)
    for name in DIM_NAMES:
        print(f"    {name:7}: E[z^2] = {ez2_per_dim[name]:.4f}")

    mean_nll = (0.5 * (z_real ** 2).sum(dim=-1) + 3.5 * math.log(2 * math.pi) + ls_real.sum(dim=-1)).mean().item()
    sum_mean_ls = sum(mean_ls_real_per_dim.values())
    sum_ez2 = sum(ez2_per_dim.values())
    nll_pred = 3.5 * math.log(2 * math.pi) + 0.5 * sum_ez2 + sum_mean_ls

    print("\n" + "=" * 100)
    print("PHAN RA NLL (doi chieu voi Muc 20.4, tren token that)")
    print("=" * 100)
    print(f"    sum_i mean(log_scale_i)        = {sum_mean_ls:.4f}")
    print(f"    sum_i E[z_i^2]                  = {sum_ez2:.4f}")
    print(f"    NLL do truc tiep (mean, token that) = {mean_nll:.4f}")
    print(f"    NLL du doan tu cong thuc            = {nll_pred:.4f}")
    lech_nll = abs(mean_nll - nll_pred)
    print(f"    |lech|                              = {lech_nll:.4f} nat  "
          f"({'DANG TIN' if lech_nll < 0.2 else 'CON NGUON CHUA HIEU -- DUNG'})")

    print("\n" + "=" * 100)
    print("TIEU CHI DOC (plan.md M5): R2_i = 1 - sigma_i^2, sigma_i = exp(mean(log_scale_i)) tren token that")
    print("=" * 100)
    n_low_r2 = 0
    for name in QUAT_DIMS:
        sigma_i = math.exp(mean_ls_real_per_dim[name])
        r2_i = 1 - sigma_i ** 2
        flag = r2_i < 0.15
        n_low_r2 += int(flag)
        print(f"    {name:7}: sigma={sigma_i:.4f}  R2={r2_i:.4f}  {'<0.15 (khong hoc)' if flag else '>=0.15'}")

    print(f"\n    So chieu quay (qw,qx,qy,qz) co R2 < 0.15: {n_low_r2}/4")
    if n_low_r2 >= 3:
        print("    => 20.4 DUOC CHUNG MINH LAI: dau hinh hoc khong hoc gi ve huong ngoai rang buoc")
        print("       ||q||=1. Sang P3a -> diagonal head -> G5 nhu ke hoach.")
    elif n_low_r2 <= 1:
        print("    => 20.4 BI BAC: so bi nhiem padding da tao ra hinh mau gia. Phai doc lai toan")
        print("       bo Muc 20-23 truoc khi lam gi tiep.")
    else:
        print("    => KHONG KET LUAN (2/4 chieu). Bao cao va dung.")


if __name__ == "__main__":
    main()
