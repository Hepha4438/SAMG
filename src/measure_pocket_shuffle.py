"""
measure_pocket_shuffle.py -- M7 (plan.md "PLAN Rev 7 -- sau M6"; RESEARCH_CONTEXT Muc 25):
xac nhan tren tap VALID (du lieu CHUA he dung de doc tieu chi) ket qua tham do cua M6 (tap
train), va doi chung kich thuoc hoc cho rieng chieu `d`.

M6 dung CONG THUC SAI: R2 = 1 - sigma^2 do DO KHONG CHAC CHAN MO HINH TU KHAI BAO, khong
phai sai so thuc (RESEARCH_CONTEXT 25.2). Cong thuc DUNG, voi muc tieu da chuan hoa
(variance=1): MSE_i = sigma_i^2 * E[z_i^2]  =>  R2_i = 1 - sigma_i^2 * E[z_i^2]. Script nay
sua cong thuc do o MOI noi bao cao, moi nhanh; cot "1-sigma2" cu van duoc in nhung doi nhan
ro rang KHONG phai variance explained.

Bon nhanh, forward tren CUNG mot batch (pocket_shuffle_perm cua SAMGLightningModule.forward
-- KHONG sua step0/step1/step2/step3/step4/step5):
    full                 : pocket_shuffle_perm=None
    shuffled             : derangement NGAU NHIEN tren CA batch
    roll1                : torch.roll(arange(B), 1) -- hoan vi vong co dinh
    shuffled_sizematched : derangement TRONG TUNG NHOM tu phan vi cua pocket_extent (M7-b,
                           xem ham make_sizematched_perm) -- kiem xem hieu ung lon cua `d`
                           co phai chi la "doc kich co hoc" thay vi "doc hinh dang hoc".

pocket_extent lay tu CHINH tensor h_target ma attention hub (step3) THUC SU nhan duoc (mot
forward PRE-hook dang ky trong script, KHONG tinh lai bang code khac) tren NHANH full, chay
truoc tat ca cac nhanh khac trong moi batch.

KHONG train: torch.no_grad() quanh moi lan forward, model.train() CHI de hook log_scale co
san trong step2 chay (gate self.training) -- khong backward, khong optimizer.step().

VE VIEC VO HIEU HOA DEQUANTIZATION NOISE: ke thua ky thuat cua measure_logscale_realtokens.py
(M5) va M6 vi cung ly do ky thuat -- voi MAF, mu/log_scale cua chieu k phu thuoc cac chieu
dung truoc qua gia tri z DA GIAI NGHICH cua chung, nen de tinh z = (target_scaled-mu)/sigma
cho dung, target_scaled phai khop CHINH XAC voi gia tri thuc su dua vao log_prob() luc do.
Patch tam thoi torch.randn_like -> zeros_like quanh moi lan goi model.forward (restore ngay
sau bang try/finally).

Cong thuc NLL (xac minh truc tiep tren pyro-ppl==1.9.1, xem M4/M5):
    NLL = 0.5*sum_i(z_i^2) + 3.5*ln(2*pi) + sum_i(log_scale_i)

KIEM TRA KHONG-DOI (thay cho doi chieu Muc 24.1 cu -- RESEARCH_CONTEXT 25.1 chi ra do la
phep so sai: hai MAU batch khac nhau vi hai script dung shuffle khac nhau o loader, nen LUON
lech bat ke code co dung hay khong). Phep kiem MOI, CHINH XAC, TRONG CUNG mot batch: forward
voi pocket_shuffle_perm=None va forward voi pocket_shuffle_perm=torch.arange(B) (hoan vi dong
nhat) phai cho mu/log_scale BIT-IDENTICAL (torch.equal), vi ca hai la CUNG mot phep hoan vi ve
mat toan hoc. Chay tren 3 batch dau, in PASS/FAIL.

Tieu chi doc M7-a (ghi cung TRUOC khi chay, plan.md, CHI co y nghia xac nhan khi --split valid):
    delta R2(d) > 0.10  VA  delta R2(theta) > 0.015  VA  delta R2(phi) > 0.015
    VA ca ba deu DUONG o nhanh roll1 (cung dau voi shuffled)
        => XAC NHAN: hoc protein chi phoi tinh tien. Bang chung dau tien duoc XAC NHAN.
    delta R2(d) < 0.05
        => BAC: hieu ung o M6 (tap train) la hien vat cua tap train, khong tong quat hoa.
    Trung gian => khong ket luan, bao cao va dung.
    Rieng quaternion: nguong KHONG doi -- ky vong delta R2 < 0.02 o ca bon. Mot chieu vuot
    0.05 => ket luan M6 ve quaternion sai, phai doc lai.

Tieu chi doc M7-b (so voi CHINH delta R2(d) cua nhanh shuffled trong CUNG lan chay nay --
KHONG so voi hang so 0.171 cua M6/train, vi do la mau khac):
    delta R2_sizematched(d) < mot nua delta R2_shuffled(d)
        => phan lon hieu ung cua d la KICH CO hoc, khong phai hinh dang.
    delta R2_sizematched(d) > hai phan ba delta R2_shuffled(d)
        => mo hinh doc thong tin hoc DAC THU, khong chi kich co.

Cach chay (server GPU, dung python -u):
    python -u src/measure_pocket_shuffle.py --split valid [--ckpt PATH] [--batch-size N] | tee log.txt
"""
import os
import sys
import glob
import math
import time
import argparse
import itertools
import subprocess
import pickle

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]
QUAT_DIMS = ["qw", "qx", "qy", "qz"]
SEED = 1234


def print_header(device):
    samg_root = os.path.dirname(CURRENT_DIR)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=samg_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")
    print(f"SEED: {SEED}")
    print(f"device: {device}")


def default_ckpt(samg_root):
    preferred = os.path.join(samg_root, "saved_checkpoints_flow", "periodic-epoch=024.ckpt")
    if os.path.exists(preferred):
        return preferred
    ckpts = glob.glob(os.path.join(samg_root, "saved_checkpoints_flow", "*.ckpt"))
    return max(ckpts, key=os.path.getmtime) if ckpts else None


def make_derangement(n, device, max_tries=10):
    """torch.randperm cua n phan tu, khong co diem bat dong. Tra ve (perm, used_fallback)."""
    if n <= 1:
        return torch.roll(torch.arange(n, device=device), 1), True
    idx = torch.arange(n, device=device)
    for _ in range(max_tries):
        perm = torch.randperm(n, device=device)
        if not bool((perm == idx).any()):
            return perm, False
    return torch.roll(idx, 1), True


def make_sizematched_perm(ext, device, max_tries=10):
    """
    Chia B mau thanh 4 nhom theo tu phan vi cua ext (argsort roi chia 4 doan bang nhau),
    derangement TRONG tung nhom. Nhom size <= 1 giu nguyen (identity). Tra ve (perm, n_identity).
    """
    B = ext.shape[0]
    order = torch.argsort(ext)
    groups = torch.tensor_split(order, 4)
    perm = torch.arange(B, device=device)
    n_identity = 0
    for g in groups:
        gs = g.shape[0]
        if gs <= 1:
            n_identity += gs
            continue
        sub_perm, _ = make_derangement(gs, device, max_tries=max_tries)
        perm[g] = g[sub_perm]
    return perm, n_identity


def run_branch(model, batch, device, perm, debug=False):
    """Forward 1 lan voi pocket_shuffle_perm=perm (hoac None). Tra ve (mu, log_scale) tu hook
    rieng dang ky tren arn trong luc goi nay (go ngay sau). debug=True truyen tiep
    pocket_shuffle_debug=True vao step6 (M8) de in shape/is_contiguous() cua h_target/
    target_mask NGAY TAI diem ap perm -- CHI dung trong run_noop_check, KHONG dung o vong
    do chinh (se spam qua nhieu dong cho 200+ batch x 4 nhanh)."""
    captured = {}
    arn = model.generator.geometric_head.arn
    h = arn.register_forward_hook(
        lambda m, i, o: captured.update(mu=o[0].detach().clone(), log_scale=o[1].detach().clone())
    )
    try:
        with torch.no_grad():
            model.forward(batch, pocket_shuffle_perm=perm, pocket_shuffle_debug=debug)
    finally:
        h.remove()
    return captured["mu"].double(), captured["log_scale"].double()


def run_noop_check(model, loader, device, n_check=3):
    """
    M8 (RESEARCH_CONTEXT 26, thay cho phep kiem torch.equal cu -- 26.2 chi ra bit-identical
    la dieu KHONG THE dat duoc qua advanced indexing tren GPU ngay ca khi logic dung, nen
    torch.equal khong co kha nang PASS va khong phan xu duoc gi).

    forward pocket_shuffle_perm=None vs pocket_shuffle_perm=torch.arange(B) (hoan vi dong
    nhat, ve mat toan hoc la no-op) TRONG CUNG mot batch. In max|delta mu|, max|delta
    log_scale|, torch.allclose(atol=1e-5, rtol=1e-4) -> PASS/FAIL, VA van giu torch.equal
    nhung doi nhan ro no KY VONG FAIL tren GPU (khong phai dau hieu loi). debug=True tren
    ca hai lan goi de step6 in shape/is_contiguous() cua h_target/target_mask NGAY TAI diem
    ap perm (doc cung voi max|delta| de phan xu (i) vo hai vs (ii) pha huy).
    """
    print("\n" + "=" * 90)
    print(f"KIEM TRA KHONG-DOI (M8): perm=None vs perm=torch.arange(B), TRONG CUNG batch, "
          f"{n_check} batch dau. pocket_shuffle_debug=True (xem cac dong [pocket_shuffle_debug]).")
    print("=" * 90)
    max_deltas = []
    for i, batch in enumerate(itertools.islice(loader, n_check)):
        batch = batch.to(device)
        num_graphs = batch.protein_element_batch.max().item() + 1
        identity_perm = torch.arange(num_graphs, device=device)

        print(f"\n  --- batch {i}: perm=None ---")
        mu_none, ls_none = run_branch(model, batch, device, None, debug=True)
        print(f"  --- batch {i}: perm=torch.arange(B) ---")
        mu_id, ls_id = run_branch(model, batch, device, identity_perm, debug=True)

        max_delta_mu = (mu_none - mu_id).abs().max().item()
        max_delta_ls = (ls_none - ls_id).abs().max().item()
        max_deltas.extend([max_delta_mu, max_delta_ls])

        allclose_ok = (torch.allclose(mu_none, mu_id, atol=1e-5, rtol=1e-4)
                       and torch.allclose(ls_none, ls_id, atol=1e-5, rtol=1e-4))
        bitwise_equal = torch.equal(mu_none, mu_id) and torch.equal(ls_none, ls_id)

        print(f"    max|delta mu|        = {max_delta_mu:.3e}")
        print(f"    max|delta log_scale| = {max_delta_ls:.3e}")
        print(f"    torch.allclose(atol=1e-5, rtol=1e-4): {'PASS' if allclose_ok else 'FAIL'}")
        print(f"    bit-identical (torch.equal) (ky vong FAIL tren GPU, KHONG phai loi): "
              f"{'PASS' if bitwise_equal else 'FAIL'}")

    overall_max_delta = max(max_deltas)
    print(f"\n  max|delta| toan bo ({n_check} batch, ca mu va log_scale) = {overall_max_delta:.3e}")

    print("\n  TIEU CHI DOC (plan.md M8 -- PHAI doi chieu CA voi shape in boi cac dong "
          "[pocket_shuffle_debug] o tren):")
    if overall_max_delta < 1e-4:
        print(f"    max|delta| = {overall_max_delta:.3e} < 1e-4. NEU dong [pocket_shuffle_debug]")
        print("    o tren cho thay shape tai diem ap la 3 chieu [B, R_max, H+3]")
        print("    => (i) VO HAI: chi la lam tron do layout bo nho. Muc 25 va M7 DUOC PHUC HOI,")
        print("       doc binh thuong.")
    elif overall_max_delta > 1e-2:
        print(f"    max|delta| = {overall_max_delta:.3e} > 1e-2")
        print("    => (ii) PHA HUY: perm dang duoc ap vao tensor sai hinh dang. Muc 25 va M7 BI")
        print("       HUY HOAN TOAN, phai sua step6 roi chay lai M6 va M7.")
    else:
        print(f"    max|delta| = {overall_max_delta:.3e} o khoang trung gian (1e-4 - 1e-2)")
        print("    => KHONG KET LUAN. Xem max|delta mu| va max|delta log_scale| rieng cho tung")
        print("       batch o tren de truy (hai 'lop' duy nhat co the tach duoc o muc nay).")
    print("    Bat ke max|delta| la bao nhieu: NEU dong [pocket_shuffle_debug] cho thay shape")
    print("    KHONG phai 3 chieu => (ii) PHA HUY, ghi de len ket luan tu max|delta| o tren.")

    return overall_max_delta


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(device)
    torch.manual_seed(SEED)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Mac dinh: saved_checkpoints_flow/periodic-epoch=024.ckpt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-batches", type=int, default=200,
                     help="So batch toi da (bo qua neu --split valid: dung het tap valid)")
    ap.add_argument("--split", choices=["train", "valid", "test"], default="train")
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

    # Checkpoint nay la MAF (RESEARCH_CONTEXT 20.9) -- geo_head TUONG MINH (giong het
    # measure_logscale_realtokens.py dong 113-114).
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

    # --split: dung CHINH filtering/phan chia cua SAMGOptimizedDataset (import lai, khong tu chia).
    dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode=args.split, pos_scale=pos_scale
    )
    from datasets.pl_data import ProteinLigandDataLoader
    # shuffle=False LUON: cac nhanh/lan chay/phep kiem khong-doi phai doc CUNG thu tu batch.
    loader = ProteinLigandDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.split == "valid":
        n_batches_target = len(loader)
        print(f"[*] --split valid: dung het {n_batches_target} batch cua valid (bo qua --n-batches).")
    else:
        n_batches_target = args.n_batches

    print(f"\n[*] CAU HINH LOADER: split={args.split}  shuffle=False  batch_size={args.batch_size}  "
          f"n_batches_du_kien={n_batches_target}")

    # Yeu cau 4: kiem tra khong-doi TRUOC khi do (dung mot vong lap loader rieng -- DataLoader
    # voi shuffle=False la deterministic nen vong sau van doc dung thu tu batch tu dau).
    run_noop_check(model, loader, device, n_check=3)

    gen_shift = model.generator.shift_factors.view(1, 7).double()
    gen_scale = model.generator.scale_factors.view(1, 7).double()

    branches = ["full", "shuffled", "roll1", "shuffled_sizematched"]
    ls_chunks = {b: [] for b in branches}
    z_chunks = {b: [] for b in branches}

    # Yeu cau 3: pre-hook TREN attention hub (step3) de bat h_target/target_mask ma no THUC SU
    # nhan duoc tren nhanh full -- KHONG tinh lai bang code khac. with_kwargs=True vi target_mask
    # duoc step4 truyen nhu tham so tu khoa (xem step4_ligand_generator.py).
    captured_ht = {}

    def _ht_pre_hook(module, args_, kwargs_):
        captured_ht["h_target"] = args_[1].detach().clone()
        captured_ht["target_mask"] = kwargs_["target_mask"].detach().clone()

    ht_hook_handle = model.generator.attention_hub.register_forward_pre_hook(_ht_pre_hook, with_kwargs=True)

    original_randn_like = torch.randn_like

    def _zero_randn_like(*a, **kw):
        return torch.zeros_like(*a, **kw)

    n_batches_done = 0
    n_real_so_far = 0
    n_fallback_shuffled = 0
    n_sizematched_identity = 0
    n_total_graphs = 0
    t_start = time.time()
    try:
        torch.randn_like = _zero_randn_like
        for i, batch in enumerate(loader):
            if i >= n_batches_target:
                break
            batch = batch.to(device)

            num_graphs = batch.protein_element_batch.max().item() + 1
            n_total_graphs += num_graphs

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
            target_scaled_clean = (target_7d_raw.view(-1, 7) - gen_shift) / gen_scale
            target_real = target_scaled_clean[pad_mask]

            # NHANH full TRUOC (bat buoc chay dau) -- hook pre-hook bat h_target/target_mask
            # cua CHINH nhanh nay de tinh pocket_extent cho shuffled_sizematched.
            mu_full, ls_full = run_branch(model, batch, device, None)
            h_target_full = captured_ht["h_target"]
            target_mask_full = captured_ht["target_mask"]
            ext = (h_target_full[..., -3:].norm(dim=-1) * target_mask_full.to(h_target_full.dtype)).amax(dim=1)

            roll1_perm = torch.roll(torch.arange(num_graphs, device=device), 1)
            shuffled_perm, used_fallback = make_derangement(num_graphs, device)
            if used_fallback:
                n_fallback_shuffled += 1
            sizematched_perm, n_identity = make_sizematched_perm(ext, device)
            n_sizematched_identity += n_identity

            branch_mu_ls = {"full": (mu_full, ls_full)}
            for branch, perm in [("shuffled", shuffled_perm), ("roll1", roll1_perm),
                                  ("shuffled_sizematched", sizematched_perm)]:
                branch_mu_ls[branch] = run_branch(model, batch, device, perm)

            for branch in branches:
                mu, log_scale = branch_mu_ls[branch]
                if log_scale.shape[0] != pad_mask.shape[0]:
                    raise RuntimeError(
                        f"[{branch}] log_scale ({log_scale.shape[0]} hang) va pad_mask "
                        f"({pad_mask.shape[0]} hang) khong khop."
                    )
                ls_real = log_scale[pad_mask]
                mu_real = mu[pad_mask]
                z_real = (target_real - mu_real) / ls_real.exp()
                ls_chunks[branch].append(ls_real)
                z_chunks[branch].append(z_real)

            n_batches_done += 1
            n_real_so_far += int(pad_mask.sum().item())

            if (i + 1) % 20 == 0:
                print(f"batch {i + 1}/{n_batches_target}, n_real={n_real_so_far} "
                      f"(elapsed {time.time() - t_start:.1f}s)", flush=True)
    finally:
        torch.randn_like = original_randn_like
        ht_hook_handle.remove()

    n_real_total = sum(t.shape[0] for t in ls_chunks["full"])
    print(f"\n[*] CAU HINH LOADER (thuc dung): split={args.split}  shuffle=False  "
          f"batch_size={args.batch_size}  n_batches={n_batches_done}  n_token_that={n_real_total}")
    print(f"[*] So lan nhanh 'shuffled' fallback sang roll1 (khong sinh duoc derangement sau 10 lan): "
          f"{n_fallback_shuffled}/{n_batches_done}")
    pct_identity = n_sizematched_identity / n_total_graphs * 100 if n_total_graphs > 0 else 0.0
    print(f"[*] So mau KHONG di chuyen duoc trong sizematched (nhom tu phan vi size<=1): "
          f"{n_sizematched_identity}/{n_total_graphs} ({pct_identity:.2f}%)")
    if pct_identity > 20.0:
        print("    [!!!] CANH BAO: > 20% mau khong di chuyen duoc trong shuffled_sizematched -- nhanh")
        print("    nay bi yeu di va KHONG doc duoc. Tang --batch-size de cac nhom tu phan vi co nhieu")
        print("    mau hon.")

    stats = {}  # stats[branch][dim] = {"mean_ls","sigma","ez2","r2","one_minus_sigma2"}
    mean_nll = {}
    for branch in branches:
        ls = torch.cat(ls_chunks[branch], dim=0).cpu()
        z = torch.cat(z_chunks[branch], dim=0).cpu()
        n_real = ls.shape[0]
        print(f"[*] [{branch}] n token THAT: {n_real}")

        stats[branch] = {}
        for i, name in enumerate(DIM_NAMES):
            mean_ls = ls[:, i].mean().item()
            sigma = math.exp(mean_ls)
            ez2 = (z[:, i] ** 2).mean().item()
            # Yeu cau 1: R2 DUNG = 1 - sigma^2 * E[z^2] (RESEARCH_CONTEXT 25.2), KHONG phai
            # 1 - sigma^2 (do do khong chac chan tu khai bao, khong phai variance explained).
            r2 = 1 - (sigma ** 2) * ez2
            one_minus_sigma2 = 1 - sigma ** 2
            stats[branch][name] = {
                "mean_ls": mean_ls, "sigma": sigma, "ez2": ez2,
                "r2": r2, "one_minus_sigma2": one_minus_sigma2,
            }

        mean_nll[branch] = (0.5 * (z ** 2).sum(dim=-1) + 3.5 * math.log(2 * math.pi) + ls.sum(dim=-1)).mean().item()

    for branch in branches:
        print("\n" + "=" * 100)
        print(f"NHANH: {branch}")
        print("=" * 100)
        print(f"{'chieu':8}{'mean(log_sigma)':>18}{'sigma':>10}{'E[z^2]':>10}{'R2 (dung)':>12}"
              f"{'1-sigma2 (do tu khai bao, KHONG phai variance explained)':>60}")
        for name in DIM_NAMES:
            s = stats[branch][name]
            print(f"{name:8}{s['mean_ls']:18.4f}{s['sigma']:10.4f}{s['ez2']:10.4f}{s['r2']:12.4f}"
                  f"{s['one_minus_sigma2']:60.4f}")
        print(f"    mean(NLL) = {mean_nll[branch]:.4f}")

    print("\n" + "=" * 100)
    print("BANG CHENH LECH R2 (DUNG): full-shuffled, full-roll1, full-sizematched "
          "(sap giam dan theo full-shuffled)")
    print("=" * 100)
    diffs = []
    for name in DIM_NAMES:
        r2_full = stats["full"][name]["r2"]
        d_shuf = r2_full - stats["shuffled"][name]["r2"]
        d_roll = r2_full - stats["roll1"][name]["r2"]
        d_size = r2_full - stats["shuffled_sizematched"][name]["r2"]
        diffs.append((name, d_shuf, d_roll, d_size))
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"{'chieu':8}{'full-shuffled':>16}{'full-roll1':>14}{'full-sizematched':>20}")
    for name, d_shuf, d_roll, d_size in diffs:
        print(f"{name:8}{d_shuf:16.4f}{d_roll:14.4f}{d_size:20.4f}")

    print("\n" + "=" * 100)
    print("TIEU CHI DOC M7-a (plan.md 'PLAN Rev 7') -- d/theta/phi, doi chung CHINH la 'shuffled'")
    print("=" * 100)
    dR2 = lambda name, branch: stats["full"][name]["r2"] - stats[branch][name]["r2"]
    d_d_shuf, d_theta_shuf, d_phi_shuf = dR2("d", "shuffled"), dR2("theta", "shuffled"), dR2("phi", "shuffled")
    d_d_roll, d_theta_roll, d_phi_roll = dR2("d", "roll1"), dR2("theta", "roll1"), dR2("phi", "roll1")
    print(f"    delta R2(d)     shuffled={d_d_shuf:.4f}      roll1={d_d_roll:.4f}")
    print(f"    delta R2(theta) shuffled={d_theta_shuf:.4f}      roll1={d_theta_roll:.4f}")
    print(f"    delta R2(phi)   shuffled={d_phi_shuf:.4f}      roll1={d_phi_roll:.4f}")
    same_sign_positive = d_d_roll > 0 and d_theta_roll > 0 and d_phi_roll > 0
    if d_d_shuf > 0.10 and d_theta_shuf > 0.015 and d_phi_shuf > 0.015 and same_sign_positive:
        print("    => XAC NHAN: hoc protein chi phoi tinh tien. Bang chung dau tien DUOC XAC NHAN cua du an.")
    elif d_d_shuf < 0.05:
        print("    => BAC: hieu ung o M6 (tap train) la hien vat cua tap train, khong tong quat hoa. Doc lai.")
    else:
        print("    => TRUNG GIAN: khong ket luan. Bao cao va dung.")

    print("\n    Rieng quaternion (nguong KHONG doi, ky vong delta R2 < 0.02 o ca bon; "
          "mot chieu vuot 0.05 => ket luan ve quaternion sai, phai doc lai):")
    quat_warn = False
    for name in QUAT_DIMS:
        d_q = dR2(name, "shuffled")
        flag = ""
        if d_q > 0.05:
            flag = "  <-- VUOT 0.05, CANH BAO"
            quat_warn = True
        print(f"        delta R2({name}) = {d_q:.4f}{flag}")
    if quat_warn:
        print("    [!!!] CO CHIEU QUATERNION VUOT 0.05 -- ket luan M6 ve quaternion (khong chi phoi)")
        print("    co the sai, phai doc lai RESEARCH_CONTEXT Muc 24-25.")

    print("\n" + "=" * 100)
    print("TIEU CHI DOC M7-b (plan.md 'PLAN Rev 7') -- doi chung kich thuoc cho chieu d")
    print("=" * 100)
    d_d_size = dR2("d", "shuffled_sizematched")
    half = d_d_shuf / 2
    two_thirds = d_d_shuf * 2 / 3
    print(f"    delta R2(d) shuffled             = {d_d_shuf:.4f}")
    print(f"    delta R2(d) shuffled_sizematched = {d_d_size:.4f}")
    print(f"    mot nua delta R2(d) shuffled      = {half:.4f}")
    print(f"    hai phan ba delta R2(d) shuffled  = {two_thirds:.4f}")
    if d_d_size < half:
        print("    => TUT XUONG DUOI MOT NUA: phan lon hieu ung cua d la KICH CO hoc, khong phai hinh dang.")
    elif d_d_size > two_thirds:
        print("    => GIU TREN HAI PHAN BA: mo hinh doc thong tin hoc DAC THU cua pocket, khong chi kich co.")
    else:
        print("    => O GIUA mot nua va hai phan ba -- khong ket luan ro rang theo hai moc da ghi trong plan.md.")


if __name__ == "__main__":
    main()
