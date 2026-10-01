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

VE NHIEU DEQUANTIZATION (M10, plan.md "PLAN Rev 9"; RESEARCH_CONTEXT Muc 27.2): M5/M6/M7 zero
hoa nhieu (torch.randn_like -> zeros_like) de z = (target_scaled-mu)/sigma tinh dung (MAF can
target_scaled khop CHINH XAC gia tri dua vao log_prob() do tinh autoregressive). M8's
run_noop_check KHONG zero nhieu va phat hien nguyen nhan FAIL cua no la step4_ligand_
generator.py:123 ve mot nhieu MOI moi lan forward (27.2). M10 THAY phep zero-hoa bang KHOA
SEED: truoc MOI lan forward cua MOI nhanh trong CUNG mot batch, dat lai torch.manual_seed(1234
+ batch_idx) + torch.cuda.manual_seed_all(...) (xem run_branch(seed=...)) de moi nhanh ve
CUNG mot lan nhieu -- giu nhieu that (khong zero) de phan phoi input giong luc huan luyen, chi
khoa cho no GIONG HET nhau giua cac nhanh thay vi khac nhau ngau nhien.

Cong thuc NLL (xac minh truc tiep tren pyro-ppl==1.9.1, xem M4/M5):
    NLL = 0.5*sum_i(z_i^2) + 3.5*ln(2*pi) + sum_i(log_scale_i)

KIEM TRA KHONG-DOI: tu M10 CHI con la CHAN DOAN (khong phai tieu chi quyet dinh nua -- 27.2 cho
thay FAIL cua M8 la do nhieu dequant chua khoa, khong phai loi logic perm). Tieu chi THAT nam o
nhanh `noop` (M10-2, so sanh trong don vi R2) o cuoi script.

NHANH `noop` (M10-2, plan.md "PLAN Rev 9"): them nhanh thu nam pocket_shuffle_perm=torch.
arange(B) -- DONG NHAT VE LOGIC voi `full`, cung chay qua toan bo loader/vong lap chinh (khong
chi 3 batch dau nhu phep kiem CHAN DOAN o tren). delta R2(full-noop) o CA 7 chieu la SAN NHIEU
(noise floor) dung don vi cua tieu chi M7-a/M7-b -- xem "TIEU CHI DOC M10" cuoi script.

Tieu chi doc M10 (ghi cung TRUOC khi chay, plan.md "PLAN Rev 9", RESEARCH_CONTEXT Muc 27) --
PHAI doc TRUOC M7-a/M7-b, ca ba dieu kien, KHONG tu sua nguong sau khi thay so:
    1. delta R2(full-noop) < 0.002 o MOI chieu (7/7), VA max|delta mu|/max|delta log_scale|
       (full vs noop, chan doan) < 1e-4
        => KHONG dat: con nguon ngau nhien chua khoa. DUNG, tim tiep, KHONG doc gi khac.
    2. Dat (1) thi moi doc M7-a/M7-b, VOI dieu kien bo sung: delta R2(d, shuffled) phai LON HON
       delta R2(d, noop) it nhat 10 lan. Duoi 10 lan => tin hieu khong tach khoi san nhieu,
       khong ket luan (du dat nguong M7-a).
    3. Nguong M7-a/M7-b (xem duoi) GIU NGUYEN, khong duoc sua sau khi thay so.

Tieu chi doc M7-a (ghi cung TRUOC khi chay, plan.md, CHI co y nghia xac nhan khi --split valid
VA M10 dieu kien 1+2 o tren DAT):
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
    --loader-shuffle (M10-b, RESEARCH_CONTEXT 27.6): them co nay de chay --split train voi
    shuffle=True (giai nghich ly valid > train -- xem tieu chi M10-b trong plan.md).
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


def run_branch(model, batch, device, perm, debug=False, seed=None):
    """Forward 1 lan voi pocket_shuffle_perm=perm (hoac None). Neu seed khac None, dat lai
    torch.manual_seed(seed) + torch.cuda.manual_seed_all(seed) NGAY TRUOC lan forward nay
    (M10-1, RESEARCH_CONTEXT 27.2/plan.md "PLAN Rev 9") de nhanh nay ve CUNG mot lan nhieu
    dequant (step4_ligand_generator.py:123) nhu cac nhanh khac GOI VOI CUNG seed trong CUNG
    mot batch -- khong con zero-hoa nhieu nhu M5-M7. Tra ve (mu, log_scale) tu hook rieng dang
    ky tren arn trong luc goi nay (go ngay sau). debug=True truyen tiep pocket_shuffle_debug=
    True vao step6 (M8) de in shape/is_contiguous() cua h_target/target_mask NGAY TAI diem ap
    perm -- CHI dung trong run_noop_check, KHONG dung o vong do chinh (se spam qua nhieu dong
    cho 200+ batch x 5 nhanh)."""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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
    CHAN DOAN (M10, plan.md "PLAN Rev 9"; RESEARCH_CONTEXT 27.2) -- KHONG CON LA TIEU CHI.
    M8 dung phep nay lam tieu chi (i)/(ii) va FAIL; 27.2 xac dinh nguyen nhan FAIL la nhieu
    dequant ve lai moi lan forward (step4_ligand_generator.py:123), khong phai loi logic perm.
    Tu M10, seed duoc khoa truoc moi forward (xem run_branch(seed=...)) nen hai nhanh nay ve
    CUNG mot nhieu; cac so duoi day chi con dung de CHAN DOAN shape/contiguous/do lon -- TIEU
    CHI THAT nam o bang delta R2(full-noop) cuoi script (ham main()).
    """
    print("\n" + "=" * 90)
    print(f"CHAN DOAN KHONG-DOI (khong con la tieu chi, M10): perm=None vs perm=torch.arange(B), "
          f"seed khoa (1234+batch_idx), TRONG CUNG batch, {n_check} batch dau.")
    print("=" * 90)
    max_deltas = []
    for i, batch in enumerate(itertools.islice(loader, n_check)):
        batch = batch.to(device)
        num_graphs = batch.protein_element_batch.max().item() + 1
        identity_perm = torch.arange(num_graphs, device=device)

        print(f"\n  --- batch {i}: perm=None ---")
        mu_none, ls_none = run_branch(model, batch, device, None, debug=True, seed=SEED + i)
        print(f"  --- batch {i}: perm=torch.arange(B) ---")
        mu_id, ls_id = run_branch(model, batch, device, identity_perm, debug=True, seed=SEED + i)

        max_delta_mu = (mu_none - mu_id).abs().max().item()
        max_delta_ls = (ls_none - ls_id).abs().max().item()
        max_deltas.extend([max_delta_mu, max_delta_ls])

        allclose_ok = (torch.allclose(mu_none, mu_id, atol=1e-5, rtol=1e-4)
                       and torch.allclose(ls_none, ls_id, atol=1e-5, rtol=1e-4))
        bitwise_equal = torch.equal(mu_none, mu_id) and torch.equal(ls_none, ls_id)

        print(f"    max|delta mu|        = {max_delta_mu:.3e}  [chan doan]")
        print(f"    max|delta log_scale| = {max_delta_ls:.3e}  [chan doan]")
        print(f"    torch.allclose(atol=1e-5, rtol=1e-4) [chan doan]: {'PASS' if allclose_ok else 'FAIL'}")
        print(f"    bit-identical (torch.equal) [chan doan, ky vong FAIL tren GPU vi advanced "
              f"indexing, KHONG phai loi]: {'PASS' if bitwise_equal else 'FAIL'}")

    overall_max_delta = max(max_deltas)
    print(f"\n  max|delta| toan bo ({n_check} batch, ca mu va log_scale) = {overall_max_delta:.3e} [chan doan]")
    print("  (Tieu chi THAT nam o bang 'delta R2(full-noop)' cuoi script -- dung don vi cua cau hoi.)")

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
    ap.add_argument("--loader-shuffle", action="store_true", default=False,
                     help="M10-b (RESEARCH_CONTEXT 27.6): shuffle=True cho loader, kiem xem "
                          "nghich ly valid>train co phai hien vat cua tap con co thu tu khong")
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
    # shuffle theo --loader-shuffle (mac dinh False; M10-b/27.6 -- kiem nghich ly valid>train
    # co phai hien vat cua "200 batch DAU theo thu tu dataset" cua train khong).
    loader = ProteinLigandDataLoader(
        dataset, batch_size=args.batch_size, shuffle=args.loader_shuffle, num_workers=0
    )

    if args.split == "valid":
        n_batches_target = len(loader)
        print(f"[*] --split valid: dung het {n_batches_target} batch cua valid (bo qua --n-batches).")
    else:
        n_batches_target = args.n_batches

    print(f"\n[*] CAU HINH LOADER: split={args.split}  shuffle={args.loader_shuffle}  "
          f"batch_size={args.batch_size}  n_batches_du_kien={n_batches_target}")

    # CHAN DOAN khong-doi TRUOC khi do (dung mot vong lap loader rieng; neu --loader-shuffle
    # thi lan lap nay va vong do chinh ben duoi KHONG doc cung thu tu batch -- khong sao vi
    # day chi con la chan doan, khong phai tieu chi, xem run_noop_check()).
    run_noop_check(model, loader, device, n_check=3)

    gen_shift = model.generator.shift_factors.view(1, 7).double()
    gen_scale = model.generator.scale_factors.view(1, 7).double()

    branches = ["full", "shuffled", "roll1", "shuffled_sizematched", "noop"]
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

    n_batches_done = 0
    n_real_so_far = 0
    n_fallback_shuffled = 0
    n_sizematched_identity = 0
    n_total_graphs = 0
    noop_diag_max_deltas = []  # M10-2: chan doan rieng (don vi mu/log_scale), xem M10 dieu kien 1
    t_start = time.time()
    try:
        for i, batch in enumerate(loader):
            if i >= n_batches_target:
                break
            batch = batch.to(device)
            seed_i = SEED + i  # M10-1: moi nhanh trong batch nay dung CUNG seed -- CUNG nhieu dequant

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
            mu_full, ls_full = run_branch(model, batch, device, None, seed=seed_i)
            h_target_full = captured_ht["h_target"]
            target_mask_full = captured_ht["target_mask"]
            ext = (h_target_full[..., -3:].norm(dim=-1) * target_mask_full.to(h_target_full.dtype)).amax(dim=1)

            roll1_perm = torch.roll(torch.arange(num_graphs, device=device), 1)
            shuffled_perm, used_fallback = make_derangement(num_graphs, device)
            if used_fallback:
                n_fallback_shuffled += 1
            sizematched_perm, n_identity = make_sizematched_perm(ext, device)
            n_sizematched_identity += n_identity
            noop_perm = torch.arange(num_graphs, device=device)  # M10-2: san nhieu

            branch_mu_ls = {"full": (mu_full, ls_full)}
            for branch, perm in [("shuffled", shuffled_perm), ("roll1", roll1_perm),
                                  ("shuffled_sizematched", sizematched_perm), ("noop", noop_perm)]:
                branch_mu_ls[branch] = run_branch(model, batch, device, perm, seed=seed_i)

            mu_noop, ls_noop = branch_mu_ls["noop"]
            noop_diag_max_deltas.append((mu_full - mu_noop).abs().max().item())
            noop_diag_max_deltas.append((ls_full - ls_noop).abs().max().item())

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
        ht_hook_handle.remove()

    n_real_total = sum(t.shape[0] for t in ls_chunks["full"])
    print(f"\n[*] CAU HINH LOADER (thuc dung): split={args.split}  shuffle={args.loader_shuffle}  "
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
    print("BANG CHENH LECH R2 (DUNG): full-shuffled, full-roll1, full-sizematched, full-noop "
          "(sap giam dan theo full-shuffled; full-noop = SAN NHIEU, M10)")
    print("=" * 100)
    dR2 = lambda name, branch: stats["full"][name]["r2"] - stats[branch][name]["r2"]
    diffs = []
    for name in DIM_NAMES:
        d_shuf = dR2(name, "shuffled")
        d_roll = dR2(name, "roll1")
        d_size = dR2(name, "shuffled_sizematched")
        d_noop = dR2(name, "noop")
        diffs.append((name, d_shuf, d_roll, d_size, d_noop))
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"{'chieu':8}{'full-shuffled':>16}{'full-roll1':>14}{'full-sizematched':>20}{'full-noop (SAN NHIEU)':>24}")
    for name, d_shuf, d_roll, d_size, d_noop in diffs:
        print(f"{name:8}{d_shuf:16.4f}{d_roll:14.4f}{d_size:20.4f}{d_noop:24.4f}")

    print("\n" + "=" * 100)
    print("TIEU CHI DOC M10 (plan.md 'PLAN Rev 9', RESEARCH_CONTEXT Muc 27) -- SAN NHIEU TRUOC M7-a/M7-b")
    print("=" * 100)
    d_noop_by_dim = {name: dR2(name, "noop") for name in DIM_NAMES}
    max_abs_d_noop = max(abs(v) for v in d_noop_by_dim.values())
    noop_diag_overall_max_delta = max(noop_diag_max_deltas) if noop_diag_max_deltas else float("nan")
    for name in DIM_NAMES:
        print(f"    delta R2(full-noop)[{name}] = {d_noop_by_dim[name]:.4f}")
    print(f"    delta R2(full-noop) toi da tren 7 chieu = {max_abs_d_noop:.4f}  (dieu kien 1: < 0.002 o MOI chieu)")
    print(f"    max|delta mu|/max|delta log_scale| (full vs noop, toan bo {n_batches_done} batch) = "
          f"{noop_diag_overall_max_delta:.3e}  (dieu kien 1: < 1e-4)")
    m10_cond1_ok = max_abs_d_noop < 0.002 and noop_diag_overall_max_delta < 1e-4
    if not m10_cond1_ok:
        print("    => DIEU KIEN 1 KHONG DAT: con nguon ngau nhien chua khoa. DUNG, tim tiep, KHONG")
        print("       doc M7-a/M7-b ben duoi (cac so van duoc in de tham khao/debug).")
    else:
        print("    => DIEU KIEN 1 DAT: san nhieu da triet tieu. Doc tiep M7-a/M7-b voi dieu kien 2")
        print("       bo sung: delta R2(d,shuffled) phai > 10 x delta R2(d,noop).")

    print("\n" + "=" * 100)
    print("TIEU CHI DOC M7-a (plan.md 'PLAN Rev 7') -- d/theta/phi, doi chung CHINH la 'shuffled'")
    print("=" * 100)
    d_d_shuf, d_theta_shuf, d_phi_shuf = dR2("d", "shuffled"), dR2("theta", "shuffled"), dR2("phi", "shuffled")
    d_d_roll, d_theta_roll, d_phi_roll = dR2("d", "roll1"), dR2("theta", "roll1"), dR2("phi", "roll1")
    print(f"    delta R2(d)     shuffled={d_d_shuf:.4f}      roll1={d_d_roll:.4f}")
    print(f"    delta R2(theta) shuffled={d_theta_shuf:.4f}      roll1={d_theta_roll:.4f}")
    print(f"    delta R2(phi)   shuffled={d_phi_shuf:.4f}      roll1={d_phi_roll:.4f}")
    same_sign_positive = d_d_roll > 0 and d_theta_roll > 0 and d_phi_roll > 0

    d_noop_d = d_noop_by_dim["d"]
    signal_vs_noise_ratio = abs(d_d_shuf) / abs(d_noop_d) if d_noop_d != 0 else float("inf")
    signal_vs_noise_ok = signal_vs_noise_ratio > 10
    print(f"    (M10 dieu kien 2) delta R2(d,shuffled) / delta R2(d,noop) = {signal_vs_noise_ratio:.1f}x "
          f"(can > 10x)")

    if not m10_cond1_ok:
        print("    => KHONG DOC: M10 dieu kien 1 chua dat (xem muc TIEU CHI DOC M10 o tren).")
    elif d_d_shuf > 0.10 and d_theta_shuf > 0.015 and d_phi_shuf > 0.015 and same_sign_positive:
        if not signal_vs_noise_ok:
            print("    => KHONG KET LUAN: dat nguong M7-a nhung KHONG dat dieu kien 2 cua M10 (ty le "
                  f"{signal_vs_noise_ratio:.1f}x <= 10x) -- tin hieu khong tach khoi san nhieu.")
        else:
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
    if not m10_cond1_ok:
        print("    (M10 dieu kien 1 chua dat -- KHONG ket luan tu cac so nay.)")
    elif quat_warn:
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
    if not m10_cond1_ok:
        print("    => KHONG DOC: M10 dieu kien 1 chua dat (xem muc TIEU CHI DOC M10 o tren).")
    elif d_d_size < half:
        print("    => TUT XUONG DUOI MOT NUA: phan lon hieu ung cua d la KICH CO hoc, khong phai hinh dang.")
    elif d_d_size > two_thirds:
        print("    => GIU TREN HAI PHAN BA: mo hinh doc thong tin hoc DAC THU cua pocket, khong chi kich co.")
    else:
        print("    => O GIUA mot nua va hai phan ba -- khong ket luan ro rang theo hai moc da ghi trong plan.md.")


if __name__ == "__main__":
    main()
