"""
measure_pocket_shuffle.py -- M6 (plan.md "PLAN Rev 6 -- SUA KE HOACH: M6 thay cho G5 o
buoc dau"; RESEARCH_CONTEXT 24.7): phep thu RE truoc khi dau tu 15 gio GPU vao G5 (train
lai 2 nhanh x 10 epoch). Dung LAI checkpoint da train san (periodic-epoch=024.ckpt, MAF),
khong train gi them -- chi forward ba lan tren cung mot batch voi dieu kien hoa pocket
(h_target/target_mask) bi hoan vi theo ba cach khac nhau, de xem `theta` va `phi` (hai
chieu DUY NHAT con lai, khong co kenh dai so ||q||=1, xem 24.2/24.7) co nhay voi danh tinh
pocket hay khong.

Ba nhanh, forward tren CUNG mot batch (pocket_shuffle_perm cua SAMGLightningModule.forward,
them o commit truoc -- KHONG sua step2/step3/step4):
    full     : pocket_shuffle_perm=None (duong chay binh thuong)
    shuffled : mot hoan vi NGAU NHIEN khong co diem bat dong (derangement) theo chieu batch
    roll1    : torch.roll(arange(B), 1) -- hoan vi vong CO DINH, khong co diem bat dong khi B>1

KHONG train: torch.no_grad() quanh moi lan forward, model.train() CHI de hook log_scale co
san trong step2 chay (gate self.training) -- khong backward, khong optimizer.step().

VE VIEC VO HIEU HOA DEQUANTIZATION NOISE: ke thua dung ky thuat cua
measure_logscale_realtokens.py (M5) va vi CUNG mot ly do ky thuat -- voi MAF, mu/log_scale
cua chieu k phu thuoc cac chieu dung truoc qua gia tri z DA GIAI NGHICH cua chung, nen de
tinh z = (target_scaled - mu)/sigma cho DUNG, target_scaled phai KHOP CHINH XAC voi gia tri
thuc su dua vao log_prob() luc do. Patch tam thoi torch.randn_like -> zeros_like trong luc
goi model.forward (restore ngay sau bang try/finally) de buoc noise = 0 mot cach co kiem
soat; xem docstring day du cua measure_logscale_realtokens.py.

Cong thuc NLL (xac minh truc tiep tren pyro-ppl==1.9.1, xem M4/M5):
    NLL = 0.5*sum_i(z_i^2) + 3.5*ln(2*pi) + sum_i(log_scale_i)

Tieu chi doc (ghi cung TRUOC khi chay, plan.md "G5 Rev 2", chi ap cho theta/phi, nhanh
shuffled la doi chung CHINH):
    R2_theta(full)-R2_theta(shuffled) > 0.05  VA  dieu tuong tu cho phi
        => hoc protein CO dong gop vao tu the. Bang chung dau tien co that. Di tiep G5 day du.
    Ca hai chenh < 0.02
        => hoc KHONG dong gop gi. Van de o encoder/conditioning, khong o head/bieu dien quay.
    Mot chieu vuot 0.05, chieu kia duoi 0.02
        => khong ket luan; bao cao va dung.

KIEM TRA KHONG-DOI: nhanh full phai cho R2 xap xi bang RESEARCH_CONTEXT 24.1 (do tren
checkpoint nay, khong shuffle). Lech > 0.02 o bat ky chieu nao => CANH BAO: them tham so
pocket_shuffle_perm co the da lam doi duong chay mac dinh, hoac loader doc batch khac.

Cach chay (server GPU, dung python -u):
    python -u src/measure_pocket_shuffle.py [--ckpt PATH] [--batch-size N] [--n-batches N] | tee log.txt
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
SEED = 1234

# RESEARCH_CONTEXT 24.1 -- bang phan ra sach tren checkpoint periodic-epoch=024.ckpt,
# KHONG shuffle. Dung de kiem tra khong-doi cho nhanh "full".
REFERENCE_R2_FULL = {
    "d": 0.518, "theta": 0.186, "phi": 0.183,
    "qw": 0.409, "qx": 0.049, "qy": 0.792, "qz": 0.235,
}


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


def make_derangement(num_graphs, device, max_tries=10):
    """torch.randperm khong co diem bat dong. Tra ve (perm, used_fallback)."""
    if num_graphs <= 1:
        return torch.roll(torch.arange(num_graphs, device=device), 1), True
    idx = torch.arange(num_graphs, device=device)
    for _ in range(max_tries):
        perm = torch.randperm(num_graphs, device=device)
        if not bool((perm == idx).any()):
            return perm, False
    return torch.roll(idx, 1), True


def run_branch(model, batch, device, perm):
    """Forward 1 lan voi pocket_shuffle_perm=perm (hoac None). Tra ve (mu, log_scale) tu hook."""
    captured = {}
    arn = model.generator.geometric_head.arn
    h = arn.register_forward_hook(
        lambda m, i, o: captured.update(mu=o[0].detach().clone(), log_scale=o[1].detach().clone())
    )
    try:
        with torch.no_grad():
            model.forward(batch, pocket_shuffle_perm=perm)
    finally:
        h.remove()
    return captured["mu"].double(), captured["log_scale"].double()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(device)
    torch.manual_seed(SEED)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Mac dinh: saved_checkpoints_flow/periodic-epoch=024.ckpt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-batches", type=int, default=200, help="So batch toi da (>= 200 nhu M5)")
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

    train_dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode="train", pos_scale=pos_scale
    )
    from datasets.pl_data import ProteinLigandDataLoader
    # shuffle=False: ba nhanh (va cac lan chay khac nhau) doc CUNG thu tu batch.
    loader = ProteinLigandDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    gen_shift = model.generator.shift_factors.view(1, 7).double()
    gen_scale = model.generator.scale_factors.view(1, 7).double()

    branches = ["full", "shuffled", "roll1"]
    ls_chunks = {b: [] for b in branches}
    z_chunks = {b: [] for b in branches}

    original_randn_like = torch.randn_like

    def _zero_randn_like(*a, **kw):
        return torch.zeros_like(*a, **kw)

    n_batches_done = 0
    n_real_so_far = 0
    n_fallback_shuffled = 0
    t_start = time.time()
    try:
        torch.randn_like = _zero_randn_like
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            batch = batch.to(device)

            num_graphs = batch.protein_element_batch.max().item() + 1

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

            roll1_perm = torch.roll(torch.arange(num_graphs, device=device), 1)
            shuffled_perm, used_fallback = make_derangement(num_graphs, device)
            if used_fallback:
                n_fallback_shuffled += 1

            perms = {"full": None, "shuffled": shuffled_perm, "roll1": roll1_perm}

            for branch in branches:
                mu, log_scale = run_branch(model, batch, device, perms[branch])
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
                print(f"batch {i + 1}/{args.n_batches}, n_real={n_real_so_far} "
                      f"(elapsed {time.time() - t_start:.1f}s)", flush=True)
    finally:
        torch.randn_like = original_randn_like

    print(f"\n[*] So batch da dung: {n_batches_done} (yeu cau toi thieu 200; dung het loader neu it hon)")
    print(f"[*] So lan nhanh 'shuffled' fallback sang roll1 (khong sinh duoc derangement sau 10 lan): "
          f"{n_fallback_shuffled}/{n_batches_done}")

    stats = {}  # stats[branch][dim] = {"mean_ls", "sigma", "r2", "ez2"}
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
            r2 = 1 - sigma ** 2
            ez2 = (z[:, i] ** 2).mean().item()
            stats[branch][name] = {"mean_ls": mean_ls, "sigma": sigma, "r2": r2, "ez2": ez2}

        mean_nll[branch] = (0.5 * (z ** 2).sum(dim=-1) + 3.5 * math.log(2 * math.pi) + ls.sum(dim=-1)).mean().item()

    for branch in branches:
        print("\n" + "=" * 90)
        print(f"NHANH: {branch}")
        print("=" * 90)
        print(f"{'chieu':8}{'mean(log_sigma)':>18}{'sigma':>12}{'R2':>12}{'E[z^2]':>12}")
        for name in DIM_NAMES:
            s = stats[branch][name]
            print(f"{name:8}{s['mean_ls']:18.4f}{s['sigma']:12.4f}{s['r2']:12.4f}{s['ez2']:12.4f}")
        print(f"    mean(NLL) = {mean_nll[branch]:.4f}")

    print("\n" + "=" * 90)
    print("BANG CHENH LECH R2: full - shuffled, va full - roll1 (sap giam dan theo full-shuffled)")
    print("=" * 90)
    diffs = []
    for name in DIM_NAMES:
        r2_full = stats["full"][name]["r2"]
        d_shuf = r2_full - stats["shuffled"][name]["r2"]
        d_roll = r2_full - stats["roll1"][name]["r2"]
        diffs.append((name, d_shuf, d_roll))
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"{'chieu':8}{'R2(full)-R2(shuffled)':>24}{'R2(full)-R2(roll1)':>22}")
    for name, d_shuf, d_roll in diffs:
        print(f"{name:8}{d_shuf:24.4f}{d_roll:22.4f}")

    print("\n" + "=" * 90)
    print("TIEU CHI DOC M6 (plan.md 'G5 Rev 2') -- CHI tren theta/phi, doi chung CHINH la 'shuffled'")
    print("=" * 90)
    d_theta = stats["full"]["theta"]["r2"] - stats["shuffled"]["theta"]["r2"]
    d_phi = stats["full"]["phi"]["r2"] - stats["shuffled"]["phi"]["r2"]
    print(f"    R2_theta(full)-R2_theta(shuffled) = {d_theta:.4f}")
    print(f"    R2_phi(full)-R2_phi(shuffled)     = {d_phi:.4f}")
    if d_theta > 0.05 and d_phi > 0.05:
        print("    => CA HAI > 0.05: hoc protein CO dong gop vao tu the. Bang chung dau tien co that.")
        print("       Di tiep: chay G5 day du (train lai 2 nhanh x 10 epoch).")
    elif d_theta < 0.02 and d_phi < 0.02:
        print("    => CA HAI < 0.02: hoc protein KHONG dong gop gi. Van de o encoder/conditioning,")
        print("       khong o head hay bieu dien quay. Phai dung va thiet ke lai duong dieu kien hoa.")
    else:
        print("    => MOT chieu vuot 0.05, chieu kia duoi 0.02 (hoac ca hai o giua): KHONG KET LUAN.")
        print("       Bao cao va dung.")

    print("\n" + "=" * 90)
    print("KIEM TRA KHONG-DOI: R2(full) do duoc vs RESEARCH_CONTEXT 24.1 (cung checkpoint, khong shuffle)")
    print("=" * 90)
    print(f"{'chieu':8}{'R2 do (full)':>16}{'R2 Muc 24.1':>16}{'lech':>10}")
    any_warn = False
    for name in DIM_NAMES:
        measured = stats["full"][name]["r2"]
        ref = REFERENCE_R2_FULL[name]
        lech = measured - ref
        warn = abs(lech) > 0.02
        any_warn = any_warn or warn
        flag = "  <-- CANH BAO" if warn else ""
        print(f"{name:8}{measured:16.4f}{ref:16.4f}{lech:+10.4f}{flag}")
    if any_warn:
        print("\n    [!!!] CO CHIEU LECH > 0.02: them tham so pocket_shuffle_perm co the da lam doi")
        print("    duong chay mac dinh (perm=None), hoac loader doc batch khac voi luc do 24.1.")
        print("    KHONG dung ket luan tieu chi M6 o tren cho den khi kiem tra lai.")
    else:
        print("\n    OK -- nhanh 'full' khop voi 24.1 trong sai so 0.02.")


if __name__ == "__main__":
    main()
