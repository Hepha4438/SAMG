"""
measure_marginal_7d.py -- M1 (plan.md "PLAN Rev 5"): thong ke marginal 7 chieu cua
target_7d tren tap train. KHONG GPU, KHONG train (~2 phut).

Muc dich (RESEARCH_CONTEXT Muc 20.5): loai hoac xac nhan gia thuyet (b) -- `qy` thoai
hoa trong DU LIEU (gan suy bien/luong cuc nen "du doan" duoc ma khong can thong tin tu
hoc). Day la giai thich tam thuong nhat cho 76,8% "tien bo" cua mo hinh o Muc 20.4 va
CHUA HE duoc loai truoc ban sua nay.

Dung CHINH XAC cung cach loc tap train voi SAMGOptimizedDataset (import tu step6_trainer,
KHONG copy code): cung pkl_by_id, cung loc frame_stable/frag_frame_stable, cung
scaler_7d.pt. CHI khac o cho KHONG goi __getitem__ day du (bo qua phan doc protein/ligand
nang -- PDBProtein, parse_sdf_file, compute_residue_transforms -- vi khong can cho thong
ke nay); thay vao do doc thang target_7d tu "sequence" cua tung pkl hop le trong
dataset.valid_indices (cung tap hop entry ma SAMGOptimizedDataset.__getitem__ se tra ve).

TIEU CHI DOC (ghi cung TRUOC khi chay, xem plan.md M1):
- Neu `qy` SAU chuan hoa co >= 50% khoi luong trong 3 bin (trong 40 bin), hoac
  |kurtosis| > 10, hoac std_raw(qy) < 0.1 * std_raw(qw) => (b) XAC NHAN: sigma=0.257 la
  hien vat du lieu, khong phai thong tin tu hoc.
- Neu qy co phan phoi tuong tu qw/qx/qz (cung bac kurtosis, cung do tan) => (b) BI LOAI.

Cach chay:
    python src/measure_marginal_7d.py
"""
import os
import sys
import subprocess
import pickle

import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]


def print_git_head():
    samg_root = os.path.dirname(CURRENT_DIR)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=samg_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")


def load_target_7d():
    """Tra ve tensor float64 [N, 7]. Dung dung filtering cua SAMGOptimizedDataset.__init__
    (import, khong copy code); chi doc thang target_7d tu pkl thay vi goi __getitem__."""
    import step6_trainer as st  # import lai -- KHONG copy logic pkl_by_id/frame_stable

    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(st.VOCAB_PATH):
        with open(st.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

    dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode="train", pos_scale=1.0
    )

    rows = []
    for i in dataset.valid_indices:
        entry = dataset.target_entries[i]
        pkl_path = dataset.pkl_by_id[entry[2].split("/")[0]]
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        for item in pkl_data["sequence"]:
            rows.append(item["spatial_tokens"])

    if not rows:
        raise RuntimeError("Khong doc duoc token nao -- kiem tra dataset/processed va split_file.")

    return torch.tensor(rows, dtype=torch.float64), len(dataset.valid_indices)


def load_scaler():
    import step6_trainer as st
    scaler_path = os.path.join(st.PROCESSED_DIR, "scaler_7d.pt")
    if not os.path.exists(scaler_path):
        print(f"[!] Khong tim thay {scaler_path} -- bo qua khong gian SAU chuan hoa.")
        return None, None
    scaler_dict = torch.load(scaler_path, map_location="cpu")
    shift = scaler_dict["shift"].flatten().double()
    scale = scaler_dict["scale"].flatten().double()
    return shift, scale


def skew_kurtosis(x):
    """x: tensor 1D float64. Tra ve (skew, excess_kurtosis) -- Normal(0,1) co ca hai = 0."""
    mean = x.mean()
    std = x.std(unbiased=False)
    if std < 1e-12:
        return 0.0, 0.0
    z = (x - mean) / std
    skew = (z ** 3).mean().item()
    kurt = (z ** 4).mean().item() - 3.0
    return skew, kurt


def mass_in_top_bins(x, n_bins=40, top_k=(1, 3)):
    """Tra ve dict {k: ty le khoi luong trong k bin lon nhat trong so n_bins}, va
    (edges, counts) de in histogram text."""
    x_np = x.numpy()
    counts, edges = np.histogram(x_np, bins=n_bins)
    total = counts.sum()
    sorted_counts = np.sort(counts)[::-1]
    out = {}
    for k in top_k:
        out[k] = float(sorted_counts[:k].sum()) / total if total > 0 else 0.0
    return out, (edges, counts)


def print_histogram_text(edges, counts, width=50):
    total = counts.sum()
    max_c = counts.max() if counts.max() > 0 else 1
    for i, c in enumerate(counts):
        bar_len = int(round((c / max_c) * width))
        pct = (c / total * 100) if total > 0 else 0.0
        print(f"    [{edges[i]:+8.3f}, {edges[i+1]:+8.3f}) {'#' * bar_len:<{width}} {c:6d} ({pct:5.2f}%)")


def report_space(name, X):
    """X: tensor [N, 7] float64. In thong ke day du cho moi chieu."""
    print(f"\n{'=' * 90}")
    print(f"KHONG GIAN: {name}")
    print("=" * 90)
    N = X.shape[0]
    for i, dim_name in enumerate(DIM_NAMES):
        x = X[:, i]
        mean, std = x.mean().item(), x.std(unbiased=False).item()
        mn, mx = x.min().item(), x.max().item()
        skew, kurt = skew_kurtosis(x)
        qs = torch.quantile(x, torch.tensor([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=torch.float64))
        n_unique = torch.unique(x).numel()
        mass, (edges, counts) = mass_in_top_bins(x)

        print(f"\n--- Chieu {dim_name} (index {i}) --- n={N}")
        print(f"    mean={mean:.4f}  std={std:.4f}  min={mn:.4f}  max={mx:.4f}")
        print(f"    skew={skew:.4f}  excess_kurtosis={kurt:.4f}")
        print(f"    percentile [1,5,25,50,75,95,99] = "
              f"[{qs[0]:.4f}, {qs[1]:.4f}, {qs[2]:.4f}, {qs[3]:.4f}, {qs[4]:.4f}, {qs[5]:.4f}, {qs[6]:.4f}]")
        print(f"    so gia tri duy nhat (torch.unique): {n_unique} / {N}")
        print(f"    ty le khoi luong trong 1 bin lon nhat: {mass[1]*100:.2f}%   "
              f"trong 3 bin lon nhat: {mass[3]*100:.2f}%")
        print(f"    histogram 40 bin:")
        print_histogram_text(edges, counts)


def main():
    print_git_head()

    print("\n[*] Dang doc target_7d tu tap train (dung filtering cua SAMGOptimizedDataset)...")
    X_raw, n_complex = load_target_7d()
    print(f"[*] So complex hop le (valid_indices): {n_complex}")
    print(f"[*] So token (hang target_7d): {X_raw.shape[0]}")

    report_space("RAW (truoc chuan hoa)", X_raw)

    shift, scale = load_scaler()
    if shift is not None:
        X_norm = (X_raw - shift) / scale
        report_space("SAU CHUAN HOA (scaler_7d.pt)", X_norm)

        print(f"\n{'=' * 90}")
        print("scaler_7d.pt: mean[i] (shift) va std[i] (scale) cho tung chieu")
        print("=" * 90)
        for i, dim_name in enumerate(DIM_NAMES):
            print(f"    {dim_name:7}: mean={shift[i].item():.6f}  std={scale[i].item():.6f}")

    print("\n[*] XONG. Doi chieu voi tieu chi doc trong docstring/plan.md M1 cho chieu qy.")


if __name__ == "__main__":
    main()
