"""
probe_pose_regression.py -- M11-a (plan.md "PLAN Rev 12"; RESEARCH_CONTEXT Muc 29): phep tham
do TOI THIEU, tach HOAN TOAN khoi generator/flow/autoregressive -- cho hoc protein (h_target
lay tu CHINH SAMGLightningModule.build_h_target(), M11-c, KHONG viet lai logic dung h_target o
day) cong danh tinh fragment (frag_id, vi tri t trong chuoi), hoi quy TRUC TIEP 7D pose chuan
hoa bang MSE. KHONG sigma, KHONG NLL, KHONG autoregressive, KHONG dua pose cua fragment truoc
vao input (lam vay se tra lai shortcut dai so ||q||=1 cua MAF va pha ca thi nghiem).

Cau hoi: encoder hien tai co trich duoc thong tin ve tu the tu hoc protein khong, doc lap voi
moi bug do luong/shortcut dai so cua pipeline day du (M6-M10).

HAI BUOC:
  1. CACHE (can GPU + checkpoint that): chay SAMGLightningModule tu checkpoint MOT LAN qua
     train/valid, voi MOI complex goi model.build_h_target(batch) (batch_size=1) va luu
     h_target [R_max, H+3], target_mask [R_max], target_7d [T,7] (CHUA chuan hoa), target_ids
     [T] ra dataset/cache_probe/{split}/*.pt. Da co cache thi BO QUA buoc nay (dung
     --rebuild-cache de buoc lam lai).
  2. HUAN LUYEN (khong can checkpoint/GPU-encoder nua, chi doc cache): ba nhanh
       pocket     -- h_target that
       blind      -- h_target ZERO HOA toan bo H+3 kenh (torch.zeros_like), GIU target_mask
                      nguyen, trong CA huan luyen va danh gia
       tokenmean  -- KHONG mo hinh: trung binh 7D chuan hoa theo tung frag_id, tinh tren
                      train, dung lam san tuyet doi
     Kien truc (pocket/blind): cross-attention 1 lop CO MASK tren h_target, query = frag_
     embedding(frag_id) + pos_embedding(t) (K/V KHONG bias -- h_target=0 o nhanh blind thi
     K=V=0 het, attn logits = 0 het -> softmax deu tren mask -> ctx = trung binh cua V=0 =
     dung 0 CHINH XAC, nen "blind" that su khong mang thong tin hoc nao qua ctx, chi con
     frag_id/t). Noi ctx voi frag_emb + pos_emb -> MLP 2 lop -> 7 so (khong gian da chuan
     hoa, scaler_7d.pt y nhu pipeline).

KHONG co "pad_mask" token-level rieng: moi chuoi cache dung DUNG do dai that T cua complex do
(cache tung complex rieng, khong ghep nhieu complex truoc khi luu nen khong co padding token
nao lot vao). Padding CHI xay ra o muc h_target/target_mask khi gop nhieu complex thanh 1
batch huan luyen (R_max khac nhau) -- target_mask (khong phai id==0) la pad_mask dung o day,
va da duoc dung trong ca masked-mean pooling LAN masked-attention.

Cach chay (buoc 1 can GPU; buoc 2 chay duoc tren CPU/GPU):
    python -u src/probe_pose_regression.py [--ckpt PATH] [--rebuild-cache] [--epochs 100]
"""
import os
import sys
import glob
import time
import argparse
import subprocess
import pickle

import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]
TRANS_DIMS = ["d", "theta", "phi"]
SEED = 1234
MAX_POS = 128  # chuoi fragment luon ngan hon nhieu; chi la bang pos-embedding, co clamp an toan
HIDDEN = 128


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


# ============================================================================================
# BUOC 1 -- CACHE
# ============================================================================================

def dir_size_bytes(d):
    total = 0
    for f in glob.glob(os.path.join(d, "*.pt")):
        total += os.path.getsize(f)
    return total


def build_cache_for_split(model, device, split, cache_dir, st, vocab, pos_scale):
    from datasets.pl_data import ProteinLigandDataLoader

    dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode=split, pos_scale=pos_scale
    )
    loader = ProteinLigandDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(cache_dir, exist_ok=True)

    n_saved = 0
    t_start = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch = batch.to(device)
            h_target, target_mask, _, _ = model.build_h_target(batch)
            item = {
                "h_target": h_target[0].detach().cpu(),
                "target_mask": target_mask[0].detach().cpu(),
                "target_7d": batch.target_7d.detach().cpu(),
                "target_ids": batch.target_ids.detach().cpu(),
            }
            torch.save(item, os.path.join(cache_dir, f"{idx:06d}.pt"))
            n_saved += 1
            if (idx + 1) % 20 == 0:
                print(f"  cache[{split}] {idx + 1}/{len(loader)} (elapsed {time.time() - t_start:.1f}s)",
                      flush=True)
    return n_saved


def ensure_cache(model, device, split, cache_root, rebuild, st, vocab, pos_scale):
    cache_dir = os.path.join(cache_root, split)
    existing = sorted(glob.glob(os.path.join(cache_dir, "*.pt"))) if os.path.isdir(cache_dir) else []
    if existing and not rebuild:
        print(f"[*] cache[{split}]: da ton tai {len(existing)} complex, BO QUA buoc cache "
              f"(dung --rebuild-cache de buoc lam lai).")
        return cache_dir, len(existing)

    if existing and rebuild:
        for f in existing:
            os.remove(f)
        print(f"[*] cache[{split}]: --rebuild-cache, da xoa {len(existing)} file cu.")

    print(f"[*] cache[{split}]: dang xay dung...")
    n = build_cache_for_split(model, device, split, cache_dir, st, vocab, pos_scale)
    return cache_dir, n


# ============================================================================================
# BUOC 2 -- HEAD MOI, HUAN LUYEN TREN CACHE
# ============================================================================================

class ProbeCacheDataset(torch.utils.data.Dataset):
    """Doc tung complex tu dataset/cache_probe/{split}/*.pt (lazy load, khong giu het trong RAM)."""

    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        assert len(self.files) > 0, f"cache rong: {cache_dir}"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu")
        return d["h_target"], d["target_mask"], d["target_7d"], d["target_ids"]


def collate_probe(batch):
    """Gop B complex thanh 1 batch huan luyen: pad h_target/target_mask theo R_max (target_mask
    LA pad_mask cho phan nay); gop TAT CA token that cua ca B complex thanh N vi du doc lap
    (N = tong do dai chuoi THAT, KHONG co token padding nao vi moi complex duoc cache rieng
    dung do dai that cua no -- xem docstring dau file)."""
    h_list, m_list, t7d_list, tid_list = zip(*batch)
    B = len(h_list)
    R_max = max(h.shape[0] for h in h_list)
    H = h_list[0].shape[1]

    h_pad = torch.zeros(B, R_max, H)
    m_pad = torch.zeros(B, R_max, dtype=torch.bool)
    for i, (h, m) in enumerate(zip(h_list, m_list)):
        R = h.shape[0]
        h_pad[i, :R] = h
        m_pad[i, :R] = m

    complex_idx, positions, frag_ids, poses = [], [], [], []
    for i, (t7d, tid) in enumerate(zip(t7d_list, tid_list)):
        T = tid.shape[0]
        complex_idx.append(torch.full((T,), i, dtype=torch.long))
        positions.append(torch.arange(T, dtype=torch.long))
        frag_ids.append(tid)
        poses.append(t7d)

    return (h_pad, m_pad, torch.cat(complex_idx), torch.cat(positions),
            torch.cat(frag_ids), torch.cat(poses, dim=0))


class PoseProbeHead(nn.Module):
    """Cross-attention 1 lop CO MASK tren h_target (plan.md M11-a): query = frag_emb(frag_id)
    + pos_emb(t); K/V chieu tu h_target KHONG bias -- khi h_target=0 het (nhanh blind), K=V=0
    het, logits attn = 0 het -> softmax deu tren mask -> ctx = trung binh cua V = 0 CHINH XAC,
    dam bao nhanh blind khong ri mot chut thong tin hoc nao qua ctx."""

    def __init__(self, h_dim, vocab_size, max_pos=MAX_POS, hidden=HIDDEN):
        super().__init__()
        self.max_pos = max_pos
        self.frag_emb = nn.Embedding(vocab_size, hidden)
        self.pos_emb = nn.Embedding(max_pos, hidden)
        self.q_proj = nn.Linear(2 * hidden, hidden)
        self.k_proj = nn.Linear(h_dim, hidden, bias=False)
        self.v_proj = nn.Linear(h_dim, hidden, bias=False)
        self.scale = hidden ** -0.5
        self.mlp = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 7))

    def forward(self, h_target, target_mask, complex_idx, positions, frag_ids):
        fe = self.frag_emb(frag_ids)
        pe = self.pos_emb(positions.clamp_max(self.max_pos - 1))
        q = self.q_proj(torch.cat([fe, pe], dim=-1))          # [N, hidden]

        k = self.k_proj(h_target)                              # [B, R, hidden]
        v = self.v_proj(h_target)                              # [B, R, hidden]
        k_c = k[complex_idx]                                   # [N, R, hidden]
        v_c = v[complex_idx]                                   # [N, R, hidden]
        mask_c = target_mask[complex_idx]                      # [N, R] bool, pad_mask cua h_target

        attn_logits = torch.einsum("nh,nrh->nr", q, k_c) * self.scale
        attn_logits = attn_logits.masked_fill(~mask_c, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)
        ctx = torch.einsum("nr,nrh->nh", attn, v_c)            # [N, hidden]

        x = torch.cat([ctx, fe, pe], dim=-1)
        return self.mlp(x)                                     # [N, 7] khong gian da chuan hoa


def zero_h_target(h_target):
    """Nhanh blind (M11-a): zero hoa TOAN BO H+3 kenh cua h_target, GIU target_mask nguyen."""
    return torch.zeros_like(h_target)


def compute_r2(pred, target):
    """R2_i = 1 - MSE_i tren khong gian DA CHUAN HOA (variance=1), dung cho ca 3 nhanh."""
    mse = ((pred - target) ** 2).mean(dim=0)
    return (1.0 - mse).tolist()


@torch.no_grad()
def eval_head(head, loader, device, shift, scale, blind):
    head.eval()
    preds, targets = [], []
    for h_pad, m_pad, c_idx, pos, frag_ids, poses in loader:
        h_pad, m_pad = h_pad.to(device), m_pad.to(device)
        c_idx, pos, frag_ids, poses = c_idx.to(device), pos.to(device), frag_ids.to(device), poses.to(device)
        if blind:
            h_pad = zero_h_target(h_pad)
        pred = head(h_pad, m_pad, c_idx, pos, frag_ids)
        target_scaled = (poses - shift) / scale
        preds.append(pred.double())
        targets.append(target_scaled.double())
    pred_all = torch.cat(preds, dim=0).cpu()
    target_all = torch.cat(targets, dim=0).cpu()
    return compute_r2(pred_all, target_all), pred_all.shape[0]


def train_branch(name, train_ds, valid_ds, h_dim, vocab_size, device, shift, scale, epochs,
                  batch_size, blind):
    """Huan luyen mot nhanh (pocket hoac blind) tu dau, seed co dinh truoc KHOI TAO va truoc
    MOI epoch (dam bao thu tu shuffle giong het nhau giua hai nhanh -- 'cung seed' theo dung
    yeu cau cua plan.md)."""
    torch.manual_seed(SEED)
    head = PoseProbeHead(h_dim, vocab_size).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g, collate_fn=collate_probe
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_probe
    )

    print(f"\n[*] --- Huan luyen nhanh '{name}' (blind={blind}), {epochs} epoch, seed={SEED} ---")
    t_start = time.time()
    for epoch in range(epochs):
        head.train()
        epoch_loss, n_tok = 0.0, 0
        for h_pad, m_pad, c_idx, pos, frag_ids, poses in train_loader:
            h_pad, m_pad = h_pad.to(device), m_pad.to(device)
            c_idx, pos, frag_ids, poses = (c_idx.to(device), pos.to(device),
                                            frag_ids.to(device), poses.to(device))
            if blind:
                h_pad = zero_h_target(h_pad)
            pred = head(h_pad, m_pad, c_idx, pos, frag_ids)
            target_scaled = (poses - shift) / scale
            loss = ((pred - target_scaled) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item() * pred.shape[0]
            n_tok += pred.shape[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            r2_valid, n_valid = eval_head(head, valid_loader, device, shift, scale, blind)
            print(f"  [{name}] epoch {epoch + 1}/{epochs}  train_loss={epoch_loss / max(n_tok, 1):.4f}  "
                  f"val_R2={['%.3f' % v for v in r2_valid]}  (elapsed {time.time() - t_start:.1f}s)",
                  flush=True)

    r2_valid, n_valid = eval_head(head, valid_loader, device, shift, scale, blind)
    r2_train, n_train = eval_head(head, train_loader_eval(train_ds, batch_size), device, shift, scale, blind)
    return head, r2_train, r2_valid, n_train, n_valid


def train_loader_eval(ds, batch_size):
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_probe)


def compute_tokenmean(train_ds, valid_ds, batch_size, shift, scale):
    """Nhanh tokenmean (plan.md M11-a): KHONG mo hinh -- trung binh 7D DA CHUAN HOA theo tung
    frag_id, tinh TREN TRAIN. frag_id khong xuat hien trong train (chi co trong valid) dung
    trung binh toan cuc cua train. San tuyet doi, khong huan luyen."""
    sums, counts = {}, {}
    global_sum = torch.zeros(7, dtype=torch.float64)
    global_n = 0
    loader = train_loader_eval(train_ds, batch_size)
    for _, _, _, _, frag_ids, poses in loader:
        target_scaled = ((poses - shift) / scale).double()
        for fid, pose in zip(frag_ids.tolist(), target_scaled):
            sums[fid] = sums.get(fid, torch.zeros(7, dtype=torch.float64)) + pose
            counts[fid] = counts.get(fid, 0) + 1
            global_sum += pose
            global_n += 1
    global_mean = global_sum / max(global_n, 1)
    mean_by_frag = {fid: sums[fid] / counts[fid] for fid in sums}

    def predict_and_r2(ds):
        preds, targets = [], []
        for _, _, _, _, frag_ids, poses in train_loader_eval(ds, batch_size):
            target_scaled = ((poses - shift) / scale).double()
            pred = torch.stack([mean_by_frag.get(fid, global_mean) for fid in frag_ids.tolist()])
            preds.append(pred)
            targets.append(target_scaled)
        pred_all = torch.cat(preds, dim=0)
        target_all = torch.cat(targets, dim=0)
        return compute_r2(pred_all, target_all), pred_all.shape[0]

    r2_train, n_train = predict_and_r2(train_ds)
    r2_valid, n_valid = predict_and_r2(valid_ds)
    return r2_train, r2_valid, n_train, n_valid


# ============================================================================================
# BAO CAO
# ============================================================================================

def print_r2_table(title, r2_by_branch):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    branches = list(r2_by_branch.keys())
    header = f"{'chieu':8}" + "".join(f"{b:>14}" for b in branches)
    print(header)
    for i, name in enumerate(DIM_NAMES):
        row = f"{name:8}" + "".join(f"{r2_by_branch[b][i]:14.4f}" for b in branches)
        print(row)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(device)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Mac dinh: saved_checkpoints_flow/periodic-epoch=024.ckpt (chi dung o buoc cache)")
    ap.add_argument("--rebuild-cache", action="store_true", default=False)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32, help="So COMPLEX moi batch huan luyen (buoc 2)")
    args = ap.parse_args()

    import step6_trainer as st
    samg_root = st.SAMG_ROOT
    cache_root = os.path.join(samg_root, "dataset", "cache_probe")

    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(st.VOCAB_PATH):
        with open(st.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

    scaler_path = os.path.join(st.PROCESSED_DIR, "scaler_7d.pt")
    scaler_dict = torch.load(scaler_path, map_location="cpu")
    shift_factors, scale_factors = scaler_dict["shift"], scaler_dict["scale"]
    pos_scale = scale_factors.flatten()[0].item()

    train_cache_dir = os.path.join(cache_root, "train")
    valid_cache_dir = os.path.join(cache_root, "valid")
    train_existing = glob.glob(os.path.join(train_cache_dir, "*.pt"))
    valid_existing = glob.glob(os.path.join(valid_cache_dir, "*.pt"))
    need_cache = args.rebuild_cache or not train_existing or not valid_existing

    # ------------------------------------------------------------------ BUOC 1: CACHE --------
    if need_cache:
        from omegaconf import OmegaConf
        ckpt_path = args.ckpt or default_ckpt(samg_root)
        if ckpt_path is None:
            raise SystemExit("[!] Khong tim thay checkpoint nao trong saved_checkpoints_flow/. Truyen --ckpt.")
        print(f"[*] Checkpoint (cho buoc cache): {ckpt_path}")

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
        model.eval()
        model.requires_grad_(False)

        train_cache_dir, n_train = ensure_cache(model, device, "train", cache_root, args.rebuild_cache,
                                                 st, vocab, pos_scale)
        valid_cache_dir, n_valid = ensure_cache(model, device, "valid", cache_root, args.rebuild_cache,
                                                 st, vocab, pos_scale)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        n_train, n_valid = len(train_existing), len(valid_existing)
        print(f"[*] cache[train]/cache[valid]: da ton tai du ca hai, BO QUA buoc cache hoan toan "
              f"(dung --rebuild-cache de buoc lam lai).")

    for split, d, n in [("train", train_cache_dir, n_train), ("valid", valid_cache_dir, n_valid)]:
        size_mb = dir_size_bytes(d) / 1e6
        print(f"[*] cache[{split}]: {n} complex, {size_mb:.1f} MB tai {d}")

    # ------------------------------------------------------------- BUOC 2: HUAN LUYEN HEAD ----
    train_ds = ProbeCacheDataset(train_cache_dir)
    valid_ds = ProbeCacheDataset(valid_cache_dir)
    h_dim = train_ds[0][0].shape[1]
    vocab_size = len(vocab)
    # tokenmean chay thuan CPU (khong dung model/device) -- giu rieng ban CPU float64 cho no;
    # train_branch/eval_head dung ban tren DEVICE, CUNG dtype voi poses/pred (float32) de tranh
    # loi lech device va ep kieu khong can thiet trong backward.
    shift_cpu = shift_factors.view(7).double()
    scale_cpu = scale_factors.view(7).double()
    shift_dev = shift_factors.view(7).to(device)
    scale_dev = scale_factors.view(7).to(device)

    print(f"\n[*] CAU HINH HUAN LUYEN: epochs={args.epochs}  batch_size(complex)={args.batch_size}  "
          f"vocab_size={vocab_size}  h_dim={h_dim}  n_train={len(train_ds)}  n_valid={len(valid_ds)}")

    print("\n[*] --- Nhanh 'tokenmean' (san tuyet doi, khong huan luyen) ---")
    tm_r2_train, tm_r2_valid, tm_n_train, tm_n_valid = compute_tokenmean(
        train_ds, valid_ds, args.batch_size, shift_cpu, scale_cpu
    )
    print(f"  tokenmean: n_train={tm_n_train}  n_valid={tm_n_valid}")

    pocket_head, pk_r2_train, pk_r2_valid, pk_n_train, pk_n_valid = train_branch(
        "pocket", train_ds, valid_ds, h_dim, vocab_size, device, shift_dev, scale_dev,
        args.epochs, args.batch_size, blind=False
    )

    print(f"\n[*] [pocket] R2(d) tren valid = {pk_r2_valid[0]:.4f} "
          f"(nguong dung lai theo plan.md Rev 11: < 0,30 => dung TRUOC khi chay 'blind')")
    if pk_r2_valid[0] < 0.30:
        print("    [!!!] R2(d) nhanh pocket < 0,30 -- mot bo hoi quy co giam sat don gian ma khong")
        print("    dat noi muc cua MAF trong pipeline day du (0,418, M10) la dau hieu M11 bi loi")
        print("    thuc thi, KHONG phai ket qua. DUNG, KHONG chay nhanh 'blind'.")
        return

    blind_head, bl_r2_train, bl_r2_valid, bl_n_train, bl_n_valid = train_branch(
        "blind", train_ds, valid_ds, h_dim, vocab_size, device, shift_dev, scale_dev,
        args.epochs, args.batch_size, blind=True
    )

    # ------------------------------------------------------------------------- BAO CAO --------
    r2_valid_by_branch = {"pocket": pk_r2_valid, "blind": bl_r2_valid, "tokenmean": tm_r2_valid}
    r2_train_by_branch = {"pocket": pk_r2_train, "blind": bl_r2_train, "tokenmean": tm_r2_train}

    print_r2_table("BANG R2 TREN VALID (7 chieu x 3 nhanh)", r2_valid_by_branch)
    print_r2_table("BANG R2 TREN TRAIN (7 chieu x 3 nhanh, de thay overfit)", r2_train_by_branch)

    print("\n" + "=" * 90)
    print("BANG CHENH LECH R2(pocket) - R2(blind) TREN VALID (sap giam dan)")
    print("=" * 90)
    diffs = [(name, pk_r2_valid[i] - bl_r2_valid[i]) for i, name in enumerate(DIM_NAMES)]
    diffs.sort(key=lambda x: x[1], reverse=True)
    for name, d in diffs:
        print(f"    {name:8}{d:10.4f}")
    d_by_dim = {name: v for name, v in diffs}

    print("\n" + "=" * 90)
    print("DOI CHIEU R2(blind) vs R2(tokenmean) TREN VALID (tung chieu)")
    print("=" * 90)
    for i, name in enumerate(DIM_NAMES):
        print(f"    {name:8} blind={bl_r2_valid[i]:.4f}   tokenmean={tm_r2_valid[i]:.4f}   "
              f"blind-tokenmean={bl_r2_valid[i] - tm_r2_valid[i]:+.4f}")

    # ---- TIEU CHI DOC M11-a (plan.md "PLAN Rev 12"), DUNG THU TU (1)(2)(3), KHONG tu dat nguong moi ----
    print("\n" + "=" * 90)
    print("TIEU CHI DOC M11-a (plan.md 'PLAN Rev 12', RESEARCH_CONTEXT Muc 29)")
    print("=" * 90)

    print("\n  (1) KIEM NHAT QUAN NOI BO: blind vs tokenmean (TRUOC khi doc tiep)")
    print("      Ly do: 'blind' co CUNG tin hieu (frag_id, t) nhu 'tokenmean' CONG mot MLP co")
    print("      huan luyen, nen VE NGUYEN TAC blind khong nen thua tokenmean dang ke tren dien")
    print("      rong. Day la BAO CAO/CHAN DOAN, KHONG dat nguong so moi.")
    n_worse = sum(1 for name in TRANS_DIMS if (bl_r2_valid[DIM_NAMES.index(name)]
                                                < tm_r2_valid[DIM_NAMES.index(name)]))
    if n_worse >= 2:
        print(f"      [!!!] blind THUA tokenmean o {n_worse}/3 chieu tinh tien -- kiem tra lai")
        print("      kien truc/huan luyen nhanh blind TRUOC khi tin bat ky ket luan nao duoi day.")
    else:
        print(f"      blind khong thua tokenmean dang ke ({n_worse}/3 chieu tinh tien) -- nhat")
        print("      quan noi bo OK, doc tiep (2).")

    print("\n  (2) TIEU CHI CHINH (plan.md Rev 10/11): R2(pocket)-R2(blind) > 0,05 o it nhat 2/3")
    print("      chieu tinh tien (d, theta, phi)")
    n_pass_005 = sum(1 for name in TRANS_DIMS if d_by_dim[name] > 0.05)
    all_below_002 = all(d_by_dim[name] < 0.02 for name in TRANS_DIMS)
    for name in TRANS_DIMS:
        print(f"      delta R2({name}) = {d_by_dim[name]:.4f}")
    if n_pass_005 >= 2:
        print("      => TIEN DE DUNG: hoc protein chua tin hieu hoc duoc ve tu the, va encoder")
        print("         hien tai trich duoc. Di tiep: sua duong dieu kien hoa cua pipeline chinh.")
    elif all_below_002:
        print("      => TIEN DE SAI: encoder bat bien hien tai khong trich duoc. Theo tieu chi BO")
        print("         o Rev 9, chuyen sang kien truc TUONG BIEN (viet lai tang encoder + bieu")
        print("         dien, khong phai viet lai repo).")
    else:
        print("      => TRUNG GIAN: khong du 2/3 chieu vuot 0,05, nhung khong ca ba deu duoi 0,02.")
        print("         Theo plan.md: tang so epoch len 40 va do lai MOT LAN; neu van trung gian")
        print("         thi bao cao va dung.")

    print("\n  (3) MOC SO SANH 0,081 (RESEARCH_CONTEXT 29, M10 tren pipeline day du co shortcut")
    print("      dai so + generator) -- CHI cho chieu d:")
    d_d = d_by_dim["d"]
    print(f"      delta R2(d) (M11, khong shortcut, khong generator) = {d_d:.4f}")
    print(f"      delta R2(d) (M10, co shortcut + generator)         = 0.0810")
    if d_d > 0.081:
        print("      => delta R2(d) M11 > 0,081: pipeline day du dang LAM MAT tin hieu so voi muc")
        print("         encoder co the cung cap -- van de nam o generator/head, dang sua duoc.")
    else:
        print("      => delta R2(d) M11 <= 0,081: pipeline khong lam mat gi dang ke -- tran nam o")
        print("         ENCODER, sua generator se KHONG giup.")

    print("\n  Rieng quaternion (bao cao, KHONG dat nguong -- du doan ~0 theo Muc 16.1/25.4):")
    for name in ["qw", "qx", "qy", "qz"]:
        i = DIM_NAMES.index(name)
        print(f"      delta R2({name}) = {pk_r2_valid[i] - bl_r2_valid[i]:.4f}")


if __name__ == "__main__":
    main()
