"""
probe_pose_regression.py -- M11-a (plan.md "PLAN Rev 14"; RESEARCH_CONTEXT Muc 31): phep tham
do TOI THIEU, tach HOAN TOAN khoi generator/flow/autoregressive -- cho hoc protein (h_target
lay tu CHINH SAMGLightningModule.build_h_target(), M11-c, KHONG viet lai logic dung h_target o
day) cong danh tinh fragment (frag_id, vi tri t trong chuoi), hoi quy TRUC TIEP 7D pose chuan
hoa. KHONG sigma, KHONG NLL, KHONG autoregressive, KHONG dua pose cua fragment truoc vao input
(lam vay se tra lai shortcut dai so ||q||=1 cua MAF va pha ca thi nghiem).

Cau hoi: encoder hien tai co trich duoc thong tin ve tu the tu hoc protein khong, doc lap voi
moi bug do luong/shortcut dai so cua pipeline day du (M6-M10).

LAN 1 (Rev 12, RESEARCH_CONTEXT 31) THAT BAI VI THIEU GIAO THUC: 100 epoch, khong early
stopping, khong weight decay, voi bang token_embedding 19.033 muc tren 101.791 token train
(moi frag_id lap trung binh 5,3 lan) -- mo hinh overfit CHO DU khong co h_target (31.2), val R2
dat dinh o epoch 1 roi suy giam don dieu ca 7 chieu (31.1). Loi dac ta, khong phai loi code.

REV 14: HAI PHEP DO DOC LAP, chay CA HAI:
  P1 (CHINH) -- linear probe co ridge, KHONG the overfit theo kieu 31.2 vi khong co bang
    embedding tu do. Dac trung moi token = concat[mean-pool(h_target,mask) [259],
    max-pool(h_target,mask) [259], embedding CO DINH NGAU NHIEN cua frag_id (dim 32, seed
    1234, KHONG hoc -- cho biet "fragment nao" ma KHONG cho hoc thuoc pose tung fragment),
    one-hot vi tri t (clip t<=15) [16]]. Mot ma tran ridge cho ca 7 chieu (closed-form),
    lambda quet tren {1e-3..1e3}, chon theo R2 trung binh 3 chieu tinh tien TREN VALID.
  P2 (PHU) -- kien truc MLP y het lan 1 (cross-attention 1 lop co mask, K/V khong bias nen
    nhanh blind cho ctx=0 chinh xac), nhung CO GIAO THUC: early stopping tren valid (danh gia
    MOI epoch, patience 10, toi da 60 epoch, giu checkpoint tot nhat theo R2 trung binh 3
    chieu tinh tien), weight_decay=1e-4, bao cao R2 tai epoch TOT NHAT (khong phai epoch
    cuoi), preload toan bo cache vao RAM (tranh torch.load tung sample moi epoch, 31.5).

Ca P1 va P2 deu chay BA nhanh: pocket / blind (zero hoa H+3 kenh cua h_target, giu target_mask)
/ tokenmean (khong mo hinh, trung binh 7D theo frag_id tinh tren train).

GUARD "R2(d)<0,30 thi dung" cua Rev 11 DA HUY tu Rev 12, KHONG dung lai (pipeline co context
tu hoi quy, M11 thi khong, nen R2 thap hon 0,418 la hop le). THAY bang KIEM GIAO THUC, doc
TRUOC moi tieu chi khac: (1) in R2(tokenmean) ca 7 chieu TRUOC TIEN -- san tuyet doi; (2) [P2]
neu epoch tot nhat la epoch 1 -> giam LR 10 lan, chay lai MOT lan; van epoch 1 -> "P2 KHONG DOC
DUOC", van tiep tuc in P1; (3) R2(blind) thap hon R2(tokenmean) qua 0,03 o chieu tinh tien nao
-> "blind CHUA HOI TU", khong doc tieu chi chinh (cho probe do). Tieu chi chinh GIU NGUYEN
Rev 12 (doc cho P1 truoc, P2 doi chieu; neu P1/P2 nguoc nhau -> KHONG KET LUAN).

KHONG co "pad_mask" token-level rieng: moi chuoi cache dung DUNG do dai that T cua complex do
(cache tung complex rieng, khong ghep nhieu complex truoc khi luu nen khong co padding token
nao lot vao). Padding CHI xay ra o muc h_target/target_mask khi gop nhieu complex thanh 1
batch (R_max khac nhau) -- target_mask (khong phai id==0) la pad_mask dung o day, dung trong
ca masked-mean/max pooling (P1) lan masked-attention (P2).

Cach chay (buoc cache can GPU; P1/P2 chay duoc tren CPU/GPU, cache da co thi KHONG can GPU):
    python -u src/probe_pose_regression.py [--ckpt PATH] [--rebuild-cache]
"""
import os
import sys
import copy
import glob
import time
import argparse
import subprocess
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# PRELOAD CACHE VAO RAM (dung chung cho P1, P2, tokenmean -- 31.5)
# ============================================================================================

def preload_split(cache_dir):
    """Nap TOAN BO cache cua 1 split vao RAM mot lan (thay torch.load tung sample moi epoch).
    Tra ve list[(h_target, target_mask, target_7d, target_ids)]."""
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    assert len(files) > 0, f"cache rong: {cache_dir}"
    items = []
    t_start = time.time()
    for idx, f in enumerate(files):
        d = torch.load(f, map_location="cpu")
        items.append((d["h_target"], d["target_mask"], d["target_7d"], d["target_ids"]))
        if (idx + 1) % 2000 == 0:
            print(f"  preload[{cache_dir}] {idx + 1}/{len(files)} (elapsed {time.time() - t_start:.1f}s)",
                  flush=True)
    print(f"[*] preload[{cache_dir}]: {len(items)} complex trong {time.time() - t_start:.1f}s")
    return items


class ProbeCacheDataset(torch.utils.data.Dataset):
    """Boc mot list da PRELOAD san trong RAM (xem preload_split) thanh torch Dataset."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


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


def zero_h_target(h_target):
    """Nhanh blind: zero hoa TOAN BO H+3 kenh cua h_target, GIU target_mask nguyen."""
    return torch.zeros_like(h_target)


def compute_r2(pred, target):
    """R2_i = 1 - MSE_i tren khong gian DA CHUAN HOA (variance=1), dung cho ca 3 nhanh."""
    mse = ((pred - target) ** 2).mean(dim=0)
    return (1.0 - mse).tolist()


def mean_trans_r2(r2_vec):
    return sum(r2_vec[DIM_NAMES.index(n)] for n in TRANS_DIMS) / 3.0


def make_loader(items, batch_size, shuffle, generator=None):
    ds = ProbeCacheDataset(items)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=generator, collate_fn=collate_probe
    )


# ============================================================================================
# TOKENMEAN (san tuyet doi, dung chung cho P1 va P2)
# ============================================================================================

def compute_tokenmean(train_items, valid_items, batch_size, shift, scale):
    """Khong mo hinh -- trung binh 7D DA CHUAN HOA theo tung frag_id, tinh TREN TRAIN. frag_id
    khong xuat hien trong train dung trung binh toan cuc cua train. shift/scale: CPU float64."""
    sums, counts = {}, {}
    global_sum = torch.zeros(7, dtype=torch.float64)
    global_n = 0
    for _, _, _, _, frag_ids, poses in make_loader(train_items, batch_size, shuffle=False):
        target_scaled = ((poses - shift) / scale).double()
        for fid, pose in zip(frag_ids.tolist(), target_scaled):
            sums[fid] = sums.get(fid, torch.zeros(7, dtype=torch.float64)) + pose
            counts[fid] = counts.get(fid, 0) + 1
            global_sum += pose
            global_n += 1
    global_mean = global_sum / max(global_n, 1)
    mean_by_frag = {fid: sums[fid] / counts[fid] for fid in sums}

    def predict_and_r2(items):
        preds, targets = [], []
        for _, _, _, _, frag_ids, poses in make_loader(items, batch_size, shuffle=False):
            target_scaled = ((poses - shift) / scale).double()
            pred = torch.stack([mean_by_frag.get(fid, global_mean) for fid in frag_ids.tolist()])
            preds.append(pred)
            targets.append(target_scaled)
        pred_all = torch.cat(preds, dim=0)
        target_all = torch.cat(targets, dim=0)
        return compute_r2(pred_all, target_all), pred_all.shape[0]

    r2_train, n_train = predict_and_r2(train_items)
    r2_valid, n_valid = predict_and_r2(valid_items)
    return r2_train, r2_valid, n_train, n_valid


# ============================================================================================
# P1 -- LINEAR PROBE CO RIDGE
# ============================================================================================

def fixed_frag_embedding(vocab_size, dim=32, seed=1234):
    """Bang embedding CO DINH NGAU NHIEN cua frag_id -- Generator RIENG (khong dung chung RNG
    toan cuc) de luon tai lap duoc bat ke thu tu goi truoc do. requires_grad=False (khong hoc):
    cho biet 'fragment nao' ma KHONG cho phep hoc thuoc pose tung fragment (31.2)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(vocab_size, dim, generator=g)


def masked_mean_pool(h, mask):
    mask_f = mask.unsqueeze(-1).to(h.dtype)
    return (h * mask_f).sum(0) / mask_f.sum(0).clamp_min(1e-8)


def masked_max_pool(h, mask):
    if not bool(mask.any()):
        return torch.zeros(h.shape[-1], dtype=h.dtype)
    neg_inf = torch.finfo(h.dtype).min
    h_masked = h.masked_fill(~mask.unsqueeze(-1), neg_inf)
    return h_masked.max(dim=0).values


def build_p1_features(items, frag_table, blind):
    """Dac trung P1 moi token: concat[mean-pool(h,mask) [259], max-pool(h,mask) [259],
    frag_table[frag_id] [32], one-hot(clip(t,15)) [16]]. Tra ve (X [N, 566], Y_raw [N, 7])."""
    feats, poses_list = [], []
    for h, m, t7d, tid in items:
        if blind:
            h = torch.zeros_like(h)
        pooled_mean = masked_mean_pool(h, m)          # [H]
        pooled_max = masked_max_pool(h, m)             # [H]
        T = tid.shape[0]
        fe = frag_table[tid]                            # [T, 32]
        pos_clip = torch.arange(T).clamp_max(15)
        pos_oh = F.one_hot(pos_clip, num_classes=16).to(h.dtype)  # [T, 16]
        mean_b = pooled_mean.unsqueeze(0).expand(T, -1)
        max_b = pooled_max.unsqueeze(0).expand(T, -1)
        feats.append(torch.cat([mean_b, max_b, fe, pos_oh], dim=-1))
        poses_list.append(t7d)
    return torch.cat(feats, dim=0), torch.cat(poses_list, dim=0)


def ridge_fit(X, Y, lam):
    """Ridge closed-form, MOT ma tran cho ca 7 chieu, khong phat (regularize) cot bias."""
    N, D = X.shape
    X_aug = torch.cat([X, torch.ones(N, 1, dtype=X.dtype)], dim=1)   # [N, D+1]
    A = X_aug.T @ X_aug
    reg = torch.eye(D + 1, dtype=X.dtype) * lam
    reg[-1, -1] = 0.0
    W = torch.linalg.solve(A + reg, X_aug.T @ Y)                     # [D+1, 7]
    return W


def ridge_predict(W, X):
    N = X.shape[0]
    X_aug = torch.cat([X, torch.ones(N, 1, dtype=X.dtype)], dim=1)
    return X_aug @ W


def run_p1_branch(name, train_items, valid_items, frag_table, shift, scale, blind, lambdas):
    """P1 cho MOT nhanh (pocket hoac blind): xay dac trung (double), quet lambda tren VALID
    theo R2 trung binh 3 chieu tinh tien, chon lambda tot nhat, tra R2 train/valid tai do."""
    print(f"\n[*] --- P1 (ridge) nhanh '{name}' (blind={blind}) ---")
    X_train, Y_train_raw = build_p1_features(train_items, frag_table, blind)
    X_valid, Y_valid_raw = build_p1_features(valid_items, frag_table, blind)
    X_train, X_valid = X_train.double(), X_valid.double()
    Y_train = ((Y_train_raw.double() - shift) / scale)
    Y_valid = ((Y_valid_raw.double() - shift) / scale)
    print(f"    X_train={tuple(X_train.shape)}  X_valid={tuple(X_valid.shape)}")

    best_lam, best_mean_trans, best_r2_valid, best_W = None, -1e18, None, None
    for lam in lambdas:
        W = ridge_fit(X_train, Y_train, lam)
        pred_valid = ridge_predict(W, X_valid)
        r2_valid = compute_r2(pred_valid, Y_valid)
        mt = mean_trans_r2(r2_valid)
        print(f"    lambda={lam:<8g}  R2_trung_binh(d,theta,phi) tren valid = {mt:.4f}  "
              f"R2={['%.3f' % v for v in r2_valid]}")
        if mt > best_mean_trans:
            best_lam, best_mean_trans, best_r2_valid, best_W = lam, mt, r2_valid, W

    pred_train = ridge_predict(best_W, X_train)
    r2_train = compute_r2(pred_train, Y_train)
    print(f"    => lambda DA CHON = {best_lam:g} (R2 trung binh tinh tien tren valid = {best_mean_trans:.4f})")
    return best_lam, r2_train, best_r2_valid, X_train.shape[0], X_valid.shape[0]


# ============================================================================================
# P2 -- MLP (kien truc y het lan 1), CO GIAO THUC
# ============================================================================================

class PoseProbeHead(nn.Module):
    """Cross-attention 1 lop CO MASK tren h_target (kien truc GIU Y HET lan 1): query =
    frag_emb(frag_id) + pos_emb(t); K/V chieu tu h_target KHONG bias -- khi h_target=0 het
    (nhanh blind), K=V=0 het, logits attn = 0 het -> softmax deu tren mask -> ctx = trung binh
    cua V = 0 CHINH XAC, dam bao nhanh blind khong ri mot chut thong tin hoc nao qua ctx."""

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


def train_mlp_once(name, train_items, valid_items, h_dim, vocab_size, device, shift, scale,
                    lr, max_epochs, patience, batch_size, blind):
    """Mot lan huan luyen MLP day du voi early stopping (danh gia MOI epoch, giu checkpoint
    tot nhat theo R2 trung binh 3 chieu tinh tien, patience, weight_decay=1e-4). Tra ve
    (head_best, best_epoch, r2_train_best, r2_valid_best, n_train, n_valid)."""
    torch.manual_seed(SEED)
    head = PoseProbeHead(h_dim, vocab_size).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = make_loader(train_items, batch_size, shuffle=True, generator=g)
    valid_loader = make_loader(valid_items, batch_size, shuffle=False)

    best_mean_trans, best_epoch, best_state, best_r2_valid = -1e18, 0, None, None
    epochs_since_best = 0
    t_start = time.time()
    for epoch in range(1, max_epochs + 1):
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

        r2_valid, _ = eval_head(head, valid_loader, device, shift, scale, blind)
        mt = mean_trans_r2(r2_valid)
        improved = mt > best_mean_trans
        if improved:
            best_mean_trans, best_epoch = mt, epoch
            best_state = copy.deepcopy(head.state_dict())
            best_r2_valid = r2_valid
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            tag = " *best*" if improved else ""
            print(f"  [{name}] epoch {epoch}/{max_epochs}  train_loss={epoch_loss / max(n_tok, 1):.4f}  "
                  f"val_R2_trung_binh_tinh_tien={mt:.4f}{tag}  (elapsed {time.time() - t_start:.1f}s)",
                  flush=True)

        if epochs_since_best >= patience:
            print(f"  [{name}] early stopping tai epoch {epoch} (best epoch={best_epoch}, "
                  f"patience={patience})")
            break

    head.load_state_dict(best_state)
    r2_train_best, n_train = eval_head(head, make_loader(train_items, batch_size, shuffle=False),
                                        device, shift, scale, blind)
    n_valid = len(valid_items)
    print(f"  [{name}] HOAN TAT: best_epoch={best_epoch}  val_R2={['%.3f' % v for v in best_r2_valid]}")
    return head, best_epoch, r2_train_best, best_r2_valid, n_train, n_valid


def train_mlp_with_protocol(name, train_items, valid_items, h_dim, vocab_size, device, shift, scale,
                             max_epochs, patience, batch_size, blind, base_lr=1e-3):
    """P2, kiem giao thuc #2 (plan.md Rev 14): neu epoch tot nhat = 1 -> giam LR 10 lan, chay
    lai MOT lan; van epoch 1 -> 'P2 KHONG DOC DUOC' cho nhanh nay (ok=False)."""
    head, best_epoch, r2_train, r2_valid, n_train, n_valid = train_mlp_once(
        name, train_items, valid_items, h_dim, vocab_size, device, shift, scale,
        base_lr, max_epochs, patience, batch_size, blind
    )
    if best_epoch == 1:
        print(f"  [{name}] GIAO THUC: epoch tot nhat = 1 -> giam LR 10 lan ({base_lr:g} -> "
              f"{base_lr / 10:g}), chay lai MOT lan.")
        head, best_epoch, r2_train, r2_valid, n_train, n_valid = train_mlp_once(
            name, train_items, valid_items, h_dim, vocab_size, device, shift, scale,
            base_lr / 10, max_epochs, patience, batch_size, blind
        )
        if best_epoch == 1:
            print(f"  [{name}] P2 KHONG DOC DUOC (van epoch 1 sau khi giam LR).")
            return head, best_epoch, r2_train, r2_valid, n_train, n_valid, False
    return head, best_epoch, r2_train, r2_valid, n_train, n_valid, True


# ============================================================================================
# BAO CAO / TIEU CHI
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


def check_blind_converged(probe_name, r2_blind, r2_tokenmean):
    """Giao thuc #3 (plan.md Rev 14): R2(blind) thap hon R2(tokenmean) qua 0,03 o chieu tinh
    tien nao -> 'blind CHUA HOI TU', khong doc tieu chi chinh cho probe nay."""
    bad = [name for name in TRANS_DIMS
           if r2_blind[DIM_NAMES.index(name)] < r2_tokenmean[DIM_NAMES.index(name)] - 0.03]
    if bad:
        print(f"\n  [{probe_name}] blind CHUA HOI TU (thap hon tokenmean qua 0,03 o: {bad}) -- "
              f"KHONG doc tieu chi chinh cho {probe_name}.")
        return False
    return True


def classify_main_criterion(probe_name, r2_pocket, r2_blind):
    """Tieu chi chinh, GIU NGUYEN Rev 12. Tra ve (label, delta_by_dim)."""
    d_by_dim = {name: r2_pocket[DIM_NAMES.index(name)] - r2_blind[DIM_NAMES.index(name)]
                for name in DIM_NAMES}
    n_pass_005 = sum(1 for name in TRANS_DIMS if d_by_dim[name] > 0.05)
    all_below_002 = all(d_by_dim[name] < 0.02 for name in TRANS_DIMS)

    print(f"\n  [{probe_name}] TIEU CHI CHINH (plan.md Rev 12): R2(pocket)-R2(blind) > 0,05 o "
          f">= 2/3 chieu tinh tien")
    for name in TRANS_DIMS:
        print(f"      delta R2({name}) = {d_by_dim[name]:.4f}")

    if n_pass_005 >= 2:
        label = "TIEN_DE_DUNG"
        print(f"      [{probe_name}] => TIEN DE DUNG: thong tin hoc co san trong dau ra encoder.")
    elif all_below_002:
        label = "CHAY_M11B"
        print(f"      [{probe_name}] => thong tin KHONG co san trong encoder dong bang -- chay M11-b.")
    else:
        label = "TRUNG_GIAN"
        print(f"      [{probe_name}] => TRUNG GIAN: bao cao va dung.")

    d_d = d_by_dim["d"]
    print(f"      [{probe_name}] moc so sanh 0,081 (M10): delta R2(d) = {d_d:.4f}")
    if d_d > 0.081:
        print(f"      [{probe_name}] => delta R2(d) > 0,081: pipeline day du dang LAM MAT tin hieu.")
    else:
        print(f"      [{probe_name}] => delta R2(d) <= 0,081: tran o ENCODER, sua generator khong giup.")

    print(f"      [{probe_name}] quaternion (bao cao, khong dat nguong):")
    for name in ["qw", "qx", "qy", "qz"]:
        print(f"          delta R2({name}) = {d_by_dim[name]:.4f}")

    return label, d_by_dim


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(device)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Mac dinh: saved_checkpoints_flow/periodic-epoch=024.ckpt (chi dung o buoc cache)")
    ap.add_argument("--rebuild-cache", action="store_true", default=False)
    ap.add_argument("--batch-size", type=int, default=32, help="So COMPLEX moi batch (P1 pool + P2 MLP)")
    ap.add_argument("--p2-max-epochs", type=int, default=60)
    ap.add_argument("--p2-patience", type=int, default=10)
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

    # ------------------------------------------------------------------ BUOC CACHE -----------
    # KHONG lam lai cache: da co (733 MB train / 33 MB valid, RESEARCH_CONTEXT 31). Chi dung
    # --rebuild-cache neu build_h_target doi (plan.md Rev 14, "Ghi nho ve uoc luong chi phi").
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

        train_cache_dir, n_train_files = ensure_cache(model, device, "train", cache_root,
                                                        args.rebuild_cache, st, vocab, pos_scale)
        valid_cache_dir, n_valid_files = ensure_cache(model, device, "valid", cache_root,
                                                        args.rebuild_cache, st, vocab, pos_scale)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        n_train_files, n_valid_files = len(train_existing), len(valid_existing)
        print(f"[*] cache[train]/cache[valid]: da ton tai du ca hai ({n_train_files}/{n_valid_files} "
              f"complex), BO QUA buoc cache hoan toan.")

    for split, d, n in [("train", train_cache_dir, n_train_files), ("valid", valid_cache_dir, n_valid_files)]:
        size_mb = dir_size_bytes(d) / 1e6
        print(f"[*] cache[{split}]: {n} complex, {size_mb:.1f} MB tai {d}")

    # ------------------------------------------------------------------ PRELOAD RAM ----------
    print(f"\n[*] --- Preload toan bo cache vao RAM (dung chung cho tokenmean/P1/P2) ---")
    train_items = preload_split(train_cache_dir)
    valid_items = preload_split(valid_cache_dir)
    h_dim = train_items[0][0].shape[1]
    vocab_size = len(vocab)

    shift_cpu = shift_factors.view(7).double()
    scale_cpu = scale_factors.view(7).double()
    shift_dev = shift_factors.view(7).to(device)
    scale_dev = scale_factors.view(7).to(device)

    print(f"\n[*] CAU HINH: batch_size={args.batch_size}  vocab_size={vocab_size}  h_dim={h_dim}  "
          f"n_train={len(train_items)}  n_valid={len(valid_items)}  "
          f"p2_max_epochs={args.p2_max_epochs}  p2_patience={args.p2_patience}")

    # ------------------------------------------------------------------ TOKENMEAN (TRUOC TIEN) --
    print("\n" + "=" * 90)
    print("GIAO THUC #1: R2(tokenmean) TRUOC TIEN (san tuyet doi, plan.md Rev 14)")
    print("=" * 90)
    tm_r2_train, tm_r2_valid, tm_n_train, tm_n_valid = compute_tokenmean(
        train_items, valid_items, args.batch_size, shift_cpu, scale_cpu
    )
    print(f"  tokenmean: n_train={tm_n_train}  n_valid={tm_n_valid}")
    for i, name in enumerate(DIM_NAMES):
        print(f"      R2(tokenmean)[{name}] = {tm_r2_valid[i]:.4f}")

    # ------------------------------------------------------------------------- P1 ------------
    print("\n" + "=" * 90)
    print("P1 (CHINH): LINEAR PROBE CO RIDGE")
    print("=" * 90)
    frag_table = fixed_frag_embedding(vocab_size, dim=32, seed=SEED)
    lambdas = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]

    p1_lam_pocket, p1_r2_train_pocket, p1_r2_valid_pocket, p1_n_train, p1_n_valid = run_p1_branch(
        "pocket", train_items, valid_items, frag_table, shift_cpu, scale_cpu, blind=False, lambdas=lambdas
    )
    p1_lam_blind, p1_r2_train_blind, p1_r2_valid_blind, _, _ = run_p1_branch(
        "blind", train_items, valid_items, frag_table, shift_cpu, scale_cpu, blind=True, lambdas=lambdas
    )

    print_r2_table("P1 -- BANG R2 TREN VALID (7 chieu x 3 nhanh)",
                    {"pocket": p1_r2_valid_pocket, "blind": p1_r2_valid_blind, "tokenmean": tm_r2_valid})
    print_r2_table("P1 -- BANG R2 TREN TRAIN (de thay overfit)",
                    {"pocket": p1_r2_train_pocket, "blind": p1_r2_train_blind, "tokenmean": tm_r2_train})

    print("\n" + "=" * 90)
    print("GIAO THUC #3 (P1): blind vs tokenmean")
    print("=" * 90)
    p1_blind_ok = check_blind_converged("P1", p1_r2_valid_blind, tm_r2_valid)
    p1_label = None
    if p1_blind_ok:
        p1_label, p1_d_by_dim = classify_main_criterion("P1", p1_r2_valid_pocket, p1_r2_valid_blind)

    # ------------------------------------------------------------------------- P2 ------------
    print("\n" + "=" * 90)
    print("P2 (PHU): MLP VOI GIAO THUC (early stopping, weight_decay, preload)")
    print("=" * 90)
    (pocket_head, pk_best_epoch, pk_r2_train, pk_r2_valid, pk_n_train, pk_n_valid,
     pk_ok) = train_mlp_with_protocol(
        "pocket", train_items, valid_items, h_dim, vocab_size, device, shift_dev, scale_dev,
        args.p2_max_epochs, args.p2_patience, args.batch_size, blind=False
    )
    (blind_head, bl_best_epoch, bl_r2_train, bl_r2_valid, bl_n_train, bl_n_valid,
     bl_ok) = train_mlp_with_protocol(
        "blind", train_items, valid_items, h_dim, vocab_size, device, shift_dev, scale_dev,
        args.p2_max_epochs, args.p2_patience, args.batch_size, blind=True
    )
    p2_readable = pk_ok and bl_ok
    if not p2_readable:
        print("\n  P2 KHONG DOC DUOC (giao thuc #2: mo hinh van chon epoch 1 sau khi giam LR o "
              "it nhat mot nhanh). P1 van la ket qua chinh, xem o tren.")

    print_r2_table(f"P2 -- BANG R2 TREN VALID TAI EPOCH TOT NHAT (pocket={pk_best_epoch}, "
                    f"blind={bl_best_epoch})",
                    {"pocket": pk_r2_valid, "blind": bl_r2_valid, "tokenmean": tm_r2_valid})
    print_r2_table("P2 -- BANG R2 TREN TRAIN TAI EPOCH TOT NHAT (de thay overfit)",
                    {"pocket": pk_r2_train, "blind": bl_r2_train, "tokenmean": tm_r2_train})

    p2_label = None
    if p2_readable:
        print("\n" + "=" * 90)
        print("GIAO THUC #3 (P2): blind vs tokenmean")
        print("=" * 90)
        p2_blind_ok = check_blind_converged("P2", bl_r2_valid, tm_r2_valid)
        if p2_blind_ok:
            p2_label, p2_d_by_dim = classify_main_criterion("P2", pk_r2_valid, bl_r2_valid)

    # ------------------------------------------------------------------------- DOI CHIEU -----
    print("\n" + "=" * 90)
    print("DOI CHIEU P1 vs P2 (tieu chi chinh GIU NGUYEN Rev 12 -- doc P1 truoc, P2 doi chieu)")
    print("=" * 90)
    print(f"  P1 label = {p1_label}")
    print(f"  P2 label = {p2_label}  (P2 khong doc duoc = {not p2_readable})")
    opposite_pair = {"TIEN_DE_DUNG", "CHAY_M11B"}
    if p1_label in opposite_pair and p2_label in opposite_pair and p1_label != p2_label:
        print("  => KHONG KET LUAN: P1 va P2 cho ket luan NGUOC nhau. Bao cao ca hai va dung.")
    elif p1_label is not None:
        print(f"  => Ket luan chinh thuc dua tren P1 (phep chinh): {p1_label}"
              + (f"; P2 dong thuan ({p2_label})." if p2_label == p1_label
                 else "; P2 khong doc duoc hoac khong doi chieu duoc o tren."))
    else:
        print("  => Khong doc duoc P1 (blind chua hoi tu) -- xem GIAO THUC #3 (P1) o tren.")


if __name__ == "__main__":
    main()
