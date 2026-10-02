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
    1234, KHONG hoc), one-hot vi tri t (clip t<=15) [16]]. Mot ma tran ridge cho ca 7 chieu
    (closed-form), lambda quet tren {1e-3..1e3}, chon theo R2 trung binh 3 chieu tinh tien
    TREN VALID.
  P2 (PHU) -- kien truc MLP y het lan 1 (cross-attention 1 lop co mask, K/V khong bias nen
    nhanh blind cho ctx=0 chinh xac), nhung CO GIAO THUC: early stopping tren valid (danh gia
    MOI epoch, patience 10, toi da 60 epoch, giu checkpoint tot nhat theo R2 trung binh 3
    chieu tinh tien), weight_decay=1e-4, bao cao R2 tai epoch TOT NHAT (khong phai epoch
    cuoi), preload toan bo cache vao RAM (tranh torch.load tung sample moi epoch, 31.5).

REV 15 (plan.md "PLAN Rev 15"; RESEARCH_CONTEXT Muc 32): P2 DAT tieu chi chinh 3/3
(delta R2 d/theta/phi = 0,0999/0,0674/0,0690), nhung P1 (phep CHINH) VOID -- loi dac ta thu
CHIN (32.4): embedding frag_id co dinh 32 chieu lam nhanh `blind` cua P1 VE CAU TRUC khong the
dat toi san tokenmean (19.033 id, embedding chi bieu dien duoc ham 32 chieu cua danh tinh
fragment, con tokenmean la bang tra cuu DAY DU) -- `pocket - blind` cua P1 bi phong dai gia
tao. SUA (chi doi dac trung cua P1, giu nguyen P2 va tieu chi doc):
  Bo embedding ngau nhien co dinh. Thay bang TARGET ENCODING: 7 dac trung
  `tokenmean_pose[frag_id]` = CHINH du doan cua nhanh tokenmean (trung binh 7D theo frag_id,
  tinh tren train; frag_id la khong co trong train dung trung binh toan cuc). CHONG RO RI bat
  buoc: voi hang TRAIN, dac trung nay PHAI tinh OUT-OF-FOLD (5 fold, seed 1234 -- xem
  kfold_assignment/build_target_encoding_oof) vi tinh truc tiep tren toan bo train se cho dac
  trung tu-du-doan-chinh-no (gan nhu ro ri target). Voi hang VALID, dung trung binh tren TOAN
  BO train (khong fold, vi valid khong duoc dung de fit). Dac trung moi token gio la
  concat[mean-pool [259], max-pool [259], tokenmean_pose[frag_id] [7], one-hot t [16]] = 541.
  Ly do: nhanh `blind` cua P1 khi do >= tokenmean VE CAU TRUC (chi can hoc he so ~1 cho 7 dac
  trung target-encoding) nen giao thuc #3 khong bi chan gia tao; va `pocket` duoc do TREN NEN
  thong tin danh tinh fragment, nen confound "hoc lam proxy cho frag_id" bi triet tieu.

REV 16 (plan.md "PLAN Rev 16"; RESEARCH_CONTEXT Muc 33, M12-a): ca hai giao thuc (P1 da sua,
P2) XAC NHAN tien de (33.1), nhung ket luan phu "pipeline dang lam mat tin hieu" (vi delta
R2(d) > 0,081 cua M10) CHUA co can cu (33.5) -- M10 va M11 do HAI DAI LUONG KHAC NHAU: M10 xao
tron luc SUY LUAN tren mot mo hinh DA huan luyen ("ham da hoc DUNG hoc bao nhieu"), con M11 so
"co hoc vs khong hoc" luc HUAN LUYEN ("co hoc thi DUOC THEM bao nhieu") -- dai luong thu hai LON
HON dai luong thu nhat MOT CACH HE THONG du khong co gi "bi mat". SUA: them nhanh thu tu
`pocket_shuf_eval` cho CA P1 va P2 -- dung DUNG model `pocket` DA HUAN LUYEN (P1: ma tran
ridge da fit; P2: checkpoint tai best_epoch), KHONG huan luyen lai gi, nhung khi DANH GIA tren
valid thi xao tron h_target theo chieu COMPLEX (derangement, seed 1234, xem
build_complex_shuffled_items/make_derangement_indices), target_mask xao tron CUNG hoan vi.
`R2(pocket) - R2(pocket_shuf_eval)` moi la dai luong CUNG LOAI voi M10, so duoc truc tiep voi
nguong 0,081. Bao cao canh nhau cot A (`R2(pocket)-R2(blind)`, train-ablation) va cot B
(`R2(pocket)-R2(pocket_shuf_eval)`, inference-shuffle) cho ca 7 chieu; nguong 0,081 CHI ap cho
cot B.

Ca P1 va P2 deu chay BON nhanh: pocket / blind (zero hoa H+3 kenh cua h_target, giu target_mask)
/ tokenmean (khong mo hinh, trung binh 7D theo frag_id tinh tren train) / pocket_shuf_eval
(model pocket da huan luyen, h_target xao tron theo complex CHI luc danh gia -- M12-a, Rev 16).

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

def fit_tokenmean(items, shift, scale):
    """mean_by_frag[fid] (tensor[7], float64, DA CHUAN HOA) + global_mean, tinh tren TOAN BO
    `items` da cho. Dung lam (a) chinh nhanh tokenmean (qua compute_tokenmean, KHONG doi) va
    (b) nguon target-encoding cho P1 (Rev 15) -- CHINH la du doan cua tokenmean, nen goi lai
    ham nay thay vi viet lai logic trung binh-theo-frag_id o noi khac."""
    sums, counts = {}, {}
    global_sum = torch.zeros(7, dtype=torch.float64)
    global_n = 0
    for h, m, t7d, tid in items:
        target_scaled = (t7d.double() - shift) / scale
        for fid, pose in zip(tid.tolist(), target_scaled):
            sums[fid] = sums.get(fid, torch.zeros(7, dtype=torch.float64)) + pose
            counts[fid] = counts.get(fid, 0) + 1
            global_sum += pose
            global_n += 1
    global_mean = global_sum / max(global_n, 1)
    mean_by_frag = {fid: sums[fid] / counts[fid] for fid in sums}
    return mean_by_frag, global_mean


def kfold_assignment(n, k, seed):
    """Gan k fold cho n hang, Generator RIENG (seed co dinh, doc lap RNG toan cuc) de luon tai
    lap duoc. Hoan vi ngau nhien roi chia vong-tron -> fold can doi."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    fold_of = torch.empty(n, dtype=torch.long)
    fold_of[perm] = torch.arange(n, dtype=torch.long) % k
    return fold_of


def build_target_encoding_oof(train_items, shift, scale, n_folds=5, seed=1234):
    """TARGET ENCODING OUT-OF-FOLD cho CAC HANG TRAIN (Rev 15, RESEARCH_CONTEXT 32.4): dac
    trung tokenmean_pose[frag_id] cua token thuoc fold k CHI duoc tinh tu 4 fold con lai --
    tinh truc tiep tren toan bo train se cho dac trung tu-du-doan-chinh-no (ro ri), lam fit
    tren train lac quan gia tao va lech viec chon lambda. Tra ve list[Tensor[T_i,7]] CUNG THU
    TU voi train_items (dac trung OOF cho cac token cua complex i, da chuan hoa)."""
    token_fid_chunks, token_pose_chunks, complex_lens = [], [], []
    for h, m, t7d, tid in train_items:
        token_fid_chunks.append(tid)
        token_pose_chunks.append((t7d.double() - shift) / scale)
        complex_lens.append(tid.shape[0])
    all_fid = torch.cat(token_fid_chunks)                 # [N]
    all_pose = torch.cat(token_pose_chunks, dim=0)         # [N, 7]
    N = all_fid.shape[0]
    fold_of = kfold_assignment(N, n_folds, seed)

    fold_sum_by_fid = [dict() for _ in range(n_folds)]
    fold_count_by_fid = [dict() for _ in range(n_folds)]
    fold_total_sum = torch.zeros(n_folds, 7, dtype=torch.float64)
    fold_total_count = torch.zeros(n_folds, dtype=torch.float64)
    for i in range(N):
        k = int(fold_of[i])
        fid = int(all_fid[i])
        pose = all_pose[i]
        fold_sum_by_fid[k][fid] = fold_sum_by_fid[k].get(fid, torch.zeros(7, dtype=torch.float64)) + pose
        fold_count_by_fid[k][fid] = fold_count_by_fid[k].get(fid, 0) + 1
        fold_total_sum[k] += pose
        fold_total_count[k] += 1

    global_sum_by_fid, global_count_by_fid = {}, {}
    for k in range(n_folds):
        for fid, s in fold_sum_by_fid[k].items():
            global_sum_by_fid[fid] = global_sum_by_fid.get(fid, torch.zeros(7, dtype=torch.float64)) + s
            global_count_by_fid[fid] = global_count_by_fid.get(fid, 0) + fold_count_by_fid[k][fid]
    global_total_sum = fold_total_sum.sum(dim=0)
    global_total_count = fold_total_count.sum()

    feats = torch.zeros(N, 7, dtype=torch.float64)
    for i in range(N):
        k = int(fold_of[i])
        fid = int(all_fid[i])
        s_out = global_sum_by_fid.get(fid, torch.zeros(7, dtype=torch.float64)) - \
            fold_sum_by_fid[k].get(fid, torch.zeros(7, dtype=torch.float64))
        c_out = global_count_by_fid.get(fid, 0) - fold_count_by_fid[k].get(fid, 0)
        if c_out > 0:
            feats[i] = s_out / c_out
        else:
            oof_total_sum = global_total_sum - fold_total_sum[k]
            oof_total_count = (global_total_count - fold_total_count[k]).clamp_min(1)
            feats[i] = oof_total_sum / oof_total_count

    out, pos = [], 0
    for L in complex_lens:
        out.append(feats[pos:pos + L])
        pos += L
    return out


def build_target_encoding_full(items, mean_by_frag, global_mean):
    """Target encoding cho CAC HANG VALID (Rev 15): dung TOAN BO train (khong fold) vi valid
    khong duoc dung de fit gi ca, nen khong co nguy co ro ri. Tra ve list[Tensor[T_i,7]]."""
    out = []
    for h, m, t7d, tid in items:
        out.append(torch.stack([mean_by_frag.get(fid, global_mean) for fid in tid.tolist()]))
    return out


def masked_mean_pool(h, mask):
    mask_f = mask.unsqueeze(-1).to(h.dtype)
    return (h * mask_f).sum(0) / mask_f.sum(0).clamp_min(1e-8)


def masked_max_pool(h, mask):
    if not bool(mask.any()):
        return torch.zeros(h.shape[-1], dtype=h.dtype)
    neg_inf = torch.finfo(h.dtype).min
    h_masked = h.masked_fill(~mask.unsqueeze(-1), neg_inf)
    return h_masked.max(dim=0).values


def build_p1_features(items, target_enc_per_complex, blind):
    """Dac trung P1 moi token (Rev 15): concat[mean-pool(h,mask) [259], max-pool(h,mask) [259],
    tokenmean_pose[frag_id] TARGET ENCODING [7], one-hot(clip(t,15)) [16]] = 541. target_enc_
    per_complex: list[Tensor[T_i,7]] CUNG THU TU voi `items` (OOF cho train, toan-bo-train cho
    valid -- xem build_target_encoding_oof/build_target_encoding_full). Target encoding KHONG
    phu thuoc `blind` (khong nam trong h_target) -- day la diem then chot: nhanh blind van GIU
    duoc tin hieu danh tinh fragment nay, nen no co the dat toi san tokenmean VE CAU TRUC.
    Tra ve (X [N, 541], Y_raw [N, 7])."""
    feats, poses_list = [], []
    for (h, m, t7d, tid), enc in zip(items, target_enc_per_complex):
        if blind:
            h = torch.zeros_like(h)
        pooled_mean = masked_mean_pool(h, m)          # [H]
        pooled_max = masked_max_pool(h, m)             # [H]
        T = tid.shape[0]
        pos_clip = torch.arange(T).clamp_max(15)
        pos_oh = F.one_hot(pos_clip, num_classes=16).to(h.dtype)  # [T, 16]
        mean_b = pooled_mean.unsqueeze(0).expand(T, -1)
        max_b = pooled_max.unsqueeze(0).expand(T, -1)
        feats.append(torch.cat([mean_b, max_b, enc.to(h.dtype), pos_oh], dim=-1))
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


def run_p1_branch(name, train_items, valid_items, train_target_enc, valid_target_enc,
                   shift, scale, blind, lambdas):
    """P1 cho MOT nhanh (pocket hoac blind): xay dac trung (double), quet lambda tren VALID
    theo R2 trung binh 3 chieu tinh tien, chon lambda tot nhat, tra R2 train/valid tai do.
    train_target_enc/valid_target_enc: target encoding da tinh SAN (OOF cho train, toan-bo-
    train cho valid) -- GIONG NHAU cho ca nhanh pocket va blind, vi no khong phu thuoc h_target."""
    print(f"\n[*] --- P1 (ridge) nhanh '{name}' (blind={blind}) ---")
    X_train, Y_train_raw = build_p1_features(train_items, train_target_enc, blind)
    X_valid, Y_valid_raw = build_p1_features(valid_items, valid_target_enc, blind)
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
    return best_lam, r2_train, best_r2_valid, X_train.shape[0], X_valid.shape[0], best_W


def make_derangement_indices(n, seed):
    """Derangement (khong diem bat dong) cua n phan tu -- Generator RIENG (seed co dinh, doc
    lap RNG toan cuc) de luon tai lap duoc. M12-a (plan.md Rev 16)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(n)
    for _ in range(10):
        perm = torch.randperm(n, generator=g)
        if not bool((perm == idx).any()):
            return perm
    return torch.roll(idx, 1)


def build_complex_shuffled_items(items, seed=SEED):
    """M12-a (plan.md Rev 16, RESEARCH_CONTEXT 33.5): hoan vi h_target/target_mask theo chieu
    COMPLEX (derangement, seed co dinh), GIU target_7d/target_ids tai vi tri goc -- dung CHO
    CA P1 va P2 de danh gia model `pocket` DA HUAN LUYEN tren pocket bi xao tron CHI luc SUY
    LUAN (khong huan luyen lai gi). Tra ve list CUNG DO DAI, vi tri i mang (h,m) cua complex
    perm[i] nhung (t7d,tid) cua CHINH complex i."""
    n = len(items)
    perm = make_derangement_indices(n, seed)
    return [(items[int(perm[i])][0], items[int(perm[i])][1], items[i][2], items[i][3])
            for i in range(n)]


def eval_p1_pocket_shuf(best_W, valid_items_shuf, valid_target_enc, shift, scale):
    """M12-a cho P1: ap DUNG ma tran ridge best_W DA FIT (khong refit) tren dac trung xay tu
    valid_items_shuf (h_target/mask da xao tron theo complex, target-encoding/vi-tri GIU
    NGUYEN vi khong phu thuoc h_target va khong doi theo hoan vi nay)."""
    X_shuf, Y_raw = build_p1_features(valid_items_shuf, valid_target_enc, blind=False)
    X_shuf = X_shuf.double()
    Y_shuf = (Y_raw.double() - shift) / scale
    pred_shuf = ridge_predict(best_W, X_shuf)
    return compute_r2(pred_shuf, Y_shuf)


def print_AB_table(probe_name, r2_pocket, r2_blind, r2_pocket_shuf):
    """M12-a: bang canh nhau cot A = R2(pocket)-R2(blind) [train-ablation], cot B =
    R2(pocket)-R2(pocket_shuf_eval) [inference-shuffle] -- cung loai voi M10. Tra ve B_by_dim."""
    print(f"\n  [{probe_name}] BANG CANH NHAU (M12-a): cot A = R2(pocket)-R2(blind) "
          f"[train-ablation]  |  cot B = R2(pocket)-R2(pocket_shuf_eval) [inference-shuffle, "
          f"CUNG LOAI voi M10]")
    print(f"      {'chieu':8}{'A (train-ablation)':>22}{'B (inference-shuffle)':>24}")
    B_by_dim = {}
    for i, name in enumerate(DIM_NAMES):
        A = r2_pocket[i] - r2_blind[i]
        B = r2_pocket[i] - r2_pocket_shuf[i]
        B_by_dim[name] = B
        print(f"      {name:8}{A:22.4f}{B:24.4f}")
    return B_by_dim


def classify_m12a_criterion(probe_name, B_by_dim):
    """Tieu chi doc M12-a (plan.md Rev 16), ap nguong 0,081 (M10) CHO COT B, KHONG cho cot A."""
    B_d = B_by_dim["d"]
    print(f"\n  [{probe_name}] TIEU CHI M12-a: B(d) = R2(pocket)-R2(pocket_shuf_eval) tai d, "
          f"doi chieu nguong 0,081 (M10) -- CHI ap cho cot B")
    print(f"      B(d) = {B_d:.4f}")
    if B_d > 0.081:
        print(f"      [{probe_name}] => XAC NHAN: pipeline khai thac KEM HON probe, CUNG "
              f"estimator -- so sanh hop le. Di theo huong 'tim cho pipeline lam mat tin hieu'.")
        return "XAC_NHAN"
    else:
        print(f"      [{probe_name}] => BAC: ket luan 32.3/33.5 ('pipeline lam mat tin hieu') "
              f"KHONG co can cu. Chenh lech o Muc 32-33 chi la hieu ung doi estimator. Huong di "
              f"KHONG phai sua generator, phai thiet ke lai buoc tiep.")
        return "BAC"


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
    print("P1 (CHINH): LINEAR PROBE CO RIDGE -- dac trung TARGET ENCODING (Rev 15)")
    print("=" * 90)
    # Rev 15 (RESEARCH_CONTEXT 32.4): bo embedding ngau nhien co dinh 32 chieu (lam nhanh blind
    # bi phong dai gia tao, vi VE CAU TRUC khong dat toi san tokenmean). Thay bang target
    # encoding = CHINH du doan cua tokenmean; OOF (5 fold, seed 1234) cho hang TRAIN de chong
    # ro ri tu-du-doan-chinh-no; toan-bo-train (khong fold) cho hang VALID.
    mean_by_frag, global_mean = fit_tokenmean(train_items, shift_cpu, scale_cpu)
    print("[*] --- P1: dang tinh target encoding OOF cho train (5 fold, seed 1234) ---")
    train_target_enc = build_target_encoding_oof(train_items, shift_cpu, scale_cpu, n_folds=5, seed=SEED)
    valid_target_enc = build_target_encoding_full(valid_items, mean_by_frag, global_mean)
    # M12-a (plan.md Rev 16, 33.3): them 1e4, 1e5 -- blind dang chon 1000 = bien tren cua day cu.
    lambdas = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]

    p1_lam_pocket, p1_r2_train_pocket, p1_r2_valid_pocket, p1_n_train, p1_n_valid, p1_W_pocket = run_p1_branch(
        "pocket", train_items, valid_items, train_target_enc, valid_target_enc,
        shift_cpu, scale_cpu, blind=False, lambdas=lambdas
    )
    p1_lam_blind, p1_r2_train_blind, p1_r2_valid_blind, _, _, _ = run_p1_branch(
        "blind", train_items, valid_items, train_target_enc, valid_target_enc,
        shift_cpu, scale_cpu, blind=True, lambdas=lambdas
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
        d_qz_p1 = p1_d_by_dim["qz"]
        print(f"\n  [P1] delta R2(qz) = {d_qz_p1:.4f}  (doi chieu RESEARCH_CONTEXT 32.5: nguong >= 0,05)")
        if d_qz_p1 >= 0.05:
            print("      => VUOT/BANG 0,05: mo lai Muc 16.1 nghiem tuc (phep quay co the bi chi "
                  "phoi boi hoc), ghi thanh muc rieng.")
        else:
            print("      => DUOI 0,05: Muc 32.5 dong lai, phep quay van la 'khong chi phoi boi hoc'.")

    # --- M12-a (plan.md Rev 16): nhanh thu tu pocket_shuf_eval, DUNG model pocket DA FIT ------
    print("\n" + "=" * 90)
    print("M12-a (P1): pocket_shuf_eval -- model pocket DA FIT, h_target xao tron theo complex "
          "CHI luc danh gia (derangement, seed 1234), KHONG huan luyen lai")
    print("=" * 90)
    valid_items_shuf = build_complex_shuffled_items(valid_items, seed=SEED)
    p1_r2_pocket_shuf = eval_p1_pocket_shuf(p1_W_pocket, valid_items_shuf, valid_target_enc,
                                             shift_cpu, scale_cpu)
    for i, name in enumerate(DIM_NAMES):
        print(f"      R2(pocket_shuf_eval)[{name}] = {p1_r2_pocket_shuf[i]:.4f}   "
              f"delta(pocket - pocket_shuf_eval) = {p1_r2_valid_pocket[i] - p1_r2_pocket_shuf[i]:.4f}")
    p1_B_by_dim = print_AB_table("P1", p1_r2_valid_pocket, p1_r2_valid_blind, p1_r2_pocket_shuf)
    p1_m12a = classify_m12a_criterion("P1", p1_B_by_dim)

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

    # --- M12-a (plan.md Rev 16): nhanh thu tu pocket_shuf_eval, DUNG checkpoint pocket best_epoch
    print("\n" + "=" * 90)
    print("M12-a (P2): pocket_shuf_eval -- checkpoint pocket tai best_epoch, h_target xao tron "
          "theo complex CHI luc danh gia (derangement, seed 1234), KHONG huan luyen lai")
    print("=" * 90)
    p2_r2_pocket_shuf, _ = eval_head(
        pocket_head, make_loader(valid_items_shuf, args.batch_size, shuffle=False),
        device, shift_dev, scale_dev, blind=False
    )
    for i, name in enumerate(DIM_NAMES):
        print(f"      R2(pocket_shuf_eval)[{name}] = {p2_r2_pocket_shuf[i]:.4f}   "
              f"delta(pocket - pocket_shuf_eval) = {pk_r2_valid[i] - p2_r2_pocket_shuf[i]:.4f}")
    p2_B_by_dim = print_AB_table("P2", pk_r2_valid, bl_r2_valid, p2_r2_pocket_shuf)
    p2_m12a = classify_m12a_criterion("P2", p2_B_by_dim)

    print("\n" + "=" * 90)
    print("DOI CHIEU M12-a: P1 vs P2 (nguong 0,081 ap cho cot B)")
    print("=" * 90)
    print(f"  P1 M12-a = {p1_m12a}   P2 M12-a = {p2_m12a}")
    if p1_m12a != p2_m12a:
        print("  => P1 va P2 cho ket luan M12-a KHAC nhau -- bao cao ca hai, khong tu quyet dinh them.")
    else:
        print(f"  => P1 va P2 DONG THUAN: {p1_m12a}.")

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
