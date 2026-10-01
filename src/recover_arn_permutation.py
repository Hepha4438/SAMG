"""
M3 (plan.md Rev 5): phuc hoi permutation THAT cua ARN tu checkpoint.

Can cu: pyro `MaskedLinear` dang ky `mask` lam buffer => mask nam trong state_dict.
Khong chay training, khong can GPU, khong can du lieu.
"""
import os, re, sys, glob, subprocess
import torch

DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]
D = len(DIM_NAMES)

ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(ROOT) == "src":
    ROOT = os.path.dirname(ROOT)

try:
    print("HEAD:", subprocess.run("git rev-parse HEAD", shell=True, capture_output=True,
                                  text=True, cwd=ROOT).stdout.strip())
except Exception:
    pass


def find_checkpoints():
    pats = [
        "save_checkpoint_flow/**/*.ckpt", "save_checkpoint*/**/*.ckpt",
        "logs/**/*.ckpt", "checkpoints/**/*.ckpt", "**/*.ckpt",
    ]
    seen, out = set(), []
    for p in pats:
        for f in glob.glob(os.path.join(ROOT, p), recursive=True):
            rp = os.path.realpath(f)
            if rp not in seen:
                seen.add(rp)
                out.append(f)
    out.sort(key=lambda f: os.path.getmtime(f), reverse=True)
    return out


def recover(masks_by_key, verbose=True):
    """masks_by_key: dict key -> 2D tensor. Tra ve (dep[D,D], multiplier) hoac None.

    Pyro create_mask: input_indices = cat((zeros(context_dim), 1 + var_index))
    => COT CONTEXT DUNG TRUOC, cot input dim la D cot CUOI. Day la diem de sai.
    """
    items = {k: v for k, v in masks_by_key.items() if torch.is_tensor(v) and v.dim() == 2}
    if not items:
        return None
    if verbose:
        print("\n-- mask buffers --")
        for k in sorted(items):
            print(f"   {k:62s} shape={tuple(items[k].shape)}")

    def layer_idx(k):
        m = re.search(r"layers\.(\d+)\.", k)
        return int(m.group(1)) if m else None

    chain_keys = sorted([k for k in items if layer_idx(k) is not None], key=layer_idx)
    skip_keys = [k for k in items if layer_idx(k) is None]

    if not chain_keys:
        print("!! khong tim thay mask nao ten '...layers.<i>.mask'")
        return None

    cur = items[chain_keys[0]].float()
    for k in chain_keys[1:]:
        m = items[k].float()
        if m.shape[1] != cur.shape[0]:
            print(f"!! shape khong noi duoc: {k} {tuple(m.shape)} vs {tuple(cur.shape)}")
            return None
        cur = m @ cur
    if verbose:
        print("   chuoi nhan:", " <- ".join(reversed(chain_keys)), "=>", tuple(cur.shape))

    dep_rows = cur > 0
    for k in skip_keys:                      # skip noi truc tiep input -> output
        m = items[k]
        if m.shape == dep_rows.shape or (m.shape[0] == dep_rows.shape[0]
                                         and m.shape[1] == dep_rows.shape[1]):
            dep_rows = dep_rows | (m > 0)
            if verbose:
                print(f"   da OR mask skip: {k}")

    nrow, ncol = dep_rows.shape
    if nrow % D != 0:
        print(f"!! so hang output ({nrow}) khong chia het cho {D}")
        return None
    mult = nrow // D
    ctx = ncol - D
    if ctx < 0:
        print(f"!! so cot ({ncol}) nho hon so chieu ({D})")
        return None
    if verbose:
        print(f"   context_dim suy ra = {ctx}; lay {D} cot CUOI lam input dim")

    dep = torch.zeros(D, D, dtype=torch.bool)
    for m in range(mult):                    # pyro: hang = m*D + i
        dep |= dep_rows[m * D:(m + 1) * D, -D:]
    return dep, mult


def report(dep):
    counts = dep.sum(1).tolist()
    print("\n-- ma tran phu thuoc (hang = output, cot = input) --")
    print("            " + " ".join(f"{n:>6s}" for n in DIM_NAMES) + "   | #")
    for i, n in enumerate(DIM_NAMES):
        row = " ".join(f"{'  X   ' if dep[i, j] else '  .   '}" for j in range(D))
        print(f"   {n:>7s}  {row}   | {counts[i]}")

    order = sorted(range(D), key=lambda i: counts[i])
    print("\n-- thu tu tu hoi quy (it phu thuoc nhat -> nhieu nhat) --")
    print("   " + " -> ".join(f"{DIM_NAMES[i]}({counts[i]})" for i in order))

    qi = DIM_NAMES.index("qy")
    others = [DIM_NAMES.index(x) for x in ("qw", "qx", "qz")]
    sees = [DIM_NAMES[j] for j in others if dep[qi, j]]
    pos = order.index(qi)

    print("\n" + "=" * 70)
    print(f"qy o vi tri {pos} / {D - 1} trong thu tu tu hoi quy")
    print(f"qy nhin thay: {sees if sees else 'KHONG chieu quaternion nao'}  (can ca 3: qw, qx, qz)")
    if len(sees) == 3:
        print("KET LUAN: (a) XAC NHAN — qy suy duoc |qy| tu ||q||=1.")
        print("          Muc 19.4 sai. Viec tiep theo: BIEU DIEN QUAY (6D), khong phai doi head.")
    else:
        print("KET LUAN: (a) BI LOAI — qy khong du thong tin de dung ||q||=1.")
        print("          Con lai (c): hoc mang thong tin ve huong. Chay tiep P3a -> M2 -> G5.")
    print("=" * 70)


def main():
    cks = find_checkpoints()
    if not cks:
        print("\nKHONG CO CHECKPOINT. M3 khong chay duoc tu du lieu da co.")
        print("=> Phai lam P3a-1 (co dinh permutation) roi chay 1 epoch moi doc duoc.")
        return
    print(f"\nTim thay {len(cks)} checkpoint (moi nhat truoc):")
    for f in cks[:10]:
        print(f"   {os.path.getmtime(f):.0f}  {os.path.relpath(f, ROOT)}")

    ck = cks[0]
    print(f"\n[*] Dung: {os.path.relpath(ck, ROOT)}")
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    sd = obj.get("state_dict", obj) if isinstance(obj, dict) else obj

    masks = {k: v for k, v in sd.items()
             if k.endswith("mask") and torch.is_tensor(v) and "arn" in k}
    if not masks:
        masks = {k: v for k, v in sd.items() if k.endswith("mask") and torch.is_tensor(v)}
    if not masks:
        print("\n!! KHONG CO mask trong state_dict. Head nay khong phai MAF,")
        print("   hoac phien ban pyro khong luu mask lam buffer. In cac key co 'arn':")
        for k in sd:
            if "arn" in k or "geometric_head" in k:
                print("   ", k, tuple(sd[k].shape) if torch.is_tensor(sd[k]) else type(sd[k]))
        return

    got = recover(masks)
    if got is None:
        return
    dep, mult = got
    print(f"\n   output_multiplier suy ra = {mult} (MAF affine ky vong 2: mean + log_scale)")
    report(dep)


if __name__ == "__main__":
    main()
