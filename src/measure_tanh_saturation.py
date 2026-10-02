"""
measure_tanh_saturation.py -- M12-b (plan.md "PLAN Rev 16"; RESEARCH_CONTEXT Muc 33): do bao
hoa cua nn.Tanh() cuoi context_proj -- nghi pham #1 cho cau hoi "neu thong tin ve tu the CO
SAN trong dau ra encoder dong bang (M11-a Rev 3, 33.1), tai sao pipeline day du khai thac KEM
HON mot linear probe/MLP 2 lop tren chinh dau ra do?" (Muc 20.8, nay len uu tien sau khi M11-a
xac nhan tien de).

context_proj (step2_mdn_module.py, dung chung cho AutoregressiveFlowLayer va DiagonalGaussian
Head) = nn.Sequential(Linear, LayerNorm, SiLU, Linear, Tanh). Neu Tanh() bao hoa (|out| gan 1
tren phan lon phan tu) thi dao ham cuc nho tai do (d/dx tanh(x) = 1-tanh(x)^2 -> 0), nghen
duong gradient tu Geo_Loss chay nguoc vao h_target/encoder -- du thong tin CO trong h_target,
pipeline khong hoc duoc cach dung no vi gradient khong toi duoc.

Checkpoint periodic-epoch=024.ckpt, geo_head="maf" (AutoregressiveFlowLayer, context_proj
giong het DiagonalGaussianHead -- do mot trong hai dai dien cho ca hai), MOT batch valid.

HAI PHAN TACH BACH:
  1. THONG KE (forward-only, torch.no_grad()): hook tren context_proj[3] (Linear ngay TRUOC
     Tanh, bat pre-activation) va context_proj[-1] (chinh Tanh, bat tanh_out).
  2. GRADIENT (mot backward() RIENG, KHONG no_grad, requires_grad=True binh thuong -- KHAC
     cac script M truoc, vi cac script do chi forward-only nen requires_grad_(False)): forward
     lai (sach, khong hook can thiet vi doc truc tiep .grad sau backward), goi
     loss_dict["loss_geo"].backward(), roi doc grad.norm() tai:
       - context_proj[0].weight, context_proj[3].weight (hai Linear cua context_proj)
       - lop cuoi cua dynamic_encoder ma THUC SU quyet dinh gia tri residue_h/h_target --
         AttentionLayerO2TwoUpdateNodeGeneral.x2h_layers (num_x2h=1) CUA base_block[-1] (lop
         CUOI trong ModuleList shared-block cua UniTransformer; h2x_layers chi cap nhat toa do,
         KHONG anh huong h_all -> residue_h, nen khong tinh vao day). Bao cao norm TONG HOP
         (L2 tren tat ca grad cua MOI tham so trong module nay), vi khong co MOT tham so don le
         dai dien "lop cuoi" ro rang hon trong kien truc GVP/attention nay.

TIEU CHI DOC (ghi cung TRUOC khi chay, plan.md Rev 16):
    ty le |tanh_out| > 0,99  VUOT 30%  => Tanh la NGHEN THAT, phai bo TRUOC moi so sanh head.
    ty le |tanh_out| > 0,99  DUOI 5%   => LOAI gia thuyet 20.8, chuyen sang nghi pham #2
                                          (v_context.detach()).
    Trung gian => bao cao, KHONG ket luan.

Cach chay:
    python -u src/measure_tanh_saturation.py [--ckpt PATH] [--batch-size N] [--split valid]
"""
import os
import sys
import glob
import argparse
import subprocess
import pickle

import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_header(device)
    torch.manual_seed(SEED)

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Mac dinh: saved_checkpoints_flow/periodic-epoch=024.ckpt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--split", choices=["train", "valid", "test"], default="valid")
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

    # Checkpoint nay la MAF (geo_head tuong minh -- giong het cac script M truoc).
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
    # KHONG requires_grad_(False): can gradient that cho PHAN GRADIENT ben duoi.

    dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode=args.split, pos_scale=pos_scale
    )
    from datasets.pl_data import ProteinLigandDataLoader
    loader = ProteinLigandDataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"\n[*] CAU HINH LOADER: split={args.split}  shuffle=False  batch_size={args.batch_size}  "
          f"n_batches=1 (chi lay batch dau)")

    batch = next(iter(loader)).to(device)
    print(f"[*] Da lay 1 batch valid: num_graphs="
          f"{batch.protein_element_batch.max().item() + 1}")

    context_proj = model.generator.geometric_head.context_proj
    print(f"[*] context_proj = {context_proj}")

    # =========================================================================================
    # PHAN 1 -- THONG KE (forward-only, no_grad)
    # =========================================================================================
    print("\n" + "=" * 90)
    print("PHAN 1: THONG KE BAO HOA TANH (forward-only, torch.no_grad())")
    print("=" * 90)

    captured = {}

    def _pre_act_hook(module, inp, out):
        captured["pre_act"] = out.detach().clone()

    def _tanh_out_hook(module, inp, out):
        captured["tanh_out"] = out.detach().clone()

    h_pre = context_proj[3].register_forward_hook(_pre_act_hook)
    h_tanh = context_proj[-1].register_forward_hook(_tanh_out_hook)
    try:
        with torch.no_grad():
            model.forward(batch)
    finally:
        h_pre.remove()
        h_tanh.remove()

    pre_act = captured["pre_act"]
    tanh_out = captured["tanh_out"]
    print(f"[*] pre_act shape={tuple(pre_act.shape)}  tanh_out shape={tuple(tanh_out.shape)}")

    n_total = tanh_out.numel()
    frac_099 = (tanh_out.abs() > 0.99).float().mean().item()
    frac_0999 = (tanh_out.abs() > 0.999).float().mean().item()
    print(f"    ty le |tanh_out| > 0,99  = {frac_099:.4f}  ({frac_099 * 100:.2f}%)  "
          f"tren {n_total} phan tu")
    print(f"    ty le |tanh_out| > 0,999 = {frac_0999:.4f}  ({frac_0999 * 100:.2f}%)")

    pre_abs = pre_act.abs()
    print(f"    pre-activation (Linear ngay truoc Tanh, context_proj[3]):")
    print(f"        mean(|pre-activation|) = {pre_abs.mean().item():.4f}")
    print(f"        std(|pre-activation|)  = {pre_abs.std().item():.4f}")
    print(f"        max(|pre-activation|)  = {pre_abs.max().item():.4f}")

    # =========================================================================================
    # PHAN 2 -- GRADIENT (mot backward() RIENG tren Geo_Loss)
    # =========================================================================================
    print("\n" + "=" * 90)
    print("PHAN 2: GRADIENT (mot backward() RIENG tren Geo_Loss, KHONG no_grad)")
    print("=" * 90)
    model.zero_grad(set_to_none=True)
    total_loss, loss_dict = model.forward(batch)
    geo_loss = loss_dict["loss_geo"]
    print(f"[*] Geo_Loss (loss_dict['loss_geo']) = {geo_loss.item():.6f}")
    geo_loss.backward()

    def grad_norm_of(name, tensor):
        if tensor.grad is None:
            print(f"    grad.norm({name}) = KHONG CO GRAD (None) -- Geo_Loss khong chay qua day")
            return None
        gn = tensor.grad.norm().item()
        print(f"    grad.norm({name}) = {gn:.6e}")
        return gn

    print("[*] grad.norm() tai cac vi tri chi dinh (plan.md Rev 16):")
    gn_w0 = grad_norm_of("context_proj[0].weight", context_proj[0].weight)
    gn_w3 = grad_norm_of("context_proj[3].weight", context_proj[3].weight)

    # "Lop cuoi cua dynamic_encoder": AttentionLayerO2TwoUpdateNodeGeneral.x2h_layers cua
    # base_block[-1] (lop CUOI trong ModuleList shared-block) -- day la lop THUC SU quyet dinh
    # gia tri h_all -> residue_h (h2x_layers chi cap nhat toa do x, khong anh huong h_all).
    last_block = model.dynamic_encoder.transformer.base_block[-1]
    last_layer = last_block.x2h_layers[-1]
    grads = [p.grad.flatten() for p in last_layer.parameters() if p.grad is not None]
    if grads:
        gn_encoder = torch.cat(grads).norm().item()
        print(f"    grad.norm(dynamic_encoder.transformer.base_block[-1].x2h_layers[-1], "
              f"TONG HOP tat ca tham so) = {gn_encoder:.6e}")
    else:
        gn_encoder = None
        print("    grad.norm(dynamic_encoder lop cuoi) = KHONG CO GRAD (None) -- Geo_Loss khong "
              "chay nguoc toi encoder")

    # =========================================================================================
    # TIEU CHI DOC (plan.md Rev 16)
    # =========================================================================================
    print("\n" + "=" * 90)
    print("TIEU CHI DOC M12-b (plan.md 'PLAN Rev 16')")
    print("=" * 90)
    print(f"    ty le |tanh_out| > 0,99 = {frac_099 * 100:.2f}%")
    if frac_099 > 0.30:
        print("    => VUOT 30%: Tanh la NGHEN THAT. Phai bo context_proj's Tanh (hoac thay the) "
              "TRUOC moi so sanh head nao -- bat ky so sanh pocket/blind/MAF/head khac deu "
              "khong dang tin neu con nghen nay.")
    elif frac_099 < 0.05:
        print("    => DUOI 5%: LOAI gia thuyet 20.8 (Tanh KHONG la nguyen nhan). Chuyen sang "
              "nghi pham #2: v_context.detach() (kiem gradient hinh hoc co chay nguoc vao "
              "attention hub / encoder hay khong).")
    else:
        print("    => TRUNG GIAN (5%-30%): bao cao, KHONG ket luan.")


if __name__ == "__main__":
    main()
