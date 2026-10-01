"""
measure_tanh_saturation.py -- M2 (plan.md "PLAN Rev 5"): do bao hoa nn.Tanh() cuoi
context_proj. KHONG train, chi 1 batch.

Muc dich (RESEARCH_CONTEXT Muc 20.8): CA HAI head (AutoregressiveFlowLayer va
DiagonalGaussianHead) ket thuc context_proj bang nn.Tanh(). Neu pre-activation lon,
Tanh bao hoa, gradient qua no -> 0, va encoder khong nhan duoc tin hieu hoc tu dau hinh
hoc -- mot giai thich kha di cho "5/7 chieu khong hoc gi" (Muc 20.4), HOAN TOAN DOC LAP
voi cau hoi MAF-vs-diagonal (ca hai head dung chung kien truc nay nen phep so sanh
diag-vs-MAF khong the phat hien no).

LUU Y VE PHEP DO GRADIENT TAI ENCODER: SAMGLightningModule.forward() hien dang
.detach() v_context TRUOC khi noi vao geo_in (C4 trong plan.md CHUA lam). Vi vay grad
tai lop cuoi cua dynamic_encoder sau backward() tren Geo_Loss duoc KY VONG BANG 0 mot
cach CAU TRUC (vi detach, khong lien quan Tanh) -- script nay do de XAC NHAN dieu do,
khong phai di tim bang chung Tanh la nguyen nhan chan dau o day.

P3b: truyen geo_head="maf" TUONG MINH khi load checkpoint (KHONG dua vao mac dinh cua
save_hyperparameters(), vi RESEARCH_CONTEXT 20.10 chi ra chua kiem duoc config co duoc
luu trong hparams hay khong -- neu khong, load mac dinh "diag_gauss" se nap SAI
state_dict cho checkpoint MAF).

Cach chay:
    python src/measure_tanh_saturation.py [--ckpt DUONG_DAN.ckpt]
Mac dinh --ckpt la file .ckpt moi sua doi gan nhat trong saved_checkpoints_flow/.
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


def print_git_head():
    samg_root = os.path.dirname(CURRENT_DIR)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=samg_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")


def find_latest_ckpt(samg_root):
    ckpts = glob.glob(os.path.join(samg_root, "saved_checkpoints_flow", "*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def grad_norm(param):
    return param.grad.norm().item() if param.grad is not None else None


def main():
    print_git_head()

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                     help="Duong dan checkpoint MAF 25-epoch. Mac dinh: .ckpt moi nhat trong saved_checkpoints_flow/")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    import step6_trainer as st
    from omegaconf import OmegaConf

    ckpt_path = args.ckpt or find_latest_ckpt(st.SAMG_ROOT)
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

    # P3b: geo_head tuong minh "maf" -- xem docstring o dau file.
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
        ckpt_path, map_location="cpu", config=config, vocab_size=len(vocab),
        shift_factors=shift_factors, scale_factors=scale_factors, strict=False,
    )
    # train() de co BatchNorm/Dropout dung che do luc train va de .backward() co y nghia;
    # KHONG goi optimizer.step() nen trong so khong doi -- day van la "khong train".
    model.train()

    train_dataset = st.SAMGOptimizedDataset(
        st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode="train", pos_scale=pos_scale
    )
    from datasets.pl_data import ProteinLigandDataLoader
    loader = ProteinLigandDataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    context_proj = model.generator.geometric_head.context_proj
    linear0 = context_proj[0]
    linear3 = context_proj[3]
    tanh_layer = context_proj[4]

    captured = {}
    h1 = linear3.register_forward_hook(lambda m, i, o: captured.__setitem__("pre_tanh", o.detach()))
    h2 = tanh_layer.register_forward_hook(lambda m, i, o: captured.__setitem__("tanh_out", o.detach()))

    total_loss, loss_dict = model.forward(batch)

    h1.remove()
    h2.remove()

    tanh_out = captured["tanh_out"]
    pre_act = captured["pre_tanh"]

    frac_99 = (tanh_out.abs() > 0.99).float().mean().item()
    frac_999 = (tanh_out.abs() > 0.999).float().mean().item()

    print("\n" + "=" * 78)
    print("DO BAO HOA TANH (context_proj[4], dau ra cuoi cung cua context_proj)")
    print("=" * 78)
    print(f"    so phan tu: {tanh_out.numel()}")
    print(f"    ty le |tanh_out| > 0.99 : {frac_99 * 100:.2f}%")
    print(f"    ty le |tanh_out| > 0.999: {frac_999 * 100:.2f}%")
    print(f"    pre-activation (context_proj[3] output, truoc Tanh):")
    print(f"        mean(|x|)={pre_act.abs().mean().item():.4f}  "
          f"std={pre_act.std().item():.4f}  max(|x|)={pre_act.abs().max().item():.4f}")

    print("\n" + "=" * 78)
    print("GRADIENT SAU backward() TREN Geo_Loss (loss_dict['loss_geo'])")
    print("=" * 78)
    model.zero_grad(set_to_none=True)
    loss_dict["loss_geo"].backward()

    g0 = grad_norm(linear0.weight)
    g3 = grad_norm(linear3.weight)
    encoder_last_layer = model.dynamic_encoder.res_inference[-1].weight
    g_enc = grad_norm(encoder_last_layer)

    print(f"    context_proj[0].weight.grad.norm() = {g0}")
    print(f"    context_proj[3].weight.grad.norm() = {g3}")
    print(f"    dynamic_encoder.res_inference[-1].weight.grad.norm() = {g_enc}")
    print("    LUU Y: grad tai encoder duoc KY VONG = 0 vi v_context.detach() trong")
    print("    generator.forward() (C4 chua lam) -- 0 o day la do THIET KE, KHONG phai")
    print("    bang chung ve Tanh bao hoa chan dau tin hieu.")

    print("\n" + "=" * 78)
    print("TIEU CHI (plan.md M2): |tanh_out| > 0.99 VUOT 30% phan tu => Tanh la nghen that,")
    print("phai bo (thay LayerNorm hoac khong gi) TRUOC khi chay bat ky so sanh head nao.")
    print("Duoi 5% => loai gia thuyet nay, khong sua.")
    if frac_99 > 0.30:
        print(f"    => XAC NHAN NGHEN: {frac_99 * 100:.2f}% > 30%.")
    elif frac_99 < 0.05:
        print(f"    => LOAI GIA THUYET: {frac_99 * 100:.2f}% < 5%.")
    else:
        print(f"    => VUNG XAM ({frac_99 * 100:.2f}%, giua 5% va 30%) -- can xem xet them.")


if __name__ == "__main__":
    main()
