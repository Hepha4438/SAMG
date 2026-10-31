# File: src/compute_scaler.py
import os
import torch
import pickle
from tqdm import tqdm
from step6_trainer import SAMGOptimizedDataset

def main():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMG_ROOT = os.path.dirname(CURRENT_DIR)
    DATASET_DIR = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "data_folder")
    PROCESSED_DIR = os.path.join(SAMG_ROOT, "dataset", "processed")
    SPLIT_FILE = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "split_druglike_dict.pkl")
    VOCAB_PATH = os.path.join(PROCESSED_DIR, "global_vocab.pkl")

    print("[*] Đang đọc Vocabulary...")
    with open(VOCAB_PATH, "rb") as f: 
        vocab = pickle.load(f)

    print("[*] Đang khởi tạo tập Train...")
    dataset = SAMGOptimizedDataset(DATASET_DIR, PROCESSED_DIR, vocab, SPLIT_FILE, split_mode="train")

    all_valid_7d = []
    n_eos_bo = 0

    print("[*] Đang vét cạn 23.052 complexes để tính Exact Mean & Std...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # S10 (Rev 22): `target_7d` nay co THEM MOT HANG o vi tri [EOS], toan 0, khong phai pose.
        # Phai loai bang `geo_mask`, neu khong moi trung binh bi pha loang theo dung ty le
        # L/(L+1): da do duoc 101791/118991 = 0.855451 tren ca 7 chieu (Muc 59).
        t7 = data.target_7d
        if hasattr(data, "geo_mask") and data.geo_mask is not None:
            keep = data.geo_mask.reshape(-1) > 0.5
            n_eos_bo += int((~keep).sum().item())
            t7 = t7[keep]
        all_valid_7d.append(t7)

    # Nối tất cả hàng triệu nguyên tử lại thành 1 Tensor khổng lồ [Tổng số nguyên tử, 7]
    all_7d_tensor = torch.cat(all_valid_7d, dim=0)
    print(f"[*] token pose dung de tinh: {all_7d_tensor.shape[0]}  "
          f"(da loai {n_eos_bo} hang [EOS])")
    if n_eos_bo == 0:
        print("[!] KHONG loai duoc hang nao: dataset chua co `geo_mask`. Neu vocab da co [EOS] thi"
              " scaler nay SAI -- K6 se bao. Chay lai step0 truoc.")
    
    # Tính toán chính xác tuyệt đối
    exact_shift = all_7d_tensor.mean(dim=0)
    exact_scale = all_7d_tensor.std(dim=0)
    
    # Đảm bảo scale không bao giờ bằng 0 (tránh chia cho 0)
    exact_scale = torch.clamp(exact_scale, min=1e-4)

    scaler_path = os.path.join(PROCESSED_DIR, "scaler_7d.pt")
    torch.save({"shift": exact_shift, "scale": exact_scale}, scaler_path)
    
    print("\n[+] Đã tính toán xong và lưu Scaler tại:", scaler_path)
    print("  -> Exact Mean (Shift):", exact_shift.tolist())
    print("  -> Exact Std (Scale): ", exact_scale.tolist())

if __name__ == "__main__":
    main()