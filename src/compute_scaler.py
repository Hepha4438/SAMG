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
    
    print("[*] Đang vét cạn 23.052 complexes để tính Exact Mean & Std...")
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        # Gom toàn bộ tọa độ thực của từng phân tử
        all_valid_7d.append(data.target_7d)

    # Nối tất cả hàng triệu nguyên tử lại thành 1 Tensor khổng lồ [Tổng số nguyên tử, 7]
    all_7d_tensor = torch.cat(all_valid_7d, dim=0)
    
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