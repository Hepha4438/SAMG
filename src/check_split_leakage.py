"""
check_split_leakage.py -- M9 (plan.md "PLAN Rev 8"; RESEARCH_CONTEXT Muc 26.3): kiem ro ri
du lieu giua train/valid/test TRUOC khi dung valid lam tap xac nhan cho bat ky phep do
generalization nao.

Can cu: R2(full) tren valid tot hon han tren train o nhieu chieu (theta 7 lan, phi 3 lan,
qx 25 lan -- RESEARCH_CONTEXT 26.3). Mot mo hinh khong the tot hon nhu vay tren du lieu
chua tung thay; day la dau hieu ro ri du lieu hoac --split valid lay sai tap.

KHONG GPU, KHONG model, KHONG train -- chi doc split_druglike_dict.pkl va cac pkl
*_sequence_7d.pkl de kiem frame_stable (qua chinh SAMGOptimizedDataset, import lai tu
step6_trainer, KHONG tu chia split hay tu viet lai logic loc).

Dinh nghia "ro ri":
- Muc pli_id: cung mot khoa dinh danh (thu muc complex, vd "1a4h__1__1.A__1.B") xuat hien
  o hai split khac nhau.
- Muc PDB (4 ky tu dau cua pli_id, tach bang pli_id.split("__")[0]): cung MOT protein (du
  khac ligand/chain) xuat hien o hai split. Day VAN la ro ri cho bai toan SBDD (structure-
  based drug design) vi mo hinh co the da thay cau truc protein do luc train, ke ca neu
  ligand dong-tinh-the khac nhau.

Doc tieu chi M9 (ghi cung TRUOC khi chay, plan.md "PLAN Rev 8"):
    train & valid RONG  VA  chong lan muc PDB RONG
        => KHONG ro ri; 26.3 phai co nguyen nhan khac (gia thuyet B: --split valid lay sai
           tap). Kiem tiep duong lay split (vd SAMGOptimizedDataset.__init__ split_mode).
    train & valid KHAC RONG (muc pli_id HOAC muc PDB)
        => RO RI DU LIEU. valid KHONG dung lam tap xac nhan duoc. Phai chia lai split theo
           PDB truoc moi phep do generalization, va moi so Vina/QED/NLL "valid" tu truoc
           den nay vo gia tri.

Cach chay:
    python src/check_split_leakage.py
"""
import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

SEED = 1234
SPLITS = ["train", "valid", "test"]
N_EXAMPLES = 10

# RESEARCH_CONTEXT 15.4 (Phase 1, do TRUOC khi frame_stable/frag_frame_stable ton tai):
# train 22151/23052, valid 1058/1071, test 451/478 -- la so entry CO pkl tuong ung, KHONG
# phai so da loc frame_stable. Dung de doi chieu trong phan giai thich chenh lech o item 4.
HISTORICAL_HAS_PKL = {"train": (22151, 23052), "valid": (1058, 1071), "test": (451, 478)}


def print_git_head():
    samg_root = os.path.dirname(CURRENT_DIR)
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=samg_root, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")
    print(f"SEED: {SEED}  (khong dung ngau nhien nao trong script nay, in de thong nhat quy uoc)")
    print("device: cpu  (M9 khong GPU, khong model -- chi doc index/pkl)")


def pdb_id_of(pli_id):
    return pli_id.split("__")[0]


def main():
    print_git_head()

    import step6_trainer as st  # import lai -- KHONG tu chia split, KHONG copy logic loc

    print(f"\n[*] CAU HINH (duong dan dung, giong het step6_trainer.py):")
    print(f"    SPLIT_FILE     = {st.SPLIT_FILE}")
    print(f"    DATASET_DIR    = {st.DATASET_DIR}")
    print(f"    PROCESSED_DIR  = {st.PROCESSED_DIR}")

    vocab = {"[SOS]": 0, "[UNK]": 1}
    if os.path.exists(st.VOCAB_PATH):
        import pickle
        with open(st.VOCAB_PATH, "rb") as f:
            vocab = pickle.load(f)

    # Dung CHINH SAMGOptimizedDataset.__init__ de lay target_entries (RAW, dung split_mode
    # cua step6) VA valid_indices (SAU loc frame_stable/frag_frame_stable) -- __init__ cua no
    # tu in ra breakdown (tong entry / co pkl / bi loai boi frame_stable / bi loai boi
    # frag_frame_stable / valid) cho tung split, dung cho muc 4 ben duoi.
    datasets = {}
    for split in SPLITS:
        print(f"\n[*] --- Dang doc split '{split}' qua SAMGOptimizedDataset (xem breakdown ben tren) ---")
        datasets[split] = st.SAMGOptimizedDataset(
            st.DATASET_DIR, st.PROCESSED_DIR, vocab, st.SPLIT_FILE, split_mode=split, pos_scale=1.0
        )

    # "Khoa dinh danh ma SAMGOptimizedDataset dung de ghep" = pli_id = entry[2].split("/")[0]
    # (dung CHINH cong thuc trong SAMGOptimizedDataset.__init__/__getitem__).
    pli_sets_raw = {
        split: set(e[2].split("/")[0] for e in datasets[split].target_entries) for split in SPLITS
    }
    pdb_sets_raw = {
        split: set(pdb_id_of(pid) for pid in pli_sets_raw[split]) for split in SPLITS
    }

    print("\n" + "=" * 90)
    print("1. SO LUONG pli_id (RAW, truc tiep tu split_druglike_dict.pkl, TRUOC loc frame_stable)")
    print("=" * 90)
    for split in SPLITS:
        print(f"    len({split:5}) = {len(pli_sets_raw[split])}")

    pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]

    print("\n" + "=" * 90)
    print("2. CHONG LAN MUC pli_id (khoa dinh danh day du)")
    print("=" * 90)
    overlap_pli = {}
    for a, b in pairs:
        inter = pli_sets_raw[a] & pli_sets_raw[b]
        overlap_pli[(a, b)] = inter
        print(f"    len({a} & {b}) = {len(inter)}")
        if inter:
            examples = sorted(inter)[:N_EXAMPLES]
            print(f"        vi du khoa chong lan: {examples}")

    print("\n" + "=" * 90)
    print("3. CHONG LAN MUC PDB (4 ky tu dau cua pli_id -- cung protein khac ligand VAN la ro ri cho SBDD)")
    print("=" * 90)
    overlap_pdb = {}
    for a, b in pairs:
        inter = pdb_sets_raw[a] & pdb_sets_raw[b]
        overlap_pdb[(a, b)] = inter
        print(f"    len(PDB {a} & PDB {b}) = {len(inter)}")
        if inter:
            examples = sorted(inter)[:N_EXAMPLES]
            print(f"        vi du PDB chong lan: {examples}")

    print("\n" + "=" * 90)
    print("4. SO COMPLEX TUNG SPLIT SAU LOC frame_stable (doi chieu voi so cu 15.4: "
          "valid 1058/1071, chi la so CO PKL, chua loc frame_stable)")
    print("=" * 90)
    for split in SPLITS:
        n_valid_now = len(datasets[split].valid_indices)
        n_total_raw = len(datasets[split].target_entries)
        hist = HISTORICAL_HAS_PKL.get(split)
        hist_str = f"(so cu, co pkl TRUOC loc frame_stable: {hist[0]}/{hist[1]})" if hist else ""
        print(f"    {split:5}: tong RAW={n_total_raw:6d}  SAU loc frame_stable={n_valid_now:6d}  {hist_str}")
    print("\n    Giai thich chenh lech (vd valid 764 vs 1.058 cu): 1.058 la so entry CO pkl tuong ung,")
    print("    DO TRUOC khi frame_stable/frag_frame_stable (P1a/P1b/P1c) ton tai trong code. So hien")
    print("    tai (vd 764) la SAU khi filter them frame_stable + frag_frame_stable, nen THAP HON la")
    print("    ky vong, KHONG phai dau hieu loi -- chenh lech la mot filter chat luong duoc them vao")
    print("    SAU khi con so cu duoc do, khong phai hai phep do mau thuan nhau.")

    print("\n" + "=" * 90)
    print("TIEU CHI DOC M9 (plan.md 'PLAN Rev 8') -- ap cho CA pli_id VA PDB, moi cap split")
    print("=" * 90)
    for a, b in pairs:
        n_pli = len(overlap_pli[(a, b)])
        n_pdb = len(overlap_pdb[(a, b)])
        if n_pli == 0 and n_pdb == 0:
            verdict = "KHONG RO RI (ca pli_id va PDB deu rong)"
        else:
            verdict = f"RO RI DU LIEU (pli_id chong lan={n_pli}, PDB chong lan={n_pdb})"
        print(f"    {a} & {b}: {verdict}")

    tv_pli, tv_pdb = len(overlap_pli[("train", "valid")]), len(overlap_pdb[("train", "valid")])
    print("\n    KET LUAN CHINH (train & valid, lien quan truc tiep RESEARCH_CONTEXT 26.3):")
    if tv_pli == 0 and tv_pdb == 0:
        print("    => train & valid RONG o CA HAI muc: KHONG ro ri. 26.3 (valid tot hon train bat")
        print("       thuong) phai co nguyen nhan KHAC -- gia thuyet (B): --split valid lay SAI tap.")
        print("       Buoc tiep theo: kiem duong lay split trong SAMGOptimizedDataset.__init__ (vd")
        print("       split_mode co bi anh xa nham sang 'train' do logic 'val'->'valid' khong?).")
    else:
        print("    => train & valid KHAC RONG: RO RI DU LIEU duoc XAC NHAN. valid KHONG dung lam tap")
        print("       xac nhan duoc. Phai chia lai split theo PDB TRUOC moi phep do generalization,")
        print("       va MOI so Vina/QED/NLL 'valid' tu truoc den nay deu VO GIA TRI.")


if __name__ == "__main__":
    main()
