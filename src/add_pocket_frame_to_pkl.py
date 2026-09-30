"""
add_pocket_frame_to_pkl.py -- MIGRATION: them pocket_R/pocket_t vao cac pkl DA CO (P2a-2)

CHI them truong, KHONG tinh lai nhan 7D nao. Dung LAI dung mot dinh nghia frame voi
step0_data_prep.py (import _get_pocket_frame/_load_pocket_geometry, KHONG copy code) de
dam bao pocket_R/pocket_t la CUNG MOT frame ma step0 da dung khi sinh nhan -- tinh lai
bang code rieng se mo duong cho lech dinh nghia giua nhan va input (dung loai loi da ton
hai o Phase 1/P0-3).

Chi phi: chi parse pocket PDB + Gram-Schmidt (khong BRICS/fragmentation) -- uoc ~15-25
phut cho 23.660 file, thay vi ~2,5 gio neu chay lai toan bo step0.

KHONG tu chay trong phien tao script nay -- chay thu cong khi can:
    python src/add_pocket_frame_to_pkl.py
"""
import os
import glob
import pickle

import numpy as np
from tqdm import tqdm

from step0_data_prep import SAMGDataPreprocessor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMG_ROOT = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(SAMG_ROOT, "dataset", "Apo2Mol_Dataset", "data_folder")
PROCESSED_DIR = os.path.join(SAMG_ROOT, "dataset", "processed")


def main():
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_sequence_7d.pkl")))
    print(f"[*] So file pkl tim thay: {len(files)}")

    n_done = 0
    n_skipped_legacy = 0
    n_mismatch = 0
    mismatches = []

    for f in tqdm(files, desc="Them pocket_R/pocket_t", unit="file"):
        pid = os.path.basename(f)[: -len("_sequence_7d.pkl")]
        with open(f, "rb") as fh:
            d = pickle.load(fh)

        if not isinstance(d, dict) or "frame_stable" not in d:
            n_skipped_legacy += 1
            continue

        apo_pocket_path = os.path.join(DATASET_DIR, pid, "receptor_apo_pocket10.pdb")
        pocket_coords, atom_names = SAMGDataPreprocessor._load_pocket_geometry(apo_pocket_path)
        R, t, stable, diag = SAMGDataPreprocessor._get_pocket_frame(pocket_coords, atom_names)

        # Kiem tra ham cho ket qua GIONG HET lan step0 da chay -- neu lech, KHONG duoc ghi
        # de tranh pocket_R/pocket_t mau thuan voi frame_stable da luu trong pkl.
        if bool(stable) != bool(d["frame_stable"]):
            n_mismatch += 1
            mismatches.append((pid, bool(stable), bool(d["frame_stable"])))
            continue

        d["pocket_R"] = np.asarray(R, dtype=np.float64)
        d["pocket_t"] = np.asarray(t, dtype=np.float64)
        with open(f, "wb") as fh:
            pickle.dump(d, fh)
        n_done += 1

    print(f"\n[*] Da them pocket_R/pocket_t cho {n_done}/{len(files)} file.")
    if n_skipped_legacy > 0:
        print(f"[!] Bo qua {n_skipped_legacy} file dinh dang CU (chua co 'frame_stable', truoc P1c).")

    if n_mismatch > 0:
        print(f"\n[!!!] DUNG -- {n_mismatch} file co 'stable' tinh lai KHAC voi 'frame_stable' da luu:")
        for pid, new_stable, old_stable in mismatches[:20]:
            print(f"      {pid}: recompute={new_stable}  saved={old_stable}")
        raise RuntimeError(
            f"{n_mismatch} file cho ket qua _get_pocket_frame khac voi lan step0 da chay "
            f"truoc do. Migration DA DUNG truoc khi ghi het cac file con lai -- kiem tra xem "
            f"step0_data_prep.py co bi doi giua hai lan chay khong (dinh nghia frame phai la "
            f"MOT, khong duoc lech giua nhan va input)."
        )


if __name__ == "__main__":
    main()
