import pickle
with open("/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG/dataset/Apo2Mol_Dataset/split_druglike_dict.pkl", "rb") as f:
    splits = pickle.load(f)
    for k, v in splits.items():
        print(f"Tập {k}: {len(v)} mẫu")