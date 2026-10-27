"""
check_step0_tokens.py -- PLAN Rev 20; RESEARCH_CONTEXT 37-pre.2 (L4) va 37-pre.5.
Kiem HAI nghi van ve step0/step7, khong GPU, khong model, vai giay.

37-pre.5: `step0:209` goi Chem.FragmentOnBonds(mol, bonds_to_break) KHONG truyen dummyLabels. Mac dinh RDKit
gan isotope cua nguyen tu dummy = CHI SO NGUYEN TU cua dau kia lien ket bi cat. Neu dung, cung mot fragment
hoa hoc o hai phan tu khac nhau cho HAI token khac nhau.
  Tieu chi (ghi cung, plan Rev 20): U = so token duy nhat sau khi thay moi [N*] -> [*]; V = so token.
    U/V < 0,50 => XAC NHAN (phai sua step0)   |   U/V > 0,90 => BAC   |   giua => khong ket luan.

L4: step7 ghep chuoi SMILES cac fragment roi .replace("*","C"). Xem phan tu thu duoc la gi. Khong co nguong.
"""
import os
import re
import sys
import pickle
import socket
import subprocess
import collections

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

DUMMY_RE = re.compile(r"\[\d+\*\]")
U_OVER_V_LO, U_OVER_V_HI = 0.50, 0.90


def strip_dummy_index(tok):
    return DUMMY_RE.sub("[*]", tok)


def main():
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(CURRENT_DIR),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception as e:
        head = f"KHONG LAY DUOC ({e})"
    print(f"git rev-parse HEAD: {head}")
    print(f"SEED: 1234  (khong dung ngau nhien)")
    print(f"device: khong dung (script khong GPU)  hostname: {socket.gethostname()}", flush=True)

    import step6_trainer as st
    with open(st.VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    special = {"[SOS]", "[UNK]", "[PAD]", "[EOS]"}
    toks = [t for t in vocab if t not in special]

    print("\n" + "=" * 90)
    print("37-pre.5: TOKEN FRAGMENT CO MANG CHI SO NGUYEN TU KHONG?")
    print("=" * 90)
    V = len(toks)
    with_dummy = [t for t in toks if DUMMY_RE.search(t)]
    stripped = [strip_dummy_index(t) for t in toks]
    U = len(set(stripped))
    print(f"    vocab_size (ke ca token dac biet) = {len(vocab)}")
    print(f"    V = so token fragment             = {V}")
    print(f"    token co dummy dang [N*]          = {len(with_dummy)}  ({len(with_dummy) / max(V, 1) * 100:.2f}%)")
    print(f"    U = so token duy nhat sau khi [N*] -> [*] = {U}")
    ratio = U / max(V, 1)
    print(f"    U / V = {ratio:.4f}")
    print(f"    5 token dau: {toks[:5]}")
    cnt = collections.Counter(stripped)
    print(f"    5 nhom lon nhat sau khi xoa chi so (fragment hoa hoc -> so bien the token):")
    for k, n in cnt.most_common(5):
        print(f"        {k!r:40} -> {n} token")
    # phan bo so bien the
    sizes = collections.Counter(cnt.values())
    print(f"    phan bo so bien the/fragment (top 8): {dict(sorted(sizes.items())[:8])}")

    print(f"\n    TIEU CHI (plan Rev 20): U/V < {U_OVER_V_LO} => XAC NHAN ; > {U_OVER_V_HI} => BAC")
    if ratio < U_OVER_V_LO:
        print("    => XAC NHAN: token mang CHI SO NGUYEN TU cua phan tu goc. Bat buoc sua step0 (truyen")
        print("       dummyLabels) truoc moi lan huan luyen lai; tu dien fragment phai dung key da xoa chi so.")
    elif ratio > U_OVER_V_HI:
        print("    => BAC: tu vung la fragment hoa hoc thuan; step0 khong can sua cho viec nay.")
    else:
        print("    => KHONG KET LUAN: bao cao, khong hanh dong.")

    print("\n" + "=" * 90)
    print("L4 (37-pre.2): GHEP CHUOI SMILES CAC FRAGMENT -- step7 dang danh gia phan tu GI?")
    print("=" * 90)
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except Exception as e:
        print(f"    [!] khong import duoc rdkit ({e}) -- bo qua L4")
        return

    pair = toks[:2] if len(toks) >= 2 else toks
    for label, parts in [("2 token dau cua vocab", pair), ("vi du tay", ["[1*]N[6*]", "[3*]C[4*]"])]:
        joined = "".join(parts)
        m = Chem.MolFromSmiles(joined)
        rep = joined.replace("*", "C")
        m2 = Chem.MolFromSmiles(rep)
        print(f"\n    [{label}] parts = {parts}")
        print(f"        ghep           = {joined!r}  -> parse {'OK' if m else 'FAIL'}"
              + (f"  n_atom={m.GetNumAtoms()}  canon={Chem.MolToSmiles(m)!r}" if m else ""))
        print(f"        sau replace *->C = {rep!r}  -> parse {'OK' if m2 else 'FAIL'}"
              + (f"  n_atom={m2.GetNumAtoms()}  canon={Chem.MolToSmiles(m2)!r}" if m2 else ""))
        if m2 is not None:
            n_iso = sum(1 for a in m2.GetAtoms() if a.GetIsotope() != 0)
            print(f"        so nguyen tu co isotope != 0 trong phan tu 'duoc danh gia': {n_iso}/{m2.GetNumAtoms()}")


if __name__ == "__main__":
    main()
