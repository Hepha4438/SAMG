"""
assemble.py -- lap rap phan tu 3D tu cac fragment da dat pose. CUA RIENG DU AN.

Vi sao khong muon `reconstruct_from_generated` cua Apo2Mol (ban dau du dinh, Muc 37.3):
  1. Server khong co repo Apo2Mol (chi co o may local), va no keo theo `openbabel`.
  2. Ham do (goc tu liGAN, GPL-2.0) **vut bo toan bo thong tin lien ket** roi de OpenBabel suy lai
     bac lien ket + thom tinh tu toa do. Day la mat mat KHONG CAN THIET o day: ta **da biet** lien
     ket trong tung fragment -- chinh SMILES cua no mang thong tin do. Chi con phai quyet dinh
     fragment noi voi fragment o dau.
  3. Lay bac lien ket tu SMILES chinh xac hon suy lai tu hinh hoc, nhat la voi vong thom va nhom tich dien.

Co so: Frag2Seq muc 3.3.1 + dinh ly surjective -- tu {fragment + pose} tai dung duy nhat phan tu 3D.
`step0:206` chi cat lien ket **don, khong trong vong, bac hai dau > 1**, nen moi lien ket LIEN-FRAGMENT
can noi lai deu la lien ket DON. `Chem.FragmentOnBonds` dat nguyen tu dummy **tai dung vi tri** cua
nguyen tu o dau kia lien ket bi cat ⇒ ghep cap dummy <-> nguyen tu that bang KHOANG CACH la dung
nguyen ban, khong phai suy doan.

Tuong ung nguyen tu: `local_coords` luu theo thu tu CANONICAL RANK cua step0, con `Chem.MolFromSmiles`
tra ve thu tu khac. Khop hai ben bang substructure match voi **bac lien ket tong quat** (do thi dau do
dung tu khoang cach cong hoa tri). Fragment doi xung co nhieu khop, nhung cac khop do la anh cua nhau
qua phep tu dang cau ⇒ cho cung mot phan tu, nen lay khop nao cung duoc.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D

# Ban kinh cong hoa tri (A) -- Cordero et al. (2008), Dalton Trans., "Covalent radii revisited".
COVALENT_R = {1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
              16: 1.05, 17: 1.02, 34: 1.20, 35: 1.20, 53: 1.39}
BOND_TOL = 1.3          # he so noi long cho tong ban kinh (quy uoc thuong dung trong perception)
LINK_TOL = 1.5          # A -- nguong ghep dummy <-> nguyen tu that giua hai fragment (quy uoc du an)


class AssemblyError(Exception):
    pass


def _plain_graph(atomic_nums, bonds):
    """Do thi TRAN: chi so hieu nguyen tu + lien ket DON. Khong dien tich hinh thuc, khong thom tinh,
    khong lap the. Dung cho CA HAI ve cua phep khop tuong ung, nen cac thuoc tinh do bi bo qua het.
    Can thiet: fragment tich dien (`*C(=O)[O-]`, `*[S+](*)C`, `[NH3+]`) truoc day khop THAT BAI vi
    substructure match mac dinh doi khop ca dien tich hinh thuc (Muc 41.2)."""
    m = Chem.RWMol()
    for z in atomic_nums:
        m.AddAtom(Chem.Atom(int(z)))
    for i, j in bonds:
        if m.GetBondBetweenAtoms(int(i), int(j)) is None:
            m.AddBond(int(i), int(j), Chem.BondType.SINGLE)
    g = m.GetMol()
    Chem.FastFindRings(g)      # dat thong tin vong ma khong sanitize (hoa tri co the bat thuong)
    return g


def _distance_bonds(coords, atomic_nums):
    n = len(atomic_nums)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            ri = COVALENT_R.get(int(atomic_nums[i]), 0.77)
            rj = COVALENT_R.get(int(atomic_nums[j]), 0.77)
            if np.linalg.norm(coords[i] - coords[j]) < BOND_TOL * (ri + rj):
                out.append((i, j))
    return out


def canon_to_smiles_map(smiles, local_coords, atomic_nums):
    """{chi so CANONICAL -> chi so trong mol parse tu SMILES}. Dung de doc `attach_self`/`attach_parent`
    (luu theo thu tu canonical cua step0) tren mol parse tu SMILES."""
    _, match = _match(smiles, local_coords, atomic_nums)
    return {int(c): i for i, c in enumerate(match)}


def _match(smiles, local_coords, atomic_nums):
    ref = Chem.MolFromSmiles(smiles, sanitize=True)
    if ref is None:
        raise AssemblyError(f"khong parse duoc SMILES fragment: {smiles!r}")
    coords = np.asarray(local_coords, dtype=np.float64)
    z = np.asarray(atomic_nums).reshape(-1)
    if ref.GetNumAtoms() != len(z):
        raise AssemblyError(f"{smiles!r}: SMILES co {ref.GetNumAtoms()} nguyen tu, template co {len(z)}")
    probe = _plain_graph(z, _distance_bonds(coords, z))
    qgraph = _plain_graph([a.GetAtomicNum() for a in ref.GetAtoms()],
                          [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in ref.GetBonds()])
    match = probe.GetSubstructMatch(qgraph, useChirality=False)
    if not match:
        raise AssemblyError(f"{smiles!r}: khong khop duoc thu tu nguyen tu giua SMILES va template")
    return ref, match


def assemble_by_channel(items, place_info=None):
    """Rev 21 (S8): lap rap bang KENH LIEN KET, khong dung hinh hoc LIEN-fragment.

    items: list theo thu tu DFS, moi phan tu la dict {smiles, local, z, parent, attach_self,
    attach_parent}. `attach_*` la chi so CANONICAL cua nguyen tu dummy (nhu step0 luu).
    place_info: list (R_g_to_w, center) de dat toa do 3D; None => giu frame fragment.

    Khac `assemble()`: o day do thi phan tu **do token quyet dinh**, nen pose sai KHONG lam mat danh
    tinh phan tu (Muc 43.7). Hinh hoc noi-fragment van dung de khop thu tu nguyen tu -- do la chinh
    xac va khong lien quan den pose.
    """
    if not items:
        raise AssemblyError("khong co fragment nao")
    mols, c2s, offs, off = [], [], [], 0
    for k, it in enumerate(items):
        m = fragment_mol_with_coords(it["smiles"], it["local"], it["z"])
        if place_info is not None:
            m = place(m, place_info[k][0], place_info[k][1])
        mols.append(m)
        c2s.append(canon_to_smiles_map(it["smiles"], it["local"], it["z"]))
        offs.append(off)
        off += m.GetNumAtoms()

    combo = mols[0]
    for m in mols[1:]:
        combo = Chem.CombineMols(combo, m)
    rw = Chem.RWMol(combo)

    def attach_atom(k, canon_idx):
        """Nguyen tu THAT ma dummy `canon_idx` cua fragment k danh dau (hang xom duy nhat cua dummy)."""
        if canon_idx is None or canon_idx < 0 or canon_idx not in c2s[k]:
            return None, None
        d_local = c2s[k][canon_idx]
        d_glob = offs[k] + d_local
        if rw.GetAtomWithIdx(d_glob).GetAtomicNum() != 0:
            return None, None
        nbrs = [a.GetIdx() for a in rw.GetAtomWithIdx(d_glob).GetNeighbors()]
        return (nbrs[0], d_glob) if len(nbrs) == 1 else (None, None)

    n_linked, n_fail = 0, 0
    for k, it in enumerate(items):
        par = it.get("parent", -1)
        if par is None or par < 0 or par >= len(items):
            continue
        a_child, _ = attach_atom(k, it.get("attach_self"))
        a_par, _ = attach_atom(par, it.get("attach_parent"))
        if a_child is None or a_par is None:
            n_fail += 1
            continue
        if rw.GetBondBetweenAtoms(a_child, a_par) is None:
            rw.AddBond(a_child, a_par, Chem.BondType.SINGLE)
            n_linked += 1

    for d in sorted([a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 0], reverse=True):
        rw.RemoveAtom(int(d))
    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise AssemblyError(f"sanitize that bai: {exc}") from exc
    diag = {"n_frag": len(items), "n_bond_lien_frag": n_linked, "n_link_that_bai": n_fail,
            "n_manh": len(Chem.GetMolFrags(mol))}
    return mol, diag


def fragment_mol_with_coords(smiles, local_coords, atomic_nums):
    """Tra ve (mol da co conformer trong frame FRAGMENT, thu tu nguyen tu theo SMILES).
    Bac lien ket va thom tinh lay tu SMILES; toa do lay tu local_coords qua phep khop tuong ung."""
    ref, match = _match(smiles, local_coords, atomic_nums)
    coords = np.asarray(local_coords, dtype=np.float64)
    out = Chem.RWMol(ref)
    conf = Chem.Conformer(out.GetNumAtoms())
    for i, pi in enumerate(match):
        x, y, zc = coords[pi]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(zc)))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out.GetMol()


def place(mol_local, R_g_to_w, center_world):
    """Dua fragment tu frame fragment sang frame the gioi: xoay quanh tam roi tinh tien."""
    conf = mol_local.GetConformer()
    P = conf.GetPositions()
    P = (np.asarray(R_g_to_w) @ (P - P.mean(axis=0)).T).T + np.asarray(center_world)
    out = Chem.Mol(mol_local)
    c = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        c.SetAtomPosition(i, Point3D(*[float(v) for v in P[i]]))
    return out


def assemble(frag_mols, link_tol=LINK_TOL):
    """Noi cac fragment DA DAT POSE tai cac nguyen tu dummy. Tra ve (mol, diag).

    Nguyen tu dummy cua fragment A nam DUNG vi tri cua nguyen tu that ma A von noi toi o fragment B.
    Nen: voi moi dummy, tim nguyen tu that gan nhat thuoc fragment KHAC; neu trong `link_tol` thi noi
    hang xom cua dummy do voi nguyen tu ay bang lien ket DON (step0 chi cat lien ket don).
    Dummy khong ghep duoc thi bo di -- hoa tri trong duoc H an bu vao.
    """
    if not frag_mols:
        raise AssemblyError("khong co fragment nao")
    combo = frag_mols[0]
    owner = [0] * frag_mols[0].GetNumAtoms()
    for k, m in enumerate(frag_mols[1:], start=1):
        combo = Chem.CombineMols(combo, m)
        owner += [k] * m.GetNumAtoms()

    rw = Chem.RWMol(combo)
    P = combo.GetConformer().GetPositions()
    owner = np.array(owner)
    is_dummy = np.array([a.GetAtomicNum() == 0 for a in rw.GetAtoms()])
    real_idx = np.where(~is_dummy)[0]

    bonds, n_linked, n_orphan = set(), 0, 0
    for d in np.where(is_dummy)[0]:
        nbrs = [a.GetIdx() for a in rw.GetAtomWithIdx(int(d)).GetNeighbors()]
        if len(nbrs) != 1:
            n_orphan += 1
            continue
        cand = real_idx[owner[real_idx] != owner[d]]
        if cand.size == 0:
            n_orphan += 1
            continue
        dist = np.linalg.norm(P[cand] - P[d], axis=1)
        j = int(cand[int(np.argmin(dist))])
        if dist.min() > link_tol:
            n_orphan += 1
            continue
        a, b = sorted((int(nbrs[0]), j))
        if (a, b) not in bonds:
            bonds.add((a, b))
            n_linked += 1

    for a, b in bonds:
        if rw.GetBondBetweenAtoms(a, b) is None:
            rw.AddBond(a, b, Chem.BondType.SINGLE)
    for d in sorted(np.where(is_dummy)[0], reverse=True):
        rw.RemoveAtom(int(d))

    mol = rw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        raise AssemblyError(f"sanitize that bai: {exc}") from exc
    if mol.GetNumAtoms() == 0:
        raise AssemblyError("phan tu rong sau khi bo dummy")
    diag = {"n_frag": len(frag_mols), "n_bond_lien_frag": n_linked, "n_dummy_mo_coi": n_orphan,
            "n_manh": len(Chem.GetMolFrags(mol))}
    return mol, diag
