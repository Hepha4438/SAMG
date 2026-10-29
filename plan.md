# PLAN.md — Rev 4 (2026-09-29)

> Doc `RESEARCH_CONTEXT.md` Muc 15 (fingerprint local/server) va Muc 16 (rang buoc thiet ke)
> truoc khi bat dau. Backup Rev 3: `plan_20260929_REV3.md`.
>
> **Pham vi: Phase 0 + Phase 1.** Muc tieu: duong **target-only sinh 3D** chay duoc.
> apo→holo va anti-target HOAN theo yeu cau nguoi dung.
>
> **Quy tac xuat xu:** local va server trung code (cung HEAD `29f0cd4`, md5 khop) nhung
> **lech du lieu** (server 23.660 pkl / vocab 19.033; local 15.817 / 15.626). Moi con so
> dataset phai do TREN SERVER. Khong dung so local lam can cu.

---

# PHASE 0 — ba sua doi mot dong, khong phu thuoc du lieu
Can cu: code local == server byte-identical (Muc 15.1). Lam duoc ngay.

## P0a. Revert P0-5 (canonicalize quaternion)
**File:** `src/step6_trainer.py`, trong `SAMGOptimizedDataset.__getitem__` (~dong 160-161)

**Xoa** hai dong:
```python
neg = data.target_7d[:, 6] < 0
data.target_7d[neg, 3:7] *= -1.0
```
**Ly do (Muc 15.6, do tren server):** `scipy.Rotation.as_quat()` DA canonicalize san theo
quy tac "thanh phan lon nhat mang dau duong" — do duoc **1,0** (100%) tren server. P0-5 lat
**34,04%** hang va **pha** canonicalization do xuong ~0,66. Nang hon: no lat theo dau cua
`target_7d[:, 6]`, ma theo Muc 6.3 index 6 la **`qz`**, khong phai `qw` — nen no khong lat
theo thanh phan nao co y nghia. Bien cat cua P0-5 nam o `|qz| ~ 0` (8,46% khoi luong trong
`|.|<0.05`), trong khi bien cua scipy nam o `|q_max| ~ 0,5-0,7` (chi 0,20% duoi 0,55) —
dac hon khoang **40 lan**.

**Acceptance:** sau khi sua, chay lai script G (`VERIFY_REV3_SCRIPTS.md`) tren server:
`argmax|q| duong` phai = **1,0** sau khi di qua `__getitem__`.

## P0b. Sua `DIM_NAMES`
**File:** `src/step2_mdn_module.py:21`
```python
DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]   # truoc: qx,qy,qz,qw
```
**Ly do (Muc 6.3):** `step0_data_prep.py:121` GHI theo thu tu `[d, theta, phi, qw, qx, qy, qz]`
va `step7_evaluation.py:344` DOC dung thu tu do. Chi `step2` la sai. Hau qua: moi nhan
per-dim trong `flow_scale_trace.csv` o cac chieu 3-6 dang bi gan sai ten.

**Acceptance:** khong doi gia tri so nao; chi doi ten cot. Ghi mot dong vao
`logs/flow_scale_trace.csv` header comment rang cac run TRUOC Rev 4 co nhan cot sai.

## P0c. Pin dependency
**File:** `requirements.txt`
```
pyro-ppl==1.9.1
```
```
kornia==0.8.2
```
Ban dang chay tren server la **0.8.2** (da xac nhan). `quaternion_to_rotation_matrix` mac
dinh **WXYZ** tu 0.6 nhung **XYZW** o 0.5.x ⇒ `kornia>=0.6.0` khong pin la rui ro sai im
lang (Muc 15 / N4). Pin dung 0.8.2.

---

# PHASE 1 — dinh nghia lai nhan 7D + tai sinh du lieu (MOT luot)
Can cu: Muc 15.10 (san nhieu quay **35,1 deg**, cung PDB **9,16 deg**, doi chung `|d|`
0,024 A), Muc 16.1 (rang buoc encoder bat bien), Muc 16.5 (frame hoc on dinh ~87%).

**Chon phuong an (A)** — canonical hoa frame hoc — theo Muc 16.5/16.6. **KHONG** chon (B)
o phase nay.

**Lam tat ca cac muc duoi day trong MOT lan tai sinh du lieu.** Khong chay `step0` hai lan.

## P1a. Frame cap fragment phai la ham cua hoa hoc
**File:** `src/step0_data_prep.py`, cho goi `self._get_local_frame(f["coords"])`

Hien tai frame dung tu 3 nguyen tu dau **theo thu tu index RDKit**, nen nhan quay khong phai
ham cua phan tu (do duoc: cung phan tu cung conformer lech **median 35,1 deg**, xem 15.10).

Sua: truoc khi dung frame, sap nguyen tu cua fragment theo **`Chem.CanonicalRankAtoms(frag,
breakTies=True)`** roi moi lay 3 diem khong cong tuyen dau tien. Canonical rank la bat bien
hoa hoc ⇒ loai bo hoan toan nguon nhieu nay.

Neu fragment co < 3 diem khong cong tuyen: **danh dau** `frag_frame_stable = False` thay vi
roi am tham vao `eye(3)`.

## P1b. Frame hoc = Gram-Schmidt tu hai huong hoa hoc (BAN CUOI — bo hoan toan PCA)
**File:** `src/step0_data_prep.py` — method moi `_get_pocket_frame`

**Lich su ba lan sua truoc, va vi sao bo PCA (16.8-16.10):** hai lan dau toi vá quy uoc dau
cho PCA (momen bac ba, roi them tang 2 tham chieu N/O, roi them phep thu jitter). Lan do
cuoi cho thay chinh **co che hai tang** la nguyen nhan: trong nhom STABLE co goc lech >30 deg,
**94,89%** la cac cap ma hai ban sao dung TANG KHAC NHAU cho it nhat mot truc (so voi 1,75%
o nhom <=2 deg). Nguong cung giua hai quy uoc tao ra mot diem gian doan; mot khac biet cau
truc nho day hai ban sao ve hai phia nguong va sinh ra cu lat 180 deg. Khong the vá bang
cach chinh nguong — phai bo cau truc co nguong.

```python
def _get_pocket_frame(self, pocket_coords, atom_names, tau=0.30):
    """Frame canonical cua hoc. Tra ve (R, t, stable, diag).
    R: cot la 3 truc. t: tam hoc. Khong PCA, khong tri rieng, khong quy uoc dau."""
```
Cai dat:
1. `BB = {"N","CA","C","O"}`; `bb` = nguyen tu co ten trong BB; `sc` = cac nguyen tu con lai.
2. `ref1 = centroid(ten=="N") - centroid(ten=="O")`; `n1 = |ref1|`.
   Neu thieu N hoac O, hoac `n1 < tau` ⇒ `stable=False`.
3. `e1 = ref1 / n1`.
4. `ref2 = centroid(sc) - centroid(bb)`; `ref2p = ref2 - (ref2 . e1) e1`; `n2 = |ref2p|`.
   Neu thieu sc/bb, hoac `n2 < tau` (hai huong gan song song) ⇒ `stable=False`.
5. `e2 = ref2p / n2`; `e3 = cross(e1, e2)`; `R = stack([e1,e2,e3], axis=1)`.
6. `t = pocket_coords.mean(0)`.
7. `diag = {"n1": n1, "n2": n2, "missing": <chuoi mo ta neu thieu nhom nguyen tu>}`.

**KHONG** dung PCA. **KHONG** dung momen bac ba. **KHONG** dung phep thu jitter (khong con
thu tu tri rieng nao de dao, nen jitter thanh khong can thiet). `tau` la tham so, mac dinh
0.30 A.

**Tinh chat (da do, 16.10):** bat bien SE(3) (hieu cac centroid bien doi nhu vector,
Gram-Schmidt giao hoan voi phep quay); tat dinh; lien tuc theo toa do nguyen tu tru dung
mot cho `ref1 || ref2`, ma buoc 4 kiem truc tiep.

**So do head-to-head (16.10):**
```
                            n      median   <2deg   >30deg
PCA 2 tang (STABLE)      28.519    0,002    87,39%   3,16%
CHEM Gram-Schmidt        24.964    0,013    83,73%   0,29%   <- giam 11 lan so ca lat
RMSD<0,05: PCA           21.113    0,001    99,98%   0,00%
RMSD<0,05: CHEM          18.684    0,009    99,98%   0,00%
CHEM ty le hoc bi loai: 15,01%
```
CHEM co median hoi cao hon (0,013 vs 0,002 deg — ca hai bang khong ve mat thuc dung) va
`<2deg` hoi thap hon tren TAT CA cap, vi frame CHEM bam theo dich chuyen centroid nen phan
anh **that** cac khac biet cau truc nho. Doi lai, `>30deg` giam tu 3,16% xuong **0,29%**:
no gan nhu khong bao gio lat. Do la danh doi dung huong — lech vai do theo cau truc that la
vo hai cho hoc may, con lat 180 deg thi khong hoc duoc.

**Ghi log ty le loai theo nguyen nhan** (`n1 < tau` / `n2 < tau` / thieu nhom nguyen tu) de
biet `tau = 0.30` co phai la rang buoc chi phoi hay khong. Chua tuning o phase nay.

## P1c. Bieu dien pose fragment trong FRAME HOC, khong phai frame noi phan tu
**File:** `src/step0_data_prep.py`, vong `for f in frag_data`

Thay `R_m_to_w, t_m_to_w = self._get_local_frame(centers)` bang frame hoc tu P1b
(goc = tam hoc). Giu nguyen dinh dang 7D `[d, theta, phi, qw, qx, qy, qz]` va **giu nguyen
thu tu nay** (khop `step0`/`step7`; xem P0b).

Ba hau qua, tat ca deu la muc tieu:
* **Muc 6.1/13.3 bien mat:** `pos 0/1/2` thoi la hang so ⇒ **48,26% nhan (do tren server)**
  tro thanh du lieu thuc thay vi entropy 0.
* **Muc 6.5 bien mat:** nhan tro thanh pose tuong doi voi hoc ⇒ Step 7 dat duoc phan tu truc
  tiep, khong phai tu che `pocket_center + pred_coords.mean()`.
* **Muc 16.1 duoc ton trong:** muc tieu tro thanh bat bien SE(3) ⇒ encoder bat bien co the
  hoc duoc.

Giu `frag_data.sort(key=smiles)` o buoc nay (no la thu tu canonical). Nhung ghi vao
`RESEARCH_CONTEXT` rang N1 (phan ra tu hoi quy theo thu tu alphabet, khong theo lien ket
hoa hoc) van la khiem khuyet chua sua, se xem lai o phase sau.

## P1d. Ghep du lieu theo `pli_id`
**File:** `src/step6_trainer.py`, `SAMGOptimizedDataset.__init__` va `__getitem__`

Hien tai: `entry = self.target_entries[real_idx]` va `pkl_path = self.available_pkls[real_idx]`
— hai danh sach khong lien quan. Do tren server: **mismatch 100,000% ca ba split** (Muc 15.3).

Sua:
1. Dung `dict` `pli_id -> pkl_path` tu `available_pkls`.
2. `self.valid_indices` chi giu cac entry co `pli_id` ton tai trong dict do. Do tren server
   (Muc 15.4): train **22151/23052**, valid **1058/1071**, test **451/478** — nen KHONG can
   chay lai step 0 cho valid/test (day la sua lai mot nhan dinh sai o luot truoc).
3. Them loc theo `frame_stable` (P1b) va `frag_frame_stable` (P1a).
4. **`assert len(self.valid_indices) > 0`** kem thong bao ro rang, cho moi split.
5. Bo vong lap `for attempt in range(max_attempts)` truot index khi doc SDF loi — no lam
   `entry` va `pkl` lech nhau lan nua. Thay bang: bo qua index loi khi dung `valid_indices`.

## P1e. Tai sinh du lieu + scaler
1. Chay lai `step0_data_prep.py` cho **ca ba split**.
2. Backup `dataset/processed` cu truoc khi ghi de.
3. Chay lai `src/compute_scaler.py`. Scaler cu **khong con dung** (nhan doi dinh nghia).
4. Xoa het `*_protein_transforms.pt` cache neu dinh nghia frame co anh huong (kiem truoc).

---

# CONG CHAN PHASE 1 — ghi cung TRUOC khi chay, do TREN SERVER
Khong chuyen sang Phase 2 neu chua dat ca bon.

| # | Phep do | Hien tai | Yeu cau |
|---|---|---|---|
| G1 | 15.10 tren nhan moi, cap **cung PDB id** | median **9,157 deg** | **median < 2 deg** |
| G2 | 16.5 tren tap DA LOC theo `frame_stable` | `<2deg` 87,44% | **`<2deg` >= 99%** |
| G3 | Doi chung `\|d\|` cua 15.10 | median 0,0239 A | giu **< 0,05 A** (chung minh khong lam hong conformer) |
| G4 | Ghep cap | mismatch 100% | **mismatch 0%** ca ba split, `len(dataset) > 0` ca ba |

Bao cao them (khong phai cong chan, de theo doi): ty le complex bi loai boi `frame_stable`
va `frag_frame_stable`; ty le token o `pos 0,1,2` (hien 48,26% — **phai giam manh**, vi
frame khong con dinh nghia bang ba fragment dau).

---

# PHASE 2 — dieu kien hoa theo hoc (Rev 4, 2026-09-30)
Phase 1 DA DONG: ca bon cong chan dat (RESEARCH_CONTEXT 18.2, 18.5). Dataset 17.200 mau train.

**Nguyen tac thu tu:** B5 va B6 deu doi input cua encoder nen deu doi hoi huan luyen lai ⇒
lam CUNG MOT LUOT. C3 re va doc lap. C4 phai sau B5. B7 chi can cho danh gia, de sau.

## P2a. B5 — `h_target` phai mang HINH HOC, khong chi feature bat bien
**File:** `src/step6_trainer.py` (~dong 225), `src/step3_attention_hub.py`

Hien tai: `h_target = scatter_mean(residue_h, prot_trans_batch, dim=0, dim_size=num_graphs)[:, :H].unsqueeze(1)`
⇒ shape `[B, 1, H]` ⇒ cross-attention chay softmax tren dung **2 key** (null-sink + 1).

**Day KHONG phai viec phu ma la dieu kien de nhan quay hoc duoc** (Muc 16.1): encoder sinh
feature **bat bien** SE(3); mot bag-of-residue-descriptors da mean-pool khong mang thong tin
huong nao. Nhan quay gio da sach (G1 = 0,0155 deg) nhung neu input khong co hinh hoc thi no
van khong hoc duoc — lan nay vi thieu thong tin dau vao, khong vi nhan nhieu.

Sua:
1. **Bo mean-pool.** `h_target` thanh chuoi key theo tung residue: `[B, R_max, H]` + mask
   `[B, R_max]` cho padding. Dung `prot_trans_batch` de tach residue theo graph roi pad.
2. **Noi toa do residue trong FRAME HOC CANONICAL.** Toa do residue = `scatter_mean(protein_pos,
   protein_atom_to_aa_group)` (tam residue — mot dong, khong can loc theo ten nguyen tu).
   Bien doi sang frame hoc `p_local = R_p^T (p - t_p)`, roi chia ca ba thanh phan cho
   `scale[0]` (= 2,4783; ca ba deu la do dai trong cung frame nen dung mot he so la dung).
   Khong tru gi vi goc frame da la tam hoc. Noi 3 so nay vao feature:
   `key_feat = concat([residue_h_invariant, p_local_normalized])` -> `[B, R_max, H+3]`.
   Chieu them vao la thu duy nhat cho phep mo hinh suy ra huong.
3. `R_p, t_p` phai la **cung mot frame** step0 dung khi sinh nhan. **KHONG tinh lai trong
   step6 bang code rieng** — tinh lai bang code khac la mo duong cho lech dinh nghia giua
   nhan va input, dung loai loi da ton hai ngay.
   **Cach lam (re, KHONG phai chay lai step0 2,5 gio):** viet mot script migration
   `src/add_pocket_frame_to_pkl.py` chi them hai truong vao cac pkl DA CO:
   * doc tung `dataset/processed/*_sequence_7d.pkl`
   * **import** `_get_pocket_frame` va `_load_pocket_geometry` tu `step0_data_prep`
     (KHONG copy code — phai la cung mot ham, cung file pocket apo, moi ra cung ket qua)
   * ghi them `pocket_R` (3x3) va `pocket_t` (3,) vao dict, luu lai
   Chi phi: chi parse pocket PDB + Gram-Schmidt, khong co BRICS/fragmentation ⇒ uoc ~15-25
   phut cho 23.660 file, thay vi 2,5 gio.
   Sau do `__getitem__` **doc** `pocket_R`/`pocket_t` tu pkl, khong tinh gi.
4. `step3_attention_hub.py`: nhan `h_target` `[B, R, H+3]` + mask; chieu `H+3 -> H` truoc khi
   lam key/value; ap mask vao softmax (`masked_fill(-inf)` truoc `F.softmax`).

**Acceptance:** in ra `h_target.shape` va `attn_weights.shape` cho mot batch — so key phai
la `R_max + 1`, khong phai 2.

## P2b. B6 — bo ligand ground-truth khoi encoder, CA train VA eval
**File:** `src/step6_trainer.py`, `src/step7_evaluation.py`, `src/step1_protein_encoder.py`

`dynamic_encoder(..., ligand_pos=data_batch.ligand_pos, ...)` va `step7:280` truyen toa do
ligand THAT vao encoder. `UniTransformer` gop protein+ligand thanh mot graph, nen vector dieu
kien **chua dap an**. Moi so Vina/QED den nay vo gia tri.

Sua:
1. **Truoc khi code: kiem** `UniTransformer` co chay duoc voi **khong nguyen tu ligand nao**
   (`mask_ligand` toan False, tensor ligand rong) hay khong. Viet mot test nho, in ket qua.
2. Neu duoc: truyen ligand rong o ca train va eval.
3. Neu khong: truyen **mot** dummy atom tai tam hoc voi mot element id rieng (them 1 vao
   `nn.Embedding(100, ...)`), va ghi ro lua chon nay vao docstring.
4. Sua ca `step6` (train) va `step7:280` (eval) **cung luc**. Sua mot phia se tao lech phan
   phoi train/eval.

**Acceptance:** `grep -n "ligand_pos" src/step6_trainer.py src/step7_evaluation.py` khong con
cho nao truyen `data_batch.ligand_pos`/`batch.ligand_pos` vao encoder.

## P2c. C3 — noi token embedding cua vi tri dang sinh vao geometric head
**File:** `src/step4_ligand_generator.py`

`v_context[t]` tinh nhan qua tu token `< t`, nhung 7D tai vi tri `t` la tu the cua fragment
`t`. Flow dang bi hoi "dat cai nay o dau" ma khong duoc cho biet "cai nay la gi".

Sua: khi teacher forcing, noi `token_embedding(target_ids[:, t])` vao context cua geometric
head (`torch.cat` roi chieu ve `hidden_dim`). Khi sampling, dung token vua duoc lay mau.
Giu `v_context.detach()` o buoc nay (C4 lam sau).

## P2d. C4 — bo `v_context.detach()` — CHI sau khi P2a xong
Khong lam truoc P2a. Voi `h_target` con la mot vector mean-pool, bo detach chi cho gradient
hinh hoc chay nguoc vao mot bieu dien da mat thong tin khong gian (Muc 8.4).

## P2e. B7 — viet lai Step 7 — de sau lan train dau
Dat tung fragment theo pose du doan trong frame hoc; bo `ETKDG` khoi duong dat pose (van
dung duoc cho hinh hoc **noi** fragment). Chi lam sau khi lan train dau cho thay co tin hieu.

---

# CONG CHAN PHASE 2 — ~~ghi cung TRUOC khi chay~~ [THAY THE boi Rev 5, xem cuoi file]

> **[HET HIEU LUC 2026-10-01]** Cong chan nay **khong doc duoc** vi hai ly do do o Audit Rev 5
> (RESEARCH_CONTEXT Muc 20): (1) khong co seed nen permutation cua ARN ngau nhien moi run, bien
> `logscale_mean` cua tung chieu thanh dai luong khong so sanh duoc giua cac run (20.3);
> (2) nguong `-0,3` khong phan biet duoc "hoc duoc tu hoc" voi "chieu do thoai hoa trong du
> lieu" (20.5). Dung cong chan Rev 5 o cuoi file.

**Tin hieu don-run (re, lam truoc):** sau P2a+P2b+C3, train 15-20 epoch va xem
`logscale_mean` cua bon chieu quay trong `logs/flow_scale_trace.csv`.
* Truoc Phase 1 (nhan nhieu, h_target mean-pool): `|logscale| <= 0,28` — **khong hoc gi**.
* Yeu cau: **`logscale_mean` cua it nhat 3 trong 4 chieu quay xuong duoi -0,3** (sigma < 0,74).
  Do la bang chung dau tien rang mo hinh trich xuat duoc thong tin huong tu hoc.

**Doi chung pocket-blind (dat, chi chay neu tin hieu don-run DAT):** chay lai voi `h_target`
bi zero hoa. Ban co dieu kien phai tot hon **ro ret** o ca `Token_Loss` va `logscale_mean`
bon chieu quay. Day la phep thu duy nhat phan biet "hoc duoc hoc" voi "hoc thong ke chuoi"
(Muc 14.2/15.7 — nguong tuyet doi cua H3 da bi vo hieu).

**Moc so sanh cu, KHONG dung truc tiep:** `logs/archive/CONTROL_pocket_blind_22ep_*`. Vocab
va dinh nghia nhan da doi nen `Token_Loss` khong so sanh duoc; chi dung `logscale_mean` cua
bon chieu quay lam moc dinh tinh ("~0 = khong hoc gi").

---

# HOAN — ghi de khong quen
* Quyet dinh Flow / MDN / Robust Regressor: hoan. Muc 11 da chi ra uu the cua MAF 1 lop la
  gia dinh chua kiem chung.
* 6D continuous rotation: khong con cap thiet (nhan da sach).
* Tuning `tau` (hien 0,30 A) de thu hoi phan trong 4.951 complex bi loai boi `frame_stable`.
* N1 — token sap theo thu tu alphabet cua SMILES ⇒ phan ra tu hoi quy khong co nghia hoa hoc.
* N2 — `updated_protein_pos` (bien dang apo->holo) duoc supervise roi bi vut.
* anti-target.

---
---

# PLAN Rev 5 (2026-10-01) — chan dinh kien truoc khi chay tiep

Can cu: RESEARCH_CONTEXT **Muc 20** (Audit Rev 5). Muc 19.2/19.3/19.4 da bi rut lai.
**Khong chay training nao nua cho den khi xong M1 va P3a.** Ly do: mot run 25 epoch mat
~19 gio va hien tai ket qua cua no **khong doc duoc** (khong co seed, chua loai thoai hoa
du lieu, logscale lan padding). Chay lai luc nay la dot 19 gio de lay mot con so vo nghia.

Thu tu bat buoc: **M1 -> P3a -> M2 -> roi moi train.**

---

## M1. Thong ke marginal 7 chieu — khong GPU, khong train, ~2 phut
**Muc dich:** loai hoac xac nhan gia thuyet (b) cua Muc 20.5 — `qy` thoai hoa trong du lieu.
Day la giai thich tam thuong nhat cho 76,8% "tien bo" cua mo hinh va **chua he duoc loai**.

**File moi:** `src/measure_marginal_7d.py`

Doc tap train y het `SAMGOptimizedDataset` dung (cung `pkl_by_id`, cung loc `frame_stable`,
cung `scaler_7d.pt`), gom toan bo `target_7d` thanh mot tensor `[N, 7]`, roi in cho **tung**
chieu, o **ca hai** khong gian (raw truoc chuan hoa, va sau chuan hoa):
* `mean, std, min, max`
* `skew, kurtosis`
* phan vi `1, 5, 25, 50, 75, 95, 99`
* histogram 40 bin (in dang text)
* **ty le khoi luong trong bin lon nhat**, va trong 3 bin lon nhat
* so gia tri **duy nhat** (`torch.unique`) — neu `qy` co it gia tri duy nhat thi no roi rac,
  khong lien tuc, va toan bo mo hinh mat do lien tuc dang sai gia thiet cho chieu do

In them `scaler_7d.pt`: `mean[i]`, `std[i]` cho ca 7 chieu, co nhan ten theo
`DIM_NAMES = ["d","theta","phi","qw","qx","qy","qz"]`.

**Tieu chi doc (ghi cung TRUOC khi chay):**
* Neu `qy` sau chuan hoa co **>= 50% khoi luong trong 3 bin** (trong 40 bin), hoac
  `|kurtosis| > 10`, hoac `std_raw(qy) < 0,1 x std_raw(qw)` ⇒ **(b) XAC NHAN**: `sigma = 0,257`
  la hien vat du lieu, khong phai thong tin tu hoc. Muc 19.4 sai hoan toan, va viec can lam la
  **bieu dien quaternion**, khong phai doi head.
* Neu `qy` co phan phoi tuong tu `qw/qx/qz` (cung bac kurtosis, cung do tan) ⇒ **(b) BI LOAI**,
  con lai (a) va (c), va P3a se phan xu.

---

## P3a. Tai lap duoc — BLOCKING, 3 sua doi nho
Day la loi phuong phap luan nang nhat con lai (Muc 20.3). **Khong so sanh run nao voi run nao
duoc cho den khi xong muc nay.**

### P3a-1. Co dinh permutation cua ARN
**File:** `src/step2_mdn_module.py`, trong `AutoregressiveFlowLayer.__init__`
```python
self.arn = ConditionalAutoRegressiveNN(out_dim, hidden_dim, [hidden_dim, hidden_dim])
```
Sua thanh truyen **permutation tuong minh**:
```python
self.arn = ConditionalAutoRegressiveNN(
    out_dim, hidden_dim, [hidden_dim, hidden_dim],
    permutation=torch.arange(out_dim),
)
```
`torch.arange` ⇒ thu tu tu hoi quy **chinh la** `DIM_NAMES`: `d, theta, phi, qw, qx, qy, qz`.
Chon thu tu nay **khong** phai vi no toi uu, ma vi no **xac dinh va doc duoc**: `d/theta/phi`
(vi tri) truoc `qw/qx/qy/qz` (huong) la thu tu nhan qua co nghia vat ly, va bon chieu quaternion
nam lien ke nhau nen co the doc truc tiep "chieu quaternion thu k thay k-1 chieu truoc".
Ghi lua chon nay vao docstring kem ly do.

### P3a-2. Seed toan cuc
**File:** `src/step6_trainer.py`, ngay dau `main()` / truoc khi khoi tao module
```python
import pytorch_lightning as pl
pl.seed_everything(SEED, workers=True)
```
voi `SEED` lay tu `config["seed"]` (them `"seed": 1234` vao config dict ~dong 383).
Them `deterministic=True` vao `pl.Trainer(...)` neu khong lam cham qua 20%; neu cham hon, bo
`deterministic` nhung **giu** `seed_everything` va ghi ro vao log.

In ra dau moi run: `SEED`, `git rev-parse HEAD`, va `permutation` thuc te cua ARN.

### P3a-3. Mask padding cho `_ls_sum` / `_pin_sum`
**File:** `src/step2_mdn_module.py`, **ca hai** head.
Commit `bcb3e99` chi sua reservoir. `_ls_sum`/`_pin_sum` van cong moi hang ke ca padding, trong
khi `Geo_Loss_Unclamped` (step5:44-45) thi mask ⇒ hai tong the khac nhau (Muc 20.6).

Trong `DiagonalGaussianHead.forward`, ap `pad_mask` cho **ca ba** bien:
```python
if pad_mask is not None:
    valid = pad_mask.detach().to(torch.bool).flatten()
    ls_v = log_sigma.detach()[valid]
else:
    ls_v = log_sigma.detach()
self._ls_sum  += ls_v.sum(0).cpu()
self._pin_sum += (ls_v <= self._ls_floor + 1e-3).float().sum(0).cpu()
self._ls_count += ls_v.shape[0]
```
Voi `AutoregressiveFlowLayer`, hook tren `self.arn` **khong** nhin thay `pad_mask`. Cach re nhat:
luu `self._cur_pad_mask = pad_mask` trong `forward` **truoc** khi goi `log_prob`, va hook doc no
(dat lai `None` o cuoi `forward`). Ghi ro trong comment rang day la coupling tam thoi.

**Acceptance P3a:** chay `step6` **hai lan** voi `max_epochs=1`, cung seed. Hai file
`logs/flow_scale_trace.csv` phai **giong nhau tung chu so** o ca 14 cot logscale/pinfrac.
Neu khac ⇒ con nguon ngau nhien chua bi bat, phai tim tiep truoc khi di xa hon.

---

## M2. Do bao hoa Tanh — khong train, 1 batch
**Muc dich:** Muc 20.8. Ca hai head ket thuc `context_proj` bang `nn.Tanh()`. Neu bao hoa thi
encoder khong nhan duoc gradient tu dau hinh hoc, va do la giai thich cho "5/7 chieu khong hoc
gi" (20.4) **doc lap** voi cau hoi MAF-vs-diagonal. Phep so sanh diag-vs-MAF **khong the** phat
hien no vi ca hai dung chung kien truc nay.

**File moi:** `src/measure_tanh_saturation.py`
Load checkpoint 25-epoch MAF, lay **mot** batch train, forward, va in:
* ty le phan tu `|tanh_out| > 0,99` va `> 0,999`
* `mean/std/max(|pre_activation|)` cua `context_proj[3]` (Linear truoc Tanh)
* sau mot `backward()` tren `Geo_Loss`: `grad.norm()` tai `context_proj[0].weight` va
  `context_proj[3].weight`, **va** tai lop cuoi cua encoder — de thay tin hieu co chay nguoc
  vao encoder hay khong

**Tieu chi:** neu `|tanh_out| > 0,99` **vuot 30%** phan tu ⇒ Tanh la nghen that, phai bo (thay
bang `LayerNorm` hoac khong gi) **truoc** khi chay bat ky so sanh head nao. Neu duoi 5% ⇒ loai
gia thuyet nay, khong sua.

---

## P3b. Kiem `step7_evaluation.py` khoi phuc dung head
Muc 20.10. `step7:264` chi co `load_from_checkpoint`. `geo_head` nam trong config step6
(dong 232 doc, dong 383 mac dinh `"diag_gauss"`).
Kiem: `grep -n "save_hyperparameters" src/step6_trainer.py` va xem `config` co nam trong
hparams duoc luu khong. Neu khong ⇒ step7 se dung mac dinh `diag_gauss` va **load sai state_dict**
cho checkpoint MAF (loi im lang hoac `unexpected keys`). Sua bang cach truyen `geo_head` tuong
minh khi load, hoac dam bao `save_hyperparameters()` luu ca config.
Khong blocking cho training, nhung blocking cho moi so Vina/QED.

---

# CONG CHAN Rev 5 — thay the cong chan Phase 2 cu

**Khong con cong chan dang "logscale_mean cua >= 3/4 chieu quay xuong duoi -0,3".** Nguong tuyet
doi do khong phan biet duoc thong tin tu hoc voi thoai hoa du lieu, va khong so sanh duoc giua
cac run khi permutation con ngau nhien.

Thay bang **mot phep thu doi chung duy nhat, chay sau khi M1/P3a/M2 xong**:

### G5. Doi chung pocket-shuffled (khong phai pocket-blind)
Chay **ba** nhanh, **cung seed**, cung so epoch, chi khac mot thu:
1. **full** — nhu binh thuong
2. **shuffled** — `h_target` bi **xao tron theo chieu batch** (ghep sai hoc voi ligand), giu
   nguyen phan phoi input
3. **zeroed** — `h_target` bi zero hoa (doi chung cu, giu de doi chieu)

Dung **shuffled** lam doi chung chinh, khong dung **zeroed**: zero hoa dua input ra ngoai
manifold huan luyen nen mo hinh co the hong vi ly do khac, con xao tron giu **y nguyen** phan phoi
bien va chi pha **su ghep doi**. Day la cach duy nhat tach "thong tin tu hoc nay" khoi "thong ke
chuoi + thong ke hoc trung binh".

**Tieu chi DAT (ghi cung truoc khi chay):**
* `Geo_Loss_Unclamped(full)` phai thap hon `Geo_Loss_Unclamped(shuffled)` it nhat **0,3 nat**
  o epoch cuoi. Duoi 0,1 nat ⇒ hoc **khong** mang thong tin ve pose, va moi thu con lai cua
  Phase 2 la sua sai cho.
* Rieng cho tung chieu: `logscale_mean_i(full)` so voi `logscale_mean_i(shuffled)`. Chieu nao
  **khong** chenh thi chieu do khong hoc gi tu hoc, bat ke gia tri tuyet doi cua no.
  Day la cach doc dung chieu `qy`: neu `logscale_mean_qy` **giong nhau** o full va shuffled thi
  `sigma = 0,257` khong den tu hoc — ket thuc tranh luan 20.5 (a)/(b)/(c).
* `Token_Loss(full)` vs `Token_Loss(shuffled)`: chenh < 0,2 nat ⇒ `Token_Loss` 3,349 la thong ke
  chuoi, khong phai dieu kien hoa theo hoc.

**Chi phi:** 3 nhanh x 10 epoch (khong can 25 — Muc 20.4 cho thay logscale on dinh tu epoch ~15,
va `lr` da giam con 1,25e-5 nen 10 epoch du de doc huong). Uoc ~3 x 7,5 gio.

### Quyet dinh MAF vs diagonal — HOAN, khong phai buoc tiep theo
Muc 19.5 de xuat doi head lam "phep thu quyet dinh re nhat". **Khong con dung** sau khi 19.3 bi
rut lai: ly do "MADE 1 lop chan 6/7 chieu" la suy dien tu mot permutation khong lien quan.
Them nua `DiagonalGaussianHead` co **it** capacity hon MAF, nen neu MAF khong hoc duoc thi no
cung khong. Chi chay so sanh head **sau** G5, va chi neu G5 cho thay hoc that su mang tin hieu.

---

# HOAN — cap nhat Rev 5
* (giu nguyen muc HOAN cua Rev 4 o tren)
* **6D continuous rotation: KHONG con "khong cap thiet".** Neu M1 cho ra (b) — `qy` thoai hoa —
  thi bieu dien quaternion la van de chinh va 6D rotation thanh viec **uu tien**.
* So sanh `AutoregressiveFlowLayer` vs `DiagonalGaussianHead`: sau G5.

---

# PLAN Rev 5 — bo sung sau M1 (2026-10-01)

**M1 DA XONG. (b) bi loai** (RESEARCH_CONTEXT 21.1-21.2). `qy` dong nhat voi `qw/qx/qz` ve moi
thong ke bien ⇒ `sigma = 0,257` den tu dieu kien hoa, khong tu phan phoi bien.

**Chen M3 len TRUOC P3a.** Ly do: M3 khong sua code, khong chay train, mat vai giay, va no
**ket thuc** tranh luan (a)/(c) ngay — trong khi P3a/M2/G5 cong lai la ~25 gio GPU. Lam viec
re nhat co kha nang ket luan cao nhat truoc.

## M3. Phuc hoi permutation THAT cua run 25 epoch tu checkpoint
**Can cu:** Pyro `MaskedLinear` dang ky `mask` lam **buffer** ⇒ mask nam trong `state_dict`.
Permutation cua dung run da tao ra `flow_scale_trace.csv` **phuc hoi duoc**, khong phai doan.

**File moi:** `src/recover_arn_permutation.py`

1. Tim checkpoint cua run 25 epoch (`save_checkpoint_flow/`, `logs/archive/`, hoac bat cu dau).
   Neu **khong co** checkpoint nao ⇒ in `KHONG CO CHECKPOINT` va dung. (Luc do M3 bat buoc
   phai chay lai 1 epoch sau P3a-1 moi doc duoc, va cau hoi (a)/(c) chuyen sang G5.)
2. Load `state_dict`, lay moi key khop `*geometric_head.arn*mask*`. In shape tung mask.
3. Ghep mask thanh ma tran phu thuoc: `D = (M_L @ ... @ M_2 @ M_1) > 0`, roi **chi lay cac cot
   ung voi input dim** (bo cac cot context — mask dau co `input_dim + context_dim` cot, cot
   input nam truoc).
4. In ma tran phu thuoc `7x7` dang bang, co nhan ten theo
   `DIM_NAMES = ["d","theta","phi","qw","qx","qy","qz"]`.
5. Tu so luong phu thuoc cua tung output, suy ra **thu tu tu hoi quy** va in ra ro rang.
6. In ro: `qy` dung o vi tri thu may, va `qy` co phu thuoc dong thoi ca `qw`, `qx`, `qz` hay khong.

**Tieu chi doc (ghi cung TRUOC khi chay):**
* `qy` phu thuoc **ca ba** `qw, qx, qz` ⇒ **(a) XAC NHAN**. Muc 19.4 sai hoan toan. Viec tiep
  theo la **bieu dien quay** (6D continuous rotation), KHONG phai doi head, KHONG phai sua
  encoder. G5 van chay nhung muc tieu doi: xac nhan `d/theta/phi`, khong con ky vong gi o
  quaternion.
* `qy` **khong** phu thuoc du ca ba ⇒ **(a) BI LOAI**. Con lai **(c)**. Chay tiep P3a -> M2 -> G5.

**Neu (a) xac nhan, lam them phep do khang dinh (re, cung checkpoint):** kiem du doan dinh luong
o 21.4 — phan phoi `log_scale` cua `qy` tren mot batch phai **luong cuc**, va `log_scale` thap
phai tuong quan voi su kien `|qy| > max(|qw|,|qx|,|qz|)`. Neu khong luong cuc thi (a) cung co
van de va phai xem lai.

## Thu tu moi
**M3 -> (tuy ket qua) -> P3a -> M2 -> G5.**
P3a (seed + co dinh permutation + mask `_ls_sum`) van **bat buoc** bat ke M3 ra gi: no la dieu
kien de moi so do ve sau co nghia (Muc 20.3).

---

# PLAN Rev 5 — bo sung sau M3 (2026-10-01)

**M3 XONG. (a) xac nhan** (RESEARCH_CONTEXT 22.1-22.2). Thang algebra khop hoan toan: luong
"hoc duoc" cua tung chieu quaternion bang dung luong `||q||=1` cho khong.
**Nhung 22.5: `pinfrac(qy) = 0` mau thuan voi phien ban don gian cua (a)** (du doan ~0,27).

## SUA LAI hanh dong da ghi cung o M3 — co ly do, khong phai doi cho tien
M3 ghi: "(a) xac nhan ⇒ viec tiep theo la bieu dien quay 6D, **khong phai doi head**".
**Toi sua lai hanh dong nay** (khong sua cach doc — cach doc dung nhu da ghi cung):

Muc tieu that su la **cat kenh dai so** de do duoc co tin hieu tu hoc hay khong.
`DiagonalGaussianHead` cat kenh do **hoan toan** — moi chieu chi dieu kien tren context, khong
chieu nao thay chieu nao ⇒ khong con duong `||q||=1` ⇒ moi `sigma` tut xuong **chi** co the do
context. Va no **da duoc viet va kiem** (Muc 20.7).
6D continuous rotation cung cat kenh do, nhung keo theo: doi `out_dim` 7->9, tinh lai scaler,
sua `DIM_NAMES`, sua `system_logger` (dang hardcode 7 ten), viet lai nghich dao o step7.
**Hai cach cho cung mot phep do; mot cach da xong.** Nen diagonal head truoc.

6D rotation **van can**, nhung vi ly do khac va de sau: double cover `q ~ -q` va ranh gioi
canonicalization lam ham mat mat khong lien tuc o `|q_max| ~ 1/sqrt(2)`, gay hai cho **sinh mau**
chu khong cho chan doan. Chuyen sang muc HOAN co dieu kien.

## M4. Kiem du doan dinh luong cua (a) — bat buoc truoc khi hanh dong
**Can cu:** 21.4 + 22.5. Dung checkpoint `saved_checkpoints_flow/periodic-epoch=024.ckpt`,
mot batch, khong train.

**File moi:** `src/measure_qy_signflip.py`
1. Load checkpoint (`geo_head="maf"` — checkpoint nay la MAF, xem 20.9).
2. Lay 1 batch train that (dung `SAMGOptimizedDataset`), forward **o che do `training=True`** de
   hook `log_scale` chay, nhung bao `torch.no_grad()`.
3. Lay `log_scale` **per-sample** cho chieu `qy` (index 5), **chi cac hang token THAT**
   (`pad_mask = target_ids != 0`).
4. Tinh chi bao tu **du lieu raw** (truoc chuan hoa): `dom = |qy| > max(|qw|,|qx|,|qz|)`.
   In ty le `dom` — du doan ~0,25.
5. In histogram 40 bin cua `log_scale_qy`, va **rieng** cho `dom=True` va `dom=False`.
6. In `mean/median` cua `log_scale_qy` trong hai nhom, va `pinfrac` trong hai nhom.

**Tieu chi doc (ghi cung TRUOC khi chay):**
* `log_scale_qy` **luong cuc**, nhom `dom=True` thap han nhom `dom=False` (chenh > 1,5 nat)
  ⇒ (a) giai thich **tron ven**. Ket thuc. Sang P3a -> diagonal head -> G5.
* `log_scale_qy` **don dinh** quanh -1,36 o **ca hai** nhom ⇒ (a) chi la **mot phan**. Mo hinh co
  thong tin ve **dau** cua `qy` tu nguon khac (token C3, `d/theta/phi`, hoac hoc). Luc do phai
  tach nguon do truoc khi ket luan gi ve hoc — va day se la manh moi dau tien co that.
* Neu `ty le dom` lech xa 0,25 ⇒ xem lai gia dinh doi xung o 21.4.

## Thu tu cuoi cung
**M4 -> P3a -> chay diagonal head (10 epoch) -> G5 (full / shuffled / zeroed).**
P3a bat buoc bat ke M4 ra gi (Muc 20.3 phan con hieu luc).

## HOAN — cap nhat sau M3
* 6D continuous rotation: **co dieu kien**, sau G5. Ly do la double cover + ranh gioi
  canonicalization gay hai cho SINH MAU, khong phai cho chan doan.
* So sanh MAF vs diagonal: **da thanh buoc bat buoc**, khong con la "phep thu quyet dinh" cua
  19.5 (ly do cu da bi rut lai) ma la **cach cat kenh dai so** de phep do co nghia.

---

# PLAN Rev 5 — bo sung sau M4 (2026-10-01)

**M4 XONG. (a) xac nhan ve CO CHE** (RESEARCH_CONTEXT 23.2: `dom=True` 83% duoi `-1,0` vs
`dom=False` 3,4%), **bac ve DO LON** (23.3). Cau hoi mo 22.5 **dong** — khong can nguon tin moi.

**Phat sinh moi, uu tien cao (23.4): `flow_scale_trace.csv` bi nhiem padding NANG** (`qy`:
`-1,357` toan hang vs `-0,768` token that, lech 43%). **Bang phan ra 20.4 khong con dung lam can
cu.** Phai do lai truoc khi quyet dinh gi.

## M5. Do lai `log_scale` 7 chieu tren TOKEN THAT, nhieu batch — thay the bang 20.4
**File moi:** `src/measure_logscale_realtokens.py`. Checkpoint `periodic-epoch=024.ckpt`,
`geo_head="maf"`. Forward-only, `torch.no_grad()`, khong backward, khong train.

1. Lap qua **it nhat 200 batch** train (hoac het mot epoch neu duoi 30 phut).
2. Hook `log_scale` per-sample tren `arn`; **loc `pad_mask = (target_ids != 0)`**.
3. Cho **ca 7 chieu**, in: `mean`, `median`, `std`, `pinfrac`, va `n` (so token that).
4. In song song: `mean` tren **moi hang** (ke ca padding) de do truc tiep do nhiem cua CSV.
5. In `mean(NLL)` tren token that, va `E[z^2]` per-dim (`z = (target - mu)/sigma`) de kiem
   calibration. Can `mu` nen phai lay ca `output[0]` cua hook.
6. Phan ra lai: `sum(ln sigma)`, NLL du doan `= 7*0,5*ln(2pi) + 0,5*sum(E[z^2]) + sum(ln sigma)`,
   doi chieu voi `mean(NLL)` do duoc. Bao lech.

**Tieu chi doc (ghi cung TRUOC khi chay) — dung do do dung dai luong (23.6):**
* Goi `R2_i = 1 - sigma_i^2` tren token that. **Dem so chieu quay (`qw,qx,qy,qz`) co
  `R2_i < 0,15`.** 
  - 3 hoac 4 chieu ⇒ ket luan 20.4 **duoc chung minh lai**: dau hinh hoc khong hoc gi ve huong
    ngoai rang buoc `||q||=1`. Sang P3a -> diagonal head -> G5 nhu ke hoach.
  - 0-1 chieu ⇒ ket luan 20.4 **bi bac**: so bi nhiem da tao ra hinh mau gia. Phai doc lai toan
    bo Muc 20-23 truoc khi lam gi tiep.
  - 2 chieu ⇒ khong ket luan; bao cao va dung.
* `|mean(NLL) - NLL du doan| < 0,2 nat` ⇒ phan ra dang tin. Lon hon ⇒ con nguon chua hieu, dung.
* Bao cao `mean` toan-hang vs token-that cho tung chieu ⇒ do chinh xac do nhiem CSV, de ghi vao
  Muc 23.4.

## Thu tu cap nhat
**M5 -> P3a -> diagonal head (10 epoch) -> G5.**
P3a-3 (mask `_ls_sum`/`_pin_sum`) gio khong con la "sua cho gon" ma la **dieu kien** de
`flow_scale_trace.csv` co y nghia o moi run ve sau (23.4).

---

# PLAN Rev 6 (2026-10-01) — sau M5

**M5 XONG. Muc 20.4 BI BAC** (tieu chi: 1/4 chieu quay co `R2 < 0,15`). Bang phan ra sach o
RESEARCH_CONTEXT **24.1**. Doc lai Muc 20-23 da lam o **24.6**.

Thay doi huong quan trong: **(a) duoc cung co, khong phai yeu di** (24.2 — thang don dieu theo so
tien to quaternion), va **hai chieu duy nhat con cho tin hieu hoc an la `theta` va `phi`** (24.7),
ca hai **khong** co kenh dai so.

## Hai hau qua cho ke hoach
1. **Khong can chay diagonal head truoc G5 nua.** Ly do o plan Rev 5 ("cat kenh dai so de phep do
   co nghia") chi ap cho **bon chieu quaternion**. Nhung cau hoi con lai nam o `theta/phi/d`, la
   nhung chieu **vo nhiem** voi kenh do. Doc G5 tren chung la sach ngay voi MAF.
   Diagonal head chuyen sang **sau** G5, de don dep phan quaternion.
2. **P3a van BAT BUOC va gio co can cu dinh luong:** `flow_scale_trace.csv` sai **17%–88%** tuy
   chieu (24.4). Khong co P3a-3 thi moi run ve sau deu khong doc duoc.

## Thu tu moi: **P3a -> G5 (2 nhanh) -> diagonal head -> (tuy) 6D rotation**

## G5 Rev 2 — rut tu 3 nhanh xuong 2, va doi cach doc
**Hai nhanh**, cung `seed`, cung `permutation`, cung so epoch, chi khac mot thu:
1. **full**
2. **shuffled** — `h_target` xao tron theo chieu batch (giu nguyen phan phoi bien, pha **su ghep
   doi** hoc <-> ligand). **Mask `target_mask` phai xao tron CUNG hoan vi** voi `h_target`, neu
   khong se lech so residue va tao nhieu khong kiem soat.

**Bo nhanh `zeroed`.** Zero hoa dua input ra ngoai manifold huan luyen nen mo hinh co the hong vi
ly do khac; `shuffled` da la doi chung dung va chat hon. Tiet kiem ~7,5 gio.

**KHONG tai dung run 25 epoch cu lam nhanh `full`**, du da co so sach cho no (24.1). Run do
**khong co seed, khong co permutation co dinh** ⇒ so sanh voi mot nhanh moi la lap lai dung loi
20.3. Chi phi tiet kiem khong bu duoc rui ro. Chay ca hai nhanh moi.

**Chi phi:** 2 nhanh x 10 epoch x ~45 phut = **~15 gio**. Chay qua dem.

### Tieu chi DAT (ghi cung TRUOC khi chay) — doc dung dai luong
Do bang **M5 tren checkpoint cuoi cua tung nhanh** (khong doc tu `flow_scale_trace.csv`, du da
sua P3a-3 — dung mot cong cu cho ca hai nhanh de loai sai lech dung cu).

**Chieu quyet dinh: `theta` va `phi`** (24.7 — vo nhiem voi kenh dai so).
* `R2_theta(full) - R2_theta(shuffled) > 0,05` **VA** dieu tuong tu cho `phi`
  ⇒ **hoc protein CO dong gop vao tu the.** Bang chung dau tien co that. Di tiep.
* Ca hai chenh `< 0,02` ⇒ **hoc KHONG dong gop gi.** `R2` 18,5% cua `theta/phi` den tu token
  fragment va thong ke chuoi. Luc do van de **khong** o head, khong o bieu dien quay, ma o
  **encoder/conditioning** — va toan bo Phase 2 la sua sai cho. Phai dung va thiet ke lai duong
  dieu kien hoa.
* Mot chieu vuot 0,05, chieu kia duoi 0,02 ⇒ khong ket luan; bao cao va dung.

**Bao cao kem (khong phai cong chan):** `R2` cua ca 7 chieu o hai nhanh, `Token_Loss` hai nhanh,
va `E[z^2]` hai nhanh. Chenh `Token_Loss` < 0,2 nat ⇒ `Token_Loss` 3,349 la thong ke chuoi.

## Yeu cau chung cho MOI script do tu day (thieu sot dac ta cua toi, 2026-10-01)
1. In `git rev-parse HEAD`, `SEED`, va **device dang dung** o dong dau.
2. In tien do moi 20 batch kem `flush=True`. Khong co dong tien do thi khong phan biet duoc
   "dang chay" voi "treo" — da mat 30 phut vi viec nay.
3. Chay tren GPU: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`,
   `map_location=device`, `model.to(device)`, `batch.to(device)`.
4. Moi so lieu per-dim phai loc `pad_mask`; neu bao cao ca ban khong loc thi phai ghi ro.
5. Lenh chay luon dung `python -u ... | tee` (khong co `-u` thi stdout bi buffer 4KB qua pipe).

## HOAN — cap nhat Rev 6
* `DiagonalGaussianHead`: sau G5.
* 6D continuous rotation: sau G5. Ly do (double cover + ranh gioi canonicalization) van dung
  nhung chi anh huong SINH MAU.
* Gia thuyet bao hoa Tanh (20.8): ha uu tien, xem 24.6.
* `E[z^2](qy) = 0,675` — mo hinh thieu tu tin o `qy`. Khong sua gi bay gio; no la hau qua cua
  Gaussian don dinh gap phan phoi hai diem, se tu bien mat khi doi bieu dien quay.

---

# PLAN Rev 6 — SUA KE HOACH: M6 thay cho G5 o buoc dau (2026-10-01)

**Toi da dat G5 sai cho.** G5 doi 2 nhanh **huan luyen** moi (~15 gio) de tra loi "hoc protein co
dong gop khong". Nhung cau hoi do co the tra loi bang **mot phep do forward-only tren checkpoint
DA CO** — vai phut, khong train gi.

## M6. Xao tron hoc o THOI DIEM SUY LUAN (inference-time pocket shuffle)
**File moi:** `src/measure_pocket_shuffle.py`. Checkpoint `periodic-epoch=024.ckpt`, `geo_head="maf"`.
Forward-only, `no_grad()`, dung **cung** bo 200 batch nhu M5, **cung** cach loc `pad_mask`.

Voi moi batch, forward **ba** lan tren **cung** mot batch:
1. **full** — nhu binh thuong.
2. **shuffled** — xao tron `h_target` theo chieu batch (`perm = torch.randperm(B)`),
   **va xao tron `target_mask` theo CUNG `perm`**. Khong duoc xao rieng mot cai.
3. **roll-1** — doi chung thu hai, re: `h_target = h_target.roll(1, dims=0)` cung voi `target_mask`.
   Dung de kiem rang ket qua cua (2) khong phu thuoc vao mot hoan vi ngau nhien cu the.

In cho **ca 7 chieu**, o **ca ba** nhanh: `mean(log sigma)`, `sigma`, `R2 = 1-sigma^2`, `E[z^2]`,
va `mean(NLL)` toan bo. In **bang chenh lech** `R2(full) - R2(shuffled)` cho tung chieu.

### Tai sao phep do nay hop le (va tai sao `shuffled` chu khong phai `zeroed`)
Xao tron giu **y nguyen** phan phoi bien cua `h_target` — moi vector van la mot hoc protein that,
chi la hoc **sai** cho ligand do. Nen input khong ra ngoai manifold huan luyen. Zero hoa thi co.
Do la ly do Rev 5 da chon `shuffled`; diem moi la **khong can train lai** moi do duoc.

### Tieu chi doc (ghi cung TRUOC khi chay)
**Chieu quyet dinh: `theta` va `phi`** (RESEARCH_CONTEXT 24.7 — vo nhiem voi kenh dai so `||q||=1`).
* `R2(full) - R2(shuffled) > 0,05` o **ca hai** chieu, **va** nhanh `roll-1` cho ket qua cung
  huong ⇒ **mo hinh DANG dung hoc protein.** Bang chung dau tien co that. G5 tro thanh khong can
  thiet o buoc nay; di tiep sang cai thien.
* Ca hai chenh `< 0,02` ⇒ **mo hinh KHONG dung hoc protein.** `R2` 18,5% cua `theta/phi` den tu
  token fragment + thong ke chuoi. Hanh dong: **dung Phase 2**, thiet ke lai duong dieu kien hoa
  (encoder -> `h_target` -> attention hub). Khong sua head, khong sua bieu dien quay — ca hai deu
  la sai cho.
* Mot chieu vuot 0,05, chieu kia duoi 0,02, hoac `roll-1` nguoc huong voi `shuffled`
  ⇒ khong ket luan; bao cao va dung.

**Bao cao kem:** chenh `R2` cua ca 7 chieu, va `mean(NLL)` ba nhanh. Neu `d` cung chenh > 0,05 thi
mot phan `51,8%` cua `d` den tu hoc chu khong chi tu hoa hoc — dang chu y nhung khong phai cong chan.

### Gioi han phai ghi ro khi doc ket qua
M6 tra loi "**mo hinh NAY co dung hoc khong**". No **khong** tra loi "trong hoc co tin hieu hay
khong". Neu M6 ra am tinh thi con hai kha nang: (i) khong co tin hieu de hoc, (ii) co tin hieu
nhung quy trinh huan luyen khong tim duoc. **Ca hai dan den cung mot hanh dong** (thiet ke lai
duong dieu kien hoa), nen khong can G5 de phan biet truoc khi hanh dong.

## Thu tu moi
**M6 (vai phut) -> P3a (bat buoc, re) -> tuy ket qua M6:**
* M6 duong tinh ⇒ giu kien truc, chay diagonal head + cai thien, G5 chi lam khi can cong bo.
* M6 am tinh ⇒ **dung Phase 2**, lam lai duong dieu kien hoa. G5 **huy** (khong con gia tri quyet dinh).

## G5 — HOAN, cho ket qua M6
Giu nguyen dac ta o Rev 6 phia tren. Khong chay truoc M6. Ly do: 15 gio GPU de lay mot cau tra
loi ma 5 phut co the lay duoc phan lon.

---

# PLAN Rev 7 (2026-10-01) — sau M6

**M6 XONG. Tieu chi M6 cua toi SAI** (`R2 = 1 - sigma^2` mu voi hieu ung nam o `mu`, xem
RESEARCH_CONTEXT 25.2). Voi `R2` dung (`1 - sigma^2 * E[z^2]`):
* **tinh tien** `d/theta/phi`: `delta R2` = **0,171 / 0,028 / 0,027** ⇒ hoc protein **CO** dong gop
* **phep quay** quaternion: `delta R2` = 0,003–0,014 ⇒ **khong** dong gop
* ca bay chieu duong o ca hai doi chung (`shuffled`, `roll1`), cung thu tu do lon
* `mean(NLL)` 8,2948 (full) vs 8,5155 (shuffled) = **+0,22 nat**

**P3a-2 da xong** trong commit `7ce0f06` (seed_everything(1234), `deterministic=True`).
Con **P3a-1** (co dinh permutation ARN) va **P3a-3** (mask `_ls_sum`/`_pin_sum`).

## Rang buoc phuong luan: 25.3 la THAM DO, chua phai xac nhan
Cong thuc `R2` dung duoc suy ra **sau khi** xem du lieu. Khong duoc dung 25.3 lam ket luan.
**M7 la buoc bat buoc truoc moi quyet dinh kien truc.**

## M7. Xac nhan tren tap VALID + doi chung kich thuoc hoc
**File:** sua `src/measure_pocket_shuffle.py` (them tham so, khong viet script moi).

### M7-a. Chay lai tren tap **valid** (1.058 complex) — du lieu chua he dung de doc
Them `--split valid`. Dung **het** tap valid, khong gioi han 200 batch. Bao cao `R2` dung
(`1 - sigma^2 * E[z^2]`) cho ca 7 chieu, ba nhanh.

**Nguong dat BAY GIO, truoc khi chay:**
* `delta R2(d) > 0,10` **VA** `delta R2(theta) > 0,015` **VA** `delta R2(phi) > 0,015`,
  **VA** ca ba cung dau o nhanh `roll1`
  ⇒ **XAC NHAN: hoc protein chi phoi tinh tien.** Bang chung dau tien duoc xac nhan cua du an.
* `delta R2(d) < 0,05` ⇒ hieu ung o 25.3 la hien vat cua tap train; **bac** 25.4, doc lai.
* Trung gian ⇒ khong ket luan, bao cao va dung.
* Rieng quaternion: nguong **khong** doi — ky vong `delta R2 < 0,02` o ca bon. Neu mot chieu
  quaternion vuot **0,05** thi 25.4 sai va phai doc lai.

### M7-b. Doi chung KICH THUOC: `d` co that su doc hinh hoc hoc, hay chi doc "hoc to/nho"?
`delta R2(d) = 0,171` la hieu ung lon nhat. Giai thich giam gia tri: `h_target` chua toa do residue
trong frame hoc (P2a), nen xao tron hoc doi luon **kich co** hoc. `d` la khoang cach. Mo hinh co
the chi doc **mot so vo huong** (hoc to bao nhieu) de dinh ty le khoang cach — van la thong tin hoc
that, nhung la loai tho nhat, khong phai nhan dang hinh dang.

Them nhanh **`shuffled_sizematched`**: tinh `pocket_extent = max||p_local||` cho tung mau trong
batch, sap xep, roi chi xao tron **trong tung nhom co kich thuoc gan nhau** (chia batch thanh 4
nhom theo tu phan vi cua `pocket_extent`, derangement **trong** moi nhom).

**Doc:**
* `delta R2(d)` o nhanh nay **tut xuong duoi mot nua** `0,171` ⇒ phan lon hieu ung cua `d` chi la
  **kich co hoc**, khong phai hinh dang. Ghi nhan trung thuc, va dieu do **ha** gia tri cua 25.4.
* `delta R2(d)` giu tren hai phan ba ⇒ mo hinh doc thong tin hoc **dac thu**, khong chi kich co.

## Sau M7 — chi lam khi M7-a XAC NHAN
### P3a-1 + P3a-3 (re, bat buoc truoc moi run huan luyen moi)
Nhu dac ta Rev 5. P3a-1: `permutation=torch.arange(out_dim)` cho `ConditionalAutoRegressiveNN`.
P3a-3: mask `pad` cho `_ls_sum`/`_pin_sum` o **ca hai** head.
Them: **khong duoc tin dong in `permutation ARN`** khi load checkpoint (25.1) — phai doc tu mask.

### E1. Chay `DiagonalGaussianHead` — NANG uu tien, co gia thuyet tuong minh
Dao lai quyet dinh o Rev 6 (ly do o 25.5): kenh dai so `||q||=1` co the **dang canh tranh** voi
tin hieu hoc o quaternion, khong chi lam nhieu phep do. Diagonal head cat kenh do hoan toan.

10 epoch, `geo_head="diag_gauss"`, seed 1234. Roi chay M6/M7 tren checkpoint cua no.

**Gia thuyet ghi cung TRUOC khi chay:** neu shortcut dai so dang canh tranh thi sau khi cat no,
**`delta R2` cua it nhat 2 trong 4 chieu quaternion phai vuot 0,03** (hien 0,003–0,014).
* Vuot ⇒ shortcut la nguyen nhan; huong di la bo bieu dien quaternion (6D rotation) cho hoan toan.
* Khong vuot ⇒ shortcut **khong** phai nguyen nhan; phep quay khong hoc duoc vi **encoder bat bien**
  (Muc 16.1), va phai giai quyet o tang bieu dien dieu kien, khong o head.
Hai nhanh nay dan den hai viec khac nhau, nen thi nghiem nay dang gia.

## G5 — HUY
Khong con gia tri: M6/M7 da tra loi cau hoi cua no bang forward-only. Tiet kiem ~15 gio.

## Bo sung yeu cau chung cho script do (them vao danh sach Rev 6)
6. **Moi `R2` phai tinh la `1 - sigma^2 * E[z^2]`**, khong bao gio la `1 - sigma^2`. Neu bao cao ca
   `1 - sigma^2` thi phai ghi ro no la "do khong chac chan tu khai bao", khong phai variance explained.
7. Khi so sanh hai lan chay, phai in **cau hinh loader** (`shuffle`, `batch_size`, `split`, `n_batches`)
   va **so token that**. Hai so token khac nhau ⇒ hai mau khac nhau, khong duoc so truc tiep (25.1).

---

# PLAN Rev 8 (2026-10-01) — M7 khong doc duoc; hai chan doan bat buoc

**Kiem khong-doi FAIL** (RESEARCH_CONTEXT 26). Muc 25 **dinh chi**. Khong doc M7-a/M7-b.
Khong quyet dinh kien truc nao cho den khi xong M8 va M9. Ca hai deu re (vai phut).

## M8. Phan xu (i) vo hai vs (ii) pha huy — sua phep kiem khong-doi
**File:** `src/measure_pocket_shuffle.py`.

Thay phep kiem `torch.equal` bang:
1. In **shape va `is_contiguous()`** cua `h_target` va `target_mask` **ngay tai diem ap `perm`**
   trong `src/step6_trainer.py` (them `print` co dieu kien, chi khi mot bien moi
   `pocket_shuffle_debug=True` duoc truyen; mac dinh tat).
2. `perm=None` vs `perm=torch.arange(B)`: in `max|delta mu|`, `max|delta log_scale|`,
   va `torch.allclose(atol=1e-5, rtol=1e-4)` -> PASS/FAIL.
3. Giu ca `torch.equal` nhung doi nhan thanh "bit-identical (ky vong FAIL tren GPU, KHONG phai loi)".

**Doc (ghi cung TRUOC khi chay):**
* `max|delta| < 1e-4` **va** shape tai diem ap la **3 chieu `[B, R_max, H+3]`**
  ⇒ **(i) vo hai**: chi la lam tron do layout. Muc 25 va M7 **duoc phuc hoi**, doc binh thuong.
* `max|delta| > 1e-2`, **hoac** shape tai diem ap khong phai 3 chieu
  ⇒ **(ii) pha huy**: `perm` ap sai tensor. Muc 25 va M7 **bi huy hoan toan**, phai sua step6 roi
  chay lai M6 va M7.
* Trung gian (`1e-4` den `1e-2`) ⇒ khong ket luan; in them `max|delta|` theo tung lop de truy.

## M9. Kiem ro ri train/valid — BAT BUOC truoc khi dung valid lam tap xac nhan
**Can cu:** 26.3 — `theta` tot hon 7 lan tren valid, `phi` 3 lan, `qx` 25 lan. Khong the la ngau nhien.

**File moi:** `src/check_split_leakage.py`. Khong GPU, khong model, chi doc index/pkl.
1. Lay tap `pli_id` (hoac khoa dinh danh ma `SAMGOptimizedDataset` dung de ghep) cua **train**,
   **valid**, **test**, theo **dung** duong dan ma step6 dung. In `len` tung tap.
2. In `len(train & valid)`, `len(train & test)`, `len(valid & test)` va vai vi du khoa chong lan.
3. Chong lan muc **PDB** (khong chi `pli_id`): tach ma PDB 4 ky tu tu khoa va lam lai buoc 2.
   Cung mot protein duoi hai ligand khac nhau **van** la ro ri o muc protein cho bai toan SBDD.
4. In so complex cua tung split **sau** filter `frame_stable` — doi chieu: valid 764 (M7 do duoc)
   so voi 1.058 (con so cu). Giai thich chenh lech.

**Doc (ghi cung TRUOC khi chay):**
* `train & valid` rong **va** chong lan muc PDB rong ⇒ khong ro ri; 26.3 phai co nguyen nhan khac
  (gia thuyet (B): `--split valid` lay sai tap). Luc do kiem tiep duong lay split.
* `train & valid` khac rong ⇒ **ro ri du lieu**. valid khong dung lam tap xac nhan duoc. Phai chia
  lai split theo **PDB** truoc moi phep do generalization, va moi so Vina/QED/NLL "valid" tu truoc
  den nay vo gia tri.

## Thu tu
**M8 va M9 chay doc lap, co the lam song song** (M9 khong dung GPU). Doc ca hai roi moi quyet dinh.
Khong chay E1 (diagonal head), khong sua bieu dien quay, khong P3a-1/P3a-3 cho den khi xong.

## Bo sung yeu cau chung cho tieu chi (them vao Rev 6/Rev 7)
8. **Truoc khi ghi cung mot tieu chi, phai tra loi duoc: "neu moi thu DUNG thi dai luong nay bang
   bao nhieu, va no CO THE dat gia tri do khong?"** Ba trong bon lan sai tieu chi hom nay deu vi
   bo qua cau hoi nay (RESEARCH_CONTEXT 26.5).

---

# PLAN Rev 9 (2026-10-01) — nhieu dequant; do SAN NHIEU truoc moi ket luan

**M9: khong ro ri** (27.1). **M8: ca (i) va (ii) bi bac; nguyen nhan la nhieu dequant ve lai moi
forward** (`step4_ligand_generator.py:123`, trong forward, input cua ARN) — 27.2.
**Hau qua: M6 va M7 deu co mot so hang nhieu khong kiem soat** (27.3). Muc 25 van dinh chi.

## QUYET DINH DU AN (nguoi dung dong y 2026-10-01): KHONG viet lai
Can cu: moi bug da sua la bug **quy uoc mien** (frame, thu tu quaternion, ghep du lieu, mask
padding, nhieu trong forward) — viet lai khong ke thua tri thuc ve quy uoc nao dung, phai tra gia
lai. Tai san dat nhat la **nhan da sach** (G1 9,157° -> 0,0155°; G3 0,0239 -> 0,0001 A; bat dinh
48% -> 0,01%), **ghep du lieu dung** (17.200 mau), **da bit ro ri ligand GT** (P2b), **hoi tu on
dinh** (pinfrac ~ 0, 25 epoch), va **RESEARCH_CONTEXT 27 muc** — thu khong nam trong code.
Khuyet diem that: **kiem tien de cot loi xuyen qua 7 tang cung luc**. Khac phuc bang phep tham do
toi thieu (M11), khong bang viet lai.
Tieu chi BO (ghi cung): neu M11 that bai ⇒ tien de "encoder hien tai trich duoc thong tin huong tu
hoc" sai ⇒ phai chuyen sang kien truc **tuong bien (equivariant)**, tuc viet lai **tang encoder +
bieu dien**, khong phai viet lai repo.

## M10. Khoa nhieu dequant + do SAN NHIEU — bat buoc truoc khi doc lai M6/M7
**File:** `src/measure_pocket_shuffle.py`.

### M10-1. Khoa nhieu giua cac nhanh
Truoc **moi** lan forward cua **moi** nhanh trong cung mot batch, dat lai cung mot seed phu thuoc
chi so batch:
```python
torch.manual_seed(1234 + batch_idx)
torch.cuda.manual_seed_all(1234 + batch_idx)
```
⇒ moi nhanh nhan **cung mot** lan ve nhieu dequant ⇒ so sanh ghep doi hoan toan, so hang nhieu
triet tieu. Giu nhieu (khong bo) de phan phoi input giong luc huan luyen.

### M10-2. Them nhanh `noop` lam SAN NHIEU, bao cao dung don vi cua tieu chi
Nhanh thu nam: `noop` = `pocket_shuffle_perm=torch.arange(B)`, tuc **dong nhat ve logic** voi `full`.
Bao cao `delta R2(full - noop)` cho ca 7 chieu, **cung bang** voi cac `delta` khac.
Giu ca `max|delta mu|`, `max|delta log_scale|`, `torch.allclose` — nhung chung la **chan doan**,
khong con la tieu chi.

### Tieu chi doc M10 (ghi cung TRUOC khi chay)
1. Sau M10-1, `delta R2(full - noop)` phai `< 0,002` o **moi** chieu, va `max|delta| < 1e-4`.
   Khong dat ⇒ con nguon ngau nhien chua khoa; **dung**, tim tiep, khong doc gi khac.
2. Dat roi thi moi doc lai M7-a/M7-b, voi dieu kien bo sung:
   **`delta R2(d, shuffled)` phai lon hon `delta R2(d, noop)` it nhat 10 lan.** Duoi 10 lan ⇒
   tin hieu khong tach khoi san nhieu, khong ket luan.
3. Nguong M7-a (`d > 0,10`; `theta > 0,015`; `phi > 0,015`; cung dau o `roll1`) **giu nguyen**,
   khong duoc sua sau khi thay so.

## M10-b. Giai nghich ly valid > train (27.6)
Chay lai `--split train` **hai lan**: mot voi `shuffle=False`, mot voi `shuffle=True` (them
`--loader-shuffle`). Neu `R2(theta)` nhay tu ~0,062 len ~0,43 ⇒ nghich ly la hien vat cua **tap con
co thu tu**, khong phai tinh chat mo hinh. Ghi lai va tu day **moi phep do dung `shuffle=True`**
hoac dung het split.

## M11. Phep tham do toi thieu — sau khi M10 xong
Bai toan co giam sat, tach **hoan toan** khoi generator: cho hoc protein + danh tinh fragment,
hoi quy 7D pose. Khong autoregressive, khong flow, khong tokenizer, khong sinh mau.
`encoder -> MLP -> 7 so`, loss MSE. Doi chung: **pocket-blind** (`h_target` zero hoa **luc huan
luyen**, khong phai luc suy luan).
Chi tiet dac ta se viet sau khi M10 cho ket qua — vi san nhieu quyet dinh kich co hieu ung can do.

## Bo sung yeu cau chung cho tieu chi (them vao Rev 6/7/8)
9. **Tieu chi phai o DUNG DON VI cua dai luong se quyet dinh.** `max|delta|` tung phan tu khong tra
   loi duoc cau hoi ve `delta R2` trung binh. Moi phep kiem "khong-doi" phai bao cao **san nhieu
   trong don vi cua tieu chi**, khong phai mot chuan vo cung (27.4).

---

# PLAN Rev 10 (2026-10-01) — M7 phuc hoi nhung khong dat nguong; M11 la buoc quyet dinh

**Doc lai M8** (RESEARCH_CONTEXT 28): phep kiem khong-doi chay o dong 297, zero-hoa nhieu bat dau
o dong 329 ⇒ **phep kiem** co nhieu song, **vong do 4 nhanh** thi khong.
⇒ 27.3 SAI; **Muc 25 va M7 PHUC HOI**. Nhung **M7-a KHONG DAT nguong** da ghi cung (28.3):
`delta R2(d)=0,0751` (can > 0,10), `theta=0,0135` (can > 0,015), `phi=0,0251` (dat).
Ket qua chinh thuc: **khong ket luan**. Khong ha nguong.

## M10 — VAN chay, nhung ha ky vong
Commit `02775b1` da kiem: thu tu seed **dung** (`run_branch` dat lai seed **ben trong**, ngay truoc
forward, nen moi tieu thu RNG truoc do — ke ca `randperm` trong `make_derangement` — bi xoa).
Gia tri con lai cua M10: do **san nhieu** (`delta R2(full - noop)`) de tra loi cau hoi 28.4 thu hai
("hieu ung co khac khong"), **khong** phai cau thu nhat ("co du lon khong").

**Luu y so sanh:** M10 bo zero-hoa, dung nhieu song + khoa seed. Nhieu nam trong input cua ARN nen
`E[z^2]` se cao hon M6/M7 mot chut (co `sigma_dequant^2 / sigma^2`). `delta R2` **trong** mot run
van ghep doi hop le, nhung **khong** so truc tiep gia tri tuyet doi M10 voi M6/M7.

Chay `--split valid` truoc. Doc theo dung ba dieu kien Rev 9. Du ket qua the nao, **sang M11**.

## M11. Phep tham do toi thieu — BUOC QUYET DINH
Can cu 28.5: M7 la phep do hoi co tren mot mo hinh co shortcut da biet ⇒ gia tri thong tin thap.
M11 huan luyen truc tiep, khong generator, khong flow, khong tokenizer, khong shortcut dai so.

### Thiet ke
**Bai toan:** cho hoc protein + danh tinh fragment (token id) + chi so vi tri, hoi quy 7D pose cua
fragment do. **Khong** autoregressive — moi fragment la mot mau doc lap.
**Kien truc:** `UniTransformer` (dung y nguyen, load tu step1) -> `h_target` `[B, R_max, H+3]` +
mask (dung y nguyen duong P2a) -> mot cross-attention 1 lop hoac mean-pool co mask -> concat
`token_embedding(frag_id)` + `pos_embedding(t)` -> MLP 2 lop -> **7 so**.
**Loss:** MSE tren 7 chieu da chuan hoa (dung `scaler_7d.pt`). **Khong** NLL, **khong** sigma —
loai bo hoan toan kha nang shortcut qua `log_scale`.
**Du lieu:** dung y nguyen `SAMGOptimizedDataset`, split train/valid. Khong tai sinh du lieu.

### Hai nhanh, cung seed, cung so epoch
1. **pocket** — binh thuong.
2. **blind** — `h_target` bi zero hoa **TRONG LUC HUAN LUYEN** (khong phai luc suy luan). Day la
   doi chung dung cho cau hoi "trong hoc co tin hieu hoc duoc khong", khac han doi chung
   inference-time cua M6/M7.

**Chi phi:** mo hinh nho hon nhieu, khong autoregressive ⇒ uoc 15-25 phut/nhanh cho 20 epoch.
Hai nhanh duoi 1 gio. Re hon M7 (M7 valid mat ~8 phut nhung chi doc duoc mot mo hinh cu).

### Tieu chi ghi cung TRUOC khi chay — dung don vi `R2` tren tap VALID
`R2_i = 1 - MSE_i` (muc tieu chuan hoa, variance = 1). Bao cao ca 7 chieu, ca hai nhanh.
* `R2(pocket) - R2(blind) > 0,05` o **it nhat 2 trong 3** chieu tinh tien (`d, theta, phi`)
  ⇒ **TIEN DE DUNG**: hoc protein chua tin hieu hoc duoc ve tu the, va encoder hien tai trich duoc.
  Di tiep: sua duong dieu kien hoa cua pipeline chinh.
* Ca ba chieu tinh tien chenh `< 0,02`
  ⇒ **TIEN DE SAI**: encoder bat bien hien tai khong trich duoc. Theo tieu chi BO o Rev 9, chuyen
  sang kien truc **tuong bien**, tuc viet lai tang encoder + bieu dien (khong phai viet lai repo).
* Trung gian ⇒ tang so epoch len 40 va do lai **mot lan**; neu van trung gian thi bao cao va dung.
* Rieng quaternion: bao cao nhung **khong** dat nguong. Muc 16.1 + 25.4 du doan chung ~0. Neu
  chung **vuot 0,05** thi 16.1 sai va do la phat hien quan trong hon ca phan tinh tien.

### Truoc M11: P3a-1 va P3a-3
P3a-1 (`permutation=torch.arange`) **khong can** cho M11 (M11 khong co MAF).
P3a-3 (mask `_ls_sum`) **khong can** cho M11 (M11 khong dung instrumentation do).
⇒ **Hoan ca hai den khi quay lai pipeline chinh.** Khong chan M11.

## Bo sung yeu cau chung (them vao Rev 6/7/8/9)
10. **Moi phep kiem phai chay TRONG cung moi truong (seed, patch, train/eval mode) voi phep do ma
    no bao ve.** Neu phep do nam trong mot patch hay context manager, phep kiem phai nam trong do.
    Vi pham dieu nay da lam dinh chi oan Muc 25 va M7 trong hai luot (28.6).

---

# PLAN Rev 11 (2026-10-01) — M10 xong; sang M11

**M10 xong** (RESEARCH_CONTEXT 29). San nhieu = **0,0011**. Ve `max|delta| < 1e-4` cua dieu kien 1
**bi loai** (vi pham chinh quy tac 27.4 viet truoc do — 29.1). Ve `delta R2 < 0,002`: **DAT**.
Dieu kien 2: **71,6x, DAT**.

**Da xac lap:** `d` co hieu ung that (~74x san nhieu, `delta R2 = 0,081` tren `R2 = 0,418`), **dac
thu theo hoc** (size-match giu 82%); `phi` nho nhung that (~20x); `theta` **khong tach duoc**;
**quay khong co gi**.
**Chua xac lap:** M7-a (`d > 0,10`, `theta > 0,015`) **KHONG DAT** — tin hieu co that nhung **khong
du lon** theo nguong da ghi cung.

## Khong lam them phep do hoi co nao nua
`--split train` cua M10: **huy**. Gia tri thong tin thap (29.7) va khong doi duoc quyet dinh.
Nghich ly valid>train (27.6) cung **huy** — khong con can thiet, va M10/M7 da chuyen sang doc
valid lam chuan.
`torch.use_deterministic_algorithms(True)`: **khong lam** bay gio. San nhieu 0,0011 da du nho so
voi 0,081. Ghi lai nhu mot tuy chon neu ve sau can san nhieu thap hon.

## M11 — BUOC QUYET DINH. Dac ta day du o plan Rev 10, bo sung moc so sanh
Giu nguyen thiet ke va tieu chi o Rev 10 (`R2(pocket) - R2(blind) > 0,05` o it nhat 2/3 chieu tinh
tien; ca ba `< 0,02` ⇒ tien de sai ⇒ chuyen sang kien truc tuong bien).

**Bo sung moc so sanh (ghi cung TRUOC khi chay):** M10 do duoc `delta R2(d) = 0,081` tren pipeline
day du **co** shortcut dai so va **co** generator. M11 bo ca hai.
* `R2(pocket) - R2(blind)` o `d` **> 0,081** ⇒ pipeline dang **lam mat** tin hieu; van de o
  generator/head, va dang sua.
* `<= 0,081` ⇒ pipeline **khong** lam mat gi dang ke; tran la o **encoder**, va sua generator se
  khong giup. Day la ket qua quan trong hon ca tieu chi chinh.

**Thu tu chay:** nhanh `pocket` truoc, doc `R2` tren valid. Neu `R2(d)` o nhanh `pocket` **thap hon
0,30** thi dung lai truoc khi chay nhanh `blind` — mot bo hoi quy co giam sat don gian ma khong dat
noi muc cua MAF trong pipeline (`0,418`) la dau hieu M11 bi loi thuc thi, khong phai ket qua.

---

# PLAN Rev 12 (2026-10-01) — SUA uoc luong chi phi M11; tach M11-a / M11-b / M11-c

**Sua loi uoc luong cua toi o Rev 10:** "15-25 phut/nhanh" la **sai ~40 lan**. M11 giu nguyen
`UniTransformer`, va encoder chinh la phan dat (~45 phut/epoch trong pipeline) ⇒ huan luyen
end-to-end 20 epoch la **~15 gio/nhanh**.

## M11-c. Refactor (DA XONG, commit `cd604cd`)
`build_h_target(self, data_batch)` tach khoi `SAMGLightningModule.forward`. Logic
`pocket_shuffle_perm` **giu o `forward`** ⇒ M11 nhan `h_target` khong bi xao tron. **Da kiem:**
`self.dynamic_encoder` duoc goi **mot** lan ca truoc (dong 288) va sau (dong 298) refactor —
khong co loi goi kep nao bi bo. Refactor thuan.
**Kiem khong-doi bat buoc truoc M11-a:** chay `measure_pocket_shuffle.py --split valid`;
`R2(full)` ca 7 chieu phai khop M10 trong **0,002**:
`d 0,4183 | theta 0,4287 | phi 0,3613 | qw 0,5318 | qx 0,1745 | qy 0,7945 | qz 0,2925`.

## M11-a. Encoder DONG BANG + `h_target` CACHE + head moi (DA XONG, commit `af87853`)
**Cau hoi:** *thong tin ve tu the co SAN trong bieu dien dau ra cua encoder hay khong?*
Encoder dong bang, cache `h_target`/`target_mask`/`target_7d`/`target_ids` (~0,9 GB), roi huan
luyen head moi tren cache (epoch chi vai giay). Cross-attention 1 lop co mask (K/V khong bias nen
nhanh `blind` cho `ctx = 0` chinh xac) -> MLP 2 lop -> 7 so, **MSE thuan**, khong sigma/NLL/
autoregressive, **khong** dua pose fragment truoc vao input.

### Ba nhanh (cung seed 1234, cung so epoch)
1. **pocket** 2. **blind** (zero hoa H+3 kenh feature, GIU `target_mask`, ca train va eval)
3. **tokenmean** (khong mo hinh: trung binh 7D theo `frag_id`, tinh tren train)

### Tieu chi ghi cung (do tren VALID, `R2_i = 1 - MSE_i`) — doc theo DUNG thu tu
**(1) Kiem nhat quan noi bo, doc TRUOC tieu chi chinh:** `R2(blind)` phai **xap xi hoac cao hon**
`R2(tokenmean)` o moi chieu. Neu `blind` **thap hon `tokenmean` qua 0,03** o bat ky chieu tinh tien
nao ⇒ `blind` **chua hoi tu** ⇒ tang epoch, **KHONG** doc tieu chi chinh.
*(Ghi chu: `af87853` chi in bao cao dinh huong cho muc nay, khong cai nguong 0,03. Nguong 0,03
duoc ap **thu cong** khi doc log — no da duoc ghi cung o day truoc khi chay.)*

**(2) Tieu chi chinh:** `R2(pocket) - R2(blind) > 0,05` o **it nhat 2 trong 3** chieu tinh tien
(`d, theta, phi`) ⇒ **TIEN DE DUNG**: thong tin hoc co san trong dau ra encoder. Sang sua duong
dieu kien hoa.
* Ca ba chieu tinh tien chenh **< 0,02** ⇒ thong tin **khong** co san trong encoder **dong bang**.
  Chua ket luan tien de sai — con kha nang encoder hoc duoc neu duoc huan luyen cho bai nay
  ⇒ **chay M11-b**.
* Trung gian ⇒ tang len 300 epoch, do lai **mot lan**; van trung gian thi bao cao va dung.

**(3) Moc so sanh (29.7):** `R2(pocket) - R2(blind)` o `d` so voi **0,081** (M10, pipeline day du).
Lon hon ⇒ pipeline dang **lam mat** tin hieu; van de o generator/head, dang sua.
`<= 0,081` ⇒ **tran o encoder**, sua generator khong giup. Quan trong hon ca tieu chi chinh.

**(4) Quaternion:** bao cao, **khong** dat nguong. Du doan ~0 (16.1, 24.2). Neu vuot 0,05 thi 16.1
sai va do la phat hien quan trong hon phan tinh tien.

**Bo guard "R2(d) < 0,30 thi dung"** cua Rev 11: **huy**. Pipeline co context tu hoi quy con M11
khong, nen `R2(d)` thap hon 0,418 la **hop le**, khong phai dau hieu loi. `tokenmean` la san dung.

## M11-b. End-to-end, encoder huan luyen — CHI neu M11-a am tinh
~15 gio/nhanh x 2 nhanh. Chi dang gia khi M11-a cho thay thong tin khong co san trong encoder
**dong bang**. Neu M11-b cung am tinh ⇒ **tieu chi BO** (Rev 9) kich hoat: chuyen sang kien truc
**tuong bien**, tuc viet lai tang encoder + bieu dien (khong phai viet lai repo).

## Da HUY
* `--split train` cua M10, va viec giai nghich ly valid>train (27.6): gia tri thong tin thap.
* `torch.use_deterministic_algorithms(True)`: khong can, san nhieu 0,0011 da du nho so voi 0,081.
* G5: da huy tu Rev 7.

---

# PLAN Rev 13 (2026-10-01) — kiem khong-doi DAT; chay M11-a

**Kiem khong-doi cho `build_h_target`: DAT** (RESEARCH_CONTEXT 30.1, lech toi da 0,0012 < 0,002).
**M10 duoc tai lap doc lap** (30.2) — lan dau trong du an.

## Viec nho, khong gap (lam khi tien, dung chan M11)
* Bo ve `max|delta| < 1e-4` khoi dieu kien 1 trong `measure_pocket_shuffle.py`; giu lam chan doan
  co nhan (30.3). Script dang in verdict gay nham.

## Tiep: chay M11-a. Tieu chi giu nguyen Rev 12, doc theo DUNG thu tu (1)-(2)-(3)-(4)
Nhac lai nguong quan trong nhat, da ghi cung:
1. `R2(blind)` thap hon `R2(tokenmean)` qua **0,03** o chieu tinh tien nao ⇒ `blind` chua hoi tu,
   **khong doc tieu chi chinh**.
2. `R2(pocket) - R2(blind) > 0,05` o **>= 2/3** chieu tinh tien ⇒ tien de dung.
   Ca ba **< 0,02** ⇒ chay M11-b.
3. So `delta R2(d)` voi **0,081** (gio co them lan tai lap: **0,0824**; dung **0,081** lam moc nhu
   da ghi cung, khong doi).
4. Quaternion: bao cao, khong dat nguong.

---

# PLAN Rev 14 (2026-10-02) — M11-a lan 1 OVERFIT; sua GIAO THUC, khong sua tieu chi

**M11-a lan 1 khong doc duoc** (RESEARCH_CONTEXT 31): `val R2` dat dinh o **epoch 1** roi suy giam
don dieu toi am o ca 7 chieu, trong khi `train_loss` giam don dieu. Script dung truoc nhanh `blind`.
**Nguyen nhan: thieu giao thuc huan luyen trong dac ta cua toi** (31.4), khong phai loi code.
**Epoch 1 cho `d 0,168 | theta 0,193 | phi 0,201`** ⇒ co tin hieu, nhung chua tach duoc khoi `frag_id`.

## M11-a Rev 2 — hai phep do, chay CA HAI
Tieu chi doc **khong doi** (Rev 12). Chi doi **giao thuc**.

### P1 (CHINH). Linear probe co ridge — khong the overfit theo kieu 31.2
Ly do lam phep chinh: cau hoi la *"thong tin co SAN trong bieu dien khong"*, va chuan muc cho cau
hoi do la **linear probe**. No khong co bang embedding tu do nen khong memorize duoc theo 31.2, va
`lambda` chon tren valid nen khong co van de model selection.

* **Dac trung cho moi token:** `concat[ mean-pool(h_target, mask), max-pool(h_target, mask),
  one-hot-nhe cua frag_id qua mot embedding CO DINH ngau nhien (khong hoc, dim 32, seed 1234),
  one-hot cua vi tri t (clip t <= 15) ]`.
  Embedding `frag_id` **co dinh khong hoc** la co y: no cho mo hinh biet "fragment nao" ma **khong**
  cho phep hoc thuoc pose cua tung fragment.
* **Mo hinh:** ridge regression (closed-form hoac `lstsq`), **mot** ma tran cho ca 7 chieu.
* `lambda` quet `[1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]`, chon theo `R2` trung binh 3 chieu tinh tien
  **tren valid**, bao cao `lambda` da chon.
* Ba nhanh nhu cu: `pocket`, `blind` (zero hoa H+3 kenh), `tokenmean`.

### P2 (PHU). MLP nhu lan 1 nhung co GIAO THUC
* **Early stopping tren valid**: danh gia **moi** epoch, giu checkpoint tot nhat theo `R2` trung binh
  3 chieu tinh tien, patience **10**, toi da **60** epoch.
* **`weight_decay=1e-4`** tren Adam.
* **Bao cao `R2` o epoch TOT NHAT**, khong phai epoch cuoi. In ro epoch nao duoc chon.
* **Preload toan bo cache vao RAM** (733 MB train + 33 MB valid) thay vi `torch.load` tung sample
  ⇒ epoch tu 25 s xuong duoi 1 s (31.5).
* Giu kien truc y het lan 1 de con so so sanh duoc.

### Thay guard Rev 11 bang KIEM GIAO THUC (doc TRUOC moi tieu chi khac)
Guard "R2(d) < 0,30 thi dung" **da huy** o Rev 12 va khong duoc dung lai: pipeline co context tu hoi
quy, M11 khong, nen `R2` thap hon 0,418 la hop le.
Thay bang, cho **ca** P1 va P2:
1. `R2(tokenmean)` phai duoc in **truoc tien**, cho ca 7 chieu. No la san tuyet doi.
2. **P2:** neu epoch tot nhat la **epoch 1** ⇒ mo hinh overfit ngay tu dau ⇒ giam LR 10 lan va chay
   lai **mot** lan; neu van epoch 1 thi **chi doc P1**, bao cao P2 la khong doc duoc.
3. `R2(blind)` thap hon `R2(tokenmean)` qua **0,03** o chieu tinh tien nao ⇒ nhanh `blind` chua hoi
   tu ⇒ khong doc tieu chi chinh (giu nguyen Rev 12).

### Tieu chi chinh — GIU NGUYEN Rev 12, ap cho P1 truoc, P2 doi chieu
* `R2(pocket) - R2(blind) > 0,05` o **>= 2/3** chieu tinh tien (`d, theta, phi`) ⇒ **TIEN DE DUNG**.
* Ca ba **< 0,02** ⇒ thong tin khong co san trong encoder **dong bang** ⇒ **chay M11-b**.
* Trung gian ⇒ bao cao va dung.
* Moc **0,081** (M10): `R2(pocket) - R2(blind)` o `d` lon hon ⇒ pipeline lam mat tin hieu;
  `<= 0,081` ⇒ tran o encoder.
* Quaternion: bao cao, khong dat nguong.
* **Neu P1 va P2 cho ket luan NGUOC nhau** ⇒ khong ket luan; bao cao ca hai va dung.

## Ghi nho ve uoc luong chi phi
Cache da xong (733 MB) nen **khong chay lai** — dung `--rebuild-cache` chi khi `build_h_target` doi.
P1 (ridge tren dac trung da pool) uoc duoi 5 phut. P2 voi preload uoc duoi 10 phut.
Lan nay **khong** uoc thap hon thuc te: moi uoc luong cua toi trong du an deu thap (31.5).

---

# PLAN Rev 15 (2026-10-02) — P2 DAT; sua P1 roi xac nhan; roi sang generator

**P2 dat tieu chi chinh 3/3** (RESEARCH_CONTEXT 32.2): `delta R2` = **0,0999 / 0,0674 / 0,0690**
cho `d / theta / phi`. Giao thuc #3 dat, `best_epoch` 8 va 4, overfit doi xung.
**`delta R2(d) = 0,0999 > 0,081`** ⇒ **pipeline day du dang LAM MAT tin hieu** (32.3).
**P1 void** do loi dac ta thu chin cua toi (32.4): embedding `frag_id` co dinh 32 chieu lam nhanh
`blind` khong the dat san `tokenmean` ⇒ `pocket - blind` bi phong dai.

## M11-a Rev 3 — sua P1, chay lai. Tieu chi doc GIU NGUYEN Rev 12
Muc dich **duy nhat**: co mot giao thuc thu hai, doc lap, de **xac nhan** P2. Khong doi tieu chi.

### Sua dac trung cua P1: thay embedding co dinh bang TARGET ENCODING
Bo embedding ngau nhien co dinh 32 chieu. Thay bang:
* **7 dac trung `tokenmean_pose[frag_id]`** — chinh du doan cua `tokenmean` (trung binh 7D theo
  `frag_id`), tinh **tren train**. Voi `frag_id` khong co trong train, dung trung binh toan cuc.
  ⇒ Nhanh `blind` khi do **>= `tokenmean` theo cau truc**, nen giao thuc #3 khong bi chan gia tao.
  ⇒ Va `pocket` duoc do **tren nen** thong tin danh tinh fragment, nen confound "hoc lam proxy cho
  `frag_id`" bi triet tieu — manh hon ca P2.
* **Chong ro ri bat buoc:** voi cac hang **train**, `tokenmean_pose` phai tinh **out-of-fold**
  (5 fold, seed 1234): dac trung cua hang thuoc fold k tinh tu 4 fold con lai. Voi cac hang
  **valid**, dung trung binh tren **toan bo** train. Khong lam vay thi dac trung tu du doan chinh no,
  lam fit tren train lac quan va lech viec chon `lambda`.
* Giu: `mean-pool(h_target, mask)` [259], `max-pool(h_target, mask)` [259], one-hot `t` (clip 15) [16].
  Tong dac trung = 259+259+7+16 = **541**.
* Ridge nhu cu, quet `lambda` nhu cu, chon theo `R2` trung binh 3 chieu tinh tien tren valid.

### Doc ket qua
Giao thuc #1, #3 va tieu chi chinh **y nguyen** Rev 12/Rev 14.
* **P1 (da sua) va P2 cung ket luan TIEN_DE_DUNG** ⇒ **XAC NHAN**. Sang muc "sau xac nhan".
* **Nguoc nhau** ⇒ khong ket luan; bao cao ca hai va dung.
* **P1 lai bi giao thuc #3 chan** ⇒ bao cao; luc do doc P2 **mot minh** la ket luan duy nhat co, va
  phai ghi ro trong bao cao rang chi mot giao thuc duoc doc.
* **`delta R2(qz)`**: neu **>= 0,05** lan nua ⇒ mo lai Muc 16.1 nghiem tuc, ghi thanh muc rieng.
  Neu **< 0,05** ⇒ 32.5 dong lai, phep quay van la "khong chi phoi boi hoc".

## SAU khi xac nhan — huong di da ro, khong con la gia thuyet
Can cu 32.3: tran **khong** o encoder. Thong tin co san; pipeline khai thac kem hon mot MLP 2 lop.
⇒ Viec tiep theo **khong** phai kien truc tuong bien, **khong** phai 6D rotation, **khong** phai doi
head. Ma la tim cho pipeline **lam mat** tin hieu giua `h_target` va `geometric_head`.

Ba nghi pham, xep theo do re khi kiem:
1. **`nn.Tanh()` cuoi `context_proj`** (Muc 20.8, da ha uu tien, nay len lai): bao hoa ⇒ nghen
   duong tu `h_target` vao head. Do: ty le `|tanh_out| > 0,99`. **Re nhat.**
2. **`v_context.detach()`** (C4, da hoan tu Rev 4): neu gradient hinh hoc khong chay nguoc vao
   attention hub / encoder thi duong dieu kien hoa **khong duoc huan luyen** cho nhiem vu hinh hoc.
3. **Attention hub mot lop + null-sink**: so key = `R_max + 1`; kiem phan phoi trong so attention —
   neu khoi luong don vao null-sink thi `h_target` bi bo qua.

**M11-b (end-to-end, ~15 gio/nhanh): HUY.** Can cu 32.3 — thong tin da chung minh co san trong
encoder **dong bang**, nen cau hoi "encoder co hoc duoc neu duoc day" khong con can thiet.
Tiet kiem ~30 gio GPU.

---

# PLAN Rev 16 (2026-10-02) — TIEN DE XAC NHAN; M12 lam khop estimator truoc khi doi huong

**Tien de XAC NHAN** qua hai giao thuc doc lap (RESEARCH_CONTEXT 33.1). `qz` duoi nguong ⇒ 32.5 dong.
**Con so cong bo lay tu P2** (bao thu hon): `d ~ 0,100 | theta ~ 0,067 | phi ~ 0,069` (33.3).
**CHUA duoc ket luan "pipeline lam mat tin hieu"** — dang so hai estimator khac nhau (33.5).

## M12. Lam KHOP estimator — bat buoc truoc moi quyet dinh kien truc
Hai phep do, ca hai **forward-only**, uoc duoi 10 phut (nhung moi uoc luong cua toi deu thap).

### M12-a. Xao tron hoc luc SUY LUAN tren chinh probe
**File:** `src/probe_pose_regression.py`. Them nhanh thu tu cho **ca P1 va P2**:
* **`pocket_shuf_eval`** — dung **dung mo hinh `pocket` da huan luyen** (P1: ma tran ridge da fit;
  P2: checkpoint tai `best_epoch`), nhung khi **danh gia tren valid** thi xao tron `h_target` theo
  chieu **complex** (derangement, seed 1234), `target_mask` xao tron **cung hoan vi**.
  **Khong huan luyen lai gi.**
* Bao cao `R2(pocket) - R2(pocket_shuf_eval)` cho ca 7 chieu. **Day moi la dai luong cung loai voi
  M10.**

**Tieu chi doc (ghi cung TRUOC khi chay):**
* `R2(pocket) - R2(pocket_shuf_eval)` o `d` **> 0,081** (M10) ⇒ **XAC NHAN pipeline khai thac kem
  hon probe**, cung estimator, so sanh hop le. Luc do moi di theo huong "tim cho pipeline lam mat".
* `<= 0,081` ⇒ **BAC** ket luan 32.3/33.5. Pipeline **khong** kem hon; chenh lech o Muc 32-33 chi la
  hieu ung doi estimator. Luc do huong di **khong** phai sua generator, va phai thiet ke lai buoc tiep.
* Bao cao kem: `R2(pocket) - R2(blind)` (train-ablation) va `R2(pocket) - R2(pocket_shuf_eval)`
  (inference-shuffle) **canh nhau** cho ca 7 chieu, de thay do lech giua hai estimator.

### M12-b. Do bao hoa `nn.Tanh()` — nghi pham #1 (Muc 20.8, nay len uu tien)
**File moi:** `src/measure_tanh_saturation.py`. Checkpoint `periodic-epoch=024.ckpt`, `geo_head="maf"`,
mot batch valid, forward-only `no_grad()` cho phan thong ke; mot `backward()` rieng cho phan gradient.
In:
* ty le phan tu `|tanh_out| > 0,99` va `> 0,999` o `context_proj[-1]` (ca hai head dung chung kien truc)
* `mean/std/max(|pre-activation|)` cua `Linear` ngay truoc `Tanh`
* sau mot `backward()` tren `Geo_Loss`: `grad.norm()` tai `context_proj[0].weight`,
  `context_proj[3].weight`, va lop cuoi cua `dynamic_encoder`

**Tieu chi (ghi cung):** `|tanh_out| > 0,99` **vuot 30%** ⇒ Tanh la nghen that, phai bo truoc moi so
sanh head nao. **Duoi 5%** ⇒ loai gia thuyet 20.8, chuyen sang nghi pham #2 (`v_context.detach()`).
Trung gian ⇒ bao cao, khong ket luan.

### Mo rong day quet `lambda` cua P1 (nho, lam cung luc)
`blind` chon `lambda = 1000` = bien tren cua day (33.3). Them `1e4, 1e5` vao day quet. Khong the doi
ket luan (bien thien 0,0005) nhung phai dung.

## Thu tu
**M12-a va M12-b doc lap, chay ca hai roi doc cung luc.** Khong quyet dinh kien truc truoc do.

## Da HUY (ghi lai de khong quay lai)
* **M11-b** (end-to-end, ~30 gio): huy tu Rev 15 — thong tin da chung minh co san trong encoder dong bang.
* **G5**: huy tu Rev 7.
* **M10 `--split train`**, nghich ly valid>train: huy tu Rev 11.
* **Kien truc tuong bien / 6D rotation / doi head**: **chua** huy nhung **khong** phai buoc tiep theo;
  ca ba chi quay lai bang neu M12-b hoac buoc sau chi ra.

---

# PLAN Rev 17 (2026-10-25) — M12 ghi cung lai TRUOC khi chay; C4 xac nhan tinh

Can cu: RESEARCH_CONTEXT **Muc 34-pre**. Thay the phan M12-a cua Rev 16; M12-b giu nguong, doi ly do.
**Chua co so lieu moi.** M12 chua chay.

## 0. Doc truoc
* **C4 xac nhan tinh (34-pre.2):** moi duong `h_target -> geometric_head` deu qua `v_context.detach()`
  (`step4:99`). Duong dieu kien hoa chi duoc huan luyen boi token loss. **Thu tu nghi pham moi:
  #1 C4, #2 Tanh (ly do hep), #3 null-sink.** Day la thu tu, khong phai thi nghiem moi.
* **34-pre.3 CHO DUYET:** dong nhat thuc `B - A = Var(m) >= 0` du doan nhanh XAC NHAN o tap `all` gan
  nhu chac chan. Muc 2 (3) duoi day chi co hieu luc neu nguoi dung duyet **truoc** khi chay.

## 1. Viec cho Claude Code — 3 commit RIENG, KHONG chay script, KHONG train
Doc: plan Rev 16 + Rev 17; RESEARCH_CONTEXT 33.5, 34-pre.2, 34-pre.4. Chi sua dung 3 file duoi day.

### T1. `src/probe_pose_regression.py`
a. Them `--n-derangements` (int, mac dinh 20). Nhanh `pocket_shuf_eval` chay K lan voi seed
   `SEED + k` (`k = 0..K-1`, tuc 1234..1253) qua `make_derangement_indices` da co. **Khong huan luyen
   lai**: P1 dung lai ma tran ridge `pocket` da fit, P2 dung lai checkpoint `pocket` tai `best_epoch`.
b. Tap token: `all`, `t0` (`positions == 0`, fragment dau tien sau `[SOS]`), `t1` (`positions >= 1`).
   Dung `positions` da co (dong ~242), khong tinh lai (R14).
c. Voi moi probe (P1, P2) va moi tap: in `n_token`; `R2(tokenmean)`, `R2(blind)`, `R2(pocket)` 7 chieu;
   cot A = `R2(pocket) - R2(blind)` 7 chieu; cot B 7 chieu duoi dang `mean ± sd (min, max)` qua K
   (`sd` mau, `ddof=1`). In them tung gia tri `B_k(d)` (K so) cho tap `all` va `t0`.
d. Thay `classify_m12a_criterion` bang tieu chi muc 2 (4) ben duoi, ap cho tap `all`. In verdict tung
   probe, roi verdict tong hop P1/P2 theo muc 2 (6) **nhung chi phan tap `all`** (phan `t0` tro ly doc tay,
   vi can so cua pipeline).
e. Dong 760-764: them tien to `"[RUT LAI -- Muc 33.5] "` vao **ca ba** dong print (dong "moc so sanh 0,081"
   va hai nhanh verdict). **Khong xoa** (R12).

### T2. `src/measure_pocket_shuffle.py`
a. Them `--n-derangements` (mac dinh 1 = hanh vi cu) va `--derangement-seed-base` (mac dinh 1234).
b. Khi K > 1: voi moi `k`, tao `torch.Generator().manual_seed(base + k)`; dung no cho **(i)** thu tu
   loader (`shuffle=True`, `generator=...`) va **(ii)** derangement trong batch (thay `torch.randperm`
   global trong `make_derangement` bang ban nhan `generator`). Ly do (i): loader `shuffle=False` ghep cac
   complex lien ke (gan giong nhau) vao cung batch, lam phep xao tron trong batch nhe hon xao tron toan
   tap cua probe. Moi `k` chay hai nhanh: `full` va `shuffled`. Bo qua `roll1`, `sizematched`, `noop`
   khi K > 1. **Giu** viec dat lai seed nhieu dequant ben trong `run_branch` (M10-1), khong doi.
c. Tap token `all` / `t0` / `t1` theo vi tri trong `target_ids` (cot 0 = fragment dau tien). Voi moi tap:
   `n_token`, `R2(full)` va `delta R2(full - shuffled)` 7 chieu dang `mean ± sd (min, max)` qua K; in
   tung `delta_k(d)` cho `all` va `t0`. R2 dung cong thuc `1 - sigma^2 * E[z^2]` nhu hien tai.
d. Dong 478-486 (dieu kien 1 dung ve `max|delta| < 1e-4`): them tien to `"[RUT LAI -- Muc 30.3] "` vao
   dong in ve `1e-4` va hai dong verdict. Khong xoa.
e. Dong dau log phai co `git rev-parse HEAD`, `SEED`, `device`, `K`, `derangement-seed-base` (R2).

### T3. `src/measure_tanh_saturation.py`
Chi doi **chu in va docstring**, khong doi phep do:
a. Docstring dong 10-12: thay co che "nghen gradient vao h_target/encoder" bang: "Tanh khong the chan
   gradient toi encoder vi `v_context.detach()` (step4:99) da chan truoc. M12-b chi do (a) mat thong tin
   chieu forward, (b) hoc trong so cua chinh `context_proj`."
b. Dong in grad tai `dynamic_encoder`: them `"(DO CAU TRUC: v_context.detach() -- KHONG phai bang chung ve Tanh)"`.

Bay tran da biet (doc RESULTS_LEDGER A8): seed nhieu dequant phai dat lai NGAY truoc forward ben trong
ham chay nhanh; moi so per-dim loc `pad_mask`; `print(..., flush=True)` moi 20 batch.

## 2. Tieu chi M12-a — ghi cung TRUOC khi chay. Doc theo DUNG thu tu
(0) Dong dau ca hai log: `device: cuda`, HEAD = commit cua T1–T3, `K = 20`. Sai ⇒ khong doc.
(1) `R2(tokenmean)` 7 chieu, tap `all`.
(2) Giao thuc #3: `R2(blind) >= R2(tokenmean) - 0,03` o moi chieu tinh tien, tap `all`, tung probe.
    Khong dat ⇒ probe do khong doc tieu chi chinh.
(3) **[CHO DUYET — 34-pre.3]** Kiem dong nhat thuc: `mean B(d) >= A(d) - 2sd`, tap `all`, tung probe.
    Khong dat ⇒ gia dinh "blind hoc dung phan khong-hoc" bi vo ⇒ bao cao, **khong** doc tieu chi chinh
    cho probe do.
(4) **Tieu chi chinh, tap `all`, tung probe**, K = 20, so voi khoang ho M10 `[0,0751 ; 0,0824]`:
    * `mean(B(d)) - 2 sd > 0,0824` ⇒ **XAC NHAN**
    * `mean(B(d)) + 2 sd < 0,0751` ⇒ **BAC**
    * con lai ⇒ **KHONG KET LUAN**
    (SUA tieu chi dac ta sai cua Rev 16, lam truoc khi thay du lieu, hop R9 — 34-pre.4.)
(5) **Tap `t0`**, tro ly doc tay tu hai log: so `B_probe(d, t0)` voi `delta_pipe(d, t0)` (T2, K = 20):
    * `mean_probe - 2sd_probe > mean_pipe + 2sd_pipe` ⇒ XAC NHAN `t0`
    * `mean_probe + 2sd_probe < mean_pipe - 2sd_pipe` ⇒ BAC `t0`
    * con lai ⇒ KHONG KET LUAN `t0`
(6) **Verdict tong hop:**
    * **XAC NHAN** chi khi P1 **va** P2 deu XAC NHAN o (4) **va** deu XAC NHAN o (5).
    * **BAC** khi P1 va P2 deu BAC o (4).
    * P1 va P2 nguoc nhau o bat ky buoc nao ⇒ **KHONG KET LUAN**. Moi truong hop khac ⇒ KHONG KET LUAN.
(7) Cau chu cua nhanh XAC NHAN: **"pipeline khai thac hoc kem hon probe o cung estimator"** — khong bao
    gio "pipeline lam mat tin hieu". Phai neu giai thich lanh tinh: thong tin du thua, va hai mo hinh
    huan luyen duong dieu kien hoa bang hai muc tieu khac nhau (34-pre.2 muc 5).
(8) Kiem cheo, khong phai cong chan: `delta_pipe(d, all)` cua T2 nen roi gan `[0,0751 ; 0,0824]`. Lech
    nhieu ⇒ ghi lai (doi loader shuffle co the doi do lon), khong doi tieu chi.
(9) Quaternion: bao cao, khong nguong.

## 3. M12-b — nguong GIU NGUYEN, ly do hep lai
`|tanh_out| > 0,99` vuot 30% ⇒ nghen that; duoi 5% ⇒ loai; trung gian ⇒ khong ket luan.
**Du doan cau truc ghi cung (34-pre.2 muc 4):** grad tai `context_proj[0]`, `context_proj[3]` khac
`None`; grad tai `dynamic_encoder` = `None`/0. **Neu grad encoder KHAC 0 ⇒ kiem tinh C4 sai ⇒ dung,
bao cao, khong doc gi khac.**

## 4. Lenh server (sau khi Claude Code push T1–T3)
```bash
cd ~/SAMG && git pull && git log --oneline -5 && conda activate samg
readlink -f ~/SAMG; ls -la logs/m12b_tanh.log 2>&1
rm -f logs/m12b_tanh.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/probe_pose_regression.py --n-derangements 20 2>&1 | tee logs/m12a_probe_rev17.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/measure_pocket_shuffle.py --split valid --n-derangements 20 2>&1 | tee logs/m12a_pipe_t0_rev17.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/measure_tanh_saturation.py 2>&1 | tee logs/m12b_tanh.log
```
Chi phi T2: 2 nhanh x 20 lan x het valid. **Khong uoc** (R13: moi uoc luong cua tro ly deu thap).
Theo doi bang dong tien do moi 20 batch.

---

# PLAN Rev 18 (2026-10-26) — sau M12: (a) vi sao M10 do thap, (b) loi ich MSE thay cho B

Can cu: RESEARCH_CONTEXT **Muc 34**. M12-a **KHONG KET LUAN**; M12-b **LOAI** Tanh. Thu tu nghi pham Rev 17
(C4/Tanh/null-sink) **tam dung**: chua co bang chung pipeline dung hoc kem (34.3 muc 4), nen chua di "tim cho mat tin hieu".
Tro ly code + commit (WORKING_RULES F). Ca hai phep do duoi day **forward-only / doc du lieu**, khong train.

## (a) Ghep cap trong batch khi loader khong xao — `src/check_valid_adjacency.py` (MOI, khong GPU, khong model)
Dung `SAMGOptimizedDataset(split_mode="valid")` (cung loc `frame_stable`) va **dung thu tu `valid_indices`** = thu tu
loader `shuffle=False`. PDB = `pdb_id_of(pli_id)` import tu `check_split_leakage.py` (R14). Tam hoc = `pocket_t` trong pkl.
"Cung hoc" = cung PDB **va** `||pocket_t_i − pocket_t_j|| < 2,0 A` (bao cao them 1,0 va 4,0 A).
Voi batch lien tiep co kich co `bs` in {16 (M10), 8}: `s(bs) = trung binh_i [ so thanh vien KHAC cung batch cung hoc /
(n_batch − 1) ]` (= xac suat cap xao tron cung hoc, vi derangement chon doi tac deu trong batch).
Bao cao `s_pdb(bs)`, `s_pocket(bs)`, va **moc ngau nhien** `s_rand` = cung dai luong nhung doi tac la moi complex khac
trong toan valid. In them: so PDB duy nhat, phan phoi so complex/PDB.

**Tieu chi ghi cung.** Gia dinh: cap cung hoc cho `delta ≈ 0`, cap khac hoc cho `delta` day du ⇒ du doan
`delta_M10 ≈ (1 − s_pocket(16)) × 0,2160`. Can `s ≈ 0,63` de ra 0,079 (giua khoang ho M10).
* `s_pocket(16) >= 0,50` **va** du doan roi trong `[0,0451 ; 0,1124]` (khoang ho M10 ± 0,03) ⇒ **XAC NHAN**: thu tu
  loader giai thich su do thap cua M10.
* `s_pocket(16) < 0,20` ⇒ **BAC**: thu tu khong phai nguyen nhan chinh; phai tim nguon khac.
* con lai ⇒ mot phan, **khong ket luan**.
Gioi han ghi truoc: PDB chi la can duoi cua "cung protein" (cung protein khac ma PDB khong bat duoc).

## (b) Loi ich MSE `G = B − Var(m)` — sua `probe_pose_regression.py` va `measure_pocket_shuffle.py`
Voi moi k: `Var_k(m) = E[(pred_pocket − pred_shuf_k)^2] / 2` tung chieu; `G_k = B_k − Var_k(m)`. Bao cao `mean ± sd`
qua K cho tap `all`/`t0`/`t1`, 7 chieu, canh A va B.
Pipeline: dung `mu` cua hook; them ban **MSE**: `R2_mse = 1 − E[(t − mu)^2]`, `B_mse = MSE(shuf) − MSE(full)`,
`Var(m) = E[(mu_full − mu_shuf)^2]/2`, `G = B_mse − Var(m)`. Giu ca B ban `1 − sigma^2 E[z^2]` de doi chieu.
Lenh chay lai **y het** M12 (K = 20, seed 1234..1253, pipeline `--batch-size 8`).

**Tieu chi ghi cung, doc theo thu tu:**
(b0) **Kiem dong nhat thuc tren probe:** `|mean G(d) − A(d)| <= 0,03`, tap `all`, tung probe. Khong dat ⇒ G khong uoc
     luong duoc loi ich train-ablation cho probe do ⇒ probe do **khong** dung o (b2).
(b1) Pipeline: `|B_mse(d) − B(d)| > 0,03` (tap `all`) ⇒ ghi lai: hai dinh nghia R2 lech, **(b2) chi doc ban MSE**.
(b2) **Chinh**, tap `all`, so `G_pipe(d)` voi `G_probe(d)` (mean ± 2sd ca hai) cho moi probe qua (b0):
     * `G_pipe + 2sd < G_probe − 2sd` voi **moi** probe qua (b0) ⇒ **"pipeline thu duoc it loi ich MSE tu hoc hon probe"**
       (cau chu mem, giong Rev 17 (7)). Chi khi do moi mo lai huong C4 / null-sink.
     * `G_pipe − 2sd > G_probe + 2sd` voi **moi** probe qua (b0) ⇒ **BAC** 32.3 dut diem: pipeline thu duoc it nhat ngang
       probe. Huong "sua cho pipeline lam mat tin hieu" **dong lai**.
     * Probe nam hai phia, chong lan, hoac khong probe nao qua (b0) ⇒ **KHONG KET LUAN**.
(b3) Tap `t0`: doc cung luat, chi bao cao. Neu `t0` nguoc verdict `all` ⇒ verdict cuoi **KHONG KET LUAN**.
(b4) Quaternion: bao cao `G(qw)`, `G(qz)`; khong nguong (34.3 muc 3 se viet nguong rieng sau (b)).
**R5:** neu pipeline khai thac ngang probe thi `G_pipe(d) ≈ A_probe(d) ≈ 0,09–0,10` (all) — dat duoc. Neu (b0) dat,
`G_probe` ≈ `A` nen moi nhanh deu co the xay ra.

## Lenh server
```bash
cd ~/SAMG && git pull && git log --oneline -3 && git status --short src/ && conda activate samg
readlink -f ~/SAMG
python -u src/check_valid_adjacency.py 2>&1 | tee logs/m13a_adjacency.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/probe_pose_regression.py --n-derangements 20 2>&1 | tee logs/m13b_probe.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/measure_pocket_shuffle.py --split valid --n-derangements 20 --batch-size 8 2>&1 | tee logs/m13b_pipe.log
```
Ket qua ghi vao **Muc 35**.

---

# PLAN Rev 19 (2026-10-27) — (a') "cung protein" theo chuan nganh (R16)

Can cu: RESEARCH_CONTEXT **Muc 35**. (a) cua Rev 18 chi bac duoc mot dinh nghia tu dat (35.1). Lam lai voi dinh
nghia co nguon. Khong GPU, khong model. Tro ly code + commit (WORKING_RULES F).

## (a') Xac suat cap derangement trong batch la "cung protein"
**Dinh nghia va nguon (R16):**
* "Cung protein" = **cung cluster MMseqs2 o 30% do trung chuoi** — chuan split cua CrossDocked2020 duoc Frag2Seq
  (PDF, phu luc du lieu: "split ... according to 30% sequence identity using MMseqs2") va 3DMolFormer (PDF, "< 30%")
  su dung. Lenh: `mmseqs easy-cluster --min-seq-id 0.3`; **cac tham so khac de mac dinh** (hai bai khong neu ⇒
  ghi ro la quy uoc). Bao cao them nguong 0,95 (gan "cung protein" theo nghia hep) — **quy uoc du an**, chi tham khao.
* Chuoi dung de cluster = chain dong gop nhieu residue nhat vao hoc `receptor_holo_pocket10.pdb` (hoc = residue trong
  10 A quanh ligand, dinh nghia cua Apo2Mol, PDF muc 5), lay tu `receptor_holo_prot.pdb` (entry[3]).
* Thu tu = `valid_indices` (loader `shuffle=False`), giong Rev 18 (a). `s(bs)` dinh nghia nhu Rev 18 (a).

**Tieu chi — nguong suy tu DU LIEU (R16 muc 2), ghi cung truoc khi chay:**
Gia dinh (quy uoc du an, R16 muc 3, se kiem nhan qua neu can): cap cung protein cho `delta ≈ 0`.
Du doan `delta_M10 = (1 − s_seq30(16)) × 0,2156` (Muc 35, `delta_pipe(d, all)`). Sai so dung `sd(delta_pipe) = 0,0185`
do duoc qua K = 20. Khoang tuong thich = ho M10 ± 2sd = **[0,0381 ; 0,1194]** ⇔ `s` trong **[0,446 ; 0,823]**.
* `0,446 <= s_seq30(16) <= 0,823` ⇒ **TUONG THICH**: thu tu loader ghep cung protein du giai thich M10 do thap.
  Chua phai bang chung nhan qua — buoc sau (neu lam) la do `delta` rieng cho cap cung/khac cluster tren GPU.
* `s_seq30(16) < 0,446` ⇒ **BAC**: cung protein khong du giai thich; phai tim nguon khac.
* `s_seq30(16) > 0,823` ⇒ **KHONG KHOP**: gia dinh "cung protein ⇒ delta ≈ 0" sai mot phan; **khong ket luan**.
Bao cao them `s_rand` (doi tac ngau nhien toan valid) va so cluster.

## Cai dat MMseqs2 tren server (khong dung vao env `samg`)
```bash
mkdir -p ~/tools && cd ~/tools && wget -q https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz && tar xzf mmseqs-linux-avx2.tar.gz
~/tools/mmseqs/bin/mmseqs version
```
Neu CPU khong co AVX2: dung `mmseqs-linux-sse41.tar.gz`. Ghi `mmseqs version` vao log.

## Lenh server
```bash
cd ~/SAMG && git pull && git log --oneline -3 && git status --short src/ && conda activate samg
readlink -f ~/SAMG
python -u src/check_valid_adjacency.py --mmseqs ~/tools/mmseqs/bin/mmseqs 2>&1 | tee logs/m14a_adjacency_seqid.log
```
Ket qua ghi vao **Muc 36**.

---

# PLAN Rev 20 (2026-10-27) — BASELINE truoc, huan luyen lai sau

Can cu: RESEARCH_CONTEXT **Muc 36** (valid 131 cluster) va **Muc 37-pre** (Step 7 co 3-4 loi nang; phep nghich
dao da xac dinh). Nguoi dung chot muc tieu cuoi = **chat luong phan tu sinh ra**, dong y viet lai step 0-6 khi
da ro cho nao doi.

**Nguyen tac thu tu (ly do, khong phai so thich):** moi so Vina/QED hien co do chat luong ETKDG chu khong phai
cua mo hinh (37-pre.2 L1). Huan luyen lai bay gio ⇒ khong doc duoc ket qua. **A truoc, C sau.**
A va B doc lap, chay song song duoc.

---

## A. Sua Step 7 + do BASELINE cua checkpoint hien tai

### A1. `src/build_fragment_dict.py` (MOI) — tu dien toa do cuc bo cua fragment
**Nguon:** Frag2Seq muc 3.3.1 — tu dien key = canonical SMILES cua fragment, value = atom types + toa do cuc bo
trong frame fragment (37-pre.3).
Duyet lai SDF cua **split train** (khong dung valid/test: tu dien la tham so hoc tu du lieu). Voi moi fragment:
* dung lai **nguyen van** `LigandProcessor.fragment_ligand_frag2seq` cho toi buoc co `coords_canon` — **import**
  tu `step0_data_prep`, khong viet lai logic cat/sap (R14). Neu ham khong lo ra `coords_canon`, tach mot ham
  `_fragment_raw()` trong step0 va goi lai — **khong copy code**.
* `R_g_to_w, t_g_to_w = _get_local_frame(coords_canon)`; `local = R_g_to_w.T @ (coords_canon - t_g_to_w)`.
* luu `{smiles: {"atom_types": [...], "local": [n,3], "n_seen": int, "rmsd_spread": float}}`.
* Voi SMILES xuat hien nhieu lan: giu conformer **dau tien** (quy uoc Frag2Seq), va tinh `rmsd_spread` =
  RMSD trung binh (sau Kabsch, tren nguyen tu da sap canonical) giua cac lan xuat hien va conformer da giu.
Luu `dataset/processed/fragment_dict.pkl`.

**Tieu chi doc (ghi cung):**
* In: so SMILES duy nhat, `%` token cua **valid** co trong tu dien (do phu), phan vi 50/90/99 cua `rmsd_spread`.
* `rmsd_spread` median **> 0,5 A** ⇒ xap xi "mot conformer moi SMILES" la nguon loi dang ke ⇒ ghi vao bao cao
  va **khong** quy ket sai lech Vina cho mo hinh. (Nguong 0,5 A: nua cua 1,0 A — nguong RMSD "tai dung thanh cong"
  cua PoseBusters/ngành docking; **quy uoc du an** suy tu do, R16 muc 3.)
* Do phu token valid **< 95%** ⇒ bao cao, va moi so Vina chi tinh tren phan tu co **moi** fragment trong tu dien.

### A2. `src/step7_evaluation.py` — viet lai duong dat pose (hang muc B7)
Sua dung ba loi 37-pre.2:
1. **Bo ETKDG khoi duong dat pose.** ETKDG **van duoc dung** cho hinh hoc **noi** fragment neu mot SMILES khong co
   trong tu dien (ghi ro ty le). Dat pose thi dung phep nghich dao 37-pre.3.
2. **Khong trung binh quaternion.** Moi fragment dat theo **quaternion cua chinh no**.
3. **Dung frame hoc:** `pocket_R`, `pocket_t` doc tu pkl (da co), **khong** tinh lai (R14), **khong** dung
   `protein_pos.mean()`.
4. **Lap rap phan tu:** noi cac fragment tai nguyen tu dummy. Buoc nay chua co dac ta — xem B2 duoi day.
**Kiem khong-doi bat buoc (R15):** voi **nhan that** cua 20 complex valid (khong phai du doan cua mo hinh), chay
phep nghich dao roi so voi SDF goc. **`RMSD < 0,5 A` o >= 18/20** ⇒ duong tai dung dung. Khong dat ⇒ **DUNG**,
phep nghich dao sai, khong doc so Vina nao. (Nguong 0,5 A nhu A1; day la kiem **tai dung**, khong phai do mo hinh.)

### A3. Danh gia theo chuan Apo2Mol
`--num_samples 5` moi hoc tren **test** (478 entry, 255 sau `frame_stable`), Vina `mode='minimize'`,
`exhaustiveness=128`. Bao cao **mean va median** cua: Vina min, QED, SA, logP, Lipinski; cong **High Affinity**,
**Validity**, **Novelty** (Tanimoto <= 0,4, Morgan). Dung tien ich cua `Apo2Mol/utils/evaluation/` (R14).
**Doi chung bat buoc, ghi cung:** cung bang so cho **ligand tham chieu** cua chinh cac hoc do, va cho
**nhan that qua duong tai dung A2** (= tran tren cua mo hinh hien tai voi tu dien fragment nay).
**Khong dat nguong PASS/FAIL cho mo hinh o buoc nay** — day la **baseline**, khong phai phep kiem gia thuyet.
Ghi so, khong ket luan.

---

## B. Uoc luong sai so theo CUM PROTEIN cho cac ket luan da co

### B1. Cluster bootstrap
**Nguon:** Field & Welsh (2007), *Bootstrapping clustered data*, JRSS-B 69(3) 369-390 — khi du lieu phan cum,
bootstrap phai **lay mau lai TOAN BO CUM** (cluster/"pairs" bootstrap), khong lay mau lai quan sat le; lay mau le
cho khoang tin cay **qua hep**. Cum o day = 131 cluster MMseqs2 30% cua valid (Muc 36.1).
Ap cho: **A6** (`delta R2` cua P1/P2), **(b2)** (`G_pipe` vs `G_P1`), **quan sat t1** (Muc 35.2).
Cai dat: hai script M12 in them **du doan va target theo tung complex**; mot script moi
`src/cluster_bootstrap.py` doc file do + nhan cluster, lay mau lai 131 cluster co hoan lai, **B = 2000** lan,
bao cao khoang phan vi 2,5%-97,5%.
**Tieu chi doc (ghi cung):** voi moi ket luan, so khoang bootstrap cum voi khoang ±2sd da dung:
* Khoang bootstrap **khong chua 0** (hoac khong chua moc so sanh) ⇒ ket luan **giu nguyen**, ghi them khoang moi.
* Khoang bootstrap **chua 0** trong khi ±2sd thi khong ⇒ ket luan do **[XEM LAI]**: bat dinh theo protein lon hon
  bat dinh theo lan xao tron.
* Bao cao ty le `do rong bootstrap / do rong 2sd` cho tung ket luan.

### B2. (nho) Doc cach Frag2Seq lap rap fragment
Doc phan tai dung trong Frag2Seq PDF (Apx., phan chung minh surjective) de lay dac ta noi nguyen tu dummy.
Ghi vao Muc 37 truoc khi code A2 buoc 4.

---

## C. Huan luyen lai — CHI sau khi co A3 va B
Danh sach ung vien (deu dang HOAN, moi muc phai co nguon truoc khi dua vao):
`P3a-1` (co dinh permutation ARN) · `P3a-3` (mask padding `_ls_sum`) · `C4` (bo `v_context.detach()`) ·
`N1` (thu tu fragment theo alphabet SMILES — Frag2Seq dung **thu tu xuat hien trong canonical SMILES**, muc 3.2,
**khac** voi `sort(key=smiles)` cua step0:230 ⇒ co nguon de doi) · **6D rotation** (double cover) ·
`tau` tuning · `N2` (`updated_protein_pos`).
**Khong chot gi o Rev 20.** Rev 21 se chot sau khi doc A3 + B.

---

## Lenh server (A1/A2 chua co code — chay khi bao)
```bash
cd ~/SAMG && git pull && git log --oneline -3 && git status --short src/ && conda activate samg
readlink -f ~/SAMG
python -u src/check_step0_tokens.py 2>&1 | tee logs/m15_step0_tokens.log
```
`check_step0_tokens.py` (Rev 20, tro ly viet) kiem **hai** nghi van, deu vai giay, khong GPU:
* **L4** (37-pre.2): ghep chuoi SMILES hai fragment roi `.replace("*","C")` — parse ra phan tu gi.
* **37-pre.5**: token fragment co mang chi so nguyen tu khong.

**Tieu chi doc, ghi cung TRUOC khi chay:**
* **37-pre.5:** goi `U` = so token duy nhat sau khi thay moi `[N*]` bang `[*]`, `V` = so token.
  `U / V < 0,50` ⇒ **XAC NHAN**: token mang chi so nguyen tu ⇒ **bat buoc sua `step0`** (truyen `dummyLabels`)
  truoc moi lan huan luyen lai, va tu dien A1 phai dung key da xoa chi so.
  `U / V > 0,90` ⇒ **BAC**: tu vung la fragment hoa hoc thuan, step0 khong can sua cho viec nay.
  Giua ⇒ bao cao, khong ket luan (co the mot phan fragment co dummy, mot phan khong).
* **L4:** in `MolFromSmiles` co parse duoc khong, `MolToSmiles` va `GetNumAtoms`. Khong co nguong —
  ghi nhan de xac dinh phan tu duoc danh gia tu truoc den nay thuc su la gi.

Ket qua ghi vao **Muc 37**.

---

## A0 (DA CODE, chay truoc A1). Chay lai `step0` voi S1-S4 — ~2,5 gio
Bon thay doi S1-S4 va **kiem khong-doi K1-K4** ghi cung o RESEARCH_CONTEXT **37.7**. Doc K1-K4 **truoc** khi
chay bat ky thu gi khac; K1 hoac K3 khong dat ⇒ **DUNG**, S1-S4 da lam doi nhan.
```bash
cd ~/SAMG && git pull && git log --oneline -3 && git status --short src/ && conda activate samg
# BACKUP truoc khi ghi de (plan Phase 1 P1e)
cp -r dataset/processed dataset/processed_backup_prevocab_$(date +%Y%m%d)
cp dataset/processed/global_vocab.pkl /tmp/global_vocab_OLD.pkl
cp dataset/processed/scaler_7d.pt     /tmp/scaler_7d_OLD.pt
# chay lai step0 (~2,5 gio) + scaler
python -u src/step0_data_prep.py 2>&1 | tee logs/m16_step0_rerun.log
python -u src/compute_scaler.py  2>&1 | tee logs/m16_scaler.log
# K1-K4
python -u src/check_step0_rerun.py --old-scaler /tmp/scaler_7d_OLD.pt --old-vocab /tmp/global_vocab_OLD.pkl \
  2>&1 | tee logs/m16_check.log
```
Sau khi K1-K4 **dat**: `python -u src/build_fragment_dict.py 2>&1 | tee logs/m16_fragdict.log` (A1).
Ket qua ghi vao **Muc 38**.

---

# PLAN Rev 20 — bo sung sau A1' (2026-10-27)

**Do phu TU VUNG = 3.981/3.981 = 1,0000** (Muc 38.3) ⇒ moi token mo hinh co the sinh ra deu co template.
Do phu valid 95,05% / phan tu 73,30% **khong** phai loi: valid co fragment chua tung thay trong train, chung
phai roi vao `[UNK]` (Frag2Seq 3.3.1, vocab train-only).

## A1b (BAT BUOC, ~20 phut). Khoi phuc `pocket_R`/`pocket_t`
Can cu Muc **38.6**: chay lai step0 da xoa hai truong nay ⇒ step6 se loi KeyError.
```bash
python -u src/add_pocket_frame_to_pkl.py 2>&1 | tee logs/m18_pocket_frame.log
```
**Doc:** script tu bao so file xu ly va so file `_get_pocket_frame` cho ket qua khac lan step0 (`n_mismatch`).
`n_mismatch > 0` ⇒ **DUNG**: frame hoc khong tai lap duoc giua hai lan chay, phai tim nguyen nhan truoc khi di tiep.

## A2a (BAT BUOC truoc khi sua Step 7). Kiem phep nghich dao tu NHAN THAT
Khong mo hinh, khong tu dien fragment (dung `local_coords` theo tung lan xuat hien). Tieu chi ghi cung:
**RMSD < 0,5 A o >= 18/20 complex**. Kem chan doan cho duoi `rmsd_spread` (Muc 38.2).
```bash
python -u src/check_reconstruction.py --split valid --n 20 --diag-smiles '*O*' 2>&1 | tee logs/m18_recon.log
```
**Doc:** DAT ⇒ phep nghich dao dung, viet lai Step 7 (A2b). KHONG DAT ⇒ **DUNG**, khong doc so Vina nao.
Chan doan: `max|d sorted| ~ 0` nhung `kabsch` lon ⇒ hoan vi thu tu nguyen tu (vo hai).
`max|d sorted|` LON ⇒ hinh hoc that su khac ⇒ truy tiep truoc khi di.

Ket qua ghi vao **Muc 39**.

---

# PLAN Rev 20 — bo sung sau A2a (2026-10-27)

**A2a DAT tuyet doi: RMSD = 0,0000 o 20/20** (Muc 39.2). Bieu dien 7D khong mat thong tin nao.
Chan doan: `rmsd_spread` la bien thien conformer THAT, khong phai hoan vi thu tu (39.3) ⇒ tu dien phai lay
**medoid**, khong lay conformer dau tien (39.4, da code).
**P3a-1 va P3a-3 da duoc ap dung tu truoc** (39.5) ⇒ huan luyen lai khong con gi chan.

## Thu tu moi (hai nhanh SONG SONG)
| nhanh | viec | chi phi |
|---|---|---|
| **GPU** | A1c (dung lai tu dien, medoid) → **C: `step6` huan luyen lai** | ~1 phut + ~19 gio |
| **CPU (tro ly viet code)** | A2b: `step7_generate_eval.py` — sinh mau + dat pose + danh gia chuan | — |

### A1c. Dung lai tu dien voi medoid (~1 phut)
```bash
python -u src/build_fragment_dict.py 2>&1 | tee logs/m19_fragdict_medoid.log
```
**Doc (ghi cung):** `do phu TU VUNG` phai van **= 1,0000**. `ngoai lai hinh hoc (span > 1,8 x n_atom)` phai
**= 0**; neu > 0 thi medoid van dinh conformer hong ⇒ bao cao, va fragment do phai bi loc truoc khi sinh mau.
(1,8 A moi lien ket la can tren rat rong rai cho lien ket cong hoa tri — quy uoc du an, R16 muc 3.)

### C. Huan luyen lai `step6` (~19 gio)
**Chi ap cac sua da co; KHONG them C4, KHONG 6D rotation** (39.5: ca hai la thi nghiem, se khong tach duoc
nguyen nhan). Tu vung moi (~3.983), nhan 7D khong doi, `pocket_R`/`pocket_t` da khoi phuc.
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/step6_trainer.py 2>&1 | tee logs/m19_train.log
```
**Kiem o dong dau log (R2):** `SEED`, `git rev-parse HEAD`, `permutation` thuc te cua ARN, `vocab_size`.
`vocab_size` phai ~3.983, **khong** phai 19.033 — neu van 19.033 thi loader doc vocab cu, **DUNG**.

Ket qua ghi vao **Muc 40**.

---

# PLAN Rev 20 — sua sau khi C that bai lan 1 (2026-10-28)

Can cu RESEARCH_CONTEXT **40-pre**. Hai loi: step6 tu dong resume checkpoint cu (tu vung lech); `geo_head`
mac dinh `diag_gauss` trong khi moi cong cu do doc MAF. Da sua ca hai.

## C (lan 2). Luu tru checkpoint cu, roi huan luyen tu dau voi MAF
```bash
cd ~/SAMG && git pull && git log --oneline -2 && conda activate samg
# Luu tru, KHONG xoa (R12): checkpoint cu thuoc tu vung 19.033, moi so Muc 24-35 gan voi chung
mkdir -p saved_checkpoints_flow_OLD_vocab19033
mv saved_checkpoints_flow/*.ckpt saved_checkpoints_flow_OLD_vocab19033/
ls saved_checkpoints_flow_OLD_vocab19033/ | head

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/step6_trainer.py --geo-head maf \
  2>&1 | tee logs/m20_train_maf.log
```
**Kiem o dau log, truoc khi de chay qua dem:**
* `vocab_size: 3983` (neu 19.033 ⇒ dang doc vocab cu, DUNG)
* `geo_head: maf  permutation ARN: tensor([0,1,2,3,4,5,6])` (P3a-1)
* `[*] Huan luyen TU DAU (khong resume).`
* `Total valid Train complexes: 17200` / `Val: 764`
Sau ~2 phut phai thay epoch 0 chay va `Token_Loss` giam. **Token_Loss dau ky vong thap hon han lan truoc**
(~ln(3983) = 8,29 thay vi ~ln(19033) = 9,85 khi doan deu) — day la **bao cao**, khong phai cong chan.

Ket qua ghi vao **Muc 40**.

---

# PLAN Rev 20 — sau khi C (lan 2) chay xong (2026-10-29)

Huan luyen 25 epoch da xong tren tu vung moi (Muc 40). Truoc khi sinh mau: **do tran cua duong lap rap**.

## A2b. Tran lap rap (khong mo hinh, ~10-20 phut) — CONG CHAN
```bash
cd ~/SAMG && git pull && git log --oneline -2 && conda activate samg
python -u src/check_assembly_ceiling.py --split test 2>&1 | tee logs/m21_ceiling_test.log
```
Tieu chi ghi cung o RESEARCH_CONTEXT **40.1**: `recon_success(gt_exact) >= 0,90` ⇒ di tiep;
`< 0,70` ⇒ **DUNG**, lap rap la nut that.
Hang `gt_template` la **tran tren cua mo hinh** — moi so cua mo hinh ve sau phai doc **so voi hang nay**,
khong phai so voi `reference`.

## Can gui kem
1. Phan duoi `logs/m20_train_maf.log` (quy dao loss, epoch tot nhat, ten checkpoint).
2. `logs/m19_fragdict_medoid.log` (do phu TU VUNG phai = 1,0000; ngoai lai hinh hoc phai = 0).

## A2c (viet sau khi A2b dat). `step7_generate_eval.py`
Sinh 5 phan tu moi hoc (Apo2Mol `--num_samples 5`), dat pose bang nghich dao 37-pre.3 + template medoid,
lap rap bang `reconstruct_from_generated`, danh gia 6 chi so Apo2Mol + Validity/Novelty.
**Van de do dai chuoi (Muc 40.2): vocab KHONG co `[EOS]`** ⇒ lay do dai tu phan phoi thuc nghiem cua train.
Ghi `[EOS]` vao Rev 21.

Ket qua ghi vao **Muc 41**.

---

# PLAN Rev 20 — sua A2b sau khi phat hien server khong co Apo2Mol (2026-10-29)

Can cu **40.4**. Bo hoan toan phu thuoc Apo2Mol cho buoc lap rap + chi so hoa hoc; viet
`src/assemble.py` cua du an. Chi buoc **Vina** con can `meeko`/`vina`/`AutoDockTools`, xu ly sau.

```bash
export SAMG=/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG
cd $SAMG && git pull && git log --oneline -1 && conda activate samg
# chay thu 20 complex truoc (~1 phut) de bat loi som
python -u src/check_assembly_ceiling.py --split test --n 20 2>&1 | tee logs/m21_ceiling_smoke.log
# neu khong no: chay het test
python -u src/check_assembly_ceiling.py --split test 2>&1 | tee logs/m21_ceiling_test.log
```
Tieu chi T1/T2/T3 giu nguyen (Muc 40.1). Doc them ba chan doan lap rap: so lien ket lien-fragment,
so dummy mo coi, ty le phan tu lien thong.
