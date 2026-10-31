# RESEARCH CONTEXT: SAMG (3D Molecular Generative Model)

> **Rev 3 — 2026-09-29.** Bản này THU HỒI phần lớn Mục 7 và hạ cấp H2′ của Rev 2.
> Lý do: phát hiện một lỗi ghép dữ liệu khiến mô hình chưa từng nhìn thấy cặp
> (hốc, phân tử) đúng, và một loạt khiếm khuyết cấu trúc trong biểu diễn 7D.
> Backup Rev 2: `RESEARCH_CONTEXT_20260928_PRE_REV3.md`.
>
> **Trạng thái các phát hiện Rev 3 (đọc kỹ trước khi hành động):** mọi mục đánh dấu
> `[ĐÃ ĐO]` đều kèm script tái lập ở Mục 10. Mục đánh dấu `[CHƯA ĐO]` là lập luận từ
> đọc code, chưa có số — **phải xác minh trước khi dùng làm căn cứ sửa code.**

---

## 0. Bản đồ hiệu lực sau Rev 3

| Mục Rev 2 | Trạng thái Rev 3 |
|---|---|
| Mục 2 "bỏ RDKit ETKDG, xây 3D Assembler" | **SAI SỰ THẬT** — code vẫn gọi `EmbedMolecule(ETKDG())`. Xem Mục 7. |
| Mục 3 (log cũ không dùng được) | **Còn hiệu lực**, và mạnh hơn: xem Mục 4. |
| Mục 4.1 (standardization no-op) | Còn hiệu lực. Đã sửa ở P0-3. |
| Mục 4.2 (suy biến 7D) | **Đúng nhưng ước lượng thấp gần 3 lần.** Thay bằng Mục 6. |
| Mục 4.3 (`L_Int` gradient 0) | Còn hiệu lực. Đã tắt ở P0-6(b). |
| Mục 4.5 (`v_context.detach()`) | Còn hiệu lực, nhưng **không phải** nguyên nhân chính. Xem Mục 8. |
| Mục 5 H2′ | **HẠ CẤP** — tiền đề đúng một phần, cơ chế chưa kiểm được, và có giải thích đơn giản hơn. Xem Mục 6.1. |
| Mục 7 (P1 baseline, 25 epoch) | **THU HỒI** — đo trên bài toán không có tín hiệu. Xem Mục 5. |

---

## 1. Project Overview & Architecture
SAMG sinh phân tử 3D (Ligand) dựa trên hình dáng hốc protein (Target Pocket), 7 bước:
* **Step 0 – Data Prep:** Tiền xử lý Apo/Holo/Ligand thành chuỗi 7D, phân mảnh kiểu
  BRICS (vocab 15.626 fragment). **Thứ tự 7 chiều khi GHI ra pkl là
  `[d, theta, phi, qw, qx, qy, qz]`** (`step0_data_prep.py:121`) — xem Mục 6.3.
* **Step 1 – Protein Encoder:** DynamicEGNN cho Target (dự đoán biến dạng),
  StaticEGNN cho Anti-target (hiện đang comment).
* **Step 2 – Geometric Module:** `ConditionalAffineAutoregressive` 1 layer (Pyro).
* **Step 3 – Attention Hub:** Multi-Differential Cross-Attention.
* **Step 4 – Ligand Generator:** Dual-Stream. Transformer → SMILES; Geometric head → 7D.
* **Step 5 – Global Loss:** `L_token + L_geo + L_pocket + L_int`.
* **Step 6 – Trainer:** PyTorch Lightning.
* **Step 7 – Evaluation:** "3D Assembler" + Vina / QED / SA.

**Phạm vi hiện tại:** anti-target tạm hoãn (`list_h_anti = []`). Ưu tiên là chạy
được đường target-only trước. **Lưu ý: mọi lỗi trong Mục 4–8 đều nằm trên đường
target-only, nên việc hoãn anti-target không giải phóng bất kỳ mục nào.**

## 2. Solved Issues (có sửa lại so với Rev 2)
* **Zombie Processes (VRAM):** quản lý qua `tmux`, dời sang GPU trống. *(cao)*
* **Chemical Validity:** `temperature=0.8` khi sampling + `v_context.detach()`. *(cao,
  nhưng detach có tác dụng phụ kiến trúc — Mục 8)*
* ~~**Vina Docking Failures: bỏ RDKit `ETKDG()`, xây 3D Assembler ở Step 7.**~~
  **RÚT LẠI.** `step7_evaluation.py:333` vẫn gọi `AllChem.EmbedMolecule(mol, ETKDG())`
  rồi `MMFFOptimizeMolecule`. Xem Mục 7.
* **Exploding Pocket Loss:** MSE → Smooth L1. *(THẤP — bằng chứng từ log batch đơn lẻ)*
* **Baseline Establishment:** tắt anti-target. *(cao)*

## 3. Measurement Integrity — log cũ KHÔNG dùng được
Giữ nguyên từ Rev 2, vẫn còn hiệu lực:
1. `epoch_losses.csv` cũ ghi giá trị batch cuối, không phải epoch-mean. Đã sửa ở P0-1.
2. CSV cũ hơn code 13 ngày; run 40 epoch sinh bởi phiên bản code không còn tồn tại.
3. Checkpoint top-k hỏng tên file (dấu `/`) ⇒ không có checkpoint nào trong vùng nổ
   (epoch 25–39). Đã sửa ở P0-2.

---

## 4. LỖI CHẶN: protein/ligand và chuỗi 7D thuộc hai complex khác nhau `[ĐÃ ĐO]`

**Vị trí:** `src/step6_trainer.py`, `SAMGOptimizedDataset.__init__` / `__getitem__`.

```python
self.target_entries = splits.get(split_mode, [])                       # theo split
self.available_pkls = sorted(glob(processed_dir + "/*_sequence_7d.pkl"))  # TAT CA split, sort theo ten
...
entry    = self.target_entries[real_idx]   # -> protein apo/holo + ligand SDF
pkl_path = self.available_pkls[real_idx]   # -> chuoi 7D + chuoi token
```

Hai danh sách không có quan hệ nào với nhau: một cái theo thứ tự split, một cái theo
thứ tự alphabet của `pli_id`, và `available_pkls` không lọc theo split.

**Đo (Mục 10.1):**
```
split keys: train=23052, valid=1071, test=478 | n pkl = 15817

train  i=0  entry=3txj__1__1.A__1.K   pkl=1a4h__1__1.A__1.B
       i=1  entry=5b59__1__1.A__1.B   pkl=1a4h__1__2.A__2.B
       MISMATCH: 2000/2000   (100%)
valid  MISMATCH: 1071/1071   (100%)
test   MISMATCH:  478/478    (100%)
```

**Hai hệ quả:**
1. Mô hình học ánh xạ *hốc protein A → chuỗi fragment của phân tử B ngẫu nhiên*.
2. Train và valid đều đọc cùng một dãy pkl bắt đầu từ index 0, nên **nhãn 7D của
   valid là tập con nhãn 7D của train**. Vừa sai cặp, vừa rò rỉ.

**Bằng chứng độc lập từ chính log P0-1 (`logs/epoch_losses.csv`):**
`ln(15626) = 9.657` là cross-entropy của phân phối đều trên vocab.
```
Epoch  Token_Loss
0      9.8086
1      9.2899      <- chi thap hon uniform 0.37 nat
```
Sau 2 epoch mô hình mới chỉ học được phân phối tần suất biên của vocab. Đó đúng là
hành vi kỳ vọng khi input không mang thông tin về target.

**Hệ quả với toàn bộ Rev 2:** mọi số trong Mục 4, 5, 7 của Rev 2 đều đo trên bài toán
không có tín hiệu. `nll_median = 9.92` khớp mốc flow-identity 9.97 **không phải** vì
flow đang khởi tạo tốt, mà vì không có gì để học. Tiêu chí thất bại "B. Chết" trong
Mục 7.3 Rev 2 gần như chắc chắn sẽ xảy ra, và sẽ bị quy sai cho `v_context.detach()`.

**Sửa:** ghép theo `pli_id`, không theo index; lọc `available_pkls` theo split. Chi
tiết ở `plan.md` P-1.

---

## 5. THU HỒI Mục 7 Rev 2 (P1 Baseline)
Tất cả số trong Mục 7 Rev 2 (`nll_median 9.92`, `nll_p99 − nll_median 2.98`,
`pinfrac_* = 0`, `logscale_mean_* < 0.06`) được đo trên dataset ghép sai (Mục 4) và ở
trạng thái smoke-test 2 epoch. Chúng không phải baseline của bất cứ thứ gì.

Phần **còn giữ lại** từ Mục 7 Rev 2: hai công cụ đo (`flow_scale_trace.csv` và
`diag_flow_state.py` v3) đã được hiệu chỉnh khớp nhau trong 0.7%, và lý do lệch 5.4 nat
trước đó đã được truy ra (thiếu canonicalize + sai `dequant`). **Công cụ đúng; đối
tượng đo thì sai.** Giữ nguyên công cụ, vứt các con số.

---

## 6. Biểu diễn 7D: khiếm khuyết cấu trúc (thay thế Mục 4.2 Rev 2)

### 6.1. 46.1% token có `(d, theta, phi)` tất định do cách dựng frame `[ĐÃ ĐO]`
`step0_data_prep._get_local_frame` lấy `p1 = centers[0]`, `x_axis ∝ (c₁−c₀)`,
`y_axis ∝ (c₁−c₀)×(c₂−c₀)`. Suy ra bằng đại số, độc lập với dữ liệu:

| Vị trí token | Ràng buộc tất định | Vì sao |
|---|---|---|
| 0 | `d = theta = phi = 0` | `rel_center ≡ 0` theo định nghĩa gốc |
| 1 | `theta = pi/2`, `phi = 0` | nằm đúng trên trục x của frame |
| 2 | `phi ∈ {0, ±pi}` | nằm trong mặt phẳng `y = 0` |

**Đo trên 18.911 token (Mục 10.2):**
```
pos 0    n= 3000 | d=0:1.000  | phi=0:1.000
pos 1    n= 3000 | theta=pi/2:0.904 | phi=0:0.904
pos 2    n= 2712 | phi = 0 hoac ±pi: 1.000
pos >=3  n=10199 | tu do
=> 46.1% token co (d,theta,phi) xac dinh hoan toan boi cach dung frame
```
(0.904 thay vì 1.000 ở vị trí 1 là do 9.6% phân tử rơi vào nhánh fallback `eye(3)` của
`_get_local_frame` — dưới 3 fragment hoặc các tâm cộng tuyến.)

Thêm: **15.9% token có `theta ∈ {0, pi}`**, tại đó `phi` vô định về mặt toán học.

**Hệ quả:** `phi` là hàm delta trên gần nửa dữ liệu. Đây là giải thích đơn giản và
đầy đủ cho hiện tượng `phi` collapse (13.7% ghim sàn ở Rev 2), **không cần** viện đến
cơ chế autoregressive của H2′.

**Hệ quả với P0-7:** mask 16.28% token anchor xử lý chưa tới 1/3 vấn đề, và còn vứt
luôn quaternion hợp lệ của các token đó. P0-7 bị huỷ.

### 6.2. Quaternion đã được scipy canonicalize sẵn — P0-5 làm hỏng thêm `[ĐÃ ĐO]`
Thống kê biên trên 25.137 token:
```
c3  mean=+0.175 std=0.471 min=-0.702 max=+0.998
c4  mean=+0.198 std=0.456 min=-0.695
c5  mean=+0.185 std=0.451 min=-0.684
c6  mean=+0.218 std=0.464 min=-0.703

thanh phan |lon nhat| mang dau DUONG : 1.0000   (25137/25137)
```
Bốn thành phần có phân phối **giống hệt nhau**, `min ≈ −1/√2 = −0.7071` ở cả bốn, và
100% quaternion có thành phần lớn nhất mang dấu dương. Đó là chữ ký của thuật toán
Shepperd mà `scipy.spatial.transform.Rotation.from_matrix().as_quat()` dùng: nó **đã
canonicalize theo quy tắc "thành phần lớn nhất mang dấu dương"**.

**Do đó P0-5 (`q[k] < 0 → flip`) là có hại, không phải chưa đủ.** Nó lật ~37%
quaternion tại `q[k] = 0` — vùng mật độ cao nhất của thành phần đó (mean 0.2, std
0.46). Cắt phân phối ngay chỗ mật độ cao nhất tạo bước nhảy không liên tục trong
`p(q | context)`, tức tạo ra đúng loại ranh giới không học được mà nó định xoá bỏ.
Analogy: đã có bản đồ múi giờ chuẩn, lại vẽ thêm đường đổi ngày xuyên qua giữa châu Âu.

**Quan sát phụ:** phân phối biên này (mean ~0.19, std ~0.46 sau gập max-positive) khớp
với phân phối của **phép quay đều trên SO(3)**. Marginal đều **không** chứng minh
rotation là không học được (nó vẫn có thể dự đoán được từ context), nhưng kết hợp với
6.4 thì đây là dấu hiệu đáng lo.

### 6.3. Sai chỉ số giữa các bước `[ĐÃ ĐO — đọc code]`
| File | Thứ tự giả định |
|---|---|
| `step0_data_prep.py:121` (GHI) | `[d, theta, phi, qw, qx, qy, qz]` |
| `step7_evaluation.py:348` (ĐỌC) | `[d, theta, phi, qw, qx, qy, qz]` ✔ khớp |
| `step2_mdn_module.py:DIM_NAMES` | `[d, theta, phi, qx, qy, qz, qw]` ✘ |
| `step6_trainer.py` canonicalize `target_7d[:,6]` | coi index 6 là `qw`, thực tế là `qz` ✘ |

Hệ quả: mọi nhãn per-dim trong chẩn đoán Rev 2 ở các chiều 3–6 đều gán sai tên
(gồm cả "`max|mu|` của `qz` = 8.68"). Không ảnh hưởng tới giá trị số, chỉ ảnh hưởng
tới việc diễn giải chúng.

### 6.4. Frame của fragment không phải hàm của token `[CHƯA ĐO — cơ chế đã chứng minh]`
`_get_local_frame(f["coords"])` dựng frame từ 3 nguyên tử không cộng tuyến đầu tiên
**theo thứ tự index RDKit**, trong khi token là SMILES **canonical**. Thứ tự nguyên tử
trong `GetMolFrags` kế thừa từ index của phân tử cha, tức từ file SDF gốc.

**Chứng minh cơ chế** (chạy `_get_local_frame` copy nguyên văn, trên MỘT đám mây
điểm duy nhất, chỉ đảo thứ tự index — Mục 10.4):
```
thu tu A              -> q = [-0.5705 -0.3915 -0.0023 +0.7219]
thu tu B (dao nguoc)  -> q = [+0.2343 +0.7805 -0.4532 +0.3612]
Goc lech giua hai frame: 159.6 do
```

**Mức độ chắc chắn — đọc kỹ:** cơ chế đã chứng minh; **tỷ lệ xảy ra trong dataset thật
thì CHƯA đo.** Phép đo "589 lần xuất hiện của `[4*]CO` → 510 quaternion phân biệt"
**KHÔNG** phải bằng chứng, vì `R_g_to_m` còn phụ thuộc frame phân tử nên vẫn phải khác
nhau kể cả khi nhãn được định nghĩa tốt. Muốn đo trực tiếp: tính lại frame từ SDF với
hai thứ tự nguyên tử khác nhau trên vài trăm complex và so quaternion thu được.

### 6.5. Frame là frame nội phân tử, không liên quan tới hốc `[ĐÃ ĐO — đọc code]`
`R_m_to_w, t_m_to_w = self._get_local_frame(centers)` với `centers` là tâm các fragment
**của chính ligand**. Toàn bộ 7D nằm trong hệ quy chiếu do ligand tự định nghĩa.
Lúc sinh phân tử, hệ quy chiếu này chưa tồn tại (nó phụ thuộc vào các fragment chưa
được sinh ra). Do đó 7D không xác định vị trí của ligand so với hốc — về mặt nguyên
tắc, một mô hình dự đoán 7D **hoàn hảo** vẫn không biết đặt phân tử ở đâu trong hốc.
Đây là lý do cấu trúc khiến Step 7 phải tự chế ra phép đặt riêng (Mục 7).

### 6.6. Sai lầm của Rev 3 đã tự sửa
Nghi ngờ ban đầu: fragment dưới 3 nguyên tử rơi vào fallback `eye(3)` tạo điểm khối
tại quaternion đơn vị. **Đo được: 0.0000.** `FragmentOnBonds` thêm dummy atom `[*]`
vào `coords`, nên hầu như mọi fragment đều có ≥3 điểm. Bỏ khỏi danh sách lo ngại.

---

## 7. Step 7 không dùng 7D per-fragment `[ĐÃ ĐO — đọc code]`
`step7_evaluation.py:333–380`:
```python
res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())      # hinh hoc noi phan tu: 100% ETKDG
AllChem.MMFFOptimizeMolecule(mol)
ai_predicted_center = pocket_center + pred_coords.mean(axis=0)
mean_quat = pred_quats.mean(axis=0); mean_quat /= norm(mean_quat)
coords = (coords - etkdg_center) @ ai_rotation_matrix.T + ai_predicted_center
```
Đầu ra 7D chỉ đóng góp **một** vector tịnh tiến trung bình và **một** quaternion trung
bình. Per-fragment pose bị vứt hoàn toàn. Ba vấn đề:
1. Trung bình cộng trên S³ rồi chuẩn hoá là sai về hình học; với phân phối gần đều
   (Mục 6.2) trung bình cộng hội tụ về ~0 và hướng thu được là nhiễu.
2. `pred_coords` ở frame nội phân tử, `pocket_center` ở frame protein — cộng hai hệ
   quy chiếu không liên quan.
3. Mô tả đúng của pipeline hiện tại: *sinh SMILES tự hồi quy + đặt conformer ETKDG
   cứng vào tâm hốc*. **Luận điểm "sinh ligand ở dạng 3D" chưa được code hỗ trợ.**

---

## 8. Điều kiện hoá theo hốc: ba lỗi khiến nó không thể hoạt động `[ĐÃ ĐO — đọc code]`

### 8.1. `h_target` là một vector duy nhất cho cả hốc
`step6_trainer.py:226`:
```python
h_target = scatter_mean(residue_h, prot_trans_batch, dim=0, dim_size=num_graphs)[:, :H].unsqueeze(1)
# shape [B, 1, H]
```
Cross-attention chạy softmax trên đúng 2 key (null-sink + 1). `MultiDifferentialCrossAttention`
hiện là một cổng scalar, không phải attention. Mean-pool trên feature bất biến của EGNN
xoá sạch thông tin hình dạng hốc: geometric head được yêu cầu dự đoán **tư thế trong
một frame cụ thể** từ một **trung bình bất biến hoán vị**.

**Hệ quả với chẩn đoán cũ:** đây là ứng viên mạnh hơn `v_context.detach()` cho failure
mode "B. Chết". Kể cả khi bỏ detach, gradient hình học chảy ngược vào một biểu diễn đã
mất thông tin không gian.

### 8.2. Ligand ground-truth bị rò vào context — cả train lẫn eval
`dynamic_encoder(..., ligand_pos=data_batch.ligand_pos, ...)`; `UniTransformer` compose
protein+ligand vào một graph; `residue_h` lấy từ graph đó. `step7_evaluation.py:280`
truyền `ligand_pos=batch.ligand_pos` **tại thời điểm đánh giá**. Vector điều kiện chứa
đáp án. Mọi số Vina/QED thu được cho tới nay là của mô hình được xem trước ligand thật.

### 8.3. Geometric head không biết token nào đang được sinh
`v_context[t]` tính nhân quả từ token `< t`, nhưng 7D tại vị trí `t` là tư thế của
fragment `t`. Flow phải marginalize trên phân phối token: bị hỏi "đặt cái này ở đâu"
mà không được cho biết "cái này" là gì. Sửa rẻ: nối `token_embedding[target_ids[t]]`
vào context của geometric head khi teacher forcing.

### 8.4. Về `v_context.detach()` (Mục 4.5 Rev 2)
Vẫn là khiếm khuyết thật: `L_geo` chỉ cấp gradient cho `context_proj` + `arn`. Nhưng
với 8.1 tồn tại, bỏ detach **một mình** sẽ không sửa được gì. Hạ ưu tiên xuống sau 8.1.

---

## 9. Hypothesis Log

### H1 — "Representation Shift Shock" (2026-09-27) — **KHÔNG CÒN Ý NGHĨA**
Dựa trên Token Loss rơi 8.0 → 2.8 tại epoch 25. Với Mục 4 (ghép sai dữ liệu), con số đó
không thể xảy ra trên dữ liệu đúng cặp. Đóng hồ sơ.

### H2 — "σ-Collapse trên manifold suy biến" (2026-09-28) — **ĐÃ BỊ BÁC** (giữ nguyên)
Tiêu chí ghi cứng trước khi đo: `log_scale ≪ 0` và >30% ghim sàn, mạnh nhất ở
`qx..qw`. Kết quả: quaternion 0% ghim sàn; chiều collapse là `phi`; tỷ lệ 13.7% < 30%.
Sai trên cả hai tiêu chí.

### H2′ — "Anchor-induced collapse trên `phi`" (2026-09-28) — **HẠ CẤP**
*Vì sao hạ cấp:*
1. **Tiền đề thiếu.** H2′ dựa trên 16.28% token anchor `d=0`. Thoái hoá thật là 46.1%
   (Mục 6.1), và nó đến từ cách dựng frame chứ không riêng anchor.
2. **Có giải thích đơn giản hơn.** `phi` là delta function trên gần nửa dữ liệu. Không
   cần cơ chế autoregressive nào để giải thích σ_phi → 0.
3. **Cơ chế chưa kiểm được và kiểm rất rẻ.** H2′ giả định `d` đứng trước `phi` trong
   permutation của ARN. `pyro.nn.ConditionalAutoRegressiveNN` mặc định dùng permutation
   **ngẫu nhiên tại init**. Một dòng `print(self.arn.permutation)` là đủ để bác hoặc
   giữ. **Chưa ai chạy dòng đó.**
4. **Đo trên dữ liệu ghép sai** (Mục 4), nên `pinfrac_phi = 13.7%` bản thân nó cũng
   không đáng tin.

*Trạng thái:* không bác bỏ, nhưng không còn là giả thuyết hiện hành. Ứng viên hiện
hành là Mục 6.1 + Mục 8.1.

### H3 — "Không có tín hiệu để học" (2026-09-29) — **GIẢ THUYẾT HIỆN HÀNH**
*Nội dung:* mô hình chưa từng được đặt vào một bài toán có tín hiệu, do (a) ghép sai
cặp dữ liệu (Mục 4), (b) 46% nhãn góc là hằng số của frame chứ không phải hoá học
(Mục 6.1), (c) context là một vector mean-pool không mang thông tin không gian
(Mục 8.1), (d) context lại chứa ligand ground-truth (Mục 8.2).
*Dự đoán kiểm được:* sau khi sửa (a) một mình, Token_Loss phải xuống rõ dưới
`ln(15626) = 9.657` trong vài epoch đầu. Nếu **không**, (c) hoặc (d) là nút thắt tiếp theo.
*Tiêu chí ghi cứng TRƯỚC khi chạy:* Token_Loss epoch-mean tại epoch 5 < 8.0.

---

## 10. Tái lập các phép đo Rev 3
Tất cả chạy được trên CPU, chỉ cần `pickle`/`glob`/`math`, trừ 10.4 cần `numpy`+`scipy`.

**10.1 — Ghép sai dữ liệu (Mục 4):** với mỗi `i`, so `entry[i][1].split("/")[0]` với
`basename(pkls[i]).replace("_sequence_7d.pkl","")`. Kỳ vọng: mismatch 100%.

**10.2 — Thoái hoá theo vị trí (Mục 6.1):** gom `spatial_tokens` theo chỉ số vị trí
trong chuỗi, đếm tỷ lệ `d==0`, `theta==pi/2`, `phi in {0, ±pi}` cho pos 0/1/2/≥3.

**10.3 — Canonicalization sẵn có (Mục 6.2):** với mỗi token, `j = argmax|q[3:7]|`;
đếm tỷ lệ `q[3+j] > 0`. Kỳ vọng: 1.0000.

**10.4 — Frame phụ thuộc thứ tự index (Mục 6.4):** copy `_get_local_frame` nguyên văn,
gọi trên `coords` và `coords[::-1]` của cùng một đám mây điểm, so hai ma trận quay.

**10.5 — Permutation của ARN (H2′):** `print(model.generator.geometric_head.arn.permutation)`.

---

## 11. Open Architectural Decision (cập nhật từ Mục 6 Rev 2)

**Quyết định Flow vs MDN vs Robust Regressor: HOÃN.** Không có dữ liệu để quyết, và
Mục 4 + 6 + 8 quyết định việc có tín hiệu hay không, đứng trên quyết định này.

Ghi chú bổ sung cho khi quay lại quyết định:
* **Lập luận "Flow giữ được đa đỉnh, regression thì không" KHÔNG đứng vững cho kiến
  trúc hiện tại.** Với MAF 1 lớp, chiều đứng đầu trong permutation có
  `p(x₁|c) = N(m₁(c), e^{2s₁(c)})` — đơn đỉnh tuyệt đối. Đa đỉnh chỉ xuất hiện yếu ở
  các chiều sau và cần ≥3–5 lớp xen permutation. Ưu thế biểu diễn so với MDN K=4–8
  hiện là giả định chưa kiểm chứng.
* Năm lỗi của phương án Robust Regressor nêu ở Rev 2 Mục 6 vẫn đúng, **trừ lỗi 1**:
  double cover đã được scipy xử lý (Mục 6.2), nên vấn đề "hồi quy về ~0" không xảy ra
  theo cách mô tả.
* **Khuyến nghị mạnh cho biểu diễn quay:** chuyển sang **6D continuous rotation**
  (Zhou et al., CVPR 2019 — hai cột đầu của ma trận quay + Gram–Schmidt). Nó liên tục,
  không có double cover, và **xoá bỏ toàn bộ bài toán canonicalization** mà P0-5 đang
  cố giải sai. Đây là thay đổi rẻ nhất với lợi ích lớn nhất trong danh sách.

---

## 12. Định hướng dự án (để ngỏ, chưa chốt)
Nguồn gốc các thành phần: biểu diễn 7D fragment-pose → Frag2Seq; apo→holo → Apo2Mol;
transformer tự hồi quy trên token 3D → 3DMolFormer; **multi-differential cross-attention
cho anti-target → đóng góp của dự án này**, và là thứ **duy nhất chưa từng chạy một lần
nào** (`list_h_anti = []` ở cả train lẫn eval; nhánh static encoder bị comment; khi còn
bật thì "anti-target" là `protein_pos + 1.5`, tức chính target dịch đi 1.5 Å).

Hai lựa chọn cần cân nhắc khi đường target-only chạy được:
* **(A)** Tiếp tục tự xây backbone 3D.
* **(B)** Fork một backbone SBDD 3D đã chạy được (Apo2Mol đã có trong repo, hoặc
  TargetDiff/DecompDiff/Pocket2Mol) và chỉ thêm cơ chế selectivity lên trên. Ablation
  trở nên sạch: *backbone* vs *backbone + selectivity*.

Khi quay lại anti-target, hai điểm đã phát hiện cần ghi nhớ:
* `align_anti_targets` ghép CA **theo chỉ số** (`ref_atoms[:min_len]` vs
  `alt_atoms[:min_len]`) — vô nghĩa với protein không đồng nguồn. Nhưng nếu attention
  hub chỉ dùng feature **bất biến** SE(3) thì bước align là thừa và câu hỏi apo/holo
  cho anti-target tự biến mất.
* Dùng holo của anti-target làm cho anti-target trông **khó bám hơn thực tế** (hốc đã
  co quanh ligand khác), tức thiên lệch theo hướng báo cáo selectivity cao giả tạo.
* Cần chốt metric selectivity và **control**: huấn luyện chỉ với target rồi đo
  selectivity. Panel hiện tại trộn homolog thật (JNK1/JNK2, HSP90/GRP94) với protein
  không liên quan (hERG, CYP3A4); chỉ nhóm homolog mới là bài toán thực sự.

---

## 13. Kết quả xác minh chéo (2026-09-29, session xác minh đối kháng)

> **[!] CANH BAO XUAT XU — DA GIAI QUYET o Muc 15. Doc Muc 15 TRUOC khi dung bat ky
> con so nao trong Muc 13.** Noi dung canh bao goc: moi phep do DU LIEU
> trong Muc 13 (13.2-13.6, 13.9 N6) va phep do entropy o 14.2 duoc chay tren **ban mirror
> local** `/Users/HIEU/Workspaces/Projects/SAMG`, KHONG phai tren server GPU noi du lieu va
> ket qua that nam. Trong phien nay da tung phat hien local lech server 13 ngay (Muc 3.2).
> **Cac con so nay o trang thai CHO XAC NHAN** cho den khi fingerprint dataset + code giua
> local va server khop (quy trinh o 13.13). Phan ngoai le, KHONG bi anh huong:
> - Muc 14 (ket qua run P1): lay tu `epoch_losses.csv` / `flow_scale_trace.csv` do **server**
>   sinh ra, nguoi dung dan truc tiep. Tin cay.
> - 13.8 (`permutation` la register_buffer): doc tu source pyro 1.9.1 upstream. Tin cay.
> - 13.7, 13.9 N1/N2/N3/N4/N5 (doc code): tin cay **neu va chi neu** server dung cung commit
>   — kiem bang `git rev-parse HEAD` (13.13).
>
> Bang chung gian tiep hien co cho thay local va server co the trung nhau: diag v3 chay tren
> server bao `d max = 17.2585` va `anchor = 0.1628`, khop **dung den chu so cuoi** voi so do
> local. Nhung day la suy luan, khong phai kiem chung.

Phương pháp: mọi phép đo dưới đây **tự cài đặt lại từ mô tả**, không tái sử dụng script
của Rev 3. Đo trên **toàn bộ** 15.817 pkl / 97.940 token (Rev 3 dùng mẫu 3.000–25.137
token). Script nguyên văn: `VERIFY_REV3_SCRIPTS.md`. Không sửa file `.py` nào, không train.

### 13.1. Bảng phán quyết

| Mục Rev 3 | Phán quyết | Số của Rev 3 | Số đo lại |
|---|---|---|---|
| 4 — ghép sai dữ liệu | **XÁC NHẬN + NẶNG HƠN** | mismatch 100% | train 99.9874%, valid 100%, test 100% |
| 6.1 — 46.1% token tất định | **XÁC NHẬN** | 46.1% | **46.98%** (toàn bộ dataset) |
| 6.2 — scipy đã canonicalize, P0-5 có hại | **XÁC NHẬN + LƯỢNG HOÁ** | 100% argmax dương | 1.000000 → **0.649347** sau P0-5 |
| 6.3 — sai chỉ số | **XÁC NHẬN chính xác** | — | index 6 = `qz`, không phải `qw` |
| 6.4 — frame phụ thuộc thứ tự nguyên tử | **XÁC NHẬN, nâng [CHƯA ĐO] → ĐÃ ĐO** | chỉ cơ chế | **median 38.6°** trên dữ liệu thật |
| 6.5 — frame nội phân tử | **XÁC NHẬN, sửa lý do** | — | frame tự định nghĩa được, nhưng mù hốc |
| 7 — ETKDG + 1 rigid transform | **XÁC NHẬN** | — | dòng 333/336/355/358 |
| 8.1 — `h_target` 1 vector, 2 key | **XÁC NHẬN** | — | `[B,1,H]`; test file tự ghi "2 keys total" |
| 8.2 — rò ligand GT ở eval | **XÁC NHẬN** | — | `step7:280` |
| 8.3 — thiếu token embedding | **XÁC NHẬN** | — | chỉ nhận `v_context_detached` |
| 10.5 — permutation ARN | **KHÔNG CHẠY ĐƯỢC + HẠ CẤP** | "chưa ai chạy" | phép kiểm không còn tính quyết định |

### 13.2. Mục 4 — nặng hơn Rev 3 báo: valid/test sẽ RỖNG sau khi sửa ghép
Đo độc lập bằng `entry[2]` (đường dẫn ligand SDF) thay vì `entry[1]`, chạy **đủ** cả ba split:
```
train  n_entries=23052  L=15817  MISMATCH=15815/15817 = 99.9874%
valid  n_entries= 1071  L= 1071  MISMATCH= 1071/1071  = 100.0000%
test   n_entries=  478  L=  478  MISMATCH=  478/478   = 100.0000%
```
Sửa nhỏ cho Rev 3: train là **99.9874%**, không phải 100% — có đúng 2 cặp khớp tình cờ.
Kết luận không đổi.

**Phát hiện mới, quan trọng hơn con số mismatch:**
```
entry co pkl tuong ung ton tai (ghep dung theo pli_id se dung duoc):
  train  15817/23052 = 68.61%
  valid      0/1071  =  0.00%     <-- RONG
  test       0/ 478  =  0.00%     <-- RONG
```
Step 0 chỉ tiền xử lý các complex thuộc split train. **Sửa ghép theo `pli_id` (P-1) sẽ
cho một tập train dùng được nhưng valid và test hoàn toàn rỗng.** Hệ quả trực tiếp:
`ModelCheckpoint(monitor="val_total_loss")` và `ReduceLROnPlateau(monitor="val/total_loss")`
sẽ không có dữ liệu. **P-1 phải bao gồm chạy lại step 0 cho valid + test**, nếu không
việc sửa ghép biến một lỗi ồn thành một lỗi im lặng.

Rò rỉ: `L = min(len(entries), len(pkls))` ⇒ train đọc `pkl[0..15816]`, valid `pkl[0..1070]`,
test `pkl[0..477]` — valid và test đều là **tập con** của train. Xác nhận.

### 13.3. Mục 6.1 — xác nhận, kèm một sửa cách diễn giải
Đo theo vị trí token trên toàn bộ 97.940 token:
```
          n      d==0  th=pi/2  phi==0  |phi|=pi  phi in{0,pi}
pos 0  15817   1.0000   0.0000  1.0000    0.0000        1.0000
pos 1  15817   0.0000   0.9093  0.9093    0.0000        0.9093
pos 2  14382   0.0000   0.0000  0.6881    0.3119        1.0000
pos>=3 51924   0.0000   0.0001  0.0002    0.0001        0.0003
=> ty le token o pos 0,1,2 = 46.98%   (Rev 3: 46.1%)
   theta o cuc {0,pi} (phi vo dinh)  = 16.15%   (Rev 3: 15.9%)
```
**Sửa cách diễn giải:** đây **không phải khiếm khuyết của `_get_local_frame`**. Frame
được định nghĩa *bằng* ba fragment đầu, nên `d=θ=φ=0` ở pos 0, `θ=π/2, φ=0` ở pos 1,
`φ∈{0,±π}` ở pos 2 là **hệ quả tất yếu của định nghĩa**, đúng như một hệ toạ độ phải
có. Khiếm khuyết thật là ở phía mô hình: `L_geo` bắt flow **học các hằng số này như thể
là dữ liệu**, và trộn chúng vào cùng một mean với các dự đoán thật. 46.98% mục tiêu có
entropy bằng 0. Điều đó cũng có nghĩa: 6.1 giải thích được `phi` collapse **không cần**
cơ chế autoregressive nào, vì `v_context` mang positional encoding nên ARN nhận ra
"đây là vị trí 0/1/2" chỉ từ context.

### 13.4. Mục 6.2 / P0-5 — xác nhận, và lượng hoá được thiệt hại
```
TRUOC P0-5: thanh phan |lon nhat| duong = 1.000000  (97940/97940)
P0-5 lat 35.0653% hang (theo dau index 6 = qz)
SAU  P0-5: thanh phan |lon nhat| duong = 0.649347   <-- canonicalization BI PHA
```
Lập luận "cắt ở vùng mật độ cao" của Rev 3 giờ có số:
```
argmax|q| : min 0.5028  mean 0.7975  | ty le < 0.55 = 0.2042%   <- bien scipy: mat do THAP
qz        : mean +0.2290 std 0.4684 | ty le |.|<0.05 = 8.4633%  <- bien P0-5: mat do CAO
```
Biên của P0-5 dày hơn biên của scipy khoảng **40 lần**. P0-5 không phải "chưa đủ" — nó
**phá** một canonicalization sạch và thay bằng một cái tệ hơn, trên một chiều mà Mục 6.3
cho thấy nó còn nhận dạng sai là `qw`. **Phải revert.**

### 13.5. Mục 6.4 — nâng lên ĐÃ ĐO, bằng một phép thử mạnh hơn đề bài
Thay vì tính lại frame từ SDF (chỉ cho *chặn trên*), tôi dùng chính đầu ra pipeline:
**các complex có chuỗi fragment-SMILES giống hệt nhau** (cùng phân tử, cùng phân mảnh,
cùng thứ tự) — 2.917 nhóm, 12.477 complex, 2.775.764 cặp token.
```
Goc lech quaternion (do):
  tat ca cap    : median  38.618  p90 150.623  >10deg 87.01%  >90deg 28.16%
  cung 1 PDB id : median   9.285  p90 124.716  >10deg 48.30%
  ty le cap lech <1 deg: 1.03%
DOI CHUNG: lech |d| = median 0.0285 A, p90 0.2285 A
```
**Đối chứng `d` là điểm mấu chốt:** `d` khớp tới 0.03 Å ⇒ đúng là cùng phân tử, cùng
conformer. Vậy chênh lệch quay **không phải** do khác conformer. Nhãn quay không phải là
hàm của phân tử.

Tách cơ chế:
```
chi lay nhom co SMILES doi mot KHAC NHAU: median 38.625  (vs 38.618)  -> KHONG doi
=> co che "fragment trung SMILES bi doi cho" KHONG phai nguyen nhan
std cua goc lech TRONG cung mot cap: median 36.72, ty le <1deg chi 1.45%
=> lech doc lap theo tung fragment, khong phai mot phep quay chung cua ca phan tu
```
Đại số: nếu chỉ frame *phân tử* đổi, `R_g_to_m = M^T G` ⇒ độ lệch tương đối `M_A^T M_B`
**không phụ thuộc `G`**, tức góc lệch phải **giống nhau ở mọi vị trí** (std = 0). Quan sát
std ≈ 36.7° ⇒ khiếm khuyết nằm ở **frame cấp fragment** `_get_local_frame(f["coords"])`,
tức thứ tự index nguyên tử trong từng fragment. Đây là chỉ dẫn trực tiếp cho chỗ phải sửa.

Phép thử SDF mà đề bài yêu cầu (tự cài lại `_get_local_frame` + Shepperd bằng numpy,
không dùng scipy, 400 complex, hoán vị ngẫu nhiên thứ tự nguyên tử):
```
median 135.444  p10 65.366  p90 174.201  max 180.000
>10 deg: 99.75%   >90 deg: 81.50%   <1 deg: 0.00%
```
Đây là **chặn trên** (hoán vị ngẫu nhiên hoàn toàn). Số thực tế trong dataset là 38.6°.

**Ngân sách tín hiệu của nhãn quay** — con số đáng lo nhất của toàn bộ Rev 3:
```
san nhieu nhan (cung phan tu, cung conformer)  : median  38.6 deg
bien do trong cung 1 loai fragment (cung SMILES): median 118.2 deg
bien do giua hai token ngau nhien               : median 132.8 deg
```
Biết token chỉ thu hẹp biên độ từ 132.8° xuống 118.2° (**11%**), trong khi sàn nhiễu là
38.6°. Nhãn quay gần như không mang cấu trúc học được có điều kiện theo token. Đây là
giải thích cho quan sát "marginal khớp phép quay đều trên SO(3)" ở Mục 6.2 Rev 3: nhãn
**thực sự** gần như ngẫu nhiên.

### 13.6. Mục 6.5 — xác nhận kết luận, sửa lý do
Rev 3 viết: *"lúc sinh phân tử, hệ quy chiếu này chưa tồn tại (nó phụ thuộc vào các
fragment chưa được sinh ra)"*. **Không chính xác.** Frame **tự định nghĩa được**: pos 0
ở gốc, pos 1 trên trục +x, pos 2 trong mặt phẳng xy — đó chính là các hằng số ở 13.3.
Nên frame khôi phục được khi sinh.

Kết luận của Mục 6.5 vẫn đúng nhưng vì lý do khác: **7D không chứa một bit nào về vị trí
của ligand so với hốc.** Một mô hình dự đoán 7D hoàn hảo vẫn không biết đặt phân tử ở
đâu. Đó là lý do cấu trúc buộc Step 7 phải tự chế `pocket_center + pred_coords.mean()`.

### 13.7. Mục 7, 8.1–8.3 — xác nhận toàn bộ
* **7:** `step7:333` `EmbedMolecule(mol, ETKDG())`, `:336` `MMFFOptimizeMolecule`,
  `:355` `pocket_center + pred_coords.mean(axis=0)`, `:358` `pred_quats.mean(axis=0)`.
  **Bổ sung:** kết hợp 13.5, phép trung bình cộng quaternion đang lấy trung bình các nhãn
  có sàn nhiễu 38.6° và marginal gần đều ⇒ hướng thu được bị nhiễu chi phối. Không chỉ
  "sai về hình học" mà còn **không có tín hiệu để trung bình**.
* **8.1:** `step6:225` `scatter_mean(...).unsqueeze(1)` ⇒ `[B,1,H]`. `step3:44`
  `cat([null_key, h_target])` ⇒ 2 key. Chính `step3_attention_hub.py:89` tự ghi
  *"Target with 1 residue + null sink -> 2 keys total"*.
* **8.2:** `step7:280` truyền `ligand_pos=batch.ligand_pos` tại thời điểm đánh giá. Xác nhận.
* **8.3:** `step4:87,97` — geometric head chỉ nhận `v_context_detached`. Xác nhận.

### 13.8. Mục 10.5 — không chạy được tại chỗ, và đã mất tính quyết định
Không có torch/pyro trên máy local và không có checkpoint local ⇒ **KHÔNG KIỂM ĐƯỢC**.

Nhưng câu hỏi nền đã giải quyết được bằng cách đọc source pyro 1.9.1: `permutation` được
`register_buffer("permutation", P)` ⇒ **nằm trong state_dict, không bị random lại khi
reload**. Loại bỏ một nghi vấn tiềm tàng (checkpoint reload làm xáo thứ tự autoregressive).

**Hạ cấp phép kiểm:** dù kết quả ra sao nó cũng không còn thay đổi kết luận, vì 13.3 cho
thấy `phi` collapse được giải thích đầy đủ mà **không cần** phụ thuộc autoregressive —
`v_context` mang positional encoding nên ARN nhận ra vị trí 0/1/2 từ context. Vẫn nên
chạy một dòng cho đủ hồ sơ, nhưng đừng coi nó là điều kiện để hành động.

### 13.9. Phát hiện MỚI (không có trong Rev 3)

**N1 — `step0:110`: token được sắp xếp theo THỨ TỰ CHỮ CÁI của SMILES `[ĐÃ ĐO — đọc code]`**
```python
frag_data.sort(key=lambda x: x["smiles"])
```
Ba hệ quả:
1. **Phân rã tự hồi quy không có ý nghĩa hoá học.** Transformer đang học *thứ tự alphabet*
   của chuỗi SMILES, không phải liên kết phân tử. Không có cạnh nào trong phân tử tương
   ứng với bước `t → t+1`.
2. **Ràng buộc chưa bao giờ được áp.** Một mẫu hợp lệ phải ra đúng thứ tự alphabet. Không
   có gì trong sampling đảm bảo điều đó; mọi mẫu vi phạm là ngoài phân phối huấn luyện.
3. Frame phân tử do 3 fragment **đứng đầu alphabet** định nghĩa ⇒ đổi một nhóm thế làm
   re-sort danh sách sẽ **quay toàn bộ hệ toạ độ**, giữa hai phân tử gần như giống nhau.

Mặt tích cực: sort làm thứ tự fragment trở thành *canonical*, và đó là lý do 13.5 đo được
rằng cơ chế "trùng SMILES đổi chỗ" không đóng góp gì.

**N2 — biến dạng apo→holo được tính, được supervise, rồi BỊ VỨT `[ĐÃ ĐO — đọc code]`**
`step1:141` gọi `apply_transforms_tensor_batch(...)` ⇒ `updated_protein_pos`, trả về ở
`step1:155`. Nhưng `grep -rn updated_protein_pos src/` cho thấy nó **chỉ được tiêu thụ
bởi `test_step1.py:116`** — không phải step6 training, không phải step7 eval. Và
`h_target` được dựng từ `residue_h`, tính trên toạ độ **apo gốc**.

⇒ `L_pocket` là **tác vụ phụ trợ thuần**, không có đường nhân quả nào tới phân tử được
sinh. Luận điểm "sinh ligand theo hốc đã biến dạng" (phần kế thừa Apo2Mol) **chưa được
code hỗ trợ**, giống hệt tình trạng của luận điểm 3D ở Mục 7. Đây là khiếm khuyết cấu
trúc thứ hai ở mức luận điểm paper, và Rev 3 chưa nêu.

**N3 — Collation `_batch` bị gán zeros: BÁC BỎ mối lo, nhưng mong manh**
`ApoHoloLigandData.__inc__` rơi về `super().__inc__()` của PyG, mà mặc định với **mọi key
chứa `'batch'`** trả về `int(value.max()) + 1`. Do đó `torch.zeros(...)` gán tay ở
`step6:143-145` hoạt động như **khuôn mẫu per-graph** và được PyG tăng thành 0,1,2,…,B−1.
Bằng chứng độc lập không cần chạy PyG: `batch_size=8`, và nếu `num_graphs` bằng 1 thì
`step3:44` `cat([null_key.expand(8,…), h_target(1,…)])` phải raise shape error — training
không crash. ⇒ đúng về mặt hàm.

**Mong manh:** phụ thuộc vào heuristic `'batch' in key` của PyG và vào việc tensor gán tay
luôn có `.max() == 0`. Nên bỏ hẳn ba dòng gán tay và dựa vào `FOLLOW_BATCH`.
*Chưa chạy được PyG tại chỗ* — lệnh xác nhận ở 13.10.

**N4 — quy ước quaternion trong `utils/data.py`: BÁC BỎ**
Chuỗi nhất quán đầu-cuối: `numpy-quaternion` → `q_array = [q.w,q.x,q.y,q.z]` (data.py:276)
⇒ `data.protein_rotations` là **(w,x,y,z)** ⇒ `loss_q = 1 − |dot|` áp đúng thứ tự đó lên
`pred_res_rot` ⇒ `apply_transforms_tensor` đưa vào kornia `quaternion_to_rotation_matrix`,
mặc định **WXYZ** từ kornia 0.6. Không có lẫn lộn.
**Mong manh:** `requirements.txt` ghi `kornia>=0.6.0` không pin; kornia 0.5.x mặc định
XYZW ⇒ sai im lặng. Nên pin.

**N5 — `[SOS]`/`[PAD]` cùng id 0: BÁC BỎ (không phải lỗi sống)**
`target_ids = input_ids_list[1:]` chỉ chứa id fragment (≥2), nên `ignore_index=0` và
`pad_mask=(target_ids!=0)` chỉ mask đúng phần padding thật. Padding ở *input* trông như
`[SOS]`, nhưng không có `src_key_padding_mask` **và không cần**: padding chỉ nằm ở đuôi,
còn causal mask khiến vị trí thật `t` chỉ thấy `≤ t`. Đặt tên mong manh, không phải khiếm khuyết.

**N6 — `align_tensor` che lệch residue: BÁC BỎ đường nghi ngờ chính**
Nghi ngờ: `compute_residue_transforms` có `except KeyError: continue` (data.py:269) bỏ qua
residue thiếu N/CA/C ⇒ `len(translations) < num_residues` ⇒ `align_tensor` đệm 0.0 vào
cuối ⇒ `prediction[i]` bị ghép với target của residue khác.
**Đo trực tiếp trên 600 pocket PDB (26.489 residue), tự parse text:**
```
residue thieu N/CA/C: 0 / 26489 = 0.0000%
so complex co >=1 residue bi bo qua: 0/600
```
Đường này **không xảy ra**. `align_tensor` vẫn là công cụ che triệu chứng (nên `raise`
thay vì đệm), nhưng không có lệch nào đo được.

### 13.10. Lệnh còn lại cần chạy trên server (cần torch/PyG, không chạy được tại chỗ)
```bash
# N3 - gia tri thuc cua cac vector _batch sau collate (dung 1 batch, KHONG train)
python - <<'EOF'
import sys; sys.path.insert(0,'src')
from step6_trainer import SAMGOptimizedDataset
from datasets.pl_data import ProteinLigandDataLoader
import pickle, os
PROC='dataset/processed'; DS='dataset/Apo2Mol_Dataset/data_folder'
SP='dataset/Apo2Mol_Dataset/split_druglike_dict.pkl'
v=pickle.load(open(os.path.join(PROC,'global_vocab.pkl'),'rb'))
d=SAMGOptimizedDataset(DS,PROC,v,SP,'train')
b=next(iter(ProteinLigandDataLoader(d,batch_size=8,shuffle=False,num_workers=0)))
for k in ('protein_element_batch','protein_translations_batch','ligand_element_batch'):
    t=getattr(b,k); print(k, tuple(t.shape), 'min',int(t.min()),'max',int(t.max()),
                          'n_unique',int(t.unique().numel()))
print('num_graphs =', int(b.protein_element_batch.max())+1, '(phai = 8)')
EOF

# 10.5 - permutation cua ARN (cho du ho so, khong con tinh quyet dinh)
python -c "
import torch
ck=torch.load('saved_checkpoints_flow/last.ckpt',map_location='cpu',weights_only=False)
p=[k for k in ck['state_dict'] if 'permutation' in k]
print(p,[ck['state_dict'][k].tolist() for k in p])"
```

### 13.11. Khuyen nghi (chi tiet o phan cuoi Muc 13)
Xem bang o 13.12.

### 13.12. Hang muc plan.md Rev 3: du can cu hay chua

| Hang muc | Phan quyet | Ly do |
|---|---|---|
| Revert P0-5 | **DU CAN CU, lam ngay** | 13.4: canonicalization 1.000 -> 0.649. Sua mot dong, khong rui ro. |
| Sua thu tu chi so (Muc 6.3) | **DU CAN CU, lam ngay** | 13.1: `DIM_NAMES` sai; phai sua cung luc voi revert P0-5. |
| Pin `kornia`, `pyro-ppl` | **DU CAN CU, lam ngay** | N4: `kornia>=0.6.0` khong pin ⇒ sai im lang neu ai cai 0.5.x. |
| P-1 ghep du lieu theo `pli_id` | **CAN SUA DAC TA TRUOC** | 13.2: valid/test se RONG. Dac ta phai bao gom chay lai step 0 cho valid+test, va mot assert `len(dataset)>0` cho moi split. |
| Bo 3 dong gan tay `_batch` | **DU CAN CU nhung uu tien thap** | N3: hien dang dung ve mat ham. Sua de giam mong manh, khong sua duoc loi nao. |
| Huy P0-7 (mask anchor) | **DU CAN CU** | 13.3: xu ly 16% trong so 47%, va mask theo chieu khong kha thi voi `to_event(1)`. |
| Chuyen sang 6D continuous rotation | **CHUA DU CAN CU** | 13.5: van de KHONG phai bieu dien (double cover da duoc scipy xu ly). Van de la **nhan quay khong phai ham cua phan tu** (san nhieu 38.6°). Doi bieu dien ma khong sua dinh nghia nhan thi chi doi cach ma hoa cung mot nhieu. **Phai sua `_get_local_frame` cap fragment truoc.** |
| Sua Muc 8.1 (h_target nhieu key) | **DU CAN CU** | 13.7 xac nhan. Nhung xem thu tu uu tien o duoi. |
| Sua Muc 8.2 (bo ligand GT khoi eval) | **DU CAN CU, lam ngay** | 13.7 xac nhan. Moi so Vina/QED cu deu vo gia tri; khong sua thi khong the bao cao ket qua nao. |
| Sua Muc 8.3 (noi token embedding) | **DU CAN CU, re** | 13.7 xac nhan. |
| Bo `v_context.detach()` | **CHUA DU CAN CU** | Dong y voi Rev 3: xep sau 8.1. |
| Step 7 dung 7D that thay ETKDG | **CHUA DU CAN CU** | 13.6: 7D khong chua thong tin vi tri so voi hoc. Viet lai Step 7 truoc khi sua dinh nghia nhan la lam hai lan. |

**Thu tu toi de nghi** (moi buoc co tieu chi kiem duoc, khong buoc nao phu thuoc gia thuyet chua kiem):

1. **Nhom "sua mot dong, khong rui ro":** revert P0-5; sua `DIM_NAMES`; pin `kornia`/`pyro-ppl`; bo `ligand_pos` khoi `step7:280`.
2. **Dinh nghia lai nhan quay** (`_get_local_frame` cap fragment) sao cho nhan la ham cua phan tu, khong cua thu tu index. Tieu chi ghi cung: lam lai phep do 13.5, **median goc lech giua hai complex cung chuoi SMILES phai < 5°** (hien 38.6°). Neu khong dat, moi viec phia sau deu vo nghia.
3. **P-1 ghep theo `pli_id` + chay lai step 0 cho valid/test** + assert `len>0` moi split.
4. Chi khi buoc 2 va 3 dat: sua 8.1 (h_target nhieu key) roi 8.3 (token embedding). Tieu chi H3 cua Rev 3 (`Token_Loss` epoch-mean tai epoch 5 < 8.0) chi co nghia sau buoc 3.
5. Quyet dinh Flow/MDN/Regressor va viet lai Step 7: **hoan** cho den khi buoc 2 dat.

**Nhan xet ve dinh huong (Muc 12):** ba trong bon lap luan cua paper hien khong duoc code
ho tro — sinh 3D (Muc 7), bien dang apo→holo (N2), va anti-target (chua chay lan nao).
Dieu do lam lua chon (B) — fork mot backbone SBDD da chay duoc roi chi them co che
selectivity — dang manh hon so voi luc Rev 3 viet.


---

## 14. Ket qua run P1 (epoch 0-21, 2026-09-29) — doc lai toan bo

### 14.1. Van de phan ky DA DUOC GIAI QUYET
```
pinfrac_* : ~0 tren ca 7 chieu, 22 epoch (max quan sat 4.5e-06)
Geo_Loss_Unclamped : 8.65 -> 7.15, khong co epoch nao bung no
```
Truoc P0-3/P0-4, NLL tho no tu 8 len 1.6e12 trong 6 epoch. Gio 22 epoch khong co dau
hieu phan ky. **H2 va H2' dong ho so vinh vien.** Du doan luc chon DEQUANT_SIGMA=0.1
("log_scale toi uu ~ -2.30, nam han tren san clip -5") **dung**: do duoc -1.23 den -1.39.

### 14.2. Token_Loss = 5.32 KHONG phai bang chung mo hinh hoc duoc gi ve hoc
Do entropy thuc te tren 97.940 token (script F, `VERIFY_REV3_SCRIPTS.md`):
```
cross-entropy phan phoi DEU  ln(15626)        = 9.657
H(token)                      unigram         = 8.188
H(token | vi tri trong chuoi)                 = 6.849
H(token | token truoc)        bigram (in-sample) = 2.920
---------------------------------------------------------
MODEL epoch 5                                 = 6.640
MODEL epoch 21                                = 5.321
```
Mo hinh vuot `H(token|vi tri)` 1.53 nat, nhung **kem mot bang tra bigram 2.4 nat**.
Vi Muc 4 (ghep sai cap) dam bao `h_target` khong mang MOT BIT nao ve phan tu, toan bo
1.53 nat do phai den tu cau truc noi tai cua chuoi — chu yeu la rang buoc thu tu alphabet
o N1. (Luu y: 2.920 la uoc luong in-sample cua bigram, lac quan do memorize; con
`H(token|vi tri)` = 6.849 thi uoc luong vung vi chi co 9 bucket.)

**=> Tieu chi ghi cung cua H3 ("Token_Loss epoch-mean tai epoch 5 < 8.0") BI VO HIEU.**
No da dat (6.640) tren du lieu ghep NGAU NHIEN. Mot nguong tuyet doi khong the phan biet
"hoc duoc hoc" voi "hoc thong ke chuoi". Tieu chi thay the o 15.3.

### 14.3. Hinh hoc: phan quay hoc duoc CHINH XAC BANG KHONG
`logscale_mean` tai epoch 21 (nho: nhan cot bi sai ten theo Muc 6.3 — thu tu that la
`[d, theta, phi, qw, qx, qy, qz]`):
```
d      -1.386   -> sigma 0.250
theta  -1.234   -> sigma 0.291
phi    -1.237   -> sigma 0.290
qw     +0.281   -> sigma 1.324   <- RONG HON phan phoi base
qx     -0.007   -> sigma 0.993
qy     -0.051   -> sigma 0.950
qz     -0.011   -> sigma 0.989
```
Bon chieu quay dung nguyen o sigma ~ 1 suot 22 epoch, tuc **dung lai o phan phoi base
N(0,1)** — khong trich xuat duoc mot bit nao. Mot chieu con rong hon base. Day la xac
nhan truc tiep, tren run thuc, cua san nhieu 38.6 do do o 13.5: **nhan quay khong hoc
duoc vi no khong phai ham cua phan tu.**

Nguoc lai `d, theta, phi` sac net that (sigma ~ 0.25-0.29). Nhung do phan lon la khai
thac 46.98% token co (d,theta,phi) la hang so (13.3), khong phai hoc hoa hoc.

### 14.4. `nll_median` DUNG YEN — cai tien chi o duoi cuc
```
epoch  0 -> 21 :  nll_median 4.446 -> 4.560   (dung yen)
                  nll_p99   13.970 -> 13.396  (dung yen)
                  Geo_Loss_Unclamped (mean) 8.654 -> 7.147  (giam 1.5)
```
Mean giam ma median va p99 deu khong doi ⇒ toan bo cai tien nam **tren p99**, tuc chi la
dep bot mot it outlier. **Token dien hinh khong hoc duoc gi.** Day la tieu chi that bai
"B. Chet" cua Rev 2 Muc 7.3, xay ra dung nhu du doan — nhung nguyen nhan KHONG phai
`v_context.detach()` (Rev 2 doan) ma la Muc 4 + 13.5.

### 14.5. LR confound da xay ra dung nhu canh bao
`lr` tut 1e-4 -> 5e-5 tai epoch 20 (ReduceLROnPlateau, patience=5). `Total_Loss` giam
14.449 -> 14.258 ngay tai epoch do. Khong co cot `lr` thi buoc giam nay se bi doc thanh
"mo hinh bat dau hoi tu". Viec them cot `lr` truoc khi chay da tra du chi phi cua no.
Ngoai ra `val_total_loss` duoc monitor lai duoc tinh tren valid vua ghep sai vua ro ri
(13.2), nen ban than tin hieu kich hoat scheduler cung vo nghia.

### 14.6. Gia tri thuc cua run nay: no la NHOM DOI CHUNG
Vi ghep cap la ngau nhien, `h_target` khong mang thong tin ve phan tu. Nen run nay
**chinh la nhom doi chung "khong co tin hieu hoc"** — mot thu ma binh thuong phai chay
rieng de co. Giu lai nguyen ven:
```
DOI CHUNG POCKET-BLIND (ghep ngau nhien, 22 epoch, lr 1e-4 -> 5e-5 @ep20):
  Token_Loss          7.601 -> 5.321
  Geo_Loss_Unclamped  8.654 -> 7.147
  nll_median          4.446 -> 4.560
  logscale_mean (4 chieu quay)  |.| <= 0.28  (khong hoc)
  Pocket_Loss         3.325 -> 2.813 (phang tu epoch 5)
```
Sau khi sua ghep cap, bat ky so nao KHONG tot hon dang ke cac muc tren deu co nghia la
dieu kien hoa theo hoc van chua hoat dong. Luu file `logs/` cua run nay vao
`logs/archive/CONTROL_pocket_blind_25ep/`.


### 13.13. Quy trinh xac nhan fingerprint local vs server (BAT BUOC truoc khi hanh dong)
Chay tren **server GPU**, doi chieu voi cot "local" duoi. Neu bat ky dong lech, moi con so
du lieu o Muc 13 va 14.2 phai do lai tren server truoc khi dung.

| Dai luong | Local do duoc |
|---|---|
| so file `*_sequence_7d.pkl` | 15817 |
| tong token | 97940 |
| vocab size | 15626 |
| `d` max | 17.2585 |
| anchor (pos0) | 15817 |
| ty le argmax\|q\| duong | 1.000000 |
| ty le index6 (qz) < 0 | 0.350653 |
| ty le token pos 0,1,2 | 0.469839 |
| `H(token\|vi tri)` | 6.8487 |
| `H(token)` unigram | 8.1881 |
| entry valid co pkl ton tai | 0 / 1071 |
| entry test co pkl ton tai | 0 / 478 |

Script fingerprint: xem `VERIFY_REV3_SCRIPTS.md` muc G.


---

## 15. Doi chieu fingerprint local vs server (2026-09-29) — SUA LAI Muc 13

Chay script G tren server GPU. Ket qua chia lam hai nua doi lap nhau.

### 15.1. CODE: trung khop tuyet doi
```
HEAD local  = HEAD server = 29f0cd41478b52f2e2cc0e33afabd5853a3ab212
md5 (12 ky tu dau), local == server tren ca 4 file:
  78798f72693c  src/step0_data_prep.py
  1bf69607ca32  src/step2_mdn_module.py
  5e8090234eec  src/step4_ligand_generator.py
  6bd5d7f91041  src/step6_trainer.py
git status: khong co file tracked nao bi sua o ca hai ben
```
⇒ **Toan bo phan doc code cua Muc 13 CO HIEU LUC cho server**: 6.3 (sai chi so), 7 (ETKDG),
8.1, 8.2, 8.3, va cac phat hien moi N1 (sort alphabet), N2 (`updated_protein_pos` bi vut),
N3 (collation), N4 (quy uoc quaternion), N5 (`[SOS]`/`[PAD]`). Khong can do lai.

### 15.2. DU LIEU: server LON HON va KHAC — moi so dataline cua Muc 13 bi thay the
| Dai luong | local | **SERVER (co tham quyen)** |
|---|---|---|
| so pkl | 15.817 | **23.660** (+49,6%) |
| vocab | 15.626 | **19.033** |
| tong token | 97.940 | **143.091** |
| `d` max | 17,2585 | **27,1296** |
| ty le token pos 0,1,2 | 46,98% | **48,26%** |
| argmax\|q\| duong | 1,000000 | **1,0** |
| index6 (`qz`) < 0 | 35,07% | **34,04%** |
| `H(token)` unigram | 8,1881 | **7,9095** |
| `H(token\|vi tri)` | 6,8487 | **6,5520** |

Suy luan cua toi hom nay ("`d max = 17.2585` khop den chu so cuoi ⇒ cung du lieu") la **SAI**:
output diag do la tu 28/09, truoc khi step 0 duoc chay lai/mo rong. Dataset da tang ke tu do.

### 15.3. B1 — XAC NHAN 100% dung nhu Rev 3; con so 99,987% cua toi la hien vat local
```
train  mismatch 23052/23052 = 100.000000%
valid  mismatch  1071/1071  = 100.000000%
test   mismatch   478/478   = 100.000000%
```
Rev 3 dung. Sua lai 13.2: khong co "2 cap khop tinh co".

### 15.4. B1b — **BAC BO**. Day la sai sot nang nhat cua toi hom nay
```
              local          SERVER
train co_pkl  15817/23052    22151/23052 = 96,09%
valid co_pkl      0/1071      1058/1071  = 98,79%
test  co_pkl      0/ 478       451/478   = 94,35%
```
Server **DA** tien xu ly valid va test. Ket luan "sua ghep theo `pli_id` se cho valid/test
rong" — thu ma toi goi la "phat hien moi quan trong hon con so mismatch" — **khong dung
tren server**. Nghia la:
* **KHONG** can chay lai step 0 cho valid/test.
* Dac ta P-1 tro lai dung nhu Rev 3 de xuat: ghep theo `pli_id` + loc `available_pkls`
  theo split. Don gian hon nhieu so voi de xuat cua toi.
* Van nen them `assert len(dataset) > 0` moi split — re, va chan dung lop loi nay.

Ro ri van con: `L = min(len(entries), len(pkls))` ⇒ train doc `pkl[0..23051]`,
valid `pkl[0..1070]`, test `pkl[0..477]` — valid/test van la tap con cua train.

### 15.5. B4 — xac nhan, hoi nang hon
Ty le token co `(d, theta, phi)` tat dinh: **48,26%** tren server (local 46,98%).

### 15.6. C1 (revert P0-5) — xac nhan tren server
`argmax|q|` duong = **1,0** ⇒ scipy da canonicalize san. `index6 < 0` = **34,04%** ⇒ P0-5
lat 34% hang va pha canonicalization xuong ~0,66. Ket luan khong doi: **revert**.

### 15.7. SUA LAI 14.2 — bang entropy dung (vocab server = 19.033)
```
cross-entropy phan phoi DEU  ln(19033)     = 9,854
H(token)                      unigram      = 7,910
H(token | vi tri trong chuoi)              = 6,552
---------------------------------------------------
MODEL epoch 5                              = 6,640   <- KEM baseline vi tri
MODEL epoch 21                             = 5,321   <- hon baseline vi tri 1,23 nat
```
Ket luan cua 14.2 **manh hon** sau khi sua: tai epoch 5 mo hinh con **te hon mot bang tra
(vi tri -> tan so token)**. Sau 22 epoch no hon baseline do 1,23 nat, va vi ghep cap la
100% ngau nhien, toan bo 1,23 nat phai den tu cau truc noi tai cua chuoi. Tieu chi H3
("< 8.0 tai epoch 5") van **vo hieu** — chinh baseline vi tri da la 6,55.

### 15.8. Con lai CHUA XAC NHAN: san nhieu quay 38,6 do
Day la con so **duy nhat con treo**, va no la co so cua toan bo khuyen nghi Phase 1
(dinh nghia lai nhan). Co che thi chac chan (code byte-identical, `_get_local_frame` dung
thu tu index nguyen tu), nhung **bien do phai do lai tren server** — nhat la vi dataset
lon hon 49,6%. Script H o `VERIFY_REV3_SCRIPTS.md`.

### 15.9. Bai hoc quy trinh
Local va server trung **code** nhung lech **du lieu**. Do la to hop te nhat: moi phep doc
code deu dung, nen do tin cay tong the bi nang len mot cach gia tao. Tu nay: moi con so
dataset phai do tren server; local chi dung de doc code va viet tai lieu.


### 15.10. B2 — XAC NHAN TREN SERVER. Muc 13 khong con muc nao treo.
```
                              SERVER        (local)
n_nhom >=2 cung chuoi SMILES   3.646        2.917
n_cap token                29.639.990    2.775.764
tat ca      : median           35,088       38,618
              p90              143,408     150,623
              >90 deg          23,73%      28,16%
cung PDB id : median            9,157        9,285
              p90              124,516     124,716
DOI CHUNG lech |d| : median  0,0239 A     0,0285 A
                     p90     0,1916 A     0,2285 A
std goc lech trong 1 cap: median 34,752 (<1deg 0,44%)   36,72
BASELINE hai token ngau nhien: median 132,95            132,80
```
Ca ba tieu chi giu nguyen tren 29,6 trieu cap token:
1. **Doi chung `|d|` = 0,0239 A** ⇒ cung phan tu, cung conformer. Chenh lech quay KHONG
   phai do khac conformer.
2. **San nhieu 35,1 deg** (toan bo), **9,16 deg** (cung PDB id — cung phan tu, cung hoc).
3. **std trong cung mot cap = 34,75 deg, chi 0,44% duoi 1 deg** ⇒ lech doc lap theo tung
   fragment ⇒ khiem khuyet o **frame cap fragment**, khong phai cap phan tu.

Ngan sach tin hieu tren server: san nhieu 35,1 / bien do tong 132,95.

**Muc 13 gio da xac nhan hoac bac bo het. Khong con muc nao o trang thai treo.**

---

## 16. Thiet ke lai nhan 7D: rang buoc phai ton trong `[LAP LUAN — chua do]`

Truoc khi viet `plan.md` cho Phase 1, co mot rang buoc kien truc ma khuyen nghi o luot
truoc cua toi da bo qua. Neu bo qua no, ta se thay mot nhan khong xac dinh bang mot nhan
khong xac dinh khac.

### 16.1. Encoder bat bien KHONG THE du doan phep quay trong mot frame co dinh
EGNN/UniTransformer sinh ra **feature bat bien SE(3)** (`residue_h`). Goi `f` la feature
do. Neu quay ca phuc hop mot goc `R`, thi `f` **khong doi**, nhung pose dung cua ligand
trong he toa do the gioi **doi thanh `R q`**. Mot head khong-equivariant doc `f` buoc phai
xuat cung mot gia tri cho hai muc tieu khac nhau ⇒ **san loi khong the vuot qua**, bat ke
kien truc head (Flow, MDN hay regression) va bat ke du lieu nhieu it ra sao.

⇒ Chi co hai duong hop le, phai chon mot:

**(A) Canonical hoa frame cua hoc.** Dinh nghia frame tu chinh cac nguyen tu hoc (vi du
PCA tensor quan tinh cua hoc), roi bieu dien CA translation VA rotation trong frame do.
Khi do muc tieu tro thanh **bat bien SE(3)**, va encoder bat bien co the hoc duoc.
*Chi phi:* re. *Rui ro:* xem 16.2.

**(B) Lam head equivariant.** Du doan vi tri bang cac cap nhat vector cua EGNN (nhu
TargetDiff / DiffSBDD / Pocket2Mol). Bo hoan toan nhu cau canonical hoa.
*Chi phi:* viet lai Step 2 + Step 4. *Loi:* khong con bai toan frame nao ca.

Phuong an (A) la duong re nhat de co mot pipeline chay duoc. Phuong an (B) la duong ma
cac mo hinh SBDD dang hoat dong deu di.

### 16.2. Neu chon (A): PCA cua hoc co the tao ra dung loai nhieu vua go bo
PCA co hai diem yeu, va ca hai deu tai tao lai B2 duoi dang khac:
* **Dau cua vector rieng khong xac dinh.** `v` va `-v` deu la vector rieng. Phai co quy
  uoc dau tat dinh (vi du: ep hinh chieu len mot vector tham chieu suy ra tu thanh phan
  hoc — tong vector tu tam hoc toi cac CA — phai duong). Khong co quy uoc nay thi frame
  lat ngau nhien giua cac complex ⇒ **san nhieu quay quay tro lai**.
* **Gia tri rieng gan bang nhau.** Hoc gan hinh cau ⇒ truc chinh khong xac dinh ⇒ frame
  khong on dinh. Day la B2 duoi dang khac.

**Phep do BAT BUOC truoc khi cam ket phuong an (A)** (chay tren server, chua co so):
voi moi hoc, tinh tri rieng `l1>=l2>=l3` cua tensor quan tinh cac nguyen tu hoc; bao cao
phan phoi cua `l2/l1` va `l3/l2`. Neu ty le token co `l2/l1 > 0.9` (hoac `l3/l2 > 0.9`)
la dang ke, PCA khong dung duoc va phai chuyen sang (B).

### 16.3. Frame cap fragment van phai sua rieng
Doc lap voi (A) hay (B): neu con du doan mot **quaternion cho tung fragment**, frame cua
fragment phai la ham cua hoa hoc, khong cua thu tu index. Sua: sap nguyen tu theo
**canonical rank cua RDKit** (`Chem.CanonicalRankAtoms`) truoc khi chon 3 diem dung frame.
Canonical rank la bat bien hoa hoc nen loai bo toan bo B2.

*Luu y:* neu chon (B) va du doan truc tiep toa do tung nguyen tu thi **khong con can
quaternion nao ca**, va B2 bien mat cung voi bieu dien 7D. Day la ly do (B) hap dan hon
so voi luc toi de xuat Phase 1 o luot truoc.

### 16.4. Tieu chi cong chan cho Phase 1 (cap nhat)
Do lai phep do 15.10 tren nhan MOI:
* **cung PDB id: median goc lech < 2 deg** (hien **9,157**) — day la tieu chi chinh, vi
  cung PDB nghia la cung phan tu va cung hoc.
* doi chung `|d|` phai giu nguyen co ~0,02 A (chung minh khong lam hong conformer).
* Them: sau khi sua ghep cap, `mismatch = 0%` ca ba split va `len(dataset) > 0`.


### 16.5. Do on dinh frame hoc (server, 29.503 cap ban sao) — (A) KHA THI co dieu kien
```
RMSD superpose: median 0.001 A   <- cac cap la ban sao gan nhu chinh xac (doi chung tot)
TAT CA                      : n=29503 median 0.002 deg p90 2.840 | <2deg 87.44% | >30deg 2.96%
chi hoc KHONG suy bien(<0.9): n=27619 median 0.001 deg p90 2.546 | <2deg 88.16% | >30deg 2.66%
```
**Doi chieu tieu chi ghi cung truoc khi do:** `median < 2 deg` ⇒ **DAT** (0,002). `<2deg > 90%`
⇒ **KHONG DAT** (87,44%). Ket qua roi vao khe ma bang tieu chi cua toi khong luong truoc:
**median chinh xac bang 0 nhung co duoi nang nho.**

**Hinh dang cua phan phoi chi ra nguyen nhan.** Khong phai suy bien dan dan — phan bo la
luong cuc: hoac dung chinh xac (median 0,002 deg, tuc bang khong ve mat so hoc), hoac sai
han (2,96% vuot 30 deg). Do la chu ky cua **lat dau vector rieng**, khong phai cua tri rieng
gan nhau. Bang chung: loc bo hoc suy bien hau nhu khong giup (2,96% -> 2,66%).

Quy uoc dau bang momen bac ba that bai khi momen bac ba **xap xi 0** — luc do dau do nhieu
so quyet dinh.

**Phan quyet: (A) kha thi, voi hai dieu kien bat buoc.**
1. **Cung hoa quy uoc dau.** Momen bac ba lam quy uoc chinh; khi `|m3|` duoi nguong, dung
   quy uoc du phong dua tren mot huong xac dinh bang **hoa hoc** (vi du: tam cac nguyen to N
   khung tru tam cac nguyen to O khung — bat bien SE(3), gan nhu khong bao gio bang 0).
   Ghi lai quy uoc nao da dung cho tung hoc.
2. **Tinh diem on dinh frame khi tien xu ly va LOC.** Moi complex nhan mot co `frame_stable`
   dua tren `|m3|` chuan hoa va khe tri rieng. Loai (hoac danh dau) cac complex khong on dinh.
   Chi phi ~3-13% dataset — re, va bien mot nguon nhieu im lang thanh mot bo loc tuong minh.

**So sanh muc do cai thien:** nhan hien tai co san nhieu **median 35,1 deg** tren 100% token.
Nhan de xuat co **median 0,002 deg** tren ~87% va duoi nang 3%. Sau khi loc theo
`frame_stable`, muc tieu la <2 deg cho >=99%.

**Cap nhat cong chan Phase 1:** tren tap DA LOC, `<2deg >= 99%` cho phep do 16.5, va
`cung PDB id median < 2 deg` cho phep do 15.10 tren nhan moi (hien 9,157).

### 16.6. (B) van la dich den dai han, khong phai buoc tiep theo
(A) cho duong ngan nhat den mot pipeline chay duoc — dung uu tien nguoi dung dat ra. (B)
(head equivariant, du doan toa do nguyen tu) van uu viet hon ve nguyen tac va xoa bo hoan
toan bai toan frame, nhung no doi viet lai Step 2 + Step 4. Ghi nhan de quay lai khi can
so sanh voi cac baseline SBDD.

### 16.7. Sua phan loai: B6 (ro ligand GT) KHONG phai viec mot dong
Luot truoc toi xep "bo `ligand_pos` khoi `step7:280`" vao Phase 0. **Sai.** `UniTransformer`
gop protein+ligand vao mot graph, va mo hinh da duoc **HUAN LUYEN** voi ligand GT trong
context. Bo ligand chi o eval se tao lech phan phoi train/eval. Sua dung doi hoi bo ligand
o **ca hai** phia va huan luyen lai ⇒ B6 thuoc Phase 2, khong phai Phase 0.
Hieu luc khong doi: moi so Vina/QED thu duoc den nay deu vo gia tri.


### 16.8. Do quy uoc dau hai tang (server, 30.386 cap) — G2 KHONG DAT, nhung vi loi DAC TA
```
n_cap ung vien cung PDB id: 42.396 -> sau loc RMSD<=1,0 A: 30.386  (RMSD median 0,0007 A)
TAT CA : median 0,002 deg  p90 5,797  | <2deg 83,14%  | >30deg 7,24%
STABLE : median 0,002 deg  p90 1,544  | <2deg 91,67%  | >30deg 0,15%
bi loai boi stable=False: 54,48% (16.553/30.386)
```
**Cong chan G2 yeu cau `<2deg >= 99%` tren nhom stable. Do duoc 91,67% ⇒ KHONG DAT.**

Nhung hai con so khac cho thay loi nam o dac ta cua toi, khong o phuong an (A):

**Loi dac ta 1 — `stable` coi "dung tang 2" la bat on dinh. Sai.**
Toi viet trong P1b: *"stable = True khi KHONG phai dung tang 2 cho bat ky truc nao"*. Tang 2
(huong tham chieu N/O khung) la mot quy uoc **tat dinh va bat bien SE(3)** — no la fallback
HOAT DONG, khong phai dau hieu that bai. Bang chung so: do PCA truoc do cho thay chi ~10%
hoc co khe tri rieng >= 0,9, nhung `stable=False` loai toi **54,48%** ⇒ khoang 45 diem phan
tram den tu dieu kien `|m3n| < m3_thresh`, tuc tu viec dung tang 2. Dat `m3_thresh = 0.05`
roi coi do la that bai la **tu loai bo mot nua dataset ma khong co ly do**.
*Sua:* `stable` phai la: voi TUNG truc, **it nhat mot** trong hai quy uoc cho dau dut khoat
(`|m3n|` tren nguong HOAC `|v.ref|` tren nguong), VA hai khe tri rieng OK.

**Loi dac ta 2 — cong chan tron lan hai nguyen nhan khac ban chat.**
`>30deg` tut tu 7,24% (tat ca) xuong **0,15%** (stable) — giam 48 lan. Nghia la co `stable`
bat rat tot cac ca lat dau tham hoa. Phan con lai (8,3% nam trong dai **2-30 deg**) khong
phai lat dau (180 deg) cung khong phai doi truc (90 deg) — do la lech **tu tu**. Ung vien
giai thich manh nhat: **khac biet cau truc that**. RMSD median la 0,0007 A nhung nguong loc
la 1,0 A, nen nhieu cap co RMSD hang phan muoi Angstrom; khi do tensor quan tinh **that su**
khac nhau va frame khac nhau vai do la frame dang phan anh dung mot khac biet vat ly, khong
phai nhieu nhan.

⇒ Cong chan `<2deg >= 99%` nhu toi viet **do lan ca vat ly vao**, nen no khong phan biet
duoc "frame bat on dinh" voi "hai hoc that su khac nhau". Dac ta sai, khong phai (A) sai.

**Tieu chi thay the, ghi cung TRUOC khi do (16.9):** chi xet cac cap co **RMSD < 0,05 A**
(cau truc gan nhu trung khop tuyet doi — moi chenh lech frame o day la nhieu nhan thuan
khiet). Tren nhom do, yeu cau `<2deg >= 99%`. Dong thoi phan tich `stable=False` theo
NGUYEN NHAN (khe tri rieng vs dung tang 2) de biet ty le loai thuc su can thiet la bao nhieu.


### 16.9. Do lai voi tieu chi dung (server, 29.503 cap) — **G2 DAT, (A) KHA THI**
```
*** CONG CHAN: RMSD<0,05 A VA stable(dn moi) -> n=19.828  <2deg=99,99%  >30deg=0,00%
    (yeu cau ghi cung truoc khi do: <2deg >= 99%)   => DAT
```
**Phan tich nguyen nhan `stable=False` (tren tung hoc, dn moi):**
```
tang 1 du (m3 dut khoat)                        47,10%
dung tang 2 (ref N/O) — VAN STABLE              43,87%
khe tri rieng >= 0,9                             6,04%
dau khong dut khoat (ca 2 tang that bai)         2,99%
--------------------------------------------------------
ty le loai THUC SU can thiet                     9,03%   (dac ta cu loai 54,48%)
```
⇒ Xac nhan **Loi dac ta 1**: coi "dung tang 2" la bat on dinh da tu loai bo **45 diem phan
tram** dataset ma khong co ly do. Huong tham chieu `centroid(N khung) - centroid(O khung)`
hoat dong tot: no giai quyet dut diem 43,87% hoc ma momen bac ba khong ket luan duoc.

**Phan tang theo RMSD — xac nhan Loi dac ta 2:**
```
dai RMSD (A)      n     median    p90     <2deg   >30deg
0,00-0,01     21.452     0,001    0,003   99,46%   0,54%
0,01-0,05        332     0,144    0,609   97,29%   1,20%
0,05-0,20      1.354     0,637    2,858   86,93%   3,91%
0,20-0,50      3.031     1,316  175,182   63,25%  10,39%
0,50-1,00      3.334     4,248  178,469   25,10%  19,11%
```
Goc lech frame tang **don dieu** theo RMSD ⇒ cong chan cu (`<2deg>=99%` tren moi cap co
RMSD toi 1,0 A) that su da do lan khac biet cau truc vat ly vao. Dac ta sai, khong phai (A) sai.

**Rui ro con lai, phai ghi ro:** o dai RMSD 0,20-0,50 A, `p90 = 175 deg` — do la **lat han**,
khong phai lech tu tu. Nghia la mot nhieu dong cau truc ~0,3 A co the day thu tu tri rieng
qua ranh gioi va lam doi truc. Frame la ham **khong lien tuc** cua cau truc o nhung hoc do,
tuc B2 duoi dang khac. Nguong khe tri rieng dat tay (0,9) la mot **proxy**, khong phai phep
do truc tiep cua tinh chat nay.

**=> Sua P1b lan hai: thay nguong dat tay bang PHEP THU JITTER truc tiep.** Thay vi doan
"khe 0,9 la du", hay do: nhieu dong toa do hoc bang Gaussian ~0,3 A vai lan, tinh lai frame,
va yeu cau moi frame khop nhau trong vai do. Dat truc tiep len tinh chat ta can (frame lien
tuc theo cau truc) thay vi len mot dai luong tuong quan voi no. Chi phi: ~5 lan PCA tren vai
tram nguyen tu — khong dang ke.


### 16.10. Chan doan xac nhan + chot thiet ke frame hoc (server, 29.503 cap)
**Chan doan "ngat tang" — xac nhan dut khoat:**
```
nhom STABLE, goc lech >30 deg : n=  901 | hai ban sao dung TANG KHAC NHAU = 94,89%
nhom STABLE, goc lech <=2 deg : n=24.923| hai ban sao dung TANG KHAC NHAU =  1,75%
```
Lam giau 54 lan. Nguyen nhan cac cu lat 180 deg **khong** phai suy bien tri rieng ma la
**nguong cung giua hai tang quy uoc dau**: mot khac biet cau truc nho day hai ban sao ve hai
phia nguong, va hai tang khong dong y ve dau. Phep thu jitter (sigma 0,3 A, 5 mau) khong du
suc bat vi nhieu dong co the khong day `|m3n|` qua ranh gioi.

**Head-to-head:**
```
                            n      median   <2deg   >30deg
PCA 2 tang (STABLE)      28.519    0,002    87,39%   3,16%
CHEM Gram-Schmidt        24.964    0,013    83,73%   0,29%
RMSD<0,05: PCA           21.113    0,001    99,98%   0,00%
RMSD<0,05: CHEM          18.684    0,009    99,98%   0,00%
CHEM: ty le hoc bi loai = 15,01%
```
**Doi chieu tieu chi ghi cung truoc khi do:** CHEM `<2deg >= 99%` ⇒ DAT (99,98%).
`>30deg <= 0,1%` o RMSD<0,05 ⇒ DAT (0,00%). Dieu kien thoat sang (B) la `>30deg > 0,5%` tren
tat ca cap ⇒ KHONG kich hoat (0,29%). Ty le loai `<= 15%` ⇒ **15,01%, sat ranh gioi** — toi
ghi ro thay vi lam tron: nguong 15% la phan doan mem cua toi, khong phai rang buoc vat ly,
va mat 15% complex la chap nhan duoc (dac ta dau tien cua toi mat 54%).

**CHOT: dung CHEM Gram-Schmidt. Bo PCA khoi P1b. Bo luon phep thu jitter** (khong con thu tu
tri rieng nao de dao). Dac ta cuoi o `plan.md` P1b.

**Ghi nhan quy trinh:** ba lan sua P1b deu la vá mot thiet ke co che do hong noi tai. Cach
ra khoi vong lap khong phai do thêm lan nua ma la **chon mot cau truc khong co ngưỡng**.
Day dung la loi da phe phan o `Log-Clamp` cua Rev 2 — va toi lap lai no ba lan truoc khi
nhan ra.


### 16.11. Xac nhan trien khai P1a-P1d (2026-09-29) — DAT, du de tai sinh du lieu
**Doi chieu `verify_frame_stability.py` (Claude Code trien khai) voi phep do doc lap 16.10:**
```
                    trien khai    tham chieu 16.10
median                 0,013           0,013      khop
<2deg (STABLE)        83,52%          83,73%      lech 0,21 pp
>30deg (STABLE)        0,32%           0,29%      lech 0,03 pp
ty le loai            15,15%          15,01%      lech 0,14 pp
```
Trong sai so. Trien khai khop dac ta.
Phan tich nguyen nhan loai (moi): `n1 < tau` 7,30% · `n2 < tau` 8,09% · thieu nhom nguyen tu
0,29%. Hai nguyen nhan can bang nhau ⇒ `tau = 0,30 A` la rang buoc chi phoi ca hai huong
tham chieu. **Co the tuning `tau` de thu hoi dataset ve sau; KHONG phai blocker.**

**Doc code xac nhan (local == server, cung HEAD):**
* P1a: `CanonicalRankAtoms(frag, breakTies=True)` co try/except; `coords_canon` duoc dung o
  `step0:211`. DAT.
* P1b: khong con `eigh`/momen bac ba/jitter trong code (chi con trong comment lich su). DAT.
* P1c: `_get_pocket_frame` goi o `step0:205`, thay dung frame phan tu cu. Dinh dang pkl doi
  thanh dict `{"frame_stable", "frame_diag", "sequence"}`. DAT.
  **Xac nhan them (quan trong): frame duoc dung tu pocket APO** (`_load_pocket_geometry(apo_pocket_path)`,
  `step0:204`), khong phai holo ⇒ khong ro thong tin ligand vao dinh nghia frame.
* P1d: `pkl_by_id` ghep theo `entry[2].split("/")[0]`; loc `frame_stable`; `assert len > 0`;
  vong lap `max_attempts` truot index DA BI BO. DAT.
  Loc token `frag_frame_stable` **dong bo** giua `input_ids_list` va `target_7d_list` (cung
  mot vong lap, `continue` truoc ca hai `append`) ⇒ khong lech chi so. DAT.

**Mot van de NGU NGHIA con lai (khong chan tai sinh du lieu):** `step6` dang loai tung
fragment co `frag_frame_stable=False` ra khoi giua chuoi. Dieu do day cho mo hinh mot phan tu
**thieu fragment** — te hon la giu fragment do voi nhan quay nhieu. Nen doi thanh **loai ca
complex**. Sua nay nam trong `step6`, khong anh huong `step0`, nen lam song song trong luc
tai sinh du lieu.


---

## 17. Ket qua Phase 1 lan 1 (server, 2026-09-30) — nhan MOI tot hon nhan cu 500 lan

### 17.1. P1c (frame hoc) THANH CONG
```
ty le nhan tat dinh (pos 0,1,2) : ~48%  ->  0,0140%
d (pocket-relative)             : mean 5,083  std 2,469  max 23,662 A
                                  (cu, intramolecular: mean 3,94  max 27,13)
```
Muc 6.1 va 6.5 dong. 48% nhan co entropy bang 0 gio la du lieu thuc.

### 17.2. G1/G3 — phai phan tang theo do giong nhau cua HOC
Do lan dau (khong loc) cho `G1 = 27,9 deg`, `G3 = 0,2146 A` — **te hon** nhan cu (9,157 /
0,0239) va toi da bao dong sai. Nguyen nhan: nhan gio la **pocket-relative**, nen hai ban
sao co **tap nguyen tu hoc khac nhau** se co nhan khac nhau **mot cach hop le**. Script do
dau tien khong co bo loc nao ve hoc ⇒ lai chinh la Loi dac ta 2 cua Muc 16.8, lan thu ba.

Do lai co phan tang (29.388 cap):
```
dieu kien                          n      G1 med   G3 med
TAT CA                          29.388   19,085   0,1861
tap nguyen tu hoc trung >=99%   10.052    0,018   0,0001
  + RMSD hoc <= 0,3 A            8.907    0,005   0,0000
  + RMSD hoc <= 0,05 A           7.906    0,002   0,0000
nhan CU (tham chieu)                       9,157   0,0239
```
**Khi hoc thuc su trung nhau: G1 = 0,018 deg (nhan cu 9,157) — tot hon 500 lan.
G3 = 0,0001 A (nhan cu 0,0239) — gan nhu chinh xac tuyet doi.**

**Ca ba cong chan DAT ngay tren du lieu dang co tren dia:** G1 0,018 < 2 · G3 0,0001 < 0,05
· [A] 0,014% tat dinh.

### 17.3. SUA LAI ket luan 13.5 — nguon nhieu chinh la frame PHAN TU, khong phai frame fragment
Muc 13.5 ket luan: *"khiem khuyet nam o frame cap fragment"*, dua tren lap luan dai so
(neu chi frame phan tu doi thi do lech phai giong nhau o moi vi tri, std = 0; do duoc std
36,7 deg ⇒ phai la cap fragment).

**Lap luan dung nhung du lieu dau vao bi nhiem.** `std = 36,7 deg` do tren cac cap gom theo
`smiles_tuple` **khong loc theo hoc**, nen ca hai frame deu khac nhau va std bi thoi phong.

Bang chung quyet dinh tu 17.2: P1a **KHONG co hieu luc** trong lan chay nay (`coords_canon`
roi ve `coords` vi `CanonicalRankAtoms` nem o 100% fragment), tuc `R_g_to_w` **y nguyen nhu cu**.
Thay duy nhat la `R_m` (frame phan tu tu 3 fragment dau theo alphabet) -> `R_p` (frame hoc).
Va G1 tut tu 9,157 xuong 0,018. ⇒ **9,14 deg trong 9,157 deg nhieu cu den tu frame PHAN TU.**

Frame fragment (thu tu index nguyen tu) khong gay lech giua cac ban sao doi xung — hop ly,
vi cac ban sao doc tu cac file SDF do cung mot pipeline ghi ra, thu tu nguyen tu trung nhau.
Co che B2 van that (M6: hoan vi ngau nhien -> median 135 deg), nhung **muc do no bieu hien
trong dataset nay thi chua do duoc**, va ro rang nho hon nhieu so voi uoc luong cua toi.

### 17.4. Cot "tam hoc lech ~50 A" la loi thiet ke phep do cua toi
Toi muon do dich chuyen tam hoc **sau khi superpose**, nhung lai tinh tren toa do tuyet doi,
nen no do khoang cach giua hai ban sao trong o mang tinh the — vo nghia. `G3 = 0,0001 A` da
tra loi dung cau hoi do: `t_p` on dinh.


## 18. Phase 1 lan 2 (server, 2026-09-30) — 3/4 cong chan DAT

### 18.1. Ket qua
```
step0:
  frame_stable=False       : 5441/23660  (23,00%)   [n1<tau 3292 · n2<tau 2438 · thieu nhom 84]
  frag_frame_stable=False  :    0/143091 fragment (0,00%)     <-- UpdatePropertyCache DA CHUA
  complex co >=1 fragment nhu vay: 0/23660 (0,00%)
  vocab 19.033 · n_token 143.091

verify_label_quality:
  [A] tat dinh : 20/143091 = 0,01%          (truoc Phase 1: ~48%)
  [B] d        : mean 5,0830 std 2,4690 max 23,6622
  [C] G1/G3 phan tang theo do giong nhau cua hoc (37.926 cap complex):
      TAT CA                     n_token 166.142   G1 17,5620   G3 0,1520
      frac>=0,99                 n_token  56.478   G1  0,0155   G3 0,0001
      frac>=0,99 & RMSD<=0,3     n_token  49.606   G1  0,0049   G3 0,0000
      frac>=0,99 & RMSD<=0,05    n_token  43.943   G1  0,0019   G3 0,0000
      (nhan CU tham chieu: G1 9,157   G3 0,0239)

verify_frame_stability: y nguyen (frame hoc khong doi)
  STABLE: median 0,013 deg · <2deg 83,52% · >30deg 0,32% · loai 15,15%
```

### 18.2. Doi chieu cong chan
| # | Yeu cau | Ket qua | |
|---|---|---|---|
| G1 | cung PDB median < 2 deg | **0,0155** | DAT |
| G2 | RMSD<0,05 + stable: <2deg >= 99% | 99,98% | DAT |
| G3 | \|d\| < 0,05 A | **0,0001** | DAT |
| G4 | mismatch 0% ca 3 split, len(dataset) > 0 | **chua co so** | CHO |

G4 can output cua `compute_scaler.py` (no in `__init__` cua dataset truoc assert nho muc 3
cua prompt) — chua duoc dan ve.

### 18.3. Ke toan trung thuc ve chi phi 2,5 gio chay lai
P1a chi cai thien G1 tu **0,018 -> 0,0155 deg**. Tuc lan chay lai **khong mua duoc cai thien
do luong duoc nao dang ke** — dung nhu 17.3 da du doan sau khi sua lai ket luan 13.5: frame
fragment khong phai nguon nhieu chinh giua cac ban sao doi xung.

Cai no mua duoc la **bao dam theo cau truc**: nhan gio bat bien thu tu index nguyen tu
**bang thiet ke** (canonical rank), khong phai nho cac file SDF tinh co co cung thu tu. Phep
do G1 chi nhin duoc cac ban sao cung PDB — dung cai tap con ma thu tu nguyen tu chac chan
trung — nen no **mu** voi truong hop P1a can xu ly. Do la ly do da neu truoc khi chay, va no
van dung; nhung phai ghi ro: day la chi phi tra cho mot bao dam khong quan sat duoc, khong
phai cho mot con so tot hon.

### 18.4. Con lai cua Phase 1
Chi G4. Sau do Phase 1 dong, va nut that dau tien cua Phase 2 la **B5**: `h_target` phai
mang feature bat bien **kem toa do residue trong frame hoc canonical**. Voi (A) da chot,
day khong phai viec phu ma la dieu kien de nhan quay hoc duoc — encoder bat bien khong the
suy ra huong tu mot bag-of-residue-descriptors khong co hinh hoc (Muc 16.1).


### 18.5. G4 DAT — Phase 1 DONG (2026-09-30)
```
[train] tong entry 23052 · co pkl 22151 · loai boi frame_stable 4951 · loai boi frag_frame_stable 0 · valid 17200
        (22151 - 4951 = 17200, so hoc khop)
```
Ghep theo `pli_id` nen mismatch **khong the xay ra ve cau truc**, khong con la con so phai do.
`valid > 0`. **G4 DAT ⇒ ca bon cong chan Phase 1 DAT.**

**Scaler moi:**
```
shift = [5,1614 · 1,5316 · 0,3913 · 0,2004 · 0,2043 · 0,2018 · 0,1857]
scale = [2,4783 · 0,6649 · 1,7598 · 0,4565 · 0,4581 · 0,4573 · 0,4643]
```
Ba xac nhan doc lap tu chinh scaler nay:
1. `shift[0]` (d) = **5,16** (cu 3,79) — khop du doan 5,08 tu [B]. Lech nho vi compute_scaler
   chay tren tap train DA LOC (17.200) con [B] do tren toan bo 23.660.
2. `shift[1]` (theta) = 1,5316, gan `pi/2 = 1,5708` — theta gio la bien tu do phan bo quanh
   pi/2 nhu mot huong gan dang huong, khong con la hang so cua frame.
3. `scale[2]` (phi) = **1,7598**, rat gan `pi/sqrt(3) = 1,814` la do lech chuan cua phan bo
   DEU tren [-pi, pi]. Tuc `phi` gio gan nhu dong deu — mot bien tu do thuc su.
   **Truoc Phase 1 `phi` la delta function tren gan nua du lieu.** Day la bang chung manh
   nhat, va doc lap voi phep do [A].

**P0a (revert P0-5) da duoc ap dung** — suy ra tu chinh scaler: bon thanh phan quaternion co
shift ~0,19-0,20 va scale ~0,456-0,464, tuc chu ky canonicalize "argmax duong" cua scipy.
Neu P0-5 con hieu luc thi `shift[6]` se la ~0,44 nhu lan do 29/09.

### 18.6. Trang thai sau Phase 1
| Muc | Trang thai |
|---|---|
| 4 (ghep sai cap) | **DONG** — ghep theo `pli_id` |
| 6.1 (48% nhan tat dinh) | **DONG** — 0,01% |
| 6.2 / P0-5 | **DONG** — da revert |
| 6.3 (sai chi so) | **DONG** — DIM_NAMES da sua |
| 6.4 (nhan quay khong phai ham cua phan tu) | **DONG** — G1 9,157 -> 0,0155 deg |
| 6.5 (frame noi phan tu, mu hoc) | **DONG** — pose gio pocket-relative |
| 8.1 (h_target 1 vector mean-pool) | MO — Phase 2, nut that dau tien |
| 8.2 (ro ligand GT) | MO — Phase 2 |
| 8.3 (thieu token embedding) | MO — Phase 2 |
| 7 (Step 7 ETKDG) | MO — Phase 2 |
| N1 (token sap theo alphabet) | MO — chua sua, ghi nhan |
| N2 (bien dang apo->holo bi vut) | HOAN theo yeu cau |
| anti-target | HOAN theo yeu cau |

Dataset: **17.200 mau train** (tu 23.052 entry; mat 901 do thieu pkl, 4.951 do frame_stable).


---

## 19. Ket qua Phase 2 lan 1 (server, 2026-10-01) — cong chan TRUOT, nhung nguyen nhan da ro

> **[CANH BAO — DOC MUC 20 TRUOC]** Cac tieu muc 19.1, 19.2, 19.3, 19.4 **DA BI RUT LAI**
> ngay 2026-10-01 boi Audit Rev 5 (Muc 20). Hai tien de cua chung deu sai:
> (1) P0b **DA** duoc ap dung trong run (Muc 20.1), nen khong can "doc theo vi tri cot";
> (2) permutation cua ARN **ngau nhien moi process** vi khong co seed (Muc 20.3), nen con so
> `[4,0,2,1,6,3,5]` do o mot process khac **khong noi gi** ve run nay.
> Van ban goc duoc giu nguyen lam dau vet sai sot, **khong duoc dung lam can cu**.
> Phan con gia tri: cac gia tri `logscale_mean` — xem phan ra lai o Muc 20.4.

### 19.1. ~~Gate: 1/4 (can >=3/4)~~ [RUT LAI — xem 20.1, 20.2]
Doc theo VI TRI cot (P0b chua duoc ap dung nen ten cot van sai: file ghi `qx,qy,qz,qw`
cho vi tri 4-7, thu tu 7D that la `[d, theta, phi, qw, qx, qy, qz]`).

### 19.2. ~~Permutation cua ARN — GIA THUYET CUA TOI BI BAC~~ [RUT LAI — xem 20.3]
```
permutation = [4, 0, 2, 1, 6, 3, 5]
```
Pyro `create_mask` dat `var_index[permutation[k]] = k`, nen **thu tu tu hoi quy** (truoc ->
sau) la: `dim4, dim0, dim2, dim1, dim6, dim3, dim5` = **qx, d, phi, theta, qz, qw, qy**.

Toi du doan: *"qx (index 4) phai dung SAU ca 3, 5, 6 thi moi hoc duoc rang buoc ||q||=1"*.
Thuc te **qx dung DAU TIEN**, dieu kien hoa tren KHONG gi ngoai context. Gia thuyet
"chieu sup do hoc rang buoc chuan tu ba chieu kia" **BI BAC** — no khong co quyen truy cap
vao ba chieu kia.

### 19.3. ~~Phat hien thay the: luong hoc duoc phu thuoc VI TRI TU HOI QUY~~ [RUT LAI — xem 20.3, 20.4]
Day la bat hinh mau tren mot phep hoan vi NGAU NHIEN khong lien quan den run. Thay the: 20.4.
Sap xep `logscale_mean` (epoch 24) theo thu tu tu hoi quy:
```
thu tu  chieu     logscale   sigma
  0     qx         -1,357    0,257     <-- duy nhat hoc duoc
  1     d          -0,196    0,822
  2     phi        -0,052    0,949
  3     theta      -0,053    0,948
  4     qz         -0,043    0,958
  5     qw         -0,023    0,977
  6     qy         -0,044    0,957
```
**Don dieu theo vi tri**, khong theo loai dai luong. Ba chieu quay "khong hoc" nam o vi tri
4, 5, 6; chieu quay "hoc duoc" nam o vi tri 0. `d` o vi tri 1 hoc duoc mot it (-0,196);
`theta`/`phi` o vi tri 2-3 thi khong.

**Giai thich:** trong MADE 1 lop, chieu o vi tri 0 co duong di context -> (mu, sigma)
**khong bi mask**; cac chieu sau phai di qua hidden unit bi mask theo thu tu input, lam
nghen duong truyen tin hieu context. Nen cai quyet dinh "hoc duoc bao nhieu" la **vi tri
trong permutation**, chu khong phai thong tin co san.

### 19.4. ~~He qua: hoc CO mang thong tin ve phep quay~~ [RUT LAI — xem 20.3, 20.5]
**KHONG co bang chung nao cho ket luan nay.** Chieu sup do la `qy` (khong phai `qx`), vi tri
tu hoi quy cua no KHONG BIET, va con it nhat hai giai thich tam thuong hon (20.5).
Chieu duy nhat co duong di sach (qx, vi tri 0) dat `sigma = 0,257` so voi base 1,0 — tuc
**context du doan duoc qx voi do lech chuan 0,257** (chuan hoa) = 0,117 don vi quaternion.
Do la thong tin that, trich tu hoc.

⇒ Ket luan **khong** phai "phep quay khong hoc duoc". Ket luan la **`ConditionalAffineAutoregressive`
1 lop dang chan 6/7 chieu khong cho dung thong tin do.**

### 19.5. Giai quyet quyet dinh treo tu Muc 11
Muc 11 hoan quyet dinh Flow/MDN/Regressor vi "khong co du lieu de quyet". **Gio co.**
Muc 11 cung da ghi: voi MAF 1 lop, chieu dau trong permutation co
`p(x1|c) = N(m1(c), e^{2s1(c)})` — don dinh tuyet doi, nen "uu the da dinh cua Flow" la gia
dinh chua kiem chung. Du lieu 19.3 cho thay con te hon the: 6 chieu con lai gan nhu khong
duoc huan luyen.

**Phep thu quyet dinh, re nhat:** thay `AutoregressiveFlowLayer` bang mot **head Gaussian
duong cheo** — `mu(c), log_sigma(c)` cho ca 7 chieu, khong co coupling tu hoi quy. Moi chieu
deu co duong di sach y het chieu vi tri 0 hien nay.
* Neu ca 7 `logscale` deu tut ⇒ chan doan 19.3 xac nhan, va van de da duoc giai quyet luon.
* Neu chi mot vai chieu tut ⇒ phan con lai that su khong du doan duoc tu hoc, va luc do moi
  xet toi bieu dien (6D rotation) hoac MDN.

### 19.6. Hai quan sat phu
* `nll_median` phang (10,66 -> 10,21 qua 25 epoch) va `nll_p99` phan ky (25 -> 216).
  **Nhung phep do nay bi hong:** reservoir lay mau tu `(-log_prob).flatten()` **truoc khi
  mask padding**, con `Geo_Loss_Unclamped` thi sau mask. Bang chung: mean (8,13) < median
  (10,21), khong the xay ra voi phan phoi co p99 = 184. Phai sua reservoir mask truoc khi
  lay mau.
* `Token_Loss` 7,21 -> **3,349**, vuot `H(token|vi tri)` = 6,552 ba nat va tiem can bigram
  in-sample 2,920. Run cu (ghep ngau nhien) chi toi 5,321. **Co the** la dieu kien hoa theo
  hoc da hoat dong cho luong ngu nghia, nhung du lieu cung da doi (17.200 mau sach thay vi
  23.052 ghep sai) nen chua ket luan duoc — doi chung pocket-blind van la phep thu duy nhat.

---

## 20. Audit Rev 5 (2026-10-01) — rut lai 19.1-19.4; ba loi ha tang

Can cu: `logs/audit_rev5.log` (server, HEAD `a1bfe91`) + doc code tren local HEAD `bcb3e99`.
Khong chay training, khong doi code. Chi doc va tinh.

### 20.1. P0b DA duoc ap dung trong run — 19.1 sai tien de
```
95a505a  2026-09-29 18:35:10 +0700  "P0b: sua thu tu DIM_NAMES..."
95a505a in HEAD (a1bfe91, 2026-09-30 11:19:10)  : YES
run bat dau   : logs/phase2_0930_1731.log  => 2026-09-30 17:31
step2 DIM_NAMES do truc tiep tren server       : ['d','theta','phi','qw','qx','qy','qz']
```
⇒ `DIM_NAMES` **dung** trong suot run. Cau "P0b chua duoc ap dung" o 19.1 la sai.

### 20.2. Ten cot trong `flow_scale_trace.csv` LA DUNG — co che tra cuu theo TEN
`src/utils/system_logger.py` hardcode `["d","theta","phi","qx","qy","qz","qw"]` o **ca** dong 33
(header) va dong 54 (row). Nhung dong 60-61 tra cuu bang:
```python
row += [stats.get(f"logscale_mean_{d}", "") for d in dim_names]
```
tuc tra cuu **theo TEN key**, khong theo vi tri. Con `step2_mdn_module.py:103-105` sinh key
**theo ten** tu `DIM_NAMES` da dung:
```python
for i, name in enumerate(DIM_NAMES):
    stats[f"logscale_mean_{name}"] = mean_ls[i].item()
```
⇒ Moi cot chua dung gia tri cua chieu ma ten cot goi. Thu tu **cot** khac thu tu `DIM_NAMES`
(mot phep hoan vi cua cung mot tap key), nhung **khong cot nao bi gan sai nhan**.

⇒ **Chieu sup do (`logscale_mean = -1,357`) la `qy`, index 5 cua `target_7d`.** Khong phai `qx`.

### 20.3. KHONG CO SEED — permutation cua ARN ngau nhien moi process
```python
self.arn = ConditionalAutoRegressiveNN(out_dim, hidden_dim, [hidden_dim, hidden_dim])
```
**Khong truyen `permutation`.** Pyro mac dinh sinh `torch.randperm(input_dim)` tu RNG toan cuc.
Va `grep -rn "seed|deterministic" src/*.py src/utils/*.py` cho duong **train**: **KHONG CO GI**.
(Chi `diag_flow_state.py:219,229` va `diag_phi_tail.py:59` co `manual_seed` — do la script **do**,
chay trong process rieng.)

⇒ Permutation `[4,0,2,1,6,3,5]` do o mot process khac co xac suat **1/5040 = 0,02%** trung voi
permutation cua run. **Muc 19.2, 19.3, 19.4 bi rut lai.**

⇒ Nang hon pham vi Muc 19: **ca du an khong tai lap duoc.** Moi run co permutation khac, init
khac, thu tu shuffle khac. Moi so sanh giua cac run qua cac lan sua code tu truoc den nay deu bi
nhieu boi mot bien gay nhieu khong kiem soat, voi bien do khong biet. Day la loi phuong phap
luan nang nhat con lai trong du an.

### 20.4. Phan ra NLL — thay cho 19.3, DOC LAP voi permutation
Muc tieu da chuan hoa (variance = 1), nen baseline "khong hoc gi" la `N(0,1)^7`:
`H = 7 x 0,5 ln(2 pi e) = 9,933` nat.

Epoch 24, ten cot **dung** theo 20.2:

| chieu  | logscale_mean | sigma | var. giai thich (1-s^2) | dong gop vao gain |
|--------|---------------|-------|-------------------------|-------------------|
| d      | -0,1965       | 0,822 | **32,5%**               | 0,197  (11,1%)    |
| theta  | -0,0525       | 0,949 | 10,0%                   | 0,053  (3,0%)     |
| phi    | -0,0515       | 0,950 | 9,8%                    | 0,052  (2,9%)     |
| qw     | -0,0428       | 0,958 | 8,2%                    | 0,043  (2,4%)     |
| qx     | -0,0228       | 0,977 | 4,5%                    | 0,023  (1,3%)     |
| qy     | **-1,3574**   | 0,257 | **93,4%**               | **1,357 (76,8%)** |
| qz     | -0,0440       | 0,957 | 8,4%                    | 0,044  (2,5%)     |

`sum(ln sigma) = -1,7674`. NLL du doan **chi tu cac sigma**, gia dinh calibrated (`E[z^2]=1`):
```
7 x 0,5 ln(2 pi)  +  7 x 0,5  +  sum(ln sigma)
   6,4326         +    3,5     +   (-1,7674)      =  8,165
```
Mean NLL thuc do = **8,13**. Lech 0,4% ⇒ `E[z^2] ~ 0,99`, mo hinh **calibrated**.

**Ket luan (dung bat ke permutation):** toan bo NLL duoc giai thich **chi bang cac sigma**.
- **5/7 chieu — `theta, phi, qw, qx, qz` — co sigma 0,95-0,98, tuc dung NGUYEN tai phan phoi
  marginal sau 25 epoch.** Giai thich 4-10% variance = nhieu.
- `d` giai thich 32,5%. Co hoc, it — va `d` la chieu de nhat (bi rang buoc boi do dai lien ket
  hoa hoc, doan duoc khong can biet hoc).
- **Mot chieu duy nhat (`qy`) chiem 76,8% toan bo tien bo.**

⇒ Cach doc dung: **dau hinh hoc khong hoc duoc gi ve pose tu hoc trong 25 epoch**, va chieu duy
nhat co hoc la chieu dang nghi nhat. Nguoc hoan toan voi 19.4.

### 20.5. Ba giai thich cho `qy`, va phep do re nhat de phan xu
- **(a) rang buoc dai so `||q||=1`** — chi kha thi neu `qy` o cuoi permutation. KHONG BIET (20.3).
- **(b) thoai hoa trong DU LIEU** — phan phoi `qy` gan suy bien hoac luong cuc, nen "du doan"
  duoc ma khong can thong tin nao tu hoc. Day la giai thich tam thuong nhat va **chua he duoc
  loai**.
- **(c) thong tin tu hoc** — gia thuyet cua 19.4.

**Phep do phai lam TRUOC khi chay lai bat cu thu gi** (khong GPU, khong train): thong ke marginal
7 chieu `target_7d` tren tap train — mean/std/skew/kurtosis **truoc** chuan hoa, histogram, va ty
le khoi luong trong mode lon nhat. Neu `qy` suy bien hoac luong cuc ⇒ (b) xac nhan, va 76,8%
"tien bo" cua mo hinh la hien vat du lieu.

### 20.6. `_ls_sum` / `_pin_sum` KHONG mask padding, con `Geo_Loss` thi co
`step4_ligand_generator.py:128-129` truyen `pad_mask_flat = (target_ids != 0).view(-1)` vao head,
nhung head **chi dung no cho reservoir NLL**. `_ls_sum += ls.sum(0)` va `_ls_count += ls.shape[0]`
cong **moi hang**, ke ca padding. Con `step5_constraints_and_loss.py:44-45`:
```python
pad_mask = (target_ids != 0).float()
loss_geo_unclamped = (loss_geo_raw * pad_mask).sum() / (pad_mask.sum() + 1e-8)
```
**co** mask. ⇒ `logscale_mean` va `Geo_Loss_Unclamped` tinh tren **hai tong the khac nhau**.

Anh huong len 20.4 la **nho**, co can cu: target cua hang padding bang 0 o **ca 7 chieu**, nen
neu padding chiem phan lon va duoc hoc voi `sigma -> 0` thi **ca 7** `logscale` phai tut va
`pinfrac` phai lon. Thuc te `pinfrac = 0` va 6/7 chieu o `sigma 0,95-0,98`. ⇒ padding **khong**
la nguyen nhan cua hinh mau 20.4. Nhung van phai mask de so lieu doc duoc chinh xac.

Commit `bcb3e99` da sua reservoir nhung **chua** sua `_ls_sum`/`_pin_sum`. Sua chua hoan chinh.

### 20.7. `DiagonalGaussianHead` — da kiem, DUNG
Doc `src/step2_mdn_module.py` (local `bcb3e99`):
- NLL `= 0,5 z^2 + log_sigma + 0,5 ln(2 pi)`, `.sum(-1)` ⇒ shape `[N]`. **Dung.**
- `log_sigma.clamp(-5,0, 3,0)`. **Dung** (cung san voi MAF).
- `pad_mask` loc reservoir. **Dung.**
- Nhanh sampling: `mu + log_sigma.exp() * randn_like(mu)`, khong chuan hoa quaternion — nhat quan
  voi MAF, step4 lo sau inverse-scale. **Dung.**
- `_ls_sum` la tensor thuong, khong phai buffer ⇒ khong vao `state_dict`. **Dung.**
- `context_proj` **y het** MAF ⇒ so sanh cong bang ve kien truc context. Nhung xem 20.8.
- Thieu: `_ls_sum`/`_pin_sum` cung khong mask padding (cung loi 20.6).

### 20.8. `nn.Tanh()` cuoi `context_proj` — co che chua bao gio duoc do
**Ca hai** head ket thuc `context_proj` bang `nn.Tanh()`. Neu pre-activation lon, Tanh bao hoa,
gradient qua no `-> 0`, va **encoder khong nhan duoc tin hieu hoc tu dau hinh hoc**. Day la mot
giai thich kha di cho "5/7 chieu khong hoc gi" (20.4) **hoan toan doc lap** voi cau hoi
MAF-vs-diagonal.

Va vi ca hai head dung **chung** kien truc nay, phep so sanh diag-vs-MAF **khong the** phat hien
no. Phai do rieng: ty le `|tanh_out| > 0,99` va chuan gradient tai `context_proj[0].weight`.

### 20.9. Server tre 4 commit so voi local
```
server a1bfe91  <-- run 25 epoch nay
local  bcb3e99
thieu: 91c4d9a, 4bd8aa6 (DiagonalGaussianHead), b58f1a2 (geo_head), bcb3e99 (reservoir mask)
```
Xac nhan tren server: `DiagonalGaussianHead` = **NOT FOUND**; `grep -rn geo_head src/` chi khop
chinh script audit.
⇒ **Run 25 epoch nay la MAF**, khong phai `diag_gauss`. Phan tich 20.4 ap cho MAF. Dung.

### 20.10. `step7_evaluation.py` — con treo
Chi co `SAMGLightningModule.load_from_checkpoint` (dong 264). `geo_head` nam trong config cua
step6 (dong 232 doc, dong 383 mac dinh `"diag_gauss"`). Neu `save_hyperparameters()` luu ca config
thi step7 tu khoi phuc dung head; neu khong, step7 se dung mac dinh `diag_gauss` va **load sai
state_dict** cho checkpoint MAF. **Chua kiem.** Phai kiem truoc khi chay step7 lan dau.

### 20.11. Ke toan trung thuc
Toi da ket luan "day la bang chung cung dau tien rang hoc protein mang thong tin ve huong xoay"
dua tren mot permutation do o process khac, trong khi doan code sinh permutation nam ngay trong
file toi da doc nhieu lan va khong he truyen `permutation`. Loi khong phai thieu du lieu — la
khong kiem tien de truoc khi xay ket luan len no. **Day la lan thu ba trong du an toi suy dien
thay vi do** (xem 15.4, 16.8, 17.2). Quy tac bo sung: truoc khi dung bat ky con so do nao de ket
luan, phai tra loi duoc "con so nay co phai cua DUNG run dang xet khong?".

---

## 21. M1 — thong ke marginal 7D (server, 2026-10-01): gia thuyet (b) BI LOAI

`logs/marginal_7d.log`. n = 101.791 token / 17.200 complex.

### 21.1. Doc theo tieu chi ghi cung truoc khi chay (plan Rev 5, M1)
| tieu chi (b) xac nhan neu...        | nguong    | `qy` do duoc | ket qua |
|-------------------------------------|-----------|--------------|---------|
| khoi luong trong 3/40 bin lon nhat  | >= 50%    | **10,49%**   | KHONG   |
| `|excess kurtosis|`                 | > 10      | **1,19**     | KHONG   |
| `std_raw(qy)` so voi `std_raw(qw)`  | < 0,1x    | **1,002x**   | KHONG   |

⇒ **(b) BI LOAI.** `qy` khong thoai hoa, khong roi rac, khong luong cuc.

### 21.2. `qy` khong phan biet duoc voi ba chieu quaternion con lai
| | `qw` | `qx` | **`qy`** | `qz` |
|---|---|---|---|---|
| mean (raw)        | 0,2004 | 0,2043 | **0,2018** | 0,1857 |
| std (raw)         | 0,4565 | 0,4581 | **0,4573** | 0,4643 |
| skew              | -0,0535 | -0,0711 | **-0,0579** | -0,0111 |
| excess kurtosis   | -1,1805 | -1,1875 | **-1,1917** | -1,2111 |
| gia tri duy nhat  | 95.462 | 95.468 | **95.503** | 95.494 |
| bin lon nhat      | 3,65% | 3,76% | **3,75%** | 3,40% |

Bon chieu **dong nhat ve moi thong ke bien**. Khong co gi trong du lieu khien `qy` de du doan
hon `qw/qx/qz`. ⇒ `sigma = 0,257` **phai** den tu dieu kien hoa, khong tu phan phoi bien.

### 21.3. Xac nhan phu: canonicalization quaternion hoat dong dung
`min(qy) = -0,7035 ~ -1/sqrt(2) = -0,7071` o ca bon chieu quaternion. Dung nhu du doan cua quy
uoc "thanh phan lon nhat mang dau duong": mot thanh phan khong the xuong duoi `-1/sqrt(2)`, vi
khi do no se la thanh phan lon nhat ve do lon va bi lat dau. Can cu doc lap rang P0a (revert
P0-5) da dung va canonicalization cua scipy con nguyen.

### 21.4. Con lai (a) va (c) — va (a) co mot du doan DINH LUONG kiem duoc
- **(a) rang buoc dai so `||q||=1`** — kha thi **chi khi** `qy` dung sau ca `qw, qx, qz` trong
  permutation. Xac suat tien nghiem = 1/4.
- **(c) thong tin tu hoc protein** — tien de cua ca du an.

Neu (a) dung thi co **mot** du doan dinh luong, khong phai suy dien dinh tinh:
biet `qw, qx, qz` ⇒ `qy^2 = 1 - qw^2 - qx^2 - qz^2` ⇒ `|qy|` xac dinh **chinh xac**. Chi con
**dau**. Va dau cung xac dinh duoc trong mot phan truong hop: neu `|qy| > max(|qw|,|qx|,|qz|)`
thi theo quy uoc canonicalization `qy` **phai** duong (neu am thi no da bi lat). Ty le `qy` la
thanh phan troi ~ **1/4** theo doi xung.
⇒ Phan phoi dieu kien cua `qy` la mot **hon hop**: ~25% gan nhu tien dinh (`sigma -> 0`), ~75%
luong cuc `+-|qy|`. Mot Gaussian don dinh khop vao hon hop do cho `mean(log_sigma)` am manh
trong khi `pinfrac` van co the = 0 (mang neural khong dat tan san `-5`).
**Du doan kiem duoc:** phan phoi `log_scale` cua `qy` phai **luong cuc**, va `log_scale` thap
phai tuong quan voi su kien `|qy| > max(|qw|,|qx|,|qz|)`.

### 21.5. Phep do quyet dinh: lay permutation THAT tu checkpoint — xem plan Rev 5 M3
Pyro luu mask cua `MaskedLinear` lam **buffer** ⇒ mask nam trong `state_dict` ⇒ **permutation
cua dung run 25 epoch phuc hoi duoc tu checkpoint**, khong can doan, khong can chay lai.
* `qy` dung sau ca `qw, qx, qz` ⇒ **(a) XAC NHAN**. Ket thuc tranh luan. Viec can lam la
  **bieu dien quay** (6D continuous rotation), khong phai doi head, khong phai sua encoder.
* `qy` **khong** dung sau ca ba ⇒ **(a) BI LOAI**, con lai **(c)**. Do se la bang chung dau
  tien — lan nay co can cu — rang hoc mang thong tin ve huong. G5 van phai chay de xac nhan.

---

## 22. M3 — permutation THAT cua run 25 epoch (server, 2026-10-01): (a) XAC NHAN

`logs/m3_permutation.log`, checkpoint `saved_checkpoints_flow/periodic-epoch=024.ckpt`.
Mask lay tu `state_dict` ⇒ day la permutation cua **dung** run da tao ra `flow_scale_trace.csv`.

### 22.1. Thu tu tu hoi quy that
```
qx(0) -> d(1) -> phi(2) -> theta(3) -> qz(4) -> qw(5) -> qy(6)
```
`qy` o **vi tri 6/6**, nhin thay ca `qw, qx, qz`. Dung truong hop 1/4 ma 21.4 neu.
⇒ Theo tieu chi ghi cung o plan Rev 5 M3: **(a) XAC NHAN.**

### 22.2. Bang chung manh hon tieu chi: THANG ALGEBRA khop hoan toan
Doc ma tran phu thuoc cho **tung** chieu quaternion, roi doi chieu voi luong thong tin ma
`||q||^2 = qw^2+qx^2+qy^2+qz^2 = 1` cho phep suy ra:

| chieu | vi tri AR | thay duoc (quaternion) | rang buoc con lai | du suy ra? | `sigma` do duoc |
|-------|-----------|------------------------|-------------------|-----------|-----------------|
| `qx`  | 0 | — | `qw^2+qy^2+qz^2 = 1-qx^2` (3 an) | **khong** | 0,977 |
| `qz`  | 4 | `qx` | `qw^2+qy^2+qz^2 = 1-qx^2` (3 an) | **khong** | 0,957 |
| `qw`  | 5 | `qx, qz` | `qw^2+qy^2 = 1-qx^2-qz^2` (2 an) | **khong** | 0,958 |
| `qy`  | 6 | `qx, qz, qw` | `qy^2 = 1-qx^2-qz^2-qw^2` (1 an) | **CO**, `|qy|` chinh xac | **0,257** |

Luong "hoc duoc" cua tung chieu quaternion **bang dung** luong ma rang buoc dai so cho khong:
khong, khong, khong, day du. Ba chieu chua du thong tin deu nam o `sigma ~ 0,96`; chieu duy nhat
du thong tin tut xuong `0,257`.

⇒ **Mo hinh hoc duoc rang buoc chuan don vi cua quaternion, va KHONG GI KHAC.** Khong co bang
chung nao cho thay hoc protein dong gop vao du doan huong xoay.

### 22.3. 19.3 sai theo chieu NGUOC LAI, khong phai sai do permutation
Voi ten cot dung, `logscale_mean` theo vi tri tu hoi quy:
```
vi tri 0  qx     -0,0228     <- 19.3 noi vi tri 0 hoc duoc NHIEU NHAT
vi tri 1  d      -0,1965
vi tri 2  phi    -0,0515
vi tri 3  theta  -0,0525
vi tri 4  qz     -0,0440
vi tri 5  qw     -0,0428
vi tri 6  qy     -1,3574     <- thuc te chi vi tri CUOI hoc duoc
```
19.3 ket luan "chieu o vi tri 0 co duong di context khong bi mask nen hoc duoc nhieu nhat".
**Nguoc hoan toan.** Vi tri 0 (`qx`) hoc duoc **it nhat** trong bon chieu quaternion. Khong co
hieu ung "MADE 1 lop chan 6/7 chieu" nao. Vi tri 6 hoc duoc nhieu nhat, va ly do la dai so.

### 22.4. SUA LAI Muc 20.3 — quy sai nguyen nhan
Permutation phuc hoi tu checkpoint quy ve `[4, 0, 2, 1, 6, 3, 5]` — **trung khop chinh xac** voi
con so do o luot truoc. Nen con so do **khong** sai; thu gay ra loi 19.2/19.3/19.4 la **anh xa
ten cot** (`qx` <-> `qy`), khong phai permutation.

Muc 20.3 noi "con so `[4,0,2,1,6,3,5]` khong noi gi ve run" — **phan quy nguyen nhan do la sai**.
Phan con lai cua 20.3 **van dung va van nghiem trong**: khong co seed o duong train, va
`ConditionalAutoRegressiveNN` khong duoc truyen `permutation` ⇒ cac run ve sau co the co thu tu
khac, nen P3a van bat buoc. Nhung no khong phai nguyen nhan cua sai sot o Muc 19.
(Hoac phep do luot truoc da doc tu checkpoint — khong con nguon de xac dinh; du sao ket luan
hien tai khong phu thuoc vao dieu do.)

### 22.5. CHUA KHOP: `pinfrac = 0` mau thuan voi phien ban don gian cua (a)
Neu (a) la **toan bo** co che thi phan phoi dieu kien cua `qy` phai la hon hop:
* `|qy| > max(|qw|,|qx|,|qz|)` (~25% theo doi xung) ⇒ dau xac dinh ⇒ `sigma -> 0`
* con lai (~75%) ⇒ dau tu do ⇒ hai diem `+-|qy|`, Gaussian don dinh tot nhat co `sigma ~ |qy|`
  ⇒ chuan hoa ~0,98 ⇒ `log sigma ~ -0,02`

Giai cho ty le `f` o san: `f*(-5,0) + (1-f)*(-0,02) = -1,357` ⇒ `f = 0,268`.
⇒ Du doan `pinfrac(qy) ~ 0,27`. **Do duoc `pinfrac(qy) = 0,0`** (tru hai epoch le te ~7,7e-7).

⇒ `log sigma` cua `qy` **khong** luong cuc nhu phien ban don gian cua (a) doi hoi. Mo hinh dang
lam gi do ngoai viec khoi phuc `|qy|`: no co thong tin **mot phan** ve dau cho phan lon mau
(co the tu token fragment qua C3, tu `d/theta/phi`, hoac tu hoc).

**Day la cau hoi mo duy nhat con lai cua Muc 19-22, va 21.4 da ghi cung phep thu cho no.**
Xem plan Rev 5 M4. Khong duoc ket luan "(a) giai thich tron ven" truoc khi M4 chay.

---

## 23. M4 (server, 2026-10-01): (a) xac nhan ve CO CHE, bac ve DO LON — va mot loi do luong moi

`logs/m4_qy_signflip.log`, checkpoint `periodic-epoch=024.ckpt`, **n = 88 token that (1 batch)**.

### 23.1. Tieu chi ghi cung cua TOI dung SAI thong ke
M4 ghi: "chenh `mean` > 1,5 nat ⇒ (a) tron ven". Do duoc `1,2019` ⇒ script in "(a) CHI LA MOT
PHAN". **Ket luan tu dong do sai**, va loi la cua toi khi dat tieu chi.

Du doan o 21.4 la ve **tinh LUONG CUC** — mot phan phoi hai dinh. `mean` la thong ke **sai** cho
gia thuyet do: hai nhom co the tach gan hoan toan ma hieu `mean` van nho, va nguoc lai. Dang le
phai ghi cung mot do do **tach biet** (AUC / do chong lan), khong phai hieu trung binh.

### 23.2. Doc lai bang histogram: hai nhom gan NHU ROI NHAU
| | `dom=True` (n=30) | `dom=False` (n=58) |
|---|---|---|
| mean | **-1,560** | **-0,358** |
| median | **-1,752** | **-0,330** |
| mien gia tri | [-2,15 ; -0,26] | [-1,33 ; +0,20] |
| ty le duoi `-1,0` | **25/30 = 83%** | **2/58 = 3,4%** |

Nguong `-1,0` tach 83% vs 3,4%. AUC uoc tu histogram ~**0,95** (chua tinh chinh xac, n nho).
⇒ **Du doan 21.4 dung ve co che.** `sigma` tut xuong **dung** o nhung mau ma
`|qy| > max(|qw|,|qx|,|qz|)`, tuc dung o nhung mau ma canonicalization **xac dinh dau** cua `qy`.
Do la du doan dac trung cua (a), va no khop.

Luu y chi bao nay chong lai quan sat chu khong ho tro no: `dom=True` nghia la `|qy|` **lon**, va
mot Gaussian khop vao `+-m` co `sigma = m` ⇒ `|qy|` lon thi `sigma` phai LON. Do duoc nguoc lai.
Nen quan sat chi co the giai thich bang "dau da xac dinh", khong bang "bien do nho".

### 23.3. Do lon bi BAC — va dieu do giai thich `pinfrac = 0` ma khong can nguon tin nao moi
22.5 du doan `sigma -> 0` o nhanh xac dinh (`log sigma -> -5`, `pinfrac ~ 0,27`).
Do duoc: `dom=True` chi xuong `-2,15`, `pinfrac = 0` o **ca hai** nhom.

Giai thich don gian hon gia thuyet "co nguon thong tin khac" cua 22.5:
`ConditionalAffineAutoregressive` mot lop cho phan phoi dieu kien **Gaussian don dinh**. Phan
phoi dung o nhanh dau tu do la **hai diem** `+-|qy|`. Gaussian **khong the** bieu dien hai diem.
Nen mo hinh thoa hiep: co `sigma` o vung `|qy|` lon (dau xac dinh), giu `sigma ~ 0,72` o cho con
lai, va **khong bao gio** dat san. Them nua `sqrt(1 - qw^2 - qx^2 - qz^2)` la ham mo hinh phai
**xap xi**, khong phai tinh chinh xac.

⇒ **Cau hoi mo o 22.5 DONG.** Khong can "nguon thong tin ve dau tu C3/hoc". (a) la co che, duoc
thuc hien **xap xi**. Van **khong** co bang chung nao ve dong gop cua hoc protein.

### 23.4. LOI DO LUONG MOI — `flow_scale_trace.csv` bi nhiem padding NANG, khong nhe
| | gia tri |
|---|---|
| `logscale_mean_qy` trong CSV (epoch 24, **moi hang**) | **-1,357** |
| `mean(log_scale_qy)` tren **token THAT** (M4) | **-0,768** |

Lech **43%**. ⇒ Muc 20.6 ket luan "anh huong len 20.4 la **nho**" — **SAI**. Lap luan cua toi o
20.6 ("neu padding chi phoi thi CA 7 chieu phai tut") khong loai duoc kha nang padding tut **vua
phai** o moi chieu, va so lieu cho thay no tut that.

⇒ **Toan bo phan ra NLL o Muc 20.4 dung so bi nhiem.** Ket luan dinh tinh ("5/7 chieu dung o
marginal, mot chieu chiem phan lon gain") **chua** bi bac nhung **chua** duoc chung minh lai.
Khong duoc dung bang 20.4 lam can cu cho bat cu quyet dinh nao cho den khi do lai.

Doi chieu them cho thay chua khop: neu `qy` that = `-0,768` thi `sum(ln sigma) = -1,178` va NLL du
doan = `8,754`, trong khi `Geo_Loss_Unclamped` (da mask) = `8,13`. Hai so khong khop ⇒ cac chieu
khac cung bi nhiem theo huong chua biet, hoac `E[z^2] < 1`. **Khong suy dien them** — do lai.

### 23.5. `n = 88` qua nho
88 token tu **mot** batch. `ty le dom = 30/88 = 34,09%`, sai so chuan `5,1%` ⇒ khop voi du doan
`25%` o khoang `1,8 sigma`. Histogram 40 bin tren 88 mau phan lon la bin rong hoac 1 mau.
Moi con so o 23.2-23.4 phai do lai tren nhieu batch truoc khi ghi thanh ket luan dinh luong.

### 23.6. Ke toan
Hai lan trong hai luot lien tiep toi dat tieu chi dinh luong **sai thong ke** cho gia thuyet minh
muon kiem: nguong `-0,3` tuyet doi cho `logscale` (het hieu luc, Muc 20 / plan Rev 5), va nguong
`1,5 nat` tren hieu `mean` cho mot gia thuyet ve **tinh luong cuc** (23.1). Quy tac bo sung:
**tieu chi phai do dung dai luong ma gia thuyet noi ve** — gia thuyet ve hinh dang phan phoi thi
tieu chi phai la do do hinh dang, khong phai moment bac mot.

---

## 24. M5 (server, 2026-10-01): phan ra SACH. Muc 20.4 BI BAC; (a) duoc CUNG CO

`logs/m5_logscale.log`. 200 batch, **n = 18.890 token that** / 38.912 hang (⇒ padding **51,45%**).
Phan ra NLL khop **chinh xac**: NLL do = NLL du doan = `7,9361`, lech `0,0000` nat. So dang tin.

### 24.1. Bang phan ra dung — thay the hoan toan bang 20.4
| chieu | `mean(log sigma)` | `sigma` | `R2 = 1-sigma^2` | dong gop vao `sum(ln sigma)` | 20.4 (bi nhiem) noi |
|-------|-------------------|---------|------------------|------------------------------|---------------------|
| `d`     | -0,3651 | 0,694 | **51,8%** | 20,5% | 32,5% |
| `theta` | -0,1032 | 0,902 | **18,6%** | 5,8%  | 10,0% |
| `phi`   | -0,1010 | 0,904 | **18,3%** | 5,7%  | 9,8%  |
| `qw`    | -0,2633 | 0,768 | **40,9%** | 14,8% | 8,2%  |
| `qx`    | -0,0249 | 0,975 | **4,9%**  | 1,4%  | 4,5%  |
| `qy`    | -0,7862 | 0,456 | **79,2%** | 44,2% | 93,4% |
| `qz`    | -0,1341 | 0,875 | **23,5%** | 7,5%  | 8,4%  |

**Tieu chi M5: so chieu quay co `R2 < 0,15` = 1/4** (chi `qx`) ⇒ theo tieu chi ghi cung,
**20.4 BI BAC**.

Cu the cai gi bi bac: cau "**5/7 chieu dung nguyen o marginal**". Sai. Chi `qx` o marginal
(4,9%). `qz` 23,5%, `qw` 40,9%, `theta/phi` ~18,5%, `d` 51,8%. Va `qy` chiem **44,2%** gain chu
khong phai 76,8%.

Nguyen nhan: `_ls_sum` cong ca padding (51,45% hang), do nhiem **17%–88%** tuy chieu (24.4).

### 24.2. Thang algebra: (a) duoc CUNG CO, va 22.2 phai viet lai
Sap bon chieu quaternion theo **so chieu quaternion di truoc** trong thu tu tu hoi quy
(`qx(0) -> d -> phi -> theta -> qz(4) -> qw(5) -> qy(6)`):

| chieu | so quaternion di truoc | thay | `R2` |
|-------|------------------------|------|------|
| `qx`  | **0** | — | **4,9%** |
| `qz`  | **1** | `qx` | **23,5%** |
| `qw`  | **2** | `qx, qz` | **40,9%** |
| `qy`  | **3** | `qx, qz, qw` | **79,2%** |

**Don dieu tang hoan hao theo so tien to.** 22.2 doc nhi phan ("khong du suy ra ⇒ khong hoc gi")
va do la **sai**: `||q||=1` voi tien to thieu khong cho gia tri chinh xac nhung cho **chan tren**
(`|qz| <= sqrt(1-qx^2)`, v.v.), va chan cung lam giam phuong sai. Voi so bi nhiem, ba chieu dau
deu trong nhu nhau (~0,96) nen toi doc thanh "khong hoc gi"; voi so sach, moi tien to them mua
duoc mot luong giam phuong sai that.

⇒ **(a) khong chi con dung ma manh hon:** toan bo cau truc hoc duoc cua bon chieu quaternion la
mot ham don dieu cua luong rang buoc dai so san co. Khong can vien den hoc protein.

### 24.3. Loai bo nham lan "vi tri AR" — 19.3 van sai, nhung co can cu moi
Co the nghi thang 24.2 chi la hieu ung "vi tri cang sau cang hoc duoc nhieu" (gia thuyet 19.3).
**Bac bang `d, phi, theta`:** vi tri 1, 2, 3 nhung `R2` = **51,8%**, 18,3%, 18,6%.
`d` o vi tri **1** co `R2` cao nhat trong ba chieu nay. Vi tri **khong** giai thich duoc.

⇒ Thang 24.2 la hieu ung **dai so**, khong phai hieu ung **vi tri**. 19.3 van bi bac.
`d` (51,8%) hop ly vi do dai lien ket cong hoa tri bi rang buoc chat boi hoa hoc, doan duoc tu
**token fragment** (C3) ma khong can biet hoc.

### 24.4. Do nhiem padding, do truc tiep
| chieu | `mean(real)` | `mean(ALL)` | lech |
|-------|--------------|-------------|------|
| `d`     | -0,3651 | -0,1724 | 52,8% |
| `theta` | -0,1032 | -0,0213 | 79,4% |
| `phi`   | -0,1010 | -0,0620 | 38,7% |
| `qw`    | -0,2633 | -0,0308 | 88,3% |
| `qx`    | -0,0249 | -0,0206 | 17,1% |
| `qy`    | -0,7862 | **-1,4327** | 82,2% |
| `qz`    | -0,1341 | -0,0198 | 85,3% |

Chu y **huong** lech: `qy` la chieu **duy nhat** co `mean(ALL)` am HON `mean(real)`; sau chieu
kia deu am IT hon. Giai thich: target cua hang padding la mot diem co dinh (vector 0), nen trong
khong gian chuan hoa no la mot gia tri duy nhat. `qy` o cuoi thu tu AR, thay ca ba chieu
quaternion dang o gia tri padding ⇒ xac dinh hoan toan ⇒ `sigma` rat nho. Cac chieu khac thay it
tien to hon nen khong xac dinh duoc. Tuc **hang padding tai tao dung thang 24.2, phong dai**.

⇒ Muc 20.6 ("anh huong nho", voi lap luan "neu padding chi phoi thi CA 7 chieu phai tut") **sai
ca ket luan ca lap luan**: padding tut **chon loc** theo dung thu tu AR, khong deu.

### 24.5. `E[z^2]`: chi `qy` lech — dau van tay thu hai cua co che dau
| chieu | `d` | `theta` | `phi` | `qw` | `qx` | **`qy`** | `qz` |
|---|---|---|---|---|---|---|---|
| `E[z^2]` | 1,014 | 0,966 | 0,984 | 0,942 | 1,014 | **0,675** | 0,968 |

Sau chieu calibrated (0,94–1,01). Rieng `qy` = **0,675** ⇒ residual thuc **nho hon** `sigma` mo
hinh dua ra ⇒ mo hinh **thieu tu tin** o `qy`: no con co the giam `sigma` de duoc NLL tot hon.

Khop dung co che 23.3: Gaussian don dinh khong bieu dien duoc hai diem `+-|qy|`, nen no phong
`sigma` len de phu ca hai nhanh; o tap con ma dau da xac dinh (`dom=True`, 34%) residual rat nho
⇒ `E[z^2]` tong tut xuong duoi 1. **Day la bang chung doc lap thu hai** cho co che dau, khong
dung chung so lieu voi 23.2.

### 24.6. Doc lai Muc 20-23 theo yeu cau cua tieu chi M5
| muc | trang thai sau M5 |
|-----|-------------------|
| 20.1 provenance, 20.2 ten cot | **con hieu luc** (doc lap voi M5) |
| 20.3 khong co seed | **con hieu luc** la khuyet diem; quy nguyen nhan da sua o 22.4 |
| **20.4 phan ra** | **BI BAC** ⇒ thay bang **24.1** |
| 20.5 ba gia thuyet (a)(b)(c) | khung van dung |
| **20.6 "padding anh huong nho"** | **BI BAC** ⇒ thay bang **24.4** |
| 20.7 `DiagonalGaussianHead` dung | **con hieu luc** |
| 20.8 gia thuyet bao hoa Tanh | **yeu di**: cac chieu CO hoc (18–52%) nen gradient co chay qua Tanh. Van chua loai kha nang Tanh chan **rieng** duong tu hoc. Giu lam gia thuyet, ha uu tien |
| 20.9 server tre, run la MAF | **con hieu luc** |
| 20.10 step7 `geo_head` | **GIAI QUYET**: truyen `config=` tuong minh vao `load_from_checkpoint` (M5 dong 113-114 da lam dung) |
| 21 M1, (b) bi loai | **con hieu luc** (chi dung du lieu) |
| 22.1 thu tu AR | **con hieu luc** |
| **22.2 bang "du/khong du suy ra"** | **VIET LAI** ⇒ **24.2** (doc nhi phan la sai; thuc te don dieu theo so tien to) |
| 22.3 19.3 sai | **con hieu luc**, can cu moi o **24.3** |
| 22.5 `pinfrac=0` | **DONG** o 23.3; them bang chung doc lap o **24.5** |
| 23 M4 (`dom`) | **con hieu luc** |

### 24.7. Con lai gi CHUA giai thich duoc — va do la cho duy nhat tin hieu hoc co the an
Sau khi tru (i) `qy/qw/qz` giai thich bang thang dai so 24.2 va (ii) `d` giai thich bang hoa hoc
qua token:
* `theta` **18,6%** va `phi` **18,3%** — **khong** co kenh dai so nao (chung khong phai quaternion),
  va khong ro bao nhieu den tu token fragment, bao nhieu tu hoc.
* phan cua `d` vuot qua muc token giai thich duoc.

**Day la hai chieu duy nhat con lai ma hoc protein co the dang dong gop**, va chung cung la hai
chieu **khong** bi nhieu boi kenh dai so ⇒ **doc chung trong G5 la sach**. Nen G5 van la phep thu
quyet dinh, va khong can doi head truoc.

---

## 25. M6 (server, 2026-10-01): HOC PROTEIN CO DUOC DUNG — va tieu chi cua toi sai LAN THU BA

`logs/m6_pocket_shuffle.log`. 200 batch, 19.104 token that, ba nhanh forward tren **CUNG** batch.
Derangement thanh cong 200/200, khong fallback.

### 25.1. Canh bao khong-doi: nguyen nhan la LOADER, khong phai code
* `measure_logscale_realtokens.py:128` dung `shuffle=True`; `measure_pocket_shuffle.py:177` dung
  `shuffle=False` ⇒ **hai bo 200 batch khac nhau**. Token that 18.890 (M5) vs 19.104 (M6) xac nhan.
* Thay doi o `step6_trainer.py` nam trong `if pocket_shuffle_perm is not None:` ⇒ **no-op chung
  minh duoc bang doc code** khi `perm=None`. Da doc diff commit `7ce0f06`.
* `gradient_clip_val=1.0`, `accumulate_grad_batches=2` **con nguyen** (dong 495-496); them
  `deterministic=True` (dong 499). Khong mat tham so nao.
⇒ Canh bao **khong** vo hieu hoa M6, vi M6 la thiet ke **ghep doi**: ba nhanh chay tren cung batch.

**Phu luc:** commit `7ce0f06` cung lam luon **P3a-2** (`pl.seed_everything(1234)`, `"seed"` vao
config, `deterministic=True`, in `git HEAD`/`SEED`). Nhieu hon yeu cau nhung dung viec can.
**CANH BAO su dung:** dong in `permutation ARN: ...` **KHONG dang tin khi load checkpoint**.
`arn.permutation` la attribute thuong, **khong** phai buffer ⇒ khong nam trong `state_dict`.
Khi load, model duoc dung moi (permutation = randperm cua RNG hien tai) roi nap `state_dict`, va
`state_dict` chi phuc hoi **mask** (buffer). Mask moi la thu quyet dinh tinh toan, nen phep do
dung; nhung con so **in ra** la cua lan dung moi, khong phai cua checkpoint. Day la ly do M3 phai
doc mask chu khong doc attribute.

### 25.2. TIEU CHI CUA TOI SAI: `R2 = 1 - sigma^2` chi dung khi mo hinh CALIBRATED
`R2 = 1 - sigma^2` do **do khong chac chan mo hinh TU KHAI BAO**, khong do **sai so thuc**.
Phuong sai du thuc (muc tieu da chuan hoa, variance = 1) la:
```
MSE_i = E[(x_i - mu_i)^2] = sigma_i^2 * E[z_i^2]        =>  R2_i = 1 - sigma_i^2 * E[z_i^2]
```
Xao tron hoc **khong** doi `sigma` bao nhieu (mo hinh van tu tin nhu cu khi duoc dua mot hoc
**that nhung sai**) nhung lam `mu` **sai di** ⇒ hieu ung hien o `E[z^2]`, khong o `sigma`.
Tieu chi `1 - sigma^2` cua toi **mu** voi dung loai hieu ung can do.

Bang chung so: `mean(NLL)` full `8,2948` vs shuffled `8,5155` ⇒ **+0,2207 nat**. Phan ra:
`0,5 * sum(delta E[z^2]) ~ 0,19` + `delta sum(ln sigma) ~ 0,03` = `0,22`. Khop. Hieu ung **co that**
va nam gan hoan toan trong `E[z^2]`.

### 25.3. Bang `R2` DUNG (`= 1 - sigma^2 * E[z^2]`)
| chieu | `R2(full)` | `R2(shuffled)` | **delta shuffled** | `R2(roll1)` | **delta roll1** |
|-------|-----------|----------------|--------------------|-------------|------------------|
| `d`     | 0,3856 | 0,2150 | **+0,1706** | 0,2386 | **+0,1470** |
| `theta` | 0,0621 | 0,0345 | **+0,0276** | 0,0390 | **+0,0232** |
| `phi`   | 0,1146 | 0,0874 | **+0,0272** | 0,0917 | **+0,0228** |
| `qw`    | 0,4123 | 0,3979 | +0,0143 | 0,3996 | +0,0127 |
| `qy`    | 0,8420 | 0,8285 | +0,0135 | 0,8305 | +0,0115 |
| `qz`    | 0,2468 | 0,2362 | +0,0105 | 0,2394 | +0,0073 |
| `qx`    | 0,0065 | 0,0038 | +0,0027 | 0,0042 | +0,0023 |

**Ca bay chieu duong, o CA HAI doi chung, cung thu tu do lon.** `roll1` nho hon `shuffled` mot cach
he thong — hop ly vi `shuffle=False` khien cac phan tu lien ke trong batch la lan can trong thu tu
dataset, nen `roll(1)` ghep voi lan can giong nhau hon ⇒ pha it hon.

### 25.4. Ket luan thuc: hoc protein chi phoi TINH TIEN, khong chi phoi PHEP QUAY
Phan tach dung theo y nghia hinh hoc cua 7D `[d, theta, phi | qw, qx, qy, qz]`:
* **Tinh tien** (`d, theta, phi`): `delta R2` = **0,171 / 0,028 / 0,027**. Hoc protein **co** dong gop.
* **Phep quay** (quaternion): `delta R2` = 0,003–0,014, tuc **gan nhu khong co gi**; va toan bo cau
  truc hoc duoc cua chung da duoc giai thich bang thang dai so `||q||=1` (24.2).

⇒ **Day la xac nhan THUC NGHIEM cho Muc 16.1**, von la lap luan ly thuyet: *encoder bat bien SE(3)
khong the du doan phep quay trong mot frame co dinh*. 16.1 noi tren giay; 25.3 do duoc.

Luu y: `theta/phi` la **huong tinh tien** (goc cau cua vector dich chuyen), **khong** phai phep quay.
Moi cho nao trong Muc 19-24 goi `theta/phi` la "chieu quay" deu la cach goi sai cua toi.

### 25.5. Gia thuyet moi, co co che: SHORTCUT DAI SO DANG CANH TRANH voi tin hieu hoc
Duong dieu kien hoa **khong** chet: `d` co `delta R2 = 0,171`, tuc thong tin hoc **co** chay qua
encoder -> `h_target` -> attention hub -> head. Nhung nos khong den duoc quaternion.

Giai thich kha di: voi quaternion, **duong dai so re hon duong hoc**. Gradient descent lay duong
re. `||q||=1` cho san 86% phuong sai cua `qy` ma khong can doc `h_target`; hoc cach doc hoc protein
de suy huong xoay thi kho hon nhieu. Day la **shortcut learning** kinh dien.

⇒ **Dao lai quyet dinh o plan Rev 6:** truoc do toi ha `DiagonalGaussianHead` xuong "sau G5" vi
"cau hoi con lai nam o `theta/phi`, vo nhiem voi kenh dai so". Lap luan do dung nhung **khong day
du**: cat kenh dai so khong chi lam phep do sach hon, no co the **go bo thu dang canh tranh** voi
tin hieu hoc o quaternion. Nang `DiagonalGaussianHead` len thanh **thi nghiem tiep theo**, voi gia
thuyet tuong minh va kiem duoc.

### 25.6. SUA bang 24.1 — cung dung sai cong thuc `R2`
`R2` dung (`1 - sigma^2 * E[z^2]`) tren so lieu M5:
| chieu | `R2` trong 24.1 (sai) | **`R2` dung** |
|-------|----------------------|---------------|
| `d`     | 51,8% | **51,1%** |
| `theta` | 18,6% | **21,4%** |
| `phi`   | 18,3% | **19,6%** |
| `qw`    | 40,9% | **44,4%** |
| `qx`    | 4,9%  | **3,5%**  |
| `qy`    | 79,2% | **86,0%** |
| `qz`    | 23,5% | **26,0%** |
Thang dai so 24.2 **van don dieu** (3,5% -> 26,0% -> 44,4% -> 86,0%) ⇒ ket luan 24.2 khong doi.

### 25.7. Ke toan — va mot ranh gioi phuong phap luan phai ton trong
**Lan thu ba** toi dat tieu chi dinh luong sai (nguong tuyet doi `-0,3`; hieu `mean` cho gia thuyet
luong cuc; nay `1 - sigma^2` cho hieu ung nam o `mu`). Lan nay hau qua nang hon: tieu chi tu dong
in ra **"hoc protein KHONG dong gop gi"**, trong khi so lieu cho dieu **nguoc lai**.

Ranh gioi: toi suy ra cong thuc `R2` dung **SAU KHI** xem du lieu. Nen moi con so o 25.3 la
**tham do (exploratory)**, khong phai xac nhan. **Khong duoc** coi 25.4 la ket luan da chung minh.
Phai xac nhan lai tren du lieu **chua dung** (tap valid, 1.058 complex) voi nguong dat **truoc**.
Xem plan M7. Neu bo qua buoc nay thi day dung la retrofit nguong cho vua du lieu — dung loi ma
toan bo Muc 19-24 da pha.

---

## 26. M7-a (server, 2026-10-01): KIEM KHONG-DOI **FAIL**. Muc 25 bi DINH CHI

`logs/m7_valid.log`. split=valid, shuffle=False, batch_size=16, 48 batch, **4.869 token that**
(764 complex). Derangement 48/48, sizematched 0/764 khong di chuyen duoc (0,00% — tot).

```
KIEM TRA KHONG-DOI: perm=None vs perm=torch.arange(B), 3 batch dau
    batch 0: FAIL    batch 1: FAIL    batch 2: FAIL
```

Theo tien dang ky: **khong doc M7-a, khong doc M7-b.** Va vi M6 dung **cung** co che
`pocket_shuffle_perm`, **toan bo Muc 25 bi dinh chi** cho den khi giai quyet xong.

### 26.1. Hai nguyen nhan kha di, phan biet duoc bang DO LON cua chenh lech
**(i) Lam tron do layout bo nho — VO HAI.**
`h_target[torch.arange(B)]` dung ve gia tri nhung tao tensor **moi, contiguous**. `h_target` tu
`to_dense_batch` co the khong contiguous. Cac phep sau (matmul, attention) tren tensor co stride
khac nhau co the goi **kernel CUDA khac**, cho lam tron dau phay dong khac o nhung bit cuoi.
Dau hieu: `max|delta|` co bac **1e-7** (epsilon cua float32).

**(ii) `perm` dang duoc ap vao tensor SAI HINH DANG — PHA HUY TAT CA.**
Neu tai diem ap, `h_target` **khong** phai `[B, R_max, H+3]` ma da bi lam phang thanh
`[B*R_max, ...]`, thi `[torch.arange(B)]` **chi chon B hang dau** va bo phan con lai. Voi `perm`
ngau nhien thi nhanh `shuffled` khong phai "xao tron hoc" ma la "cat bo gan het hoc" ⇒ moi so o
Muc 25 va M7 **la rac**.
Dau hieu: `max|delta|` co bac **1e-2 tro len**, va shape tai diem ap khong phai 3 chieu.

⇒ **Chinh `max|delta|` phan xu, khong phai `torch.equal`.**

### 26.2. Dac ta cua TOI sai — lan thu tu
Toi yeu cau "bit-identical (`torch.equal`)". Tren GPU, qua advanced indexing lam doi layout bo nho,
**bit-identical la dieu khong the dat duoc** ngay ca khi logic hoan toan dung. Nen phep kiem cua
toi **khong co kha nang PASS** trong truong hop (i) — tuc no khong phan biet duoc vo hai voi pha
huy, dung thu no duoc tao ra de lam.
Phep kiem dung: `torch.allclose(atol=1e-5, rtol=1e-4)` **kem bao cao `max|delta|`** va shape.

### 26.3. CO DAU HIEU DOC LAP rang co gi sai — lech train/valid khong the la ngau nhien
Doi chieu nhanh `full` (khong he dung `perm`, nen khong bi anh huong boi 26.1) giua M6 (train) va
M7 (valid):
| chieu | `R2(full)` train (M6) | `R2(full)` valid (M7) | `sigma` train | `sigma` valid | `E[z^2]` train | `E[z^2]` valid |
|-------|----------------------|----------------------|---------------|---------------|----------------|----------------|
| `d`     | 0,386 | 0,420 | 0,775 | **0,604** | 1,024 | **1,592** |
| `theta` | 0,062 | **0,431** | 0,930 | **0,772** | 1,085 | 0,954 |
| `phi`   | 0,115 | **0,361** | 0,946 | **0,804** | 0,989 | 0,989 |
| `qx`    | 0,007 | **0,175** | 0,982 | **0,955** | 1,030 | 0,905 |

`theta` tot hon **7 lan** tren tap valid. `phi` tot hon 3 lan. `qx` tot hon 25 lan. Mo hinh
**khong** the tot hon nhu vay tren du lieu chua thay — day khong phai bien thien mau.

Hai gia thuyet, va ca hai deu nghiem trong:
* **(A) Ro ri du lieu:** tap "valid" chong lan tap train (cung `pli_id`, hoac cung complex duoi id
  khac). Mo hinh da thay chung ⇒ valid **khong** dung lam tap xac nhan duoc, va moi con so
  "generalization" tu truoc den nay vo gia tri.
* **(B) `--split valid` khong lay dung tap valid** (vi du roi ve mot slice cua train, hoac bo
  filter khac).
Luu y `E[z^2](d) = 1,592` tren valid lai **nguoc huong**: `d` te hon han tren valid. Mot tap vua
bi ro ri vua te hon o mot chieu la to hop kho giai thich, cang cung co rang co loi quy trinh.

### 26.4. Trang thai cac ket luan
| muc | trang thai |
|-----|------------|
| Muc 25 (hoc chi phoi tinh tien) | **DINH CHI** — phu thuoc `pocket_shuffle_perm`, chua biet (i) hay (ii) |
| Muc 24 (phan ra sach) | **con hieu luc** — khong dung `perm`; nhung `R2` phai doc theo cong thuc sua o 25.6 |
| Muc 22, 23 (thang dai so, co che dau) | **con hieu luc** — khong dung `perm` |
| Muc 21 (loai (b)) | **con hieu luc** — chi du lieu |
| M7-a, M7-b | **khong doc** |
| "valid la tap xac nhan" | **DINH CHI** — xem 26.3 |

### 26.5. Ke toan
Lan thu tu trong mot ngay toi dat tieu chi/dac ta sai: nguong tuyet doi `-0,3`; hieu `mean` cho gia
thuyet luong cuc; `1-sigma^2` cho hieu ung o `mu`; va nay `torch.equal` cho mot phep bien doi
khong the bao toan bit. **Ba trong bon lan, loi cung mot dang:** toi chon mot dai luong nghe co ve
chat che ma khong kiem xem no co the nhan gia tri can thiet trong truong hop "dung" hay khong.
Quy tac bo sung, ap cho moi tieu chi tu day: **truoc khi ghi cung mot tieu chi, phai tra loi duoc
"neu moi thu DUNG thi dai luong nay bang bao nhieu, va no co the dat gia tri do khong?"**

---

## 27. M8 + M9 (server, 2026-10-01): nguyen nhan la NHIEU DEQUANT, khong phai (i) cung khong phai (ii)

### 27.1. M9 — KHONG ro ri. Gia thuyet (A) cua 26.3 bi bac
`logs/m9_leakage.log`. Duong dan dung `split_druglike_dict.pkl` giong step6.
```
len(train)=23052  len(valid)=1071  len(test)=478
muc pli_id : train&valid=0   train&test=0   valid&test=0
muc PDB    : train&valid=0   train&test=0   valid&test=0
```
**Khong chong lan o ca hai muc.** Chenh lech 764 vs 1.058 duoc giai thich dung: 1.058 la so entry
**co pkl**, do **truoc** khi `frame_stable`/`frag_frame_stable` ton tai; 764 la **sau** filter do.
Hai phep do khong mau thuan.
⇒ "valid la tap xac nhan" **duoc phuc hoi** ve mat phan chia du lieu.

### 27.2. M8 — ca (i) va (ii) deu SAI. Nguyen nhan la (iii): NHIEU NGAU NHIEN trong forward
```
TRUOC khi ap perm: h_target shape=(16, 49, 259) is_contiguous=True
SAU   khi ap perm: h_target shape=(16, 49, 259) is_contiguous=True
max|delta mu| = 8.281e-01     max|delta log_scale| = 1.290e+00
```
Shape **dung 3 chieu**, contiguous, **khong doi** truoc/sau ⇒ **(ii) bi bac**: `perm` ap dung cho
`[B, R_max, H+3]` tren dim 0, dung nhu thiet ke.
`arange(16)` tren dim 0 cua tensor contiguous la phep **dong nhat chinh xac ve toan hoc** ⇒ chenh
lech `O(1)` **khong the** den tu phep indexing ⇒ **(i) cung bi bac**.

**Nguyen nhan thuc, do duoc trong code:** khong co `Dropout` o bat ky file nao
(`step1/2/3/4` deu khong). Nhung `src/step4_ligand_generator.py:123`:
```python
noise = torch.randn_like(target_scaled) * DEQUANT_SIGMA      # DEQUANT_SIGMA = 0.1
```
**nam TRONG forward**, va `target_scaled` chinh la **input cua ARN** khi tinh `log_prob`
(MAF goi ARN theo chieu nghich, voi target lam input). Nen **moi lan forward ve mot nhieu MOI**
⇒ `mu` va `log_scale` khac nhau giua hai lan goi, du moi thu khac y het.
`sigma = 0,1` trong khong gian chuan hoa, qua ARN khuech dai len `O(1)` — khop voi `1,29`.

### 27.3. Hau qua: KHONG chi phep kiem, ma CA M6 va M7
Bon nhanh (`full`, `shuffled`, `roll1`, `sizematched`) moi nhanh **ve nhieu dequant RIENG**. Nen moi
`delta R2` do duoc chua:
* hieu ung hoc protein that, **cong**
* chenh lech giua hai lan ve nhieu doc lap

Nhieu trung binh hoa qua 4.869–19.104 token nen anh huong len **trung binh** nho hon nhieu so voi
`max` tung phan tu. Nhung **chua biet nho bao nhieu**, va do dung la thu phep kiem khong-doi phai
tra loi — neu no bao cao dai luong dung.

**Mot luu y giam gia tri cho bang chung "bon nhanh nhat quan":** ca bon `delta` dung **cung** mot
nhanh `full` lam tham chieu, nen mot lan ve nhieu xau o `full` se day **ca bon** `delta` cung huong.
Chung **khong** phai bang chung doc lap voi nhau. Tinh nhat quan thu tu 7 chieu van la bang chung
yeu chong lai "toan nhieu", nhung khong du.

### 27.4. Dai luong dung ma toi da khong yeu cau ngay tu dau
Phep kiem khong-doi phai bao cao **`delta R2` cua cap khong-doi**, trong **dung don vi** cua tieu
chi, chu khong phai `max|delta|` tung phan tu. Do la **san nhieu** (noise floor).
```
delta R2(no-op)  <<  delta R2(d, shuffled)   ⇒  tin hieu that
delta R2(no-op)  ~   delta R2(d, shuffled)   ⇒  toan nhieu
```
`max|delta|` khong tra loi duoc cau hoi nay vi no o don vi khac. Day la **lan thu nam** toi chon
dai luong khong khop voi cau hoi — va lan nay la dang thu hai: don vi sai, khong phai "khong the
dat gia tri dung".

### 27.5. Trang thai
| muc | trang thai |
|-----|------------|
| Muc 25, M7-a, M7-b | **VAN DINH CHI** — cho san nhieu o M10 |
| 26.1 (hai gia thuyet (i)/(ii)) | **CA HAI BI BAC** ⇒ thay bang 27.2 |
| 26.3 gia thuyet (A) ro ri | **BI BAC** boi M9 (27.1) |
| 26.3 gia thuyet (B) `--split` sai | **BI BAC** — M9 xac nhan duong lay split dung |
| 26.3 nghich ly valid > train | **CON NGUYEN, chua giai thich** — xem 27.6 |
| Muc 21, 22, 23, 24 | **con hieu luc** |

### 27.6. Nghich ly valid > train: gia thuyet con lai la TAP CON CO THU TU
M6 (train) dung `shuffle=False` ⇒ lay **200 batch DAU** theo thu tu dataset, tuc mot tap con
**co thien lech he thong**, khong phai mau ngau nhien. M7 (valid) dung **het** tap valid.
⇒ So sanh "tap con dau cua train" voi "toan bo valid" khong cung ban chat.
Kiem duoc: chay lai train voi `shuffle=True`. Neu `R2(theta)` nhay tu 0,062 len gan 0,43 thi
nghich ly la hien vat cua thu tu dataset, khong phai tinh chat cua mo hinh.

---

## 28. Doc lai M8 (2026-10-01): 27.3 SAI. M6/M7 KHONG bi nhiem nhieu — nhung M7-a VAN khong ket luan

### 28.1. Bang chung tu code cua dung ban da chay (`d10e67a`)
```
297:    run_noop_check(model, loader, device, n_check=3)        <- CHAY O DAY
317:    original_randn_like = torch.randn_like
329:        torch.randn_like = _zero_randn_like                 <- ZERO HOA BAT DAU O DAY
330:        for i, batch in enumerate(loader):                  <- vong do 4 nhanh
390:        torch.randn_like = original_randn_like
```
`run_noop_check` o **dong 297**, zero-hoa nhieu o **dong 329**.
⇒ Phep kiem khong-doi chay voi nhieu **SONG**; vong do 4 nhanh chay voi nhieu **ZERO HOA**.

### 28.2. Hai sua doi
* **27.2 (nguyen nhan la nhieu dequant): DUNG.** `step4:123` ve nhieu moi moi forward, va phep
  kiem khong-doi khong duoc bao ve boi zero-hoa ⇒ `max|delta| = 1,29`.
* **27.3 (M6/M7 bi nhiem): SAI.** Vong do chinh nam **trong** vung zero-hoa ⇒ nhieu = 0 o moi
  nhanh ⇒ so sanh 4 nhanh la **tat dinh**, khong co so hang nhieu.
  ⇒ **Muc 25 va M7-a/M7-b duoc PHUC HOI** khoi trang thai dinh chi.
* **26.2 cung sai ve ly do.** Toi noi phep kiem khong the PASS vi `torch.equal` khong bao toan bit
  qua advanced indexing. Ly do thuc: phep kiem **chay ngoai vung zero-hoa nhieu**. Doi `torch.equal`
  thanh `allclose` khong sua duoc dieu do — va dung the, M8 van FAIL voi `allclose`.
  Loi dac ta cua toi: yeu cau mot phep kiem khong-doi ma **khong hoi co che kiem soat ngau nhien
  dang nam o dau** trong script.

### 28.3. Nhung tieu chi M7-a — doc theo dung nguong da ghi cung — KHONG DAT
Nguong ghi cung (plan Rev 7, truoc khi chay): `delta R2(d) > 0,10` **VA** `delta R2(theta) > 0,015`
**VA** `delta R2(phi) > 0,015`, cung dau o `roll1`.
Do duoc tren **valid** (4.869 token, 764 complex):
| | do duoc | nguong | ket qua |
|---|---|---|---|
| `delta R2(d)`     | **0,0751** | > 0,10  | **KHONG DAT** |
| `delta R2(theta)` | **0,0135** | > 0,015 | **KHONG DAT** (sat) |
| `delta R2(phi)`   | **0,0251** | > 0,015 | DAT |

⇒ **"TRUNG GIAN: khong ket luan"** — dung nhu script in. So lieu gio **dang tin** (tat dinh,
khong ro ri, split dung, `R2` dung cong thuc) nhung **khong vuot nguong**.

**Khong duoc ha nguong sau khi thay so.** Nguong `0,10`/`0,015` da ghi cung truoc khi chay va phai
giu. Ket qua chinh thuc cua M7-a la: **khong ket luan**.

### 28.4. Hai cau hoi KHAC NHAU, khong duoc thay the cho nhau
* **"Hieu ung co DU LON de dang ke?"** — nguong `0,10`/`0,015` o M7-a. **KHONG DAT.**
* **"Hieu ung co KHAC KHONG?"** — so voi **san nhieu**, dieu kien `delta R2(d, shuffled) >
  10 x delta R2(d, noop)` da ghi cung o plan Rev 9 **truoc** run nay. **Chua do** (nhanh `noop`
  chi co tu M10).

Hai cau hoi doc lap. Tra loi cau thu hai **khong** cho phep tuyen bo cau thu nhat. Ghi ro de lan
sau khong tu lan sang nhau.

### 28.5. Y nghia thuc te: M7 la phep do HOI CO, gia tri thong tin thap
Ngay ca neu M10 cho thay hieu ung khac khong, M7 van chi noi: *mot mo hinh da huan luyen 25 epoch,
voi mot shortcut dai so da biet o quaternion va mot duong dieu kien hoa chua tung duoc kiem, co su
dung hoc protein mot luong nho cho tinh tien.* Do la thong tin yeu.

**Phep tham do toi thieu (M11) manh hon nhieu bac**: no huan luyen truc tiep hai mo hinh (co hoc /
bit mat hoc) tren mot bai toan co giam sat, khong co generator, khong co flow, khong co shortcut
dai so. Hieu ung o do lon va doc duoc truc tiep, khong can san nhieu hay nguong tinh vi.

⇒ **M10 van chay** (re, da viet xong, cho san nhieu va dong lai cau hoi 28.4), nhung **M11 la buoc
quyet dinh**, khong phai M10.

### 28.6. Ke toan
Dac ta sai lan thu sau, va la dang thu ba: toi yeu cau mot **phep kiem khong-doi** ma khong hoi
**co che kiem soat ngau nhien dang nam o dau trong script**. Phep kiem vi the do mot duong chay
khac voi duong duoc do. Hau qua: dinh chi oan Muc 25 va M7 trong hai luot.
Quy tac bo sung: **moi phep kiem phai chay TRONG cung moi truong (seed, patch, mode) voi phep do ma
no bao ve.** Neu phep do nam trong mot context manager/patch, phep kiem phai nam trong do.

---

## 29. M10 (server, 2026-10-01): san nhieu = 0,0011. `d` THAT; `theta` KHONG. M7-a van khong dat nguong

`logs/m10_valid.log`, HEAD `02775b1`, split=valid, 4.869 token, 5 nhanh, seed khoa.

### 29.1. Gate cua toi tu phu dinh — loi dac ta thu BAY
```
delta R2(full-noop) toi da tren 7 chieu = 0.0011   (dieu kien 1: < 0.002)    DAT
max|delta mu| / max|delta log_scale|    = 2.975e-01 (dieu kien 1: < 1e-4)    KHONG DAT
```
Verdict "KHONG DAT" do **hoan toan** ve `max|delta| < 1e-4` quyet dinh — tuc chinh dai luong ma
**Muc 27.4, viet TRUOC run nay**, da ghi la "o don vi khac nen khong tra loi duoc cau hoi".
Toi goi no tro lai vao gate o plan Rev 9 **do cau tha**, mau thuan voi quy tac chinh toi vua ghi.

⇒ **Ve `max|delta| < 1e-4` bi loai khoi dieu kien 1.** Day la **sua loi**, khong phai retrofit:
nguyen tac lam no vo hieu da duoc ghi **truoc khi** thay du lieu (27.4). Ve con lai —
`delta R2(full-noop) < 0,002` o moi chieu — **DAT** (toi da 0,0011).

### 29.2. Tai sao `max|delta|` van lon du da khoa seed: CUDA scatter khong tat dinh
Khong con nguon ngau nhien nao chua khoa (khong co Dropout trong `src/`). Nguyen nhan la
**`atomicAdd` khong tat dinh** trong cac phep scatter/message-passing cua PyG: thu tu cong khac
nhau moi lan goi ⇒ lam tron khac ⇒ khuech dai qua encoder sau va qua `exp()` trong ARN.
`deterministic=True` da them vao `pl.Trainer` **khong** ap cho script doc lap.
⇒ Khong phai loi logic. Va quan trong: **san nhieu 0,0011 da bao gom no**.

### 29.3. Doc theo SAN NHIEU (cau hoi "co khac khong") — dung don vi, da tien dang ky
San nhieu bao thu = `max |delta R2(full-noop)|` tren 7 chieu = **0,0011**.
| chieu | `delta R2` (shuffled) | boi so san nhieu | doc |
|-------|----------------------|------------------|-----|
| `d`     | **0,0811** | **~74x** | **that, ro rang** |
| `phi`   | **0,0220** | **~20x** | **that, nho** |
| `qw`    | 0,0111 | ~10x | sat bien |
| `qz`    | 0,0101 | ~9x  | sat bien |
| `qy`    | 0,0033 | ~3x  | khong tach duoc |
| `theta` | **0,0017** | **~1,5x** | **KHONG tach duoc khoi nhieu** |
| `qx`    | -0,0007 | — | khong |

Dieu kien 2 (ghi cung o Rev 9): `delta R2(d, shuffled) > 10 x delta R2(d, noop)` ⇒ **71,6x, DAT**.

### 29.4. Nhung M7-a — cau hoi "co DU LON khong" — VAN KHONG DAT
Nguong ghi cung o Rev 7: `d > 0,10` **VA** `theta > 0,015` **VA** `phi > 0,015`.
| | do duoc (M10) | nguong | |
|---|---|---|---|
| `d`     | 0,0811 | > 0,10  | **KHONG DAT** |
| `theta` | 0,0017 | > 0,015 | **KHONG DAT** (te hon M7: 0,0135) |
| `phi`   | 0,0220 | > 0,015 | DAT |

⇒ **M7-a: KHONG DAT. Khong ha nguong.**
`theta` tut tu 0,0135 (M7) xuong 0,0017 (M10) vi M10 dung **nhieu song** thay vi zero-hoa ⇒
`E[z^2]` cao hon (vd `qy`: 0,989 -> 1,250). **Khong so truc tiep gia tri tuyet doi M10 voi M7.**
Nhung trong **cung** run M10, `theta` khong tach khoi san nhieu, va M10 la run co san nhieu do
duoc, nen no dang tin hon.

### 29.5. M7-b — doi chung kich thuoc: hieu ung cua `d` la DAC THU, khong chi kich co
`delta R2(d)`: `shuffled` = 0,0811; `shuffled_sizematched` = **0,0667** = **82%**, tren hai phan ba.
⇒ Khi chi xao tron giua cac hoc **cung kich co**, hieu ung chi mat 18%.
⇒ Mo hinh doc thong tin hoc **dac thu**, khong chi mot so vo huong "hoc to/nho".

### 29.6. Ket luan co the ghi — va khong duoc ghi
**Duoc ghi** (dung don vi, tren tap valid chua dung, co san nhieu do duoc, doi chung ghep doi):
* Hoc protein **co** anh huong den du doan **khoang cach `d`**, muc `delta R2 = 0,081` tren
  `R2 = 0,418`, tuc **~19% luong giai thich duoc cua `d` phu thuoc vao hoc DUNG**. Cao hon san
  nhieu ~74 lan va **dac thu theo hoc** (82% con lai sau size-match).
* `phi` (mot trong hai goc **huong tinh tien**): hieu ung nho nhung that (~20x san nhieu).
* `theta`: **khong** tach duoc khoi nhieu.
* **Phep quay (quaternion): khong co hieu ung dang ke** — khop Muc 16.1 va 24.2.

**KHONG duoc ghi:**
* Khong duoc noi M7-a "xac nhan". Nguong `0,10`/`0,015` **khong dat**; do la cau hoi khac (28.4).
* Khong duoc dung gia tri tuyet doi `R2` cua M10 so sanh voi M6/M7 (khac dieu kien nhieu).

### 29.7. Y nghia: du muc "cuu tien de", khong du cho pipeline
`19%` cua `d`, `phi` nho, `theta` khong, quay khong. Tin hieu **yeu nhung co that** cho tinh tien.
Khong du de ket luan pipeline kha thi, cung khong bac bo tien de. Dung nhu 28.5 du doan: phep do
hoi co tren mot mo hinh co shortcut chi cho tin hieu yeu.
⇒ **M11 van la buoc quyet dinh**, va gio co moc de so: neu M11 (co giam sat, khong shortcut) khong
cho `delta R2` **lon hon** 0,081 o `d`, thi tran nam o encoder, khong o generator.

### 29.8. Ke toan
Loi dac ta thu bay, va la lan **lap lai** mot loi vua duoc ghi thanh quy tac. Quy tac moi:
**sau khi ghi mot quy tac phuong phap luan vao tai lieu, phai doc lai tieu chi dang viet va kiem
no co vi pham quy tac do khong — truoc khi dua tieu chi cho Claude Code.**

### 29.9. Loi quy trinh tai lieu (phat hien 2026-10-01 21:57)
Muc 29 va plan Rev 12 **da bi mat** trong hai luot: toi append vao mot ban copy phia container roi
commit, va ban copy do lech khoi file tren may nguoi dung. Claude Code doc file **that** (chi co
den Rev 11 / Muc 28) nen da trien khai M11 theo Rev 10/11 — **khong phai loi cua no**.
Quy tac: **truoc moi lan ghi tai lieu, phai stage file THAT tu may nguoi dung, append vao ban do,
commit, roi doc lai tu may nguoi dung de xac nhan.** Khong tin ban copy container giua cac luot.

---

## 30. Kiem khong-doi cho refactor M11-c (server, 2026-10-01): DAT. Va M10 duoc TAI LAP

`logs/m11c_noop.log`, HEAD `af87853`.

### 30.1. Refactor `build_h_target` khong doi hanh vi
| chieu | `R2(full)` M10 | `R2(full)` sau refactor | lech |
|-------|----------------|-------------------------|------|
| `d`     | 0,4183 | 0,4195 | 0,0012 |
| `theta` | 0,4287 | 0,4286 | 0,0001 |
| `phi`   | 0,3613 | 0,3608 | 0,0005 |
| `qw`    | 0,5318 | 0,5318 | 0,0000 |
| `qx`    | 0,1745 | 0,1745 | 0,0000 |
| `qy`    | 0,7945 | 0,7938 | 0,0007 |
| `qz`    | 0,2925 | 0,2923 | 0,0002 |

Lech toi da **0,0012 < 0,002** (nguong ghi cung o Rev 12). **DAT.**
Xac nhan them bang doc code: `self.dynamic_encoder` duoc goi **mot** lan ca truoc (`cd604cd^`
dong 288) va sau (dong 298) refactor — khong co loi goi kep nao bi bo. Cach dien dat cua Claude
Code ("tranh goi lan thu hai") de gay hieu sai nhung khong co thay doi hanh vi.

### 30.2. M10 duoc TAI LAP doc lap — lan dau trong du an
| dai luong | M10 (`02775b1`) | M11-c run (`af87853`) |
|-----------|-----------------|------------------------|
| `delta R2(d, shuffled)` | 0,0811 | **0,0824** |
| `delta R2(phi, shuffled)` | 0,0220 | **0,0208** |
| `delta R2(theta, shuffled)` | 0,0017 | **0,0015** |
| san nhieu `max|delta R2(full-noop)|` | 0,0011 | **0,0008** |
| ty le `d` / san nhieu | 71,6x | **100,6x** |
| `delta R2(d)` sizematched | 0,0667 (82%) | **0,0681 (83%)** |

Hai lan do rieng, cung ket luan. Muc 29.3/29.5/29.6 duoc **cung co**: `d` that va dac thu theo
hoc; `phi` nho nhung that; `theta` khong tach duoc; quay khong co gi.
Day la ket qua **dau tien** cua du an duoc tai lap qua hai run doc lap. Dang ghi nhan: no co duoc
nho P3a-2 (seed) + san nhieu do duoc, hai thu chi ton tai tu hom nay.

### 30.3. Script van in verdict gay nham — can sua khi tien
`measure_pocket_shuffle.py` van in "DIEU KIEN 1 KHONG DAT" vi ve `max|delta| < 1e-4`, vế **da bi
loai** o 29.1. Ve con lai (`delta R2(full-noop) < 0,002`) **dat** (0,0008).
Viec can lam (khong gap): bo ve `max|delta| < 1e-4` khoi dieu kien 1 trong script, giu no lam
**chan doan** co nhan. Chua lam vi khong chan M11.

---

## 31. M11-a lan 1 (server, 2026-10-02): OVERFIT, khong doc duoc — nhung epoch 1 cho tin hieu

`logs/m11a_probe.log`. Cache: train 17.200 complex / **733,4 MB** (8.322 s), valid 764 / 33,3 MB (364 s).
Config: `epochs=100 batch_size=32 vocab_size=19033 h_dim=259 n_train=17200 n_valid=764`.

### 31.1. Du lieu: val R2 dat dinh o EPOCH 1 roi suy giam don dieu
`val_R2` theo thu tu `[d, theta, phi, qw, qx, qy, qz]`:
```
ep   1  train_loss=0.9588  [ 0.168,  0.193,  0.201,  0.171,  0.124,  0.060,  0.099]
ep  10  train_loss=0.6774  [ 0.098,  0.097,  0.111,  0.034,  0.038,  0.015,  0.049]
ep  20  train_loss=0.5601  [ 0.034,  0.025, -0.011, -0.030, -0.026, -0.079, -0.042]
ep  50  train_loss=0.4493  [ 0.005, -0.045, -0.113, -0.144, -0.128, -0.164, -0.153]
ep 100  train_loss=0.3846  [-0.043, -0.107, -0.164, -0.243, -0.231, -0.267, -0.273]
```
`train_loss` **giam don dieu**; `val R2` **giam don dieu** o ca 7 chieu. Overfit kinh dien.
Script dung truoc khi chay nhanh `blind` (guard Rev 11), nen **khong co so de so sanh**.

### 31.2. Co che overfit nhin thay duoc trong config
`vocab_size = 19.033` tren **101.791** token train ⇒ moi `frag_id` xuat hien trung binh **5,3 lan**.
Bang `token_embedding` mot minh du suc hoc thuoc pose cua tap train.
Va duong nay **khong phu thuoc `h_target`** ⇒ no khong lam `pocket` kem hon `blind`, no lam **ca
hai** ve am va bien hieu `pocket - blind` thanh vo nghia. Day la ly do phai sua giao thuc truoc khi
doc bat cu gi.

### 31.3. Nhung epoch 1 KHONG vo gia tri
`R2` tai epoch 1 tren valid: `d 0,168 | theta 0,193 | phi 0,201`. Day la encoder **dong bang** +
mot head chua kip memorize. Ba chieu tinh tien deu quanh **0,17-0,20**.
⇒ **Co tin hieu trong dau ra encoder.** Nhung **chua** biet bao nhieu den tu hoc va bao nhieu tu
`frag_id`, vi khong co nhanh `blind`. Khong duoc ket luan gi tu con so nay.

### 31.4. Loi dac ta thu TAM — va la dang thu tu
Toi ghi "100 epoch" ma **khong** model selection, **khong** early stopping, **khong** weight decay.
Voi mot bang embedding 19.033 muc tren 101.791 mau, do la dam bao overfit.
Va khi huy guard Rev 11 o Rev 12 (dung ly do: pipeline co context tu hoi quy, M11 khong), toi thay
no bang **khong gi ca**. Guard dung o day khong phai mot nguong `R2` tuyet doi ma la mot yeu cau
**giao thuc**: *val R2 phai duoc chon o epoch tot nhat, va phai kiem xem no co suy giam don dieu
khong*.
Dang loi moi (thu tu): **ke hoach thieu giao thuc huan luyen, khong phai thieu tieu chi doc.** Bay
loi truoc deu la "do sai dai luong"; loi nay la "thi nghiem duoc thiet ke sai tu dau".

### 31.5. Chi phi thuc te so voi uoc luong
Cache: **2 gio 24 phut** (toi uoc "vai chuc phut"). Epoch: **25 s** (toi uoc "vai giay") — vi
`torch.load` tung complex moi sample, I/O-bound tren 733 MB. Preload vao RAM se sua duoc.
Ghi lai de lan sau uoc dung: moi lan toi uoc chi phi trong du an nay, toi uoc **thap**.

---

## 32. M11-a Rev 2 (server, 2026-10-02): P2 DAT tieu chi chinh; P1 khong doc duoc (loi dac ta thu CHIN)

`logs/m11a_probe_rev2.log`, HEAD `40112e9`. Cache dung lai (733,4 MB / 33,3 MB), preload 11,9 s.
`n_train=101.791 token / 17.200 complex`, `n_valid=4.869 token / 764 complex`.

### 32.1. San tuyet doi `tokenmean` (giao thuc #1, in truoc tien)
`d 0,0670 | theta 0,0811 | phi 0,1022 | qw 0,0722 | qx 0,0560 | qy -0,0753 | qz -0,0426`
Ghi chu: `tokenmean` train `R2(d) = 0,4954` vs valid `0,0670` ⇒ ban than `tokenmean` overfit nang
(trung binh tren ~5,3 mau/id). No la san **yeu** tren valid, nhung van la san.

### 32.2. P2 (MLP co giao thuc) — TIEU CHI CHINH DAT 3/3
| chieu | `pocket` | `blind` | **delta** | `tokenmean` |
|-------|----------|---------|-----------|-------------|
| **`d`**     | 0,2299 | 0,1301 | **+0,0999** | 0,0670 |
| **`theta`** | 0,2332 | 0,1658 | **+0,0674** | 0,0811 |
| **`phi`**   | 0,2642 | 0,1952 | **+0,0690** | 0,1022 |
| `qw` | 0,1586 | 0,1486 | +0,0100 | 0,0722 |
| `qx` | 0,1195 | 0,1256 | -0,0061 | 0,0560 |
| `qy` | 0,0599 | 0,0221 | +0,0378 | -0,0753 |
| `qz` | 0,1237 | 0,0736 | **+0,0500** | -0,0426 |

`best_epoch`: `pocket` = **8**, `blind` = **4**; early stopping patience 10, toi da 60.
**Ca ba chieu tinh tien vuot 0,05** (can >= 2/3) ⇒ theo tieu chi ghi cung Rev 12: **TIEN DE DUNG.**

**Bon can cu de tin P2:**
1. **Giao thuc #3 DAT:** `blind` cao hon `tokenmean` o ca ba chieu tinh tien ⇒ baseline hoi tu that.
2. **`best_epoch` = 8 va 4**, khong phai 1 ⇒ khong con overfit tu dau nhu Muc 31.
3. **Overfit DOI XUNG:** ty le train/valid o `d` = 0,5391/0,2299 = **2,34** (`pocket`) va
   0,3171/0,1301 = **2,44** (`blind`). Gan bang nhau ⇒ hieu khong bi lech boi mot nhanh overfit nang hon.
4. **`blind` cua P2 co embedding `frag_id` HOC DUOC** ⇒ no dung day du thong tin danh tinh fragment.
   Phan `+0,0999` cua `pocket` la **them vao tren** nen do, khong thay the no. Confound
   "hoc lam proxy cho danh tinh fragment" vi the **yeu** o P2.

**Do tin cay thong ke (uoc tho):** `se(R2) ~ sqrt(2(1-R2)^2/n) ~ 0,016` voi `n=4.869`; hai nhanh
tuong quan (cung du lieu) nen `se` cua hieu nho hon. `delta(d) = 0,100` ~ 6 sigma; `theta 0,067`
~ 4 sigma. Du vung.

### 32.3. Moc so sanh 0,081 — pipeline DANG LAM MAT tin hieu
`delta R2(d)` cua P2 = **0,0999** > **0,081** (M10, pipeline day du).
Theo tieu chi ghi cung Rev 12 muc (3): **pipeline dang lam mat tin hieu**. Mot bo hoi quy don gian
tren encoder **dong bang** khai thac duoc hoc **nhieu hon** ca pipeline co generator + MAF.
⇒ Van de o **generator/head**, khong o encoder. Sua generator la dang gia, khong phai sai cho.
Day la lan dau du an co mot ket luan **chi huong hanh dong** dua tren so lieu hop le.

### 32.4. P1 (phep CHINH) KHONG DOC DUOC — loi dac ta thu CHIN
`blind` cua P1: `d 0,0368` vs `tokenmean 0,0670` (thap hon **0,030**);
`theta -0,0563` vs `0,0811` (thap hon **0,137**) ⇒ giao thuc #3 chan, dung.

**Nguyen nhan nam trong dac ta cua toi:** toi cho `frag_id` qua mot **embedding ngau nhien CO DINH
32 chieu**. Voi 19.033 id, mot mo hinh tuyen tinh chi bieu dien duoc mot ham **32 chieu** cua danh
tinh fragment, trong khi `tokenmean` la bang tra cuu **day du**. Nhanh `blind` cua P1 vi the **ve
cau truc khong the** dat toi san.

**Hai hau qua:**
* `blind` bi lam yeu gia tao ⇒ `pocket - blind` cua P1 (`d 0,169`!) **bi phong dai**, khong dung duoc.
* Khi `frag_id` bi bop, `h_target` co the dang lam **proxy cho danh tinh fragment** (hoc nao thuong
  gan fragment nao) ⇒ phan chenh khong thuan la thong tin tu the.

Toi chon embedding co dinh de chan memorize (dung nguyen nhan Muc 31.2). No chan duoc memorize
**nhung pha baseline**. Dang loi moi: **sua mot loi bang cach tao ra mot loi khac o phia doi dien.**
⇒ **Phai ghi nhan: P2 la thi nghiem duoc thiet ke TOT HON P1. Viec toi dat P1 lam phep chinh la sai.**

### 32.5. `delta R2(qz) = 0,0500` — mo lai o muc YEU, khong duoc bo qua
Nguong ghi cung: *"mot chieu quaternion **vuot** 0,05 ⇒ ket luan ve phep quay (16.1, 24.2, 25.4) sai"*.
`qz = 0,0500` **khong vuot**, bang dung nguong o do chinh xac in ra. Ba chieu kia: `0,0100`,
`-0,0061`, `0,0378`.
⇒ Khong dismiss. Ket luan "hoc khong chi phoi phep quay" **bi mo lai o muc yeu**, va se duoc xu ly
trong lan chay P1 sua (33). Neu `qz` lai >= 0,05 voi P1 da sua thi phai doc lai Muc 16.1.

### 32.6. Trang thai ket luan
**Duoc ghi:** *co bang chung tu mot giao thuc hop le (P2) rang thong tin tu the co san trong dau ra
encoder dong bang, muc `delta R2` = 0,100/0,067/0,069 cho `d/theta/phi`; va pipeline day du dang
khai thac KEM hon muc do.*
**KHONG duoc ghi:** "tien de da duoc xac nhan". Phep do **chinh** (P1) void do loi dac ta cua toi.
Phai sua P1 va chay lai (re, vai phut). Hai giao thuc doc lap cung ket luan moi la xac nhan.

### 32.7. Uoc luong chi phi — lan thu chin toi uoc thap
Toi uoc epoch "duoi 1 giay" sau preload; thuc te **14,9 s/epoch**. Preload sua duoc I/O nhung
forward/backward cua head van la phan chinh. Moi uoc luong chi phi cua toi trong du an deu thap.

---

## 33. M11-a Rev 3 (server, 2026-10-02): TIEN DE XAC NHAN qua hai giao thuc doc lap

`logs/m11a_probe_rev3.log`, HEAD `0d9bf41`. Cache dung lai. P1 da sua theo Rev 15 (target encoding
OOF 5 fold). Dac trung P1 = 541 chieu (259 mean-pool + 259 max-pool + 7 tokenmean_pose + 16 one-hot t).

### 33.1. Ket qua: CA HAI giao thuc dat tieu chi chinh
| | `delta R2(d)` | `delta R2(theta)` | `delta R2(phi)` | dat |
|---|---|---|---|---|
| **P1** (ridge, target encoding) | **0,1040** | **0,2050** | 0,0488 | **2/3** |
| **P2** (MLP, early stopping) | **0,0999** | **0,0674** | **0,0690** | **3/3** |

Ban sua P1 hoat dong: `blind` gio **cao hon** `tokenmean` o ca ba chieu tinh tien
(`0,1052 / 0,0932 / 0,1544` vs `0,0670 / 0,0811 / 0,1022`) ⇒ giao thuc #3 khong con chan.
⇒ Theo tien dang ky Rev 15: **TIEN DE XAC NHAN.** Thong tin ve tu the **co san** trong dau ra cua
encoder dong bang, cho ca ba chieu tinh tien.

### 33.2. `delta R2(qz) = -0,0049` ⇒ Muc 32.5 DONG LAI
Duoi nguong 0,05 da ghi cung. Phep quay van la "khong chi phoi boi hoc" (16.1, 24.2, 25.4).
P1 quaternion: `qw 0,0174 | qx -0,0082 | qy 0,0048 | qz -0,0049` — tat ca duoi 0,02.

### 33.3. P2 van la phep do DANG TIN HON, du toi dat no lam phu
`delta R2(theta)`: P1 = **0,2050**, P2 = **0,0674**, lech **3 lan**. Nguyen nhan o nhanh `blind`:
* P2 `blind` theta = **0,1658** (co embedding `frag_id` **hoc duoc** ⇒ dung day du danh tinh fragment)
* P1 `blind` theta = **0,0932** (chi co 7 dac trung target-encoded)
⇒ **Baseline cua P2 manh hon ⇒ `delta` cua P2 nho hon va BAO THU hon.**

Hai wart nua cua P1, deu khong co o P2:
* **`lambda` khac nhau giua hai nhanh:** `pocket` chon **10** (co dinh noi bo, co peak ro),
  `blind` chon **1000** = **gia tri LON NHAT trong day quet** (bien). Hai nhanh nhan cuong do
  regularization khac nhau ⇒ capacity hieu dung khac nhau. Bien thien qua 6 bac chi 0,0005 nen
  **khong the doi ket luan**, nhung day quet phai duoc mo rong cho dung.
* **Overfit bat doi xung:** ty le train/valid o `d` = 0,4430/0,2092 = **2,12** (`pocket`) vs
  0,3584/0,1052 = **3,41** (`blind`). P2: **2,34** vs **2,44** — doi xung hon han.

⇒ **Con so cong bo lay tu P2**, khong tu P1: `d ~ 0,100 | theta ~ 0,067 | phi ~ 0,069`.
`d` la chieu vung nhat: **0,1040** (P1) va **0,0999** (P2) gan nhu trung.

### 33.4. `phi` sat nguong — ghi ro, khong lam tron len
`delta R2(phi)` cua P1 = **0,0488**, thieu **0,0012** so voi nguong 0,05. Tieu chi can >= 2/3 nen
van dat, nhung `phi` la chieu **yeu va phu thuoc giao thuc** (P1 0,0488 vs P2 0,0690), khac han `d`
va `theta`. Khong duoc trinh bay `phi` ngang hang voi `d`.

### 33.5. CANH BAO: "pipeline dang lam mat tin hieu" CHUA co can cu chat
Ca hai script in ket luan nay vi `delta R2(d)` = 0,0999-0,1040 > **0,081** (M10). **Nhung day la so
sanh HAI ESTIMATOR KHAC NHAU:**
* **M10:** mo hinh da huan luyen, **xao tron hoc luc SUY LUAN** ⇒ do *"ham da hoc DUNG hoc bao nhieu"*.
* **M11:** **huan luyen co hoc vs huan luyen khong co hoc** ⇒ do *"co hoc thi DUOC THEM bao nhieu"*.

Voi mot mo hinh khai thac chua toi, dai luong thu hai **lon hon** dai luong thu nhat **mot cach he
thong**, ke ca khi khong co gi bi "lam mat". Them nua pipeline co `R2(d) = 0,4195` tong, **gan gap
doi** probe (0,2299), vi no co context tu hoi quy.
⇒ **Khong duoc ket luan "pipeline lam mat tin hieu" tu con so nay.** Va day dung la ket luan se
quyet dinh huong di tiep, nen khong duoc de long.

**Cach sua, re:** chay **phep xao tron luc suy luan** tren chinh probe (`h_target` xao tron theo
chieu complex luc eval, khong huan luyen lai). Khi do ca hai so cung mot estimator va so duoc truc
tiep. Xem plan M12.

### 33.6. Trang thai sau Muc 33
**Duoc ghi:**
* Thong tin tu the **co san** trong dau ra encoder dong bang — **xac nhan** qua hai giao thuc doc lap,
  tren tap valid chua dung, voi san tuyet doi (`tokenmean`) va kiem giao thuc da dat.
* Do lon (bao thu, tu P2): `d ~ 0,100`, `theta ~ 0,067`, `phi ~ 0,069`.
* Phep quay: khong chiu chi phoi cua hoc (nhat quan tu 16.1 den day).
**CHUA duoc ghi:**
* "Pipeline lam mat tin hieu" — can M12 (33.5).
* Bat ky quyet dinh kien truc nao dua tren so sanh M10 vs M11.

---

## 34-pre. Ghi cung TRUOC M12 (2026-10-25, HEAD `eb32820`): kiem tinh C4, sua co che estimator, tieu chi Rev 17
**Muc nay KHONG co so lieu do moi tu du lieu du an.** Moi thu duoi day duoc ghi **truoc** khi chay M12.
Ket qua M12 se ghi vao **Muc 34** (cung so, khong co hau to `-pre`). Ke hoach: **PLAN Rev 17**.
Bo canh: du an chuyen sang tai khoan moi; tro ly moi tiep nhan trang thai qua `00_START_HERE.md`,
`RESULTS_LEDGER.md`, `WORKING_RULES.md`, plan Rev 16, va doc code local (chi doc, khong chay).

### 34-pre.1. Loi co che cua tro ly moi (ghi lai theo R12)
Trong ban tiep nhan, tro ly viet: "`blind` duoc huan luyen de bu phan thieu nen delta train-ablation lon
hon delta inference-shuffle". **Sai chieu** (nguoi dung bat): neu `blind` bu tot thi `R2(blind)` cao va
delta train-ablation **nho** di. Co che do du doan bat dang thuc nguoc voi quan sat (0,0999 > 0,0824).
Xem 34-pre.3 cho phan tich day du — no cung cho thay ca ban sua lan dau lan cau chu 33.5 deu chua dung.

### 34-pre.2. Kiem tinh C4 (doc code, khong GPU): XAC NHAN
**Cau hoi:** co duong nao tu `h_target` toi `geometric_head` ma KHONG qua `detach()` khong?
**Doc code tai HEAD `eb32820`:**
* `step4_ligand_generator.py:88-90` — `x = token_embedding(input_ids)` + positional encoding ->
  `ar_decoder`. **Input cua ar_decoder chi co danh tinh token, KHONG co pose cua fragment truoc.**
* `step4:93` — `v_context, attn_weights = attention_hub(h_ar, h_target, ..., target_mask)`. Day la
  **cho duy nhat** `h_target` di vao generator.
* `step4:99` — `v_context_detached = v_context.detach()`.
* `step4:115` — `geo_in = geo_in_proj(cat[v_context_detached, tok_emb.detach()])`.
* `step4:~130` — `geometric_head(geo_in_flat, target=target_scaled, pad_mask=...)`. Hai input duy nhat:
  `geo_in_flat` (da qua detach) va `target_scaled` (tu `target_7d` = du lieu + nhieu dequant).
* `step6_trainer.py:380-387` + `step5` — tong loss = token + geo + pocket (`pred_res_tr/rot/chi` tu
  step1) + int; `"int": 0.0` (step6 ~dong 453).

**Ket luan cau truc (khong phai ket luan hieu nang):**
1. Gradient cua `Geo_Loss` dung tai `geo_in_proj` va `geometric_head`. **Attention hub, `ar_decoder`,
   `token_embedding` va encoder KHONG nhan gradient hinh hoc nao.**
2. **Duong dieu kien hoa `h_target -> attention_hub -> v_context` chi duoc huan luyen boi token loss**
   (cross-entropy du doan token ke tiep). Encoder duoc huan luyen boi token loss + pocket loss.
   Head hinh hoc nhan `v_context` nhu mot dac trung **co dinh tu goc nhin cua no**.
3. "Context tu hoi quy giua cac fragment" cua pipeline = **danh tinh** cac fragment dung truoc (theo thu
   tu alphabet SMILES, N1), **khong phai pose** cua chung. Vi tri `t = 0` (fragment dau tien sau `[SOS]`)
   khong co context nay. Con lai o `t = 0`: rang buoc **trong-vector** cua MAF (`d` o vi tri AR 1 thay
   `qx` that, co nhieu dequant) va embedding cua chinh token dang sinh (P2c, tuong duong `frag_id` cua probe).
4. He qua cho M12-b: `grad.norm()` tai `dynamic_encoder` sau `Geo_Loss.backward()` **bat buoc** la
   `None`/0 **do cau truc**. Do khong phai bang chung ve Tanh. Tanh chi con co the gay hai theo hai
   cach: (a) mat thong tin chieu forward (bao hoa nen mat do lon), (b) can tro viec hoc trong so cua chinh
   `context_proj`. Docstring `measure_tanh_saturation.py:10-12` gan sai co che.
5. Probe M11-a P2 thi khac: cross-attention cua no duoc huan luyen **truc tiep** bang MSE pose. Day la mot
   khac biet kien truc co san giua hai mo hinh duoc so trong M12-a.

**Gioi han:** kiem tinh chi noi duong gradient ton tai hay khong. No **khong** noi viec bo `detach()` se
cai thien gi. C4 len **dau** danh sach nghi pham (truoc Tanh) vi no da duoc xac nhan la co that ve cau
truc, con Tanh thi chua.

### 34-pre.3. Co che estimator — phan tich cua tro ly. **CHO NGUOI DUNG DUYET**
**Dong nhat thuc (cung mot mo hinh, `R2 = 1 - MSE` tren muc tieu chuan hoa):**
Viet du doan cua mo hinh co hoc la `g(x) + m(p, x)`, voi `x` = moi thu khong phai hoc, `m` = phan phu
thuoc hoc, va `r = y - g(x)`. Xao tron luc suy luan thay `p` bang `p'` doc lap voi `(x, y)` va cung phan phoi.
* inference-shuffle: `B = MSE(shuf) - MSE(pocket) = 2 Cov(r, m)`
  (khi `E[m(p',x)^2] = E[m(p,x)^2]`, dung chinh xac neu `p` doc lap voi `x`).
* train-ablation, **gia dinh** `blind` hoc duoc `g(x) + const`: `A = MSE(blind) - MSE(pocket) = 2 Cov(r, m) - Var(m)`.
* ⇒ **`B - A = Var(m) >= 0`.** Cung mot mo hinh: **delta inference-shuffle >= delta train-ablation**,
  bang nhau chi khi mo hinh khong dung hoc. Hoc sai doi chu khong chi lam mat thong tin, no **dan sai**.

**Kiem bang mo phong** (du lieu tong hop, khong phai du lieu du an; `p` doc lap `x`, `Var f = 0,10`,
mo hinh du doan `c*f` + sai so uoc luong):
```
c=1.0 nhieu=0.0: A=0,0995  B=0,1990  B-A=0,0995  Var(m)=0,1003
c=1.0 nhieu=0.2: A=0,0808  B=0,2043  B-A=0,1236  Var(m)=0,1244
c=0.5 nhieu=0.0: A=0,0744  B=0,1001  B-A=0,0257  Var(m)=0,0250
c=0.5 nhieu=0.2: A=0,0558  B=0,1020  B-A=0,0462  Var(m)=0,0473
c=0.2 nhieu=0.0: A=0,0361  B=0,0401  B-A=0,0040  Var(m)=0,0040
```

**Hau qua:**
1. **Cau chu cua 33.5** ("voi mo hinh khai thac chua toi, dai luong train-ablation LON HON inference-
   shuffle mot cach he thong") — **sai voi cung mot mo hinh**; dieu nguoc lai moi dung. 33.5 so A cua
   probe voi B cua pipeline (hai mo hinh); cau do khong co co che nao bao dam.
2. **Ban sua "co san >= duoc dung"** (nguoi dung, 2026-10-25) — dung ve **dinh tinh** (huan luyen co the
   trich chua toi), nhung **khong** thanh bat dang thuc giua `A(probe)` va `B(pipeline)`: voi mot mo hinh
   trich du (`m = f`), `B = 2 Var(f)` trong khi `A <= Var(f)`. Nen `B(pipeline)` co the vuot `A(probe)` toi 2 lan.
3. **M12-a van la so sanh hop le**: `B(probe)` vs `B(pipeline)` = so `2 Cov(r, m)` cua hai mo hinh.
4. **R5 cho tieu chi M12-a o tap `all`:** voi `A(probe, d)` da do = 0,0999 (P2) / 0,1040 (P1), dong nhat
   thuc du doan `B(probe, d) >= ~0,10 > 0,0824`. ⇒ **Nhanh XAC NHAN o tap `all` gan nhu chac chan**, tru
   khi gia dinh bi vo. Thong tin moi cua M12-a nam o: (i) kiem chinh dong nhat thuc (`B >= A`), (ii) so
   sanh tai `t = 0`.
5. Neu dong nhat thuc dung, du lieu **da co** da nghieng ve huong 32.3 (probe dung hoc nhieu hon pipeline).
   **Khong ghi thanh ket luan** cho den khi M12 chay va nguoi dung duyet muc nay.

**Gia dinh co the vo:**
* `blind` **underfit** so voi phan khong-hoc `g` cua mo hinh `pocket` ⇒ `A` bi phong dai, co the `A > B`.
  Giao thuc #3 (`blind >= tokenmean - 0,03`) chi kiem mot phan.
* `p` tuong quan voi `x` (hoc tuong quan voi danh tinh fragment) va `m` co tuong tac `p*x` ⇒ dong nhat
  thuc chi con xap xi. Mo phong tren chua phu truong hop nay.

### 34-pre.4. Tieu chi M12 ghi cung (chi tiet o PLAN Rev 17)
* **P1/P2 bat dong** ve verdict B(d) ⇒ **KHONG KET LUAN** (khoi phuc quy tac Rev 15 ma Rev 16 bo sot).
* **Sai so:** K = 20 derangement (seed 1234..1253), bao cao mean ± sd cua B(d). So voi **khoang quan sat
  cua ho M10 `[0,0751 ; 0,0824]`**, khong so voi diem 0,081: `mean - 2sd > 0,0824` ⇒ XAC NHAN;
  `mean + 2sd < 0,0751` ⇒ BAC; con lai ⇒ KHONG KET LUAN.
  **Day la SUA mot tieu chi dac ta sai** (Rev 16 so mot diem, khong co sai so, trong khi chinh gia tri M10
  da dao dong 0,0751–0,0824), lam **truoc khi thay du lieu**, hop R9 — **khong phai retrofit**.
* **Tach theo vi tri** `t = 0` / `t >= 1`, ca probe va pipeline (`measure_pocket_shuffle.py`). Nhanh
  XAC NHAN doc o `t = 0` (34-pre.2 muc 3: khu context danh tinh giua fragment; KHONG khu rang buoc
  trong-vector `qx -> d`).
* **Dien giai mem:** XAC NHAN = "pipeline khai thac hoc kem hon probe o cung estimator". **Khong** phai
  "pipeline lam mat tin hieu". Giai thich lanh tinh con lai: thong tin du thua. Them nua, 34-pre.2 muc 5:
  hai mo hinh huan luyen duong dieu kien hoa bang hai muc tieu khac nhau.
* **M12-b:** nguong giu nguyen; ly do hep lai (34-pre.2 muc 4).

### 34-pre.5. Sua tai lieu kem (P8, nguoi dung yeu cau)
* `WORKING_RULES.md`: dau file "10 quy tac" -> 15 (R1–R15) + R10b.
* `RESULTS_LEDGER.md` A4: ghi nhan split — 86,0% cua `qy` la M5 (`split=train`); A3 (0,7938) la valid.
* `RESULTS_LEDGER.md` A6: `theta` cung phu thuoc giao thuc (0,2050 P1 vs 0,0674 P2, lech ~3x), nang hon
  `phi` (0,0488 vs 0,0690).
* `00_START_HERE.md`: `logs/m12b_tanh.log` (84 byte, 13/10 11:52) thay o **LOCAL**; server **chua kiem**.
  Quan he `~/SAMG` va `/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG`: **chua xac minh**.
* Phat hien kem khi doc code: `probe_pose_regression.py:760-764` van in verdict cu (cot A so voi 0,081)
  — chinh ket luan da rut o 33.5. Giao Claude Code gan nhan `[RUT LAI -- Muc 33.5]` (Rev 17 T1).

### 34-pre.6. Trang thai sau 34-pre
**Duoc ghi:** ket luan cau truc C4 (34-pre.2). Tieu chi M12 Rev 17.
**Cho duyet:** 34-pre.3 (dong nhat thuc `B - A = Var(m)` va du doan R5).
**Chua chay:** M12-a, M12-b. Can Claude Code lam T1–T3 (Rev 17) truoc.

---

## 34. M12 (server, 2026-10-26, HEAD `53b4778`): M12-a KHONG KET LUAN; M12-b LOAI Tanh; moc M10 bi do THAP ~2,6x
Logs: `logs/m12a_probe_rev17.log`, `logs/m12a_pipe_t0_rev17.log` (`--batch-size 8`), `logs/m12b_tanh.log`.
Ca ba: `device: cuda`, HEAD `53b4778`, `K = 20` (hai log M12-a). Tieu chi: PLAN Rev 17 muc 2–3 (ghi cung o 34-pre).
**Con thieu tu nguoi dung:** output `readlink -f ~/SAMG`, `ls logs/m12b_tanh.log`, va `git status --short src/` tren
server (log Tanh ghi `batch_size=8` — can xac nhan code chay dung commit, R2).

### 34.1. M12-b: LOAI gia thuyet Tanh (20.8)
`|tanh_out| > 0,99` = **0,00%** (18.432 phan tu, 1 batch valid, 8 complex, 72 token). `max|pre-activation|` = 1,91
(`tanh(1,91) = 0,956`), `mean` 0,33. Duoi 5% ⇒ **LOAI** theo tieu chi ghi cung.
Gradient: `context_proj[0].weight` 5,93; `context_proj[3].weight` 3,65; `dynamic_encoder` lop cuoi = **None**
— dung du doan cau truc 34-pre.2 muc 4 ⇒ **kiem tinh C4 duoc xac nhan khi chay**.

### 34.2. M12-a theo dung thu tu Rev 17 muc 2
**(1) tokenmean** (all): `d 0,0670 | theta 0,0811 | phi 0,1022` — khop A6.
**(2) Giao thuc #3:** dat ca hai probe (P1 blind `0,1203/0,0776/0,1671`; P2 `0,1301/0,1658/0,1952`).
* P2 **tai lap tung chu so**: `delta R2 = 0,0999 / 0,0674 / 0,0690`.
* P1 **doi** so voi 33.1 vi day quet `lambda` mo rong (Rev 16): `blind` chon `1e4` thay `1e3` ⇒
  `delta R2 = 0,0889 / 0,2206 / 0,0361` (truoc: 0,1040 / 0,2050 / 0,0488). Van 2/3 ⇒ tien de giu nguyen.
  Cau 33.3 "bien thien 0,0005, khong the doi ket luan" dung ve ket luan, **sai ve do lon** (`delta R2(d)` doi 0,015).
**(3) Dong nhat thuc 34-pre.3 `B >= A - 2sd`:** dat o `d` ca 3 tap x 2 probe. Tren 7 chieu x 3 tap x 2 probe:
**40/42** o dat; hai o vi pham deu la `qz`, sat bien (P1 `t1`: B 0,0057 vs A−2sd 0,0085; P2 `t0`: 0,1310 vs 0,1369).
Du doan R5 (tap `all` gan nhu chac chan XAC NHAN) **dung**. ⇒ 34-pre.3 **duoc du lieu ung ho** (nguoi dung duyet 2026-10-26).
**(4) Tap `all`, B(d) qua K = 20:** P1 **0,1573 ± 0,0140**; P2 **0,3112 ± 0,0201**. Ca hai `mean − 2sd > 0,0824` ⇒ XAC NHAN ca hai.
**(5) Tap `t0`** (764 token): pipeline `delta(d) = 0,3595 ± 0,0402` [0,2792 ; 0,4399].
* P1 **0,1238 ± 0,0184** (mean+2sd 0,1606 < 0,2792) ⇒ **BAC t0** — probe dung hoc **it hon** pipeline.
* P2 **0,2895 ± 0,0220** [0,2455 ; 0,3335] chong lan ⇒ **KHONG KET LUAN t0**.
**(6) Verdict tong hop: KHONG KET LUAN** (P1/P2 bat dong o `t0`). "Pipeline khai thac hoc kem hon probe"
**KHONG duoc xac lap.**

### 34.3. Kiem cheo (8) — phat hien quan trong nhat: moc M10 bi do thap
Pipeline, xao tron moi (loader `shuffle=True` + derangement trong batch, generator rieng moi k):
```
tap all (4869 token)  R2(full)  delta mean ± sd      M10 (loader khong xao, batch 16)
d                      0,4095   0,2160 ± 0,0183      0,0824
theta                  0,4267   0,1307 ± 0,0100      0,0015
phi                    0,3607   0,0748 ± 0,0117      0,0208
qw                     0,5294   0,0496 ± 0,0038      0,0110
qx                     0,1743   0,0070 ± 0,0014     -0,0007
qy                     0,7989   0,0159 ± 0,0059      0,0026
qz                     0,2922   0,0631 ± 0,0039      0,0098
```
`R2(full)` khop A3 (lech <= 0,01) ⇒ cung mo hinh, cung du lieu; chi doi **cach xao tron**. `delta(d)` lon gap **~2,6 lan**.
**Gia thuyet (CHUA kiem):** loader khong xao ghep cac complex lien ke — nhieu kha nang cung protein/cung hoc — vao cung
batch, nen derangement trong batch thuong doi hoc lay mot hoc gan nhu giong het. Kiem: Rev 18 (a).
**He qua:**
1. **Nguong o buoc (4) lay tu mot estimator KHAC** voi estimator cua probe (probe xao toan tap). So tren cung cach xao:
   P1 0,157 ≲ pipeline 0,216 ≪ P2 0,311 — hai probe nam **hai phia** pipeline. Khong doi verdict (da la KHONG KET
   LUAN), khong sua tieu chi sau khi thay so (R9).
2. **Do lon o A5 / Muc 29–30 la CAN DUOI va phu thuoc giao thuc.** Ca tinh chat "giu 83% khi khop kich co" (29.5)
   cung do bang cach xao yeu nay.
3. **Phep quay CO dung hoc** duoi cach xao manh: `qz 0,063 ± 0,004`, `qw 0,050 ± 0,004` (>> san nhieu 0,0011).
   Them nua P2 `A(qz) = 0,0500` cham dung nguong `>= 0,05` cua Rev 15 ("mo lai 16.1") — 33.2 chi doc P1 (−0,0049)
   nen **da bo sot**. ⇒ "phep quay khong chiu chi phoi cua hoc" (A5, A6, 33.2) **[XEM LAI — Muc 34.3]**.
4. **C4 khong con la ung vien hien nhien:** o `t0`, pipeline dung hoc cho `d` it nhat ngang ca hai probe du duong
   dieu kien hoa chi duoc huan luyen boi token loss. Chua co bang chung pipeline dung hoc kem.

### 34.4. Gioi han cua chinh dai luong B (tu dong nhat thuc 34-pre.3)
`B = 2 Cov(r, m)` **tang theo do lon cua `m`** ke ca khi `m` bi phong qua muc toi uu. Tu so do: `Var(m) = B − A`:
P1 `d`: Cov ≈ 0,079, Var(m) ≈ 0,068 (do doc hoi quy `r` theo `m` ≈ 1,15 — `m` hoi nho); P2 `d`: Cov ≈ 0,156,
Var(m) ≈ 0,211 (≈ 0,74 — `m` bi **phong qua**, khop voi overfit cua P2). ⇒ **B khong phai thuoc do sach cua "dung hoc".**
Dai luong sach hon: **loi ich MSE `G = B − Var(m) = 2 Cov − Var(m)`**, voi `Var(m) = E[(mu_full − mu_shuf)^2] / 2`
(do truc tiep tu output, khong can train). Voi probe, `G` phai ≈ `A` neu dong nhat thuc dung. Xem Rev 18 (b).
Luu y them: R2 cua pipeline o Muc 25–34 la `1 − sigma^2 E[z^2]`, chi bang `1 − MSE(mu)` khi `sigma` khong doi theo
token; dong nhat thuc viet cho MSE ⇒ (b) phai bao cao ca ban MSE cua pipeline.

### 34.5. Trang thai sau Muc 34
**Duoc ghi:** Tanh bi loai (34.1). C4 xac nhan cau truc, tinh va dong. Dong nhat thuc `B − A = Var(m)` duoc du lieu
ung ho (34.2 (3)). Tien de (A6) giu nguyen; so P1 cap nhat.
**Rut lai / xem lai:** 33.5 (co che sai, 34-pre.3); do lon A5 (can duoi, 34.3); "phep quay khong chiu chi phoi cua
hoc" (34.3 muc 3); so P1 o 33.1 (34.2 (2)).
**Chua xac lap:** pipeline khai thac hoc kem hay khong (M12-a khong ket luan). Buoc tiep: Rev 18 (a) va (b).

---

## 35. Rev 18 (server, 2026-10-27, HEAD `ba77e95`): (a) BAC trong pham vi hep; (b) KHONG KET LUAN; R2 cua pipeline KHONG phai R2 chuan
Logs: `logs/m13a_adjacency.log` (hostname `cuda12.ecs.vuw.ac.nz`), `logs/m13b_probe.log`, `logs/m13b_pipe.log`
(`--batch-size 8`). Ca ba HEAD `ba77e95`; hai log GPU `device: cuda`, K = 20. So probe/pipeline tai lap M12
tung chu so (vd `delta_pipe(d, all)` 0,2156 vs 0,2160). **Van thieu:** `readlink -f ~/SAMG`, `git status --short src/`.
Tu muc nay ap **R16** (moi tieu chi phai co co so hoc thuat — WORKING_RULES muc G).

### 35.1. (a) Ghep cap trong batch: BAC — nhung chi bac dinh nghia TU DAT
valid: 764 complex, **446 PDB**; phan phoi complex/PDB `{1: 310, 2: 78, 3: 21, 4: 10, 5: 5, 6: 2, 7: 2, 8: 18}`.
Cap lien ke cung PDB: 0,2818. `s(bs=16)`: cung PDB **0,0839**, cung PDB & tam < 2 A **0,0004** (`s_rand` 0,0000).
Theo tieu chi ghi cung: `s_pocket(16) < 0,20` ⇒ **BAC**.
**Pham vi that cua ket qua (R16):** dinh nghia "cung hoc = cung PDB & tam < 2 A" la **quy uoc tu dat, khong co
nguon nganh**. No bo sot hai truong hop co trong du lieu:
* 20 entry dau theo thu tu loader deu la ma `5s**` (`5sos`, `5srk`, `5sqj`, ...) — nhieu kha nang mot loat sang loc
  fragment tren **cung mot protein**, moi cau truc mot ma PDB ⇒ "cung PDB" khong bat duoc.
* Cung PDB khac chain (`5sqt__1__1.A` vs `5sqt__2__1.B`) ⇒ cung loai hoc nhung tam khac toa do ⇒ "tam < 2 A" bo sot.
⇒ Chi duoc ghi: **"cung PDB va cung toa do" KHONG phai nguyen nhan M10 do thap.** Gia thuyet "cung protein" **CHUA kiem**.
Lam lai bang dinh nghia nganh: Rev 19 (a').

### 35.2. (b) Loi ich MSE G: KHONG KET LUAN
**(b0)** `|G(d) − A(d)|`, tap all: P1 **0,0158 DAT**; P2 **0,0414 KHONG DAT** (G − A > 0 ⇒ `blind` cua P2 du doan tot hon
phan khong-hoc cua chinh mo hinh `pocket`, tuc `m` cua P2 tuong tac voi dac trung khac; gia dinh cua dong nhat thuc
khong giu). ⇒ chi P1 vao (b2).
**(b1)** `|B_mse(d) − B(d)|` = **0,1010 > 0,03** ⇒ chi doc ban MSE. Xem 35.3 — phat hien quan trong nhat.
**(b2)** tap all (ban MSE): `G_pipe(d)` = **0,0308 ± 0,0126** [0,0055 ; 0,0561]; `G_P1(d)` = **0,0732 ± 0,0128**
[0,0475 ; 0,0988]. Chong lan (0,0475–0,0561) ⇒ **KHONG KET LUAN** (pipeline nghieng ve thap hon, chua vuot luat 2sd).
**(b3)** tap t0: `G_pipe` 0,0695 ± 0,0236 vs `G_P1` 0,0355 ± 0,0179 — chong lan, **nguoc chieu** voi all ⇒ verdict cuoi
**KHONG KET LUAN**.
**Quan sat, khong phai ket luan:** tap t1 `G_pipe` 0,0236 ± 0,0126 (mean+2sd 0,0488) < `G_P1` 0,0802 ± 0,0143
(mean−2sd 0,0516). Pipeline thu loi ich tu hoc o fragment dau, it han o cac fragment sau — **tuong thich** voi gia
thuyet thong tin du thua (context danh tinh fragment truoc thay cho hoc). Chua co tieu chi ghi cung cho dieu nay.
**(b4)** `G_pipe(qz)` = **0,0303 ± 0,0025** (all), 0,0793 ± 0,0069 (t0) — loi ich MSE that tu hoc cho phep quay;
cung co 34.3 muc 3. P1 `G(qz)` −0,0060.

### 35.3. R2 cua pipeline o Muc 25–34 KHONG phai he so xac dinh chuan
He so xac dinh chuan: `R2 = 1 − SS_res/SS_tot` (MSE-based). Du an dung `1 − sigma_bar^2 * E[z^2]` voi
`sigma_bar = exp(mean log sigma)`, `z_i = (t_i − mu_i)/sigma_i` (A2, Muc 25.2). Hai cai chi trung nhau khi `sigma_i`
khong doi theo token; khi `sigma_i` thay doi, `E[z^2]` gan trong so `1/sigma_i^2` nen cac token mo hinh tu khai
"khong chac" (va doan sai) gan nhu khong duoc tinh.
```
pipeline, valid, tap all   R2 (du an, A3)   R2 chuan (MSE cua mu)
d                          0,4095           0,0991     <- tokenmean 0,0670; P1 0,2092; P2 0,2299
theta                      0,4267           0,2075
phi                        0,3607           0,2136
qw                         0,5293           0,3052
qx                         0,1744           0,1263
qy                         0,7989           0,5582
qz                         0,2922           0,1698
```
**He qua:**
1. Nhu mot bo du doan diem, pipeline cho `d` chi **R2 = 0,099** — kem ca hai probe.
2. **Moi so R2 / delta R2 cua pipeline o A3, A5, Muc 25–34 la dai luong co trong so sigma**, khong phai phuong sai
   giai thich. Thang dai so `||q||=1` (A4) cung do bang dai luong nay.
3. **M12-a so lech thuoc do**: probe `1 − MSE`, pipeline co trong so sigma — chenh lech estimator thu hai (them vao 34.3).
4. Dai luong co trong so sigma **khong vo nghia** — no gan voi NLL, phu hop khi danh gia mot mo hinh mat do. Nhung
   khong duoc goi la "R2" va khong duoc so voi R2 cua probe.
⇒ **A2 [RUT LAI — Muc 35.3]**: "R2 = 1 − sigma^2 E[z^2]" khong phai cong thuc R2 dung; no chi dung khi sigma hang.

### 35.4. Trang thai sau Muc 35
**Duoc ghi:** "cung PDB & cung toa do" khong giai thich M10 (35.1); R2 cua pipeline o Muc 25–34 la dai luong co
trong so sigma; R2 chuan cua pipeline cho `d` la 0,099 (35.3); loi ich MSE tu hoc cho `qz` > 0 (35.2 (b4)).
**Rut lai:** A2 (35.3).
**Chua xac lap:** pipeline thu loi ich tu hoc kem probe hay khong (35.2); vi sao M10 do thap (35.1).
**Tiep:** Rev 19 (a') — "cung protein" theo chuan nganh.

---

## 36. Rev 19 (a') (server, 2026-10-27, HEAD `88f5cc3`): TUONG THICH — ghep cung protein giai thich ~78% su do thap cua M10
Log: `logs/m14a_adjacency_seqid.log` (hostname `cuda12`, `git status --short src/` sach — chi file untracked).
MMseqs2 `easy-cluster --min-seq-id 0.3` (version `503743960fc5...`), chuoi = chain chinh cua hoc 10 A; do dai chuoi
min 36 / median 300 / max 1724, khong chuoi nao < 20 aa.
```
min-seq-id   n_cluster   s(bs=16)   s(bs=8)   s_rand
0,30         131         0,4951     0,5999    0,0543
0,95         164         0,4794     0,5830    0,0497   (quy uoc du an)
```
`s_seq30(16) = 0,4951` trong `[0,446 ; 0,823]` ⇒ **TUONG THICH** (tieu chi Rev 19, nguong suy tu du lieu).
Du doan `delta_M10 = (1 − 0,4951) × 0,2156 = 0,1089`; thuc do 0,0751–0,0824 ⇒ co che nay giai thich
`(0,2156 − 0,1089)/(0,2156 − 0,0788) ≈ 78%` chenh lech. Phan con lai (~0,03) chua ro nguon.
Nguong 0,95 cho gan nhu cung `s` ⇒ cac cap cung cluster phan lon la protein **gan trung chuoi hoan toan** (cac loat
sang loc fragment `5s**`, `7fr*/7fs*`).
**Gioi han:** gia dinh "cap cung protein cho delta ≈ 0" van la quy uoc du an (R16 muc 3), chua do nhan qua.

### 36.1. Dac tinh du lieu moi: valid chi co 131 cluster protein (30%)
764 complex, 446 PDB, **131 cluster MMseqs2 30%**. Moi khoang ± 2sd o Muc 33–35 chi do dao dong **giua cac lan xao
tron** (hoac giua cac token), **khong** do bat dinh do chon protein. Voi suy rong sang hoc moi, so mau doc lap hieu
dung gan **131** hon 764 ⇒ cac ket luan dua tren valid (ke ca A6) co the kem chac hon da ghi. Can uoc luong sai so
theo cum (PLAN Rev 20).

### 36.2. Quy tac moi rut ra (dac tinh du lieu, R16 muc 2)
Moi phep xao tron hoc ve sau phai xao **tren toan tap** (hoac loader `shuffle=True`), khong duoc xao trong batch cua
loader `shuffle=False`: o thu tu do ~50% cap trong batch la cung protein.
A5 (cac so M10) ⇒ **[BI KEO THAP — Muc 36]** do ghep cung protein trong batch.

---

## 37-pre. Ra soat Step 7 + chuan danh gia (2026-10-27, HEAD `88f5cc3`, doc code + PDF, KHONG chay gi)
**Khong co so lieu moi.** Ghi truoc khi sua code, theo R16 (moi lua chon kem nguon).
Bo canh: nguoi dung chot muc tieu cuoi la **chat luong phan tu sinh ra**, va dong y viet lai step 0-6 khi
da xac dinh ro cho nao can doi.

### 37-pre.1. Chuan danh gia cua nganh (nguon: Apo2Mol PDF + code `Apo2Mol/eval_split.py`)
Sau chi so chinh (Apo2Mol, muc "Evaluation Metrics"): **Vina min**, **QED** (Bickerton 2012), **SA**
(Ertl & Schuffenhauer 2009), **logP**, **Lipinski**, **High Affinity** (% ligand sinh ra co ai luc tot hon
ligand tham chieu). Moi chi so tru High Affinity bao **ca mean va median**.
Phu luc C.1 them **Validity** va **Novelty** (Tanimoto <= 0,4 tren Morgan fingerprint).
Chi tiet thuc thi lay tu code ho:
* `eval_split.py:54` `--docking_mode vina_score`; `:55` `--exhaustiveness 128`; `:204` chay `mode='minimize'`
  ⇒ **Vina min** = minimize roi lay affinity (khong phai `score_only`, khong phai `dock`).
* `sample_split.py:266` `--num_samples 5` ⇒ **5 phan tu moi hoc**.
* Tien ich dung lai duoc (R14): `Apo2Mol/utils/evaluation/scoring_func.py` (QED/SA/Lipinski),
  `docking_vina.py` (VinaDockingTask), `similarity.py` (Novelty), `analyze.py`.
* Lipinski (Apo2Mol Apx.): MW < 500; HBD <= 5; HBA <= 10; logP trong [-2, 5]; (dieu 5 o PDF).

### 37-pre.2. BA LOI NANG trong `src/step7_evaluation.py` — moi so Vina/QED tu truoc den nay VO NGHIA
**(L1) Tu the mo hinh du doan gan nhu bi vut bo.** Dong 373 goi `AllChem.EmbedMolecule(mol, ETKDG())` dung
hinh hoc 3D, roi dong 385-388 **gop** ca chuoi 7D thanh **mot** tinh tien `pred_coords.mean(axis=0)` va **mot**
phep quay `pred_quats.mean(axis=0)` ap cho **toan** phan tu. Tu the RIENG cua tung fragment — chinh thu mo hinh
hoc — khong duoc dung. ⇒ Vina do chat luong cua **ETKDG**, khong phai cua mo hinh. (Day la hang muc **B7**, hoan
tu plan Rev 4.)
**(L2) Trung binh quaternion theo tung thanh phan la sai toan hoc.** Dong 387-389:
`mean_quat = pred_quats.mean(axis=0)` roi chuan hoa. Quaternion nam tren mat cau don vi trong SO(3); trung binh
so hoc **khong** la phep trung binh tren nhom quay. Nang hon: do **double cover** (`q` va `-q` cung mot phep quay,
muc D), hai quaternion nguoc dau **triet tieu** nhau. Phep trung binh dung: vector rieng ung voi tri rieng lon
nhat cua `sum q_i q_i^T` (Markley et al. 2007, *Averaging Quaternions*, JGCD 30(4)) — nhung xem (L1): **khong nen
trung binh gi ca**.
**(L3) Khung quy chieu sai.** Dong 330 `pocket_center = batch.protein_pos.mean(dim=0)`; dong 386
`ai_predicted_center = pocket_center + pred_coords.mean(axis=0)`. Nhan duoc sinh trong **frame hoc**
(`pocket_R`, `pocket_t` — P1b/P2a), **khong** phai frame the gioi goc o trung binh toa do protein. Thieu han phep
quay `R_m_to_w`, va goc sai (`protein_pos.mean()` != `pocket_t` = `pocket_coords.mean(0)`, hai tap nguyen tu khac
nhau).
**(L4, can kiem tren server) Chuoi SMILES duoc GHEP CHUOI KY TU.** Dong 364:
`smi = "".join([inv_vocab[tid] ...])` roi `.replace("*", "C")`. Token la SMILES **cua tung fragment** co nguyen tu
dummy (vd `[1*]N[6*]`). Ghep chuoi hai fragment cho `[1*]N[6*][3*]C[4*]`, sau replace thanh `[1C]N[6C][3C]C[4C]`
— RDKit **co the** parse (`[1C]` = carbon dong vi 1) nhung phan tu thu duoc khong lien quan den phep lap rap
fragment. **Kiem tren server:** parse chuoi vi du nay va in `MolToSmiles` + `GetNumAtoms`.
⇒ **Khong co baseline dang tin.** Phai sua Step 7 **truoc** khi huan luyen lai, neu khong se khong doc duoc
huan luyen lai tot len hay xau di.

### 37-pre.3. Tai dung khong can ETKDG LA KHA THI — phep nghich dao da xac dinh day du
**Nguon:** Frag2Seq muc 3.3.1 (PDF): "we save each splited fragment from the training set into a dictionary with
the key to be the canonical SMILES of each fragment and the value to be the atom types and atom local coordinates
in each fragment under the associated fragment local frame"; Dinh ly surjective (Apx.): tu chuoi
`[d, θ, φ, quaternion]` **tai dung duy nhat** phan tu 3D, sai khac mot phep bien doi SE(3). Bo 7 so cua du an
**chinh la** bieu dien do.
**Phep thuan trong `step0_data_prep.py:236-242`:**
```
rel_center = R_m_to_w.T @ (center_world - t_m_to_w);  (d, θ, φ) = spherical(rel_center)
R_g_to_w, t_g_to_w = _get_local_frame(coords_canon)     # t_g_to_w = coords_canon[0], KHONG phai centroid
R_g_to_m = R_m_to_w.T @ R_g_to_w;  (qx,qy,qz,qw) = R.from_matrix(R_g_to_m).as_quat()
```
**Phep nghich dao (day du, tat dinh):**
```
rel_center = [d sinθ cosφ, d sinθ sinφ, d cosθ]        # dung voi _get_spherical_coords (dong 187-192)
center_world = R_m_to_w @ rel_center + t_m_to_w        # R_m_to_w = pocket_R, t_m_to_w = pocket_t (CO trong pkl)
R_g_to_w = R_m_to_w @ R_g_to_m
atom_world = R_g_to_w @ (local_coords - centroid_local) + center_world
```
voi `local_coords = R_g_to_w^T @ (coords_canon - coords_canon[0])` cua fragment do.
**Thieu mat xich duy nhat:** `local_coords` **KHONG duoc luu**. pkl chi co
`{frame_stable, frame_diag, sequence, pocket_R, pocket_t}`, moi item sequence chi co
`{smiles, spatial_tokens, frag_frame_stable}` (kiem tren server 2026-10-27). Nhung `coords_canon` **duoc tinh** o
`step0:223` roi vut ⇒ dung lai bang mot script duyet lai SDF, **khong** phai chay lai step0 2,5 gio.
**Xap xi phai ghi ro:** mot SMILES fragment co the co nhieu conformer khac nhau giua cac phan tu; tu dien luu
**mot** conformer cho moi SMILES (dung nhu Frag2Seq lam). Day la xap xi cua **ca Frag2Seq**, khong phai cua rieng
du an — nhung van phai do: bao cao RMSD noi-fragment giua conformer trong tu dien va conformer that.
**Con lai chua giai:** noi cac fragment tai nguyen tu dummy de ra phan tu lien thong (can cho QED/SA/Vina).
Frag2Seq co lam; chi tiet chua doc. Xem Rev 20 B2.

### 37-pre.4. Trang thai
**Duoc ghi:** chuan danh gia + chi tiet thuc thi (37-pre.1); ba loi L1-L3 trong Step 7 (37-pre.2); phep nghich
dao day du va mat xich thieu (37-pre.3).
**Can kiem tren server:** L4.
**He qua cho ke hoach:** **Step 7 phai sua TRUOC khi huan luyen lai.** Xem PLAN Rev 20.

### 37-pre.5. NGHI VAN NANG o `step0`: token fragment co the mang CHI SO NGUYEN TU cua phan tu goc
`step0_data_prep.py:209`: `Chem.FragmentOnBonds(mol, bonds_to_break)` — goi **khong** truyen `dummyLabels`.
Mac dinh cua RDKit khi `addDummies=True` va `dummyLabels=None`: nguyen tu dummy duoc gan **isotope = chi so
nguyen tu** cua dau kia cua lien ket bi cat. Du lieu khop: pkl dau tien co `smiles = '[1*]N[6*]'` — "1" va "6"
la **chi so nguyen tu trong phan tu goc**, khong phai nhan hoa hoc.
**Neu dung, hau qua rat nang:**
1. **Cung mot fragment hoa hoc o hai phan tu khac nhau ⇒ HAI token khac nhau.** Tu vung phinh to va bi phan
   manh: `vocab_size = 19.033` tren 17.200 phan tu train (~1,1 token mỗi phan tu) — **tuong thich** voi gia
   thuyet nay, va kho giai thich neu token la fragment hoa hoc thuan.
2. **Token sinh ra vo nghia ve hoa hoc:** mo hinh phat ra token mang chi so cua mot phan tu train nao do.
3. **Tu dien fragment (Rev 20 A1) bi nhiem:** key chua chi so ⇒ do phu tren valid se rat thap.
4. Giai thich duoc vi sao `Token_Loss ~ 3,349` kho giam va vi sao N1 (thu tu fragment) kho doc.
**CHUA XAC NHAN** — can kiem tren server (lenh o plan Rev 20). Neu xac nhan, day la **sua bat buoc o step0**
truoc moi lan huan luyen lai: truyen `dummyLabels=[(0,0)] * len(bonds_to_break)` (tat ca dummy thanh `[*]`),
hoac dung nhan **hoa tri/loai lien ket** nhu BRICS thay vi chi so.
**Nguon cho ban sua:** Frag2Seq muc 3.2 cat lien ket xoay duoc (3 dieu kien — **giong het** `step0:206`), va luu
tu dien theo **canonical SMILES cua fragment** (muc 3.3.1) ⇒ key phai la SMILES **hoa hoc**, khong mang chi so.

### 37-pre.6. Thu tu fragment: step0 KHAC Frag2Seq (hang muc N1, nay co nguon chinh xac)
`step0:230`: `frag_data.sort(key=lambda x: x["smiles"])` — sap **theo thu tu bang chu cai cua SMILES**.
Frag2Seq muc 3.2/3.3.1: sap theo **thu tu xuat hien trong canonical SMILES cua CA PHAN TU**, cu the theo
"the order of a fragment's first-ranked atom". Day la hai thu tu khac nhau: cua Frag2Seq phan anh **lien ket
hoa hoc doc chuoi**, cua step0 thi khong. ⇒ N1 gio co **nguon** cho ban sua, khong con la "khiem khuyet chua
biet sua the nao".

---

## 37. Kiem step0/step7 (server, 2026-10-27, HEAD `4274b0c`): 37-pre.5 **XAC NHAN**; L4 xac nhan; cach lap rap da ro
Log: `logs/m15_step0_tokens.log` (hostname `cuda12`, `git status --short src/` sach).

### 37.1. Token fragment MANG CHI SO NGUYEN TU — xac nhan, muc do nang hon du doan
```
V (token fragment)                        = 19.031
token co dummy dang [N*]                  = 18.822  (98,90%)
U (duy nhat sau khi [N*] -> [*])          =  5.334
U / V                                     = 0,2803   < 0,50 => XAC NHAN
```
Vi du token: `[3*]CO`, `*C[9*]`, `*[C@@H]1CCCN([25*])C1`.
**Phan manh tap trung dung vao cac fragment PHO BIEN NHAT:**
```
[*]C[*]            -> 672 token      [*]c1ccc([*])cc1 -> 417 token
[*]C([*])=O        -> 570 token      [*]O[*]          -> 409 token
[*]N[*]            -> 519 token
```
Phan bo so bien the/fragment: `{1: 3593, 2: 688, 3: 292, 4: 165, 5: 115, 6: 85, 7: 49, 8: 51, ...}`.
⇒ Tu vung **phinh 3,57 lan** (19.031 / 5.334). Cac fragment thuong gap nhat — cau noi methylene, carbonyl, NH,
phenyl — bi vo thanh hang tram token rieng biet, **chi khac nhau o chi so nguyen tu cua phan tu train**.
**He qua (suy luan, chua do rieng):** mot phan `Token_Loss ≈ 3,349` la **entropy khong the giam**: ngay ca khi mo
hinh doan dung fragment hoa hoc, no van phai doan dung bien the chi so. Voi `[*]C[*]`, `ln(672) = 6,51` nat.
Dieu nay **tuong thich** voi viec `Token_Loss` kho giam va voi ket qua "Token_Loss chu yeu la thong ke chuoi".
**Chua do:** phan ra `Token_Loss` thanh (entropy chon fragment hoa hoc) + (entropy chon bien the chi so).

### 37.2. L4 xac nhan: phan tu "duoc danh gia" la ghep chuoi ky tu
```
parts  = ['[3*]CO', '[4*][C@H]1OC[C@H](O)[C@@H](O)[C@@H]1O']
ghep   = '[3*]CO[4*][C@H]1OC...'            -> parse OK, 13 nguyen tu
sau .replace("*","C") = '[3C]CO[4C][C@H]1...' -> parse OK, 13 nguyen tu, 2/13 nguyen tu co isotope != 0
```
Chuoi ghep **parse duoc** nen khong he bao loi, nhung phan tu thu duoc la noi duoi hai chuoi SMILES, **khong**
phai phep lap rap fragment theo hinh hoc; cac nguyen tu dummy thanh carbon dong vi (`[3C]`, `[4C]`).
⇒ Cong voi L1-L3 (37-pre.2): **moi so Vina/QED/SA/Lipinski cua du an deu vo nghia** — ca ve hinh hoc (ETKDG +
trung binh quaternion + sai frame) lan ve **thanh phan hoa hoc** (ghep chuoi).

### 37.3. Cach lap rap fragment — GIAI, bang chuan cua nganh (dong Rev 20 B2)
**Nguon:** `Apo2Mol/eval_split.py:122` goi `reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type,
pred_aromatic)`; `Apo2Mol/utils/reconstruct.py:455` cai dat bang **OpenBabel bond perception** (suy lien ket tu
toa do 3D + so hieu nguyen tu). `eval_split.py:246` bao `recon_success` nhu mot chi so.
⇒ **Khong can bookkeeping nguyen tu dummy de lap rap.** Duong dung:
```
7D cua tung fragment --(nghich dao 37-pre.3)--> toa do 3D cua tung nguyen tu trong frame the gioi
  -> bo nguyen tu dummy -> gom (xyz, atomic_num) cua ca phan tu
  -> reconstruct_from_generated(...)  [dung code Apo2Mol, R14]  -> RDKit mol -> QED/SA/Vina
```
Day cung la ly do **co the bo chi so dummy khoi token ma khong mat gi**: lien ket giua cac fragment duoc suy tu
**hinh hoc**, khong tu nhan.

### 37.4. Ban sua `step0` da xac dinh
`step0_data_prep.py:209`: `Chem.FragmentOnBonds(mol, bonds_to_break)` ⇒
`Chem.FragmentOnBonds(mol, bonds_to_break, dummyLabels=[(0, 0)] * len(bonds_to_break))`.
Moi dummy thanh `[*]` (isotope 0) ⇒ token = **SMILES hoa hoc thuan**, tu vung `19.031 -> ~5.334`.
**Bat buoc chay lai `step0` + `compute_scaler.py`** (nhan 7D **khong** doi, nhung tu vung va token id doi hoan
toan) ⇒ **phai huan luyen lai**. Dieu kien cat cua step0:206 (lien ket don, khong trong vong, bac hai dau > 1)
**giong het** Frag2Seq muc 3.2 — **khong doi**.
**Canh bao tai dung (R15):** cache probe `dataset/cache_probe/` chua `target_ids` theo tu vung CU ⇒ **vo hieu**
sau khi doi; moi ket qua probe (A6, Muc 32-35) gan voi tu vung cu. Nhan 7D va `h_target` khong doi, nhung
`frag_id` doi ⇒ `tokenmean` va target-encoding phai tinh lai.

### 37.5. Trang thai
**Duoc ghi:** 37.1 (token mang chi so, tu vung phinh 3,57x), 37.2 (L4), 37.3 (cach lap rap), 37.4 (ban sua step0).
**He qua ke hoach:** huan luyen lai **bat buoc** (tu vung doi), nhung **van phai sua Step 7 truoc** de co baseline
doc duoc. Thu tu A -> C cua Rev 20 **giu nguyen**, bo sung ban sua step0 vao C.

### 37.6. Hai loi them o `build_global_vocab` (doc code, sua cung luc voi 37.4)
1. **Tu vung KHONG tai lap duoc.** `step0:327` dung `glob.glob(...)` **khong sap**; thu tu tra ve phu thuoc he
   thong tep ⇒ **token id khac nhau giua cac lan chay/may**. Cung nguyen tac voi P3a-2 (seed) va R2: hai run
   khong so sanh duoc neu dinh danh token khong tat dinh. ⇒ `sorted(glob.glob(...))`.
2. **Tu vung dung ca valid/test.** Vong lap duyet **moi** `*_sequence_7d.pkl`. Nguon cho ban sua: Frag2Seq muc
   3.3.1 — "we save each splited fragment **from the training set** into a dictionary". Khong gian dau ra cua mo
   hinh khong duoc dinh hinh boi du lieu danh gia; fragment chi co o valid/test phai roi vao `[UNK]`, dung nhu
   khi trien khai that. ⇒ `build_global_vocab(train_pli_ids=...)`.
**Luu y khi doc ket qua ve sau:** `[UNK]` tren valid/test se **tang** so voi truoc. Do la dung, khong phai hoi quy.

### 37.7. Bon thay doi cua Rev 20 o `step0` + kiem khong-doi ghi cung TRUOC khi chay lai
| # | Thay doi | Nguon |
|---|---|---|
| S1 | `dummyLabels=[(0,0)] * n` | 37.1 (do duoc U/V = 0,2803) + Frag2Seq 3.3.1 (key = SMILES hoa hoc) |
| S2 | Luu `local_coords`, `atomic_nums` | Frag2Seq 3.3.1; phep nghich dao 37-pre.3 |
| S3 | N1: thu tu fragment theo canonical SMILES cua **ca phan tu** | Frag2Seq muc 3.2/3.3.1 ("first-ranked atom") |
| S4 | `sorted(glob)` + vocab chi tu **train** | 37.6 |
**S1-S4 KHONG doi gia tri nhan 7D** (cung fragment, cung pose; chi doi **nhan dummy**, **thu tu**, va **tu vung**).
⇒ **Kiem khong-doi (R15), ghi cung TRUOC khi chay:**
* **K1 — scaler:** `scaler_7d.pt` moi phai khop ban cu trong **sai so tuong doi 1e-4** o ca 14 so
  (`d 5,161378/2,478297 · theta 1,531624/0,664945 · phi 0,391346/1,759804 · qw 0,200377/0,456459 ·
  qx 0,204276/0,458135 · qy 0,201783/0,457270 · qz 0,185659/0,464332`, A1). Sai so 1e-4 (khong phai 0) vi doi
  **thu tu cong** do S3 ⇒ lech dau phay dong. **Khong khop ⇒ S1-S4 da lam doi nhan ⇒ DUNG, tim loi.**
* **K2 — tu vung:** `len(vocab)` moi phai `<= 5.336` (= 5.334 fragment hoa hoc do duoc o 37.1, + `[SOS]`,
  `[UNK]`) va `>= 3.500` (train la ~92% so phan tu). Ngoai khoang ⇒ bao cao truoc khi di tiep.
* **K3 — so complex:** `frame_stable` phai giu **17.200 / 764 / 255** (A1). Doi ⇒ S1-S4 da cham vao frame hoc,
  **DUNG**.
* **K4 — truong moi:** moi item `sequence` phai co `local_coords` (shape `[n,3]`) va `atomic_nums` (shape `[n]`)
  cung do dai; `frag_order_mode == "frag2seq_canonical_appearance"` o ~100% file.
**Mat hieu luc sau khi chay lai:** `dataset/cache_probe/` (chua `target_ids` theo tu vung cu) ⇒ moi ket qua probe
Muc 32-35 (ke ca A6) gan voi tu vung cu. `*_protein_transforms.pt` **khong** anh huong (frame hoc khong doi).

---

## 38. Rev 20 A0 + A1 (server, 2026-10-27, HEAD `8196e9b`): S1-S4 DAT K1-K4; tu dien fragment dung duoc
Logs: `logs/m16_step0_rerun.log`, `m16_scaler.log`, `m16_check.log`, `m16_fragdict.log`.

### 38.1. K1-K4 DAT ⇒ S1-S4 khong lam doi nhan 7D
Scaler moi khop ban cu trong 1e-4; vocab trong `[3.500 ; 5.336]`; `frame_stable` giu 17.200/764/255;
`local_coords`/`atomic_nums` day du, `frag_order_mode = frag2seq_canonical_appearance`.
⇒ Bon thay doi (dummyLabels, luu toa do cuc bo, thu tu Frag2Seq, vocab sorted+train-only) **chi doi tu vung,
thu tu va nhan dummy**, dung nhu dac ta 37.7.

### 38.2. Tu dien fragment (lan 1): xap xi mot-conformer DAT, do phu THIEU
```
token fragment doc duoc        101.791        SMILES duy nhat trong tu dien   3.178
rmsd_spread: median 0,0620 A   p90 0,2818     p99 1,5265
do phu VALID: token 4.608/4.869 = 0,9464  |  phan tu du MOI fragment: 542/764 = 0,7094
```
* **`median(rmsd_spread) = 0,062 A` ⇒ DAT** (nguong 0,5). Xap xi "mot conformer moi SMILES" (chinh la xap xi cua
  Frag2Seq) **khong** phai nguon loi dang ke cho phan lon fragment.
* **Duoi cung co duoi nang:** `p99 = 1,53 A`; lech nhat la `*O*` 6,25 A, `*CC` 5,80 A, `*c1nnc(*)n1*` 4,13 A.
  Day deu la fragment **rat nho va doi xung**. Mot fragment 3 nguyen tu voi lien ket ~1,4 A **khong the** lech
  6 A ve hinh hoc that ⇒ nhieu kha nang la **hoan vi thu tu nguyen tu**: `CanonicalRankAtoms(breakTies=True)`
  chay tren mol chi `UpdatePropertyCache(strict=False)` + `FastFindRings` (khong sanitize day du), nen voi
  fragment doi xung thu tu co the khac nhau giua cac lan. Kabsch **khong** sua duoc hoan vi.
  **Anh huong thuc te: nho.** Luc SINH khong co "thu tu dung" de doi chieu — ta dat nguyen van template, va lien
  ket duoc suy lai tu hinh hoc (37.3). Phep kiem tai dung A2 dung `local_coords` **theo tung lan xuat hien** trong
  pkl nen khong qua tu dien ⇒ khong bi anh huong. Da them chan doan (n_atom cua cac lan lech nhat) de xac nhan.
* **Do phu la van de that:** chi **70,94%** phan tu valid co du moi fragment.

### 38.3. Nguyen nhan do phu thieu — da sua
`build_fragment_dict.py` loc `frame_stable` (do on dinh cua **frame HOC**), trong khi `build_global_vocab`
**khong** loc. Template fragment nam trong **frame FRAGMENT**, doc lap hoan toan voi frame hoc ⇒ bo loc do lam
mat do phu **vo co**. Da sua: bo loc `frame_stable`, chi bo item co `frag_frame_stable=False` (voi chung
`_get_local_frame` roi ve `eye(3)` nen `local_coords` nam trong huong **the gioi**, khong dung lam template duoc).
Them **do phu TU VUNG** — so do quan trong hon do phu valid: luc sinh, mo hinh chi phat ra token trong tu vung;
token khong co template thi **khong dat pose duoc**. Tu vung >= 3.500 (K2) nhung tu dien chi 3.178 ⇒ truoc khi
sua co **>= 322 token khong co template**.

### 38.4. He qua quan trong: KHONG CO baseline tu checkpoint cu
`periodic-epoch=024.ckpt` co `token_embedding` 19.033 x 256 va `semantic_head` xuat 19.033 logit. Tu vung moi
~3.500 ⇒ **lech shape, khong load duoc cho sinh mau**. Khong the lay baseline cua mo hinh cu duoi tu vung moi.
Ma so cu cung **vo nghia san** (37.2). ⇒ **Bo y dinh "baseline tu checkpoint cu"** (Rev 20 A3).
**Thay bang ba moc so sanh, deu co can cu:**
1. **Ligand tham chieu** cua chinh cac hoc do (Apo2Mol cung bao moc nay — PDF Fig. 8, 9).
2. **Nhan that qua duong tai dung** = tran tren cua bieu dien 7D + tu dien fragment (khong phu thuoc mo hinh).
3. **Bang baseline cua Apo2Mol** (Pocket2Mol, IPDiff, Apo2Mol...) — **cung dataset, cung split PLINDER
   23.052/1.071/478** (PDF muc 5) ⇒ so sanh truc tiep duoc, khong can chay lai baseline nao.

### 38.5. Thu tu da cap nhat
`A0 DAT` → **A1' (dung lai tu dien, ~1 phut)** → **A2 (viet lai Step 7 + kiem tai dung tu NHAN THAT)** →
**C (huan luyen lai step6, ~19 gio)** → **A3 (danh gia, so voi 3 moc o 38.4)**.
A2 chay duoc **ngay**, khong can mo hinh: no chi dung nhan that + tu dien.
**B (cluster bootstrap): HOAN.** Cache probe gan voi tu vung cu (37.7) nen phai dung lai cache truoc; gia tri
thong tin thap hon duong A2→C→A3. Quay lai sau khi co so chat luong phan tu.

### 38.6. LOI NGHIEM TRONG phat hien khi viet A2: chay lai step0 da XOA `pocket_R`/`pocket_t`
`step0_data_prep.py` **khong bao gio** ghi `pocket_R`/`pocket_t`; chung do script migration
`src/add_pocket_frame_to_pkl.py` (P2a-3, plan Rev 4) them vao **sau**. Chay lai step0 (A0) ghi de toan bo
`*_sequence_7d.pkl` ⇒ **hai truong nay da mat**.
**Hau qua:** `step6_trainer.py:177-180` doc `pkl_data["pocket_R"]` ⇒ huan luyen lai se **loi KeyError ngay**;
phep kiem tai dung A2 cung can chung. K1-K4 **khong bat duoc** vi khong kiem truong nay (thieu sot dac ta 37.7).
**Sua (hai phan):**
1. **Truoc mat:** chay `python -u src/add_pocket_frame_to_pkl.py` (~15-25 phut) — re hon chay lai step0 2,5 gio.
2. **Lau dai:** `step0` **tu ghi** `pocket_R`/`pocket_t` (da sua): `R_m_to_w`/`t_m_to_w` co san ngay tai cho tra
   ve va la **cung mot frame** dung de sinh nhan ⇒ bo han buoc migration, het ca rui ro lech frame
   (dung tinh than R14: khong tinh lai mot dai luong da co bang duong khac).
**Bai hoc cho 37.7 (them vao R15):** kiem khong-doi phai liet ke **moi truong ma downstream doc**, khong chi cac
truong moi them. Lan nay thieu `pocket_R`/`pocket_t` nen K1-K4 bao DAT trong khi pkl da hong voi step6.

---

## 39. Rev 20 A1b + A2a (server, 2026-10-27, HEAD `7a77ef6`): phep nghich dao DUNG TUYET DOI
Logs: `logs/m18_pocket_frame.log`, `logs/m18_recon.log`.

### 39.1. Khoi phuc pocket frame: 23.660/23.660 file, khong mismatch
`add_pocket_frame_to_pkl.py` chay 3 phut 32 giay, them `pocket_R`/`pocket_t` cho **toan bo** pkl, **khong** canh
bao `n_mismatch` ⇒ `_get_pocket_frame` tai lap duoc giua hai lan chay (xac nhan them cho P1b: Gram-Schmidt tat
dinh, khong co nhanh re). Tu lan chay step0 sau, buoc nay **khong con can** (step0 tu ghi, Muc 38.6).

### 39.2. Kiem tai dung tu nhan that: **RMSD = 0,0000 o 20/20** — DAT
```
20 complex valid, 15-32 nguyen tu moi phan tu
RMSD median = 0,0000   max = 0,0000   max dev = 0,0000   sai nguyen to = 0
```
Gan ghep Hungarian co rang buoc cung nguyen to; so nguyen tu tai dung **khop chinh xac** SDF goc o ca 20.
⇒ **Phep nghich dao 37-pre.3 dung tuyet doi**, khong phai "trong sai so". Chuoi
`(pocket_R, pocket_t, [d, θ, φ, quaternion], local_coords)` **tai dung day du** phan tu 3D — dung nhu dinh ly
surjective cua Frag2Seq. **Bo 7 so KHONG mat thong tin nao.**
He qua: moi sai lech o buoc danh gia ve sau **chi** den tu (a) du doan cua mo hinh, hoac (b) xap xi template
mot-conformer cua tu dien — **khong** tu bieu dien. Day la lan dau du an co mot mat xich **khong** phai nghi ngo.

### 39.3. Chan doan `*O*`: la BIEN THIEN HINH HOC THAT, khong phai hoan vi thu tu — va lo ra mot diem yeu
Tren 300 lan xuat hien dau: `kabsch` toi da **0,339 A**, nhung `max|d sorted|` = **0,668 != 0**.
Vector khoang cach doi mot **bat bien voi thu tu nguyen tu**, nen khac nhau ⇒ **hinh hoc that su khac**.
```
moc   (1a4h): sorted-distance [1,310 ; 1,454 ; 2,371]   (d1-d2 = 2,371 ⇒ goc C-O-C ≈ 112°)
lech  (1aj2): sorted-distance [1,527 ; 1,565 ; 3,039]   (d1-d2 = 3,039 ⇒ goc C-O-C ≈ 159°)
```
⇒ **Gia thuyet "hoan vi thu tu nguyen tu" o 38.2 BI BAC.** `rmsd_spread` do bien thien conformer that.
Goc ete C-O-C **159°** khong phai hinh hoc hoa hoc binh thuong (~110-120°) ⇒ chat luong cau truc nguon
(PLINDER loc `resolution <= 2,5 A`, van con cau truc tinh che kem).
**Truong hop 6,25 A chua cham toi:** chan doan dung o 300 lan xuat hien dau (cac PDB dau bang chu cai).
Voi fragment 3 nguyen tu, `kabsch = 6,25 A` doi hoi `span > 12 A` ⇒ **conformer hong** trong SDF nguon.
**Diem yeu lo ra:** tu dien lay conformer **DAU TIEN gap** lam template — mot lua chon **tuy tien**. Neu lan dau
gap roi dung vao conformer hong thi **moi** phan tu sinh ra chua fragment do deu co hinh hoc vo ly.
**Ban sua (Muc 39.4).**

### 39.4. Tu dien: lay conformer MEDOID thay vi conformer dau tien
Thay "first seen" bang **medoid**: trong toi da `K = 50` lan xuat hien duoc lay mau cho moi SMILES, chon conformer
**cuc tieu hoa tong RMSD (Kabsch) toi cac conformer con lai**. Medoid la uoc luong trung tam **ben voi ngoai lai**
(khac trung binh, no luon la mot conformer co that nen hinh hoc hoa hoc hop le). Conformer hong nam o ria ⇒
khong bao gio duoc chon tru khi phan lon lan xuat hien deu hong.
Bao cao them: `medoid_spread` (RMSD trung binh tu medoid), so fragment ma **first khac medoid > 0,5 A** (= so
fragment ma lua chon tuy tien truoc day da sai lech dang ke), va **ngoai lai hinh hoc** (`span` lon bat thuong
so voi so nguyen tu).

### 39.5. Trang thai: huan luyen lai KHONG con gi chan
P3a-1 (`permutation=torch.arange`) va P3a-3 (mask padding cho `_ls_sum`/`_pin_sum`) **da duoc ap dung** tu truoc
(`step2_mdn_module.py:53-56`, `:74-86`, `:140-142`, `:151-152`) ⇒ hai muc HOAN tu Rev 5/Rev 10 **da xong**.
`pocket_R`/`pocket_t` da khoi phuc; tu vung moi; scaler khop. ⇒ **Co the chay step6 ngay.**
**KHONG dua C4 (bo `v_context.detach()`) va 6D rotation vao lan nay:** ca hai la **thi nghiem** doi muc tieu
huan luyen, khong phai sua loi; gop vao se khong tach duoc nguyen nhan neu ket qua doi. Giu HOAN, quyet o Rev 21
sau khi co so chat luong phan tu dau tien.

---

## 40-pre. Hai loi chan huan luyen lai (server, 2026-10-28, HEAD `192e747`) — doc log, sua code
Log: `logs/m19_train.log`. `trainer.fit` **no ngay** voi `RuntimeError: Error(s) in loading state_dict`.

### 40-pre.1. `step6_trainer.py` TU DONG resume tu checkpoint cu, khong hoi
`step6:533-539` lay **file .ckpt moi nhat** trong `saved_checkpoints_flow/` roi truyen vao
`trainer.fit(..., ckpt_path=...)`. Sau khi tu vung doi (19.033 → 3.983, Muc 37.4) checkpoint cu **khong the**
load: `token_embedding.weight` `[19033,256]` vs `[3983,256]`, `semantic_head.3` tuong tu.
**Day la hanh vi nguy hiem doc lap voi loi hom nay:** mot script huan luyen **mac dinh** resume se am tham tiep
tuc mot run cu moi khi thu muc checkpoint khong rong — tuc khong bao gio chay duoc tu dau neu khong xoa file.
**Sua:** resume phai **xin tuong minh** (`--resume <path|auto>`), mac dinh huan luyen tu dau; them **guard**
doi chieu `vocab_size` cua checkpoint voi vocab hien tai va bao loi **ro rang** thay vi de Lightning no.

### 40-pre.2. `geo_head` mac dinh la `diag_gauss`, nhung toan bo cong cu do doc MAF
`step6:438` mac dinh `"diag_gauss"`; log dong 12 xac nhan `geo_head: diag_gauss  permutation ARN: None`.
Checkpoint cu la **MAF** (20.9), va **moi** so da do (A3, A4, A5, A6, Muc 24-35) deu tren MAF.
**Ly do chon MAF cho lan nay (khong phai so thich):** cong cu do cua du an —
`measure_pocket_shuffle.py` (hook tren `model.generator.geometric_head.arn`),
`measure_logscale_realtokens.py`, `recover_arn_permutation.py` — **chi chay duoc voi MAF**;
`DiagonalGaussianHead` khong co `arn`. Chay `diag_gauss` bay gio la **mat toan bo kha nang chan doan**.
So sanh MAF vs diagonal (E1, hoan tu Rev 5) van la mot thi nghiem **mot bien** rieng, lam sau.
**Sua:** them `--geo-head`, **mac dinh `maf`**.

### 40-pre.3. Kem theo
* In `vocab_size` o dau run (R2) — truoc day khong in, nen khong doi chieu duoc 3.983 vs 19.033.
* `--max-epochs` thanh tham so (truoc hard-code 25).
* **Checkpoint cu phai duoc luu tru, khong xoa** (R12): chung thuoc tu vung cu va moi so o Muc 24-35 gan voi
  chung. Chuyen sang `saved_checkpoints_flow_OLD_vocab19033/` de `default_ckpt()` cua cac script do khong
  vo tinh lay nham.

---

## 40. Huan luyen lai (server, 2026-10-28, HEAD `428c835`): KHOI DONG DUNG, 25 epoch da chay
Log: `logs/m20_train_maf.log`. Bon kiem dau run **deu dat**:
`vocab_size: 3983` · `geo_head: maf  permutation ARN: tensor([0,1,2,3,4,5,6])` (P3a-1 co hieu luc) ·
`Huan luyen TU DAU (khong resume)` · `Train 17.200 / Val 764`. Mo hinh 10,3 M tham so.
⇒ Hai loi 40-pre da duoc sua dung. Day la checkpoint **dau tien** cua du an tren tu vung hoa hoc thuan
(3.983 token) va thu tu fragment theo Frag2Seq.
**Chua doc duoc:** quy dao `Token_Loss`/`Geo_Loss`, `val_total_loss`, epoch tot nhat — can phan duoi cua log.

### 40.1. Buoc chan truoc khi sinh mau: do TRAN cua duong lap rap (khong can mo hinh)
Can cu 39.2 (tai dung RMSD = 0,0000) + 37.3 (lap rap bang `reconstruct_from_generated`). Truoc khi quy bat ky
sai lech nao cho mo hinh, phai biet **duong lap rap tu no mat bao nhieu**. Ba hang tren cung tap test:
| hang | nguon toa do | do dieu gi |
|---|---|---|
| `reference` | doc thang SDF goc | moc that |
| `gt_exact` | nhan that + `local_coords` tung lan xuat hien | **rieng** buoc lap rap (toa do chinh xac) |
| `gt_template` | nhan that + template medoid | lap rap + xap xi mot-conformer = **TRAN TREN cua mo hinh** |
`gt_template` la tran tren **that su**: khong mo hinh nao vuot qua duoc no, vi mo hinh dung **cung** template
va **cung** duong lap rap, chi khac la pose do no du doan thay vi nhan that.
**Tieu chi ghi cung:** `recon_success(gt_exact) >= 0,90` ⇒ di tiep; `< 0,70` ⇒ lap rap la nut that, sua truoc,
**khong sinh mau**; `|QED(gt_exact) − QED(reference)| <= 0,05` ⇒ lap rap giu duoc hoa hoc.
Ly do nguong: tu toa do **chinh xac**, suy lien ket la bai toan da giai (OpenBabel), nen moi that bai la mat
mat **thuan** cua phuong phap. Con so tho duoc bao cao du the nao (quy uoc du an, R16 muc 3).

### 40.2. Thieu sot kien truc phat hien khi thiet ke buoc sinh mau: **KHONG CO token EOS**
`build_global_vocab` chi tao `{"[SOS]": 0, "[UNK]": 1}` roi them SMILES fragment. **Khong co `[EOS]`.**
Them nua `pad_sequence(..., padding_value=0)` dung **cung id voi `[SOS]`**; may man la `target_ids` khong bao
gio chua `[SOS]` nen `pad_mask = (target_ids != 0)` van dung, nhung day la su trung id nguy hiem.
⇒ **Mo hinh khong co cach nao bao "phan tu da xong".** `step7` cu bu bang `max_len = 15` cung nhac — sinh du
15 fragment cho moi phan tu, trong khi train trung binh **5,9** fragment/phan tu (101.791/17.200).
**Tam thoi (cho lan nay):** lay do dai tu **phan phoi thuc nghiem so fragment/phan tu cua train**
(quy uoc du an, R16 muc 3). **Ban sua that:** them `[EOS]` vao tu vung va vao `target_ids`, can chay lai step0
+ huan luyen lai ⇒ ghi vao Rev 21, khong gop vao lan nay.

### 40.3. Duong dan server: `~/SAMG` KHONG ton tai (xac minh 2026-10-29)
Cau hoi treo tu Muc 37 (`readlink -f ~/SAMG`) **da co dap an**: `cd ~/SAMG` bao `no such file or directory`.
Duong dan that la `/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG`; khong co symlink nao.
`WORKING_RULES` muc E va moi khoi lenh tro ly dua ra tu Muc 34 den nay deu ghi `cd ~/SAMG` — **sai**.
Cac lan chay truoc van dung vi nguoi dung da o san thu muc do. Da sua mau lenh chuan sang
`export SAMG=/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG` roi `cd $SAMG`.

### 40.4. Bo PHU THUOC Apo2Mol: viet bo lap rap cua rieng du an (`src/assemble.py`)
**Su that bi bo sot:** server **khong co** repo Apo2Mol (nguoi dung da noi; tro ly van viet code import
tu `Apo2Mol/utils/`). `Apo2Mol/` chi ton tai o may **local**. Day la lan thu hai tro ly dung mot thu
chi co o local lam can cu (lan truoc: `logs/m12b_tanh.log`, Muc 37-pre).

**Nhung doi huong nay tot hon ve ky thuat, khong chi la vong tranh:**
`reconstruct_from_generated` (Apo2Mol, goc tu liGAN, GPL-2.0) **vut bo toan bo thong tin lien ket** roi
de OpenBabel suy lai bac lien ket + thom tinh **tu toa do**. O day do la mat mat **khong can thiet**:
ta **da biet** lien ket trong tung fragment — chinh SMILES cua no mang thong tin do. Chi con phai quyet
dinh fragment noi voi fragment o dau, ma `FragmentOnBonds` dat nguyen tu dummy **tai dung vi tri** cua
nguyen tu o dau kia lien ket bi cat ⇒ ghep cap bang khoang cach la **dung nguyen ban**, khong suy doan.
Them nua `step0:206` chi cat lien ket **don** ⇒ moi lien ket lien-fragment can noi lai deu la lien ket don.
Lay bac lien ket tu SMILES **chinh xac hon** suy lai tu hinh hoc, nhat la voi vong thom va nhom tich dien.
Va tranh duoc viec dua ma nguon GPL-2.0 vao du an.

**Mat xich kho duy nhat — tuong ung nguyen tu:** `local_coords` luu theo thu tu **canonical rank** cua
step0, con `Chem.MolFromSmiles` tra ve thu tu khac. Khop bang substructure match voi **bac lien ket tong
quat** (`AdjustQueryParameters.makeBondsGeneric`), do thi do dung tu ban kinh cong hoa tri
(Cordero et al. 2008). Fragment **doi xung** co nhieu khop, nhung chung la anh cua nhau qua phep tu dang
cau ⇒ cho **cung mot phan tu**, nen lay khop nao cung duoc.

**Chi so hoa hoc cung bo phu thuoc:** SA dung `sascorer` **di kem RDKit** (`RDConfig.RDContribDir/SA_Score`)
— ban chinh thuc cua Ertl & Schuffenhauer (2009), **cung nguon** ma Apo2Mol chep lai. QED/logP/Lipinski
dung rdkit truc tiep; 5 quy tac Lipinski giu **dung dinh nghia** cua `scoring_func.py:obey_lipinski`.
**Con lai phu thuoc Apo2Mol:** chi buoc **Vina** (`docking_vina.py` can `meeko`, `vina`, `AutoDockTools`).
Se xu ly rieng o buoc danh gia; `openbabel` **da co** trong env `samg`.

**Chan doan moi cua buoc lap rap** (bao cao kem, chua dat nguong): so lien ket lien-fragment, so dummy
**mo coi** (khong ghep duoc trong 1,5 A), va ty le phan tu **lien thong** (1 manh). Ty le lien thong thap
la dau hieu pose sai lam cac diem gan khong gap nhau — day la thong tin ve **chat luong pose**, khong
phai loi lap rap.

---

## 41. A2b (server, 2026-10-29, HEAD `8226ad8`): duong lap rap DAT; T3 vo hieu; do phu test 58%
Log: `logs/m21_ceiling_test.log`, `logs/assembly_ceiling_test.json`. 255 complex test (sau `frame_stable`).

### 41.1. T1/T2 DAT — bo lap rap cua du an tai tao DUNG phan tu goc
```
reference  255/255 = 1,0000   QED 0,6064  SA 2,9123  logP 1,9436  Lipinski 4,6941
gt_exact   249/255 = 0,9765   QED 0,6106  SA 2,8726  logP 1,9665  Lipinski 4,6908
```
T1 `0,9765 >= 0,90` **DAT**. T2 `|ΔQED| = 0,0042 <= 0,05` **DAT**.
Chan doan lap rap: **dummy mo coi trung binh 0,00**; **phan tu lien thong 249/249 = 1,0000**.
⇒ Phep ghep cap dummy theo khoang cach **khong truot lan nao**: moi diem gan deu tim dung doi tac,
khong manh nao roi ra. Cung voi 39.2 (tai dung RMSD = 0,0000), chuoi
`7D + local_coords → toa do → do thi phan tu` da duoc xac nhan **hai dau**.

### 41.2. **T3 VO HIEU — loi dac ta cua tro ly, vi pham R5**
Tren tap chung 144 complex, ca ba hang cho **so y het nhau toi bon chu so**:
`QED 0,5603 · SA 2,7863 · logP 1,4523 · Lipinski 4,5694`.
**Khong phai trung hop:** QED, SA, logP (Crippen), Lipinski deu la **mo ta 2D** — chung chi phu thuoc
**do thi** phan tu (nguyen to, lien ket, bac, dien tich), **khong** phu thuoc toa do 3D. `gt_exact` va
`gt_template` khac nhau **chi o conformer**, con do thi thi giong het (lap rap thanh cong, cung lien ket)
⇒ bon chi so do **buoc phai** bang nhau.
Theo **R5** ("neu moi thu DUNG thi dai luong nay bang bao nhieu, va no CO THE dat gia tri do khong?"):
neu moi thu dung thi T3 = 0 — tieu chi **khong co kha nang phat hien bat cu dieu gi**. Day la lan sai
tieu chi thu muoi cua tro ly, va dung loai loi R4/R6 da ghi: **do sai dai luong**.
**Sua:** gia cua xap xi mot-conformer la dai luong **3D**, nen do bang **RMSD tuyet doi** giua hai ban
(cung thu tu nguyen tu — cung duong dung; cung frame the gioi). Khong can chinh Kabsch: phan tu phai nam
**dung cho** trong hoc, nen do lech tuyet doi moi co nghia. Chi so 2D chi doi khi **do thi** doi — tuc
khi lap rap that bai — nen chung van huu ich o T1/T2, khong huu ich o T3.
**He qua rong hon:** moi so QED/SA/logP/Lipinski cua mo hinh ve sau cung **mu voi chat luong pose**.
Chung chi do "mo hinh chon dung fragment nao" chu **khong** do "dat dung cho khong". Chi **Vina** (va cac
phep kiem hinh hoc kieu PoseBusters) moi doc duoc phan pose. Phai ghi ro dieu nay khi bao cao ket qua.

### 41.3. Sau loi khop tuong ung o fragment TICH DIEN — da sua
`gt_exact` truot 6/255, tat ca cung mot kieu: `khong khop duoc thu tu nguyen tu giua SMILES va template`
o `*C(=O)[O-]`, `*[S+](*)C`, `*N1CCC2(CC1)Cc1ccccc1[C@H]2[NH3+]`, `*N1CCC2(CC1)Oc1ccccc1[C@H]2[NH3+]`.
**Nguyen nhan:** `GetSubstructMatch` mac dinh doi khop ca **dien tich hinh thuc**; do thi do dung tu
khoang cach co moi nguyen tu **trung hoa**, nen fragment tich dien khong bao gio khop.
**Sua:** dung do thi **TRAN** cho **ca hai ve** (chi so hieu nguyen tu + lien ket don, bo dien tich, thom
tinh, lap the), `FastFindRings` thay vi sanitize (hoa tri co the bat thuong khi bo dien tich).
Nhap nhang do doi xung (hai oxy cua carboxylate) la **vo hai**: hai khop la anh cua nhau qua tu dang cau
cua do thi tran, va hai vi tri oxy gan trung nhau ve hinh hoc.

### 41.4. Do phu tu dien tren test: 58,04% — mot GIOI HAN THAT, khong phai loi
`148/255 = 0,5804` phan tu test co **moi** fragment trong tu dien train. 107 phan tu con lai chua
fragment chi xuat hien o test (nhieu nhat: `*c1nc(Cl)nc2c1ncn2*` x20 — khung purine clo hoa).
**Doc cho dung:** mo hinh **khong bao gio sinh ra** cac fragment do (do phu TU VUNG = 100%, Muc 38.3),
nen day **khong** phai loi lap rap, cung khong phai thieu sot cua tu dien. Day la phat bieu ve **bai toan**:
tu vung fragment hoc tu train **khong voi toi 42% phan tu test**. Tran tren cua mo hinh vi the chi ton
tai tren **58% test**; phan con lai mo hinh **khong the** tai tao, du pose co hoan hao.
So sanh: valid 73,30% (Muc 38.2) vs test 58,04% — test xa train hon valid, **nhat quan** voi viec Apo2Mol
chia split **theo thoi gian** (PDF muc 5: chronological split, test la cac cau truc moi nhat).
Day la con so **phai bao cao kem** moi ket qua sinh mau, va la lap luan manh cho viec xem lai cach cat
fragment (fragment hiem/lon → it tai su dung).

---

## 42. A2b' (server, 2026-10-29, HEAD `26860fb`): duong lap rap SACH HOAN TOAN
Log: `logs/m22_ceiling_test2.log`.
```
reference   255/255 = 1,0000
gt_exact    255/255 = 1,0000   QED/SA/logP/Lipinski TRUNG KHIT reference (lech 0,0000)
gt_template 146/255 = 0,5725   (109 thieu template -- gioi han tu vung, Muc 41.4)
lap rap: dummy mo coi 0,00  |  lien thong 255/255 = 1,0000
```
**T1 = 1,0000 DAT. T2 = 0,0000 DAT.** Ban sua do thi tran (41.3) xu ly tron ven ca 6 ca fragment tich dien.
**T3 (moi, RMSD 3D tren 146 complex): median 0,1187 A · p90 0,6324 · max 1,0219.**
⇒ Gia cua xap xi mot-conformer: **median 0,12 A**, duoi xa nguong 2 A ma nganh docking coi la "pose dung"
(va duoi ca nguong 1 A chat hon o p90). Medoid (39.4) lam dung viec cua no.
**Chuoi bieu dien → toa do → do thi phan tu gio da sach ca ba khau:** nghich dao (RMSD = 0,0000),
lap rap (255/255, 0 dummy mo coi, 100% lien thong), template (median 0,12 A).
Moi sai lech ve sau **chi** co the den tu du doan cua mo hinh.

### 42.1. Hai moc de doc ket qua mo hinh (ghi cung truoc khi sinh mau)
* **Tran tren** = `gt_template [tap chung]` tren 146 complex: QED 0,5583 · SA 2,8077 · logP 1,3972 ·
  Lipinski 4,5685. Mo hinh dung **cung** template va **cung** duong lap rap, chi khac pose ⇒ **khong the
  vuot**. Nhung luu y 41.2: bon chi so nay **2D**, nen chung do "chon dung fragment nao", khong do pose.
* **San pose** = RMSD 0,12 A (median). Moi lech pose cua mo hinh phai doc **tren** nen nay.
* **Gioi han bai toan**: tu vung khong voi toi 42% phan tu test (41.4) — phai bao cao kem moi bang ket qua.

### 42.2. Loi o duong SINH MAU phat hien khi doc lai `step4` — da sua
`step4_ligand_generator.py` re nhanh theo `if target_7d is not None` de quyet dinh lay `tok_emb` tu
`target_ids` hay tu token **tu lay mau ben trong**. Luc sinh mau, `target_7d is None` ⇒ generator **tu lay
mau mot token rieng** lam `tok_emb`, trong khi token thuc su duoc giu lai la token lay mau o **ngoai** vong
(`step7` cu, dong ~455). **Hai token nay khac nhau** ⇒ 7D duoc sinh voi dieu kien **sai token**.
Dieu do pha vo dung muc dich cua P2c ("Flow dang bi hoi 'dat cai nay o dau' ma khong duoc cho biet 'cai
nay la gi'"): P2c noi `token_embedding(token dang sinh)` vao context cua geometric head, nhung luc sinh mau
thi noi nham token khac.
**Sua:** re nhanh theo `if target_ids is not None`. Luc huan luyen khong doi gi (`target_ids` luon duoc
truyen cung `target_7d`). Luc sinh mau: lay mau **toan bo chuoi token truoc**, roi mot forward **cuoi** voi
`target_ids` = chuoi do va `target_7d=None` ⇒ toan bo 7D duoc sinh dung tren token cua chinh vi tri do,
va chi ton `L+1` forward thay vi `2L`.

---

## 43. Sinh mau lan dau + chan doan pose (server, 2026-10-29/30, HEAD `f4d2d53` / `ea59aa7`)
Logs: `logs/m23_gen_smoke.log` (10 hoc x 5 mau), `logs/m24_pose_error.log` (40 hoc, K=20).

### 43.1. Sinh mau: 0/29 phan tu LIEN THONG
```
Validity (sanitize duoc) 29/50 = 0,58   |   LIEN THONG 0/29 = 0,0000
dummy mo coi trung binh  8,41            (nhan that: 0,00 -- Muc 42)
QED 0,4688 (tran 0,5583)  SA 3,8812 (tran 2,8077)  Novelty 1,0000
```
Voi ~5,9 fragment/phan tu (~10-12 diem gan), **8,41 mo coi** nghia la gan nhu **khong diem gan nao gap nhau**.
Thu sinh ra la mot **dam fragment roi rac**, khong phai phan tu. Novelty 1,0000 la he qua tam thuong cua viec
khong giong gi ca, khong phai thanh tich.
⇒ **Dinh nghia Validity cu qua long** (mot dam fragment roi van sanitize duoc). Da tach lam hai, con so that
la "sanitize VA lien thong".

### 43.2. Chan doan (teacher forcing, token THAT, pose cua mo hinh) -- 40 hoc, K=20
```
sai so TAM fragment   tung mau      median 5,976 A   p90 10,184   mean 6,325
                      trung binh K  median 3,464 A   p90  6,303   mean 3,797
      do tan cua K mau quanh trung binh    4,944 A
sai so GOC QUAY       tung mau      median 131,4 do  mean 125,7
                      trung binh K  median 121,2 do  mean 122,6
phan phoi `d`: mo hinh mean 5,167 std 2,551 | THAT (test) mean 3,383 std 1,615 | scaler (train) 5,161/2,478
```

**(C) Vuot nguong lap rap.** Nguong 1,5 A. Sai so tam **5,98 A** (tung mau), **3,46 A** (trung binh K).
⇒ 0/29 lien thong **khong phai loi moi**, ma la he qua truc tiep. Doi chieu Muc 35.3: R2(d) = 0,099,
std(d) = 2,478 ⇒ sai so du cua **rieng `d`** ~ 2,35 A, da vuot 1,5 A truoc khi tinh `theta`/`phi`.

**(B) Phan ra bias/variance** (`sqrt(3,464² + 4,944²) = 6,04` khop `5,976` do duoc):
variance **67%**, bias **33%**. Verdict tu dong cua script ("phan lon la do tan ⇒ sinh bang ky vong se cai
thien ngay") **doc chua du**: dung la variance troi, nhung **bias con lai 3,46 A van gap 2,3 lan nguong 1,5 A**.
⇒ Sinh bang ky vong la **can nhung khong du**. (Nguong `0,6 x` trong script la con so tro ly tu dat, khong co
can cu -- vi pham R16; chi dung de phan loai tho, khong duoc doc nhu ket luan.)

### 43.3. **PHEP QUAY KHONG PHAN BIET DUOC VOI NGAU NHIEN DEU** -- bang chung manh nhat tu truoc den nay
Goc giua hai phep quay **ngau nhien deu** trong SO(3) (mat do `(1−cos t)/π` tren `[0, π]`):
**mean = 126,5°, median = 132,3°**. Mo hinh do duoc: **mean 125,7°, median 131,4°**.
⇒ **Trung khop.** Dau ra quay cua mo hinh **khong mang thong tin nao**. Day la phat bieu **dinh luong, don vi
nhiem vu** cho ket luan dinh tinh da co tu A4/A5/25.4 ("phep quay khong chiu chi phoi cua hoc"), va manh hon
tat ca cac phep do `delta R2` truoc day — khong can doi chung xao tron, khong can san nhieu.

### 43.4. `d` cua mo hinh = PHAN PHOI BIEN cua train, khong phai phan phoi co dieu kien
`d` sinh ra: mean **5,167**, std **2,551**. Scaler train: **5,161 / 2,478**. **Trung khit.**
⇒ Mo hinh dang lay mau `d` tu **phan phoi bien da hoc tren train**, gan nhu khong dieu kien hoa theo hoc.
Phu hop voi R2(d) = 0,099: ~90% phuong sai khong duoc giai thich.
**Phat hien kem, moi:** `d` THAT tren test co mean **3,383** std **1,615** — **lech han** train (5,161/2,478).
Day la **dich chuyen phan phoi train→test**, nhat quan voi split **theo thoi gian** cua Apo2Mol (Muc 41.4:
do phu tu dien test 58% vs valid 73%). Mot bo du doan bien hoan hao cung se **lech he thong** tren test.
Chua do tren valid; can doi chieu truoc khi quy hoan toan cho split thoi gian.

### 43.5. Doc tong hop: ba nguyen nhan XEP CHONG, khong phai mot
1. **Phep quay: ngau nhien** (43.3). Mot minh no da du lam lap rap that bai: dat dung tam nhung xoay sai
   ~130° thi diem gan van khong gap nhau.
2. **Tinh tien: gan nhu chi la phan phoi bien** (43.4), sai so tam 3,46 A ngay ca khi bo het do tan lay mau.
3. **Lay mau tu phan phoi qua rong** (43.2): them 67% phuong sai len tren hai cai tren.
Sua (3) la viec cua buoc **sinh**; sua (1) va (2) la viec cua **kien truc/bieu dien** — khong co cach nao
sinh mau khon ngoan cuu duoc.

### 43.6. He qua nghiem trong: NGAY CA TRAN DA CHUNG MINH cung khong du de lap rap
Probe M11-a P2 (A6) dat `R2(d) = 0,2299` tren valid — day la muc trich tot nhat **da chung minh duoc**
tu dau ra encoder dong bang. Quy ra sai so:
```
R2 = 0,2299  ⇒ sai so du cua `d` = 2,478 x sqrt(1 − 0,2299) = 2,17 A
```
**Van vuot nguong lap rap 1,5 A**, va do la moi rieng `d`, chua tinh `theta`/`phi`.
⇒ **Sua C4 cho that tot, dat bang muc probe, VAN khong du de phan tu lien thong.** Khoang cach khong
phai "mo hinh kem hon probe" (M12-a: khong ket luan) ma la **ca hai deu kem hon muc nhiem vu doi hoi**.

### 43.7. Nguyen nhan GOC: bieu dien 7D KHONG CO KENH LIEN KET
Chuoi `[fragment SMILES, d, θ, φ, quaternion]` **khong ma hoa** fragment nao noi voi fragment nao.
Frag2Seq khoi phuc lien ket **bang hinh hoc** (va dinh ly surjective cua ho dung voi toa do **chinh xac**).
Khi pose co sai so, **danh tinh phan tu bi mat**, khong chi pose bi lech:
```
nhan that  (pose chinh xac)   -> 255/255 lien thong, 0 dummy mo coi   (Muc 42)
mo hinh    (sai so ~6 A)      ->   0/29  lien thong, 8,41 dummy mo coi
```
Day la **tinh chat cua bieu dien**, khong phai cua mo hinh: bieu dien doi do chinh xac pose <= ~1,5 A chi
de phan tu **ton tai**. Moi mo hinh dung bieu dien nay deu phai vuot rao do truoc khi ban toi QED/Vina.
⇒ Ba huong thoat, phai chon truoc khi huan luyen lai (Rev 21):
* **(b) Them KENH LIEN KET vao bieu dien** — moi fragment ghi ro no gan vao fragment nao, tai diem gan nao.
  Khi do do thi phan tu duoc **xac dinh boi token**, khong phu thuoc pose; pose chi con anh huong Vina,
  khong anh huong **tinh hop le**. Can sua step0 + huan luyen lai. Day la cach cac mo hinh sinh theo
  fragment thuong lam.
* **(a) Nang do chinh xac pose** — 6D rotation (Zhou et al. 2019, CVPR, arXiv:1812.07035: bieu dien quay
  trong khong gian <= 4 chieu, ke ca quaternion, **khong lien tuc** nhu ham tu SO(3) nen mang kho hoi quy;
  6D thi lien tuc) + bo `v_context.detach()` (C4). Nhung 43.6 cho thay ngay ca muc probe cung chua du ⇒
  rui ro cao, rieng no co the khong giai quyet duoc.
* **(c) Noi long lap rap** — dung pose du doan lam **rang buoc** roi de truong luc dong lien ket. Giu duoc
  bieu dien hien tai nhung lam nhoe ranh gioi "pose do mo hinh" va "pose do toi uu hoa".
**Khuyen nghi cua tro ly: (b) truoc.** No lam cho "phan tu co hop le khong" **doc lap** voi do chinh xac
pose, nen moi do luong ve sau (QED/SA/Vina) tro nen co nghia; va no re hon (a) ve rui ro. (a) van can,
nhung nen do **sau** khi (b) tach bach duoc hai nguon loi.

---

## 44. Dac tinh do thi fragment (server, 2026-10-30, HEAD `7ffcb79`): Q1 DAT, **Q2 BAC**, Q3 do duoc
Log: `logs/m25_frag_graph.log`. 1.894 phan tu train, 0 loi doc.

### 44.1. Q1: do thi fragment LUON la cay — 1.894/1.894 = 1,0000
Khop ly thuyet (`step0` chi cat lien ket **khong trong vong**). Da **do**, khong gia dinh.

### 44.2. Q2: **BAC** — thu tu canonical cua Frag2Seq KHONG dam bao cha xuat hien truoc
```
phan tu ma MOI fragment t>=1 co hang xom o vi tri nho hon:  326/1.894 = 0,1721
so VI TRI vi pham:                                        3.186/9.176 = 0,3472
```
⇒ **Hon mot phan ba so vi tri khong co cha o truoc.** Thiet ke "du doan mot chi so cha trong cac vi tri
da sinh" **khong chay duoc** voi thu tu hien tai.
**Vi sao Frag2Seq khong gap van de nay:** ho khoi phuc lien ket **bang hinh hoc**, khong bang thu tu chuoi
(Muc 43.7), nen ho chi can thu tu **canonical** (tat dinh, bat bien dong cau) — khong can
cha-truoc-con. Ta can **ca hai**. Day khong phai loi cua S3/N1 (Muc 37.6): thu tu Frag2Seq van dung cho
muc dich cua no, nhung **khong du** cho kenh lien ket.

### 44.3. Ban sua: thu tu DFS CANONICAL tren cay fragment
Goc = fragment chua nguyen tu co **canonical rank 0**; duyet con theo thu tu **rank nho nhat** cua chung.
Tat dinh va bat bien dong cau (vi `CanonicalRankAtoms` la vay), **va** cha luon xuat hien truoc **theo cau
truc**, khong phai theo xac suat.
**Nguon:** Jin, Barzilay & Jaakkola (2018), *Junction Tree Variational Autoencoder for Molecular Graph
Generation*, ICML — sinh cay con theo thu tu **DFS**, moi buoc du doan nut cha va diem gan. Day la cach
chuan cua cac mo hinh sinh phan tu theo cay/fragment.

### 44.4. Q3: kich co cac head (do duoc, R16 muc 2)
```
so dummy / fragment:  1:4.440  2:6.030  3:551  4:46  5:3     max=5  p99=3  mean=1,66
so fragment / phan tu: max=20  p99=19  mean=5,84
bac trong do thi fragment: max=5  p99=3
```
⇒ head **diem gan**: 5 lop. head **cha**: con tro tren toi da 19 vi tri truoc do.
Luu y `mean=5,84` khop `5,92` do o Muc 43 (hai cach dem khac nhau, cung co).

### 44.5. Chon kien truc: BA HEAD RIENG, khong xen token
Hai cach them kenh lien ket:
* **(i) Xen vao chuoi token** (fragment, offset-cha, dummy-self, dummy-cha): khong can head moi, nhung
  chuoi dai gap ~4 lan va **pha vo anh xa 1-1 giua vi tri va 7D** ⇒ phai sua mask cua `Geo_Loss`, tuc dong
  vao duong hinh hoc **vua moi duoc xac nhan sach** (Muc 42). Mo hinh con phai hoc ngu phap va co the sinh
  chuoi sai ngu phap.
* **(ii) Ba head rieng** `parent` (con tro), `attach_self` (5 lop), `attach_parent` (5 lop): chuoi **giu
  nguyen do dai**, 7D van 1-1 voi vi tri ⇒ **khong dong gi vao duong hinh hoc**. Luc sinh, che (mask) cac
  lua chon khong hop le ⇒ **khong the sinh ra cau truc sai**.
**Chon (ii).** Ly do quyet dinh: khong lam nhieu duong hinh hoc da xac nhan, va bao dam tinh hop le bang
che thay vi bang hoc.

---

## 45. Rev 21 S5 (server, 2026-10-30): K5/K7/K8/K9 DAT; K6 lo ra **SCALER DA LECH DU LIEU TU LAU**
Logs: `logs/m26_*`, `logs/m27_*`, `logs/m28_pkl_diff_full.log`.

### 45.1. K5 DAT sau khi chap nhan GOC MOI
`parent == -1` o `t >= 1` xuat hien **11/119.431 = 0,0092%** vi tri: ligand co **nhieu thanh phan roi
nhau** (muoi/doi ion trong SDF) ⇒ do thi fragment la **RUNG**, khong phai cay. Q1 (Muc 44.1) do tren
1.894 phan tu khong gap ca nao; tren 23.660 thi co. Hop le, da sua tieu chi. `parent[0] == -1` o 100%.

### 45.2. **K8 DAT: 399/400 = 0,9975** — KENH LIEN KET HOAT DONG
Tai dung **chi tu** `parent`/`attach_self`/`attach_parent`, **khong dung hinh hoc lien-fragment**, cho do
thi cau tao **trung khop phan tu goc**. ⇒ **Huong (b) duoc xac nhan**: do thi phan tu duoc xac dinh boi
TOKEN, doc lap voi do chinh xac pose. Day la tieu chi quyet dinh cua Rev 21.
Mot ca truot: `*O*` khong khop duoc thu tu nguyen tu (cung fragment da gay bat thuong o Muc 38.2/39.3).

### 45.3. K8 ban dau that bai vi **SO SAI DAI LUONG** — lai la loi kieu T3
Phien ban dau so `MolToSmiles` **co lap the**: 242/400 = 0,605. Lap the cua ban lap rap phu thuoc thu tu
hang xom quanh tam bat doi, ma thu tu do **doi** khi ta bo dummy va them lien ket moi ⇒ so SMILES co lap
the **khong do** dieu ta muon biet (do thi co dung khong). Doi sang **do thi cau tao** (`isomericSmiles=
False`): 0,9975. Lap the bao cao rieng: **0,605** — mot van de that, nhung rieng biet va khong chan duong.
Day la lan thu hai tro ly dat tieu chi len sai dai luong (lan truoc: T3, Muc 41.2).

### 45.4. K6: `compute_scaler.py` o Rev 20 **DA SAP**, nen K1 khi do la phep kiem RONG
```
logs/m16_scaler.log:  0%| | 0/17200
  File "step6_trainer.py", line 177, in __getitem__
    data.pocket_R = torch.tensor(pkl_data["pocket_R"], ...)
KeyError: 'pocket_R'
```
Thu tu lenh o Rev 20 A0 la `step0 → compute_scaler → check`, trong khi `pocket_R` chi duoc khoi phuc
**sau do** o A1b (Muc 38.6). ⇒ `compute_scaler` sap ngay item dau, **khong ghi file**.
⇒ `dataset/processed/scaler_7d.pt` van la ban CU, va **K1 da so file do voi mot ban sao cua chinh no**
(`/tmp/scaler_7d_OLD.pt`) ⇒ **dat mot cach vo nghia**.
**Loi thiet ke phep kiem cua tro ly:** mot phep kiem khong-doi phai xac minh **TIEN DE** cua no — rang san
pham thuc su da duoc tao lai — truoc khi tin vao ket qua so sanh. Them vao R15.

### 45.5. Trung binh 7D that qua ba phien ban du lieu — SCALER DA LECH TU LAU
Tinh **truc tiep tu pkl** (train, `frame_stable`, float64), bo qua `Dataset`/`compute_scaler`:
```
                      d         theta     phi        qw        qx        qy        qz
A1 (file scaler cu)   5,161378  1,531624  0,391346   0,200377  0,204276  0,201783  0,185659
TRUOC Rev20 (pkl)     5,161319  1,531652  0,391268   0,199063  0,202019  0,204564  0,187700
HIEN TAI    (pkl)     5,161378  1,531624  0,391346   0,198343  0,198508  0,208647  0,187691
```
* `d/theta/phi`: giong nhau o ca ba (chenh o chu so thu 5 la do ban backup thieu 1 complex).
* **Quaternion khac nhau o ca ba phien ban.**
**(i) TRUOC Rev20 → HIEN TAI: do S1 (`dummyLabels`), dung nhu du doan.** `Chem.FragmentOnBonds` khong
truyen `dummyLabels` thi dummy mang isotope = chi so nguyen tu; `CanonicalRankAtoms` **co tinh isotope**.
Bo isotope ⇒ cac dummy tro nen **tuong duong**, thu hang canonical trong fragment doi ⇒ `coords_canon` doi
⇒ `_get_local_frame` doi ⇒ **quaternion doi**. `d/theta/phi` khong doi vi chi phu thuoc **tam** fragment,
bat bien voi hoan vi. Day la he qua **da luong truoc duoc** cua S1, va K1 le ra phai bat — nhung K1 rong (45.4).
**(ii) A1 khong khop CA HAI phien ban pkl.** ⇒ file scaler tren dia da **cu hon** ca du lieu truoc Rev 20:
o mot lan chay lai `step0` **truoc do nua**, `compute_scaler` cung khong duoc chay lai. Cung mot loai loi,
som hon, va **chua tung co phep kiem nao bat**.
**Muc do anh huong:** lech ~1% o `shift` cua quaternion = **~0,005 lan mot do lech chuan** (0,002/0,46).
Rat nho cho viec hoc. Va moi run deu **tu nhat quan** (huan luyen va sinh mau dung cung mot file scaler),
nen cac so da bao cao **khong bi vo hieu**. Cai hong la **quy trinh**, khong phai ket qua.
**Bang A1 trong RESULTS_LEDGER da loi thoi** — cap nhat theo cot HIEN TAI.

### 45.6. K6 dinh nghia lai — tu so voi bang lich su sang so voi CHINH DU LIEU
Nguong cu ("khop bang A1 trong 1e-4") so voi mot **bang lich su** co the da cu ⇒ khong co kha nang phan
biet "scaler dung" voi "scaler cu". Thay bang phep kiem **TU NHAT QUAN**, co kha nang phat hien that:
**`scaler_7d.pt` phai khop trung binh/do lech chuan tinh TRUC TIEP tu pkl (train, `frame_stable`,
float64) trong sai so tuong doi 1e-4.** Khong dat ⇒ scaler cu hoac `compute_scaler` sap.
Day la **sua mot moc sai**, khong phai ha nguong (R9): moc cu doi chieu voi mot con so **khong dung voi du
lieu**; moc moi chat hon vi no khong the dat bang cach khong lam gi.

---

## Muc 46 -- Quy dao loss cua lan huan luyen Rev 20 (MAF, 25 epoch): doc lai cho dung

Nguon: CSV nguoi dung cung cap, epoch 0-18 cua lan chay `m20_train_maf`. Day la **lan chay da sinh ra
0/29 phan tu lien thong** (Muc 43), nen moi ket luan rut ra tu no phai doc duoi anh sang do.

| | epoch 0 | epoch 18 |
|---|---|---|
| Total_Loss | 15.1499 | 12.6031 |
| Token_Loss | 3.6411 | 2.3840 |
| Geo_Loss (clamped) | 7.9445 | 7.0420 |
| Geo_Loss_Unclamped | 9.7558 | 8.4456 |
| Pocket_Loss | 3.5644 | 3.1771 |
| Int_Loss | 0.0001 | 0.0001 |

### 46.1 Token_Loss 3.349 -> 2.384 **khong** phai bang chung vocab moi tot hon

Cach doc truc giac ("loss giam 0.97 nat, vay sua S1 co ich") **sai**, vi hai con so thuoc hai bo tu vung
khac nhau. Tran doan ngau nhien dong deu:

- vocab cu 19033: ln(19033) = 9.854
- vocab moi  3983: ln(3983)  = 8.290
- chenh **1.564 nat cho khong**, chi do tu vung nho di.

Muc giam thuc te chi **0.965 nat** < 1.564. Theo ty le, perplexity/|V| **tang** tu 0.00150 len 0.00272.
Ngoai ra thu tu fragment cung da doi (N1, thu tu DFS theo Jin et al. 2018), nen bai toan du doan khong
con la bai toan cu. **Ket luan: hai con so khong so sanh duoc; khong duoc tuyen bo cai thien.**
(R5 -- tieu chi phai co kha nang that bai; phep so sanh nay khong co.)

Dieu doc duoc mot cach hop le: 2.384 nat so voi tran 8.290 nat, tuc mo hinh token hoc that, con xa muc
doan mu.

### 46.2 Geo_Loss_Unclamped cho mot uoc luong **doc lap** ve tran cua kenh hinh hoc

Trong khong gian da chuan hoa, neu mo hinh chi du doan **phan bo bien** N(0,1) tren 7 chieu thi
NLL = 3.5 + 3.5*ln(2pi) = **9.9326 nat**. Quan sat o epoch 18 la **8.4456 nat**.

- khoang cach: **1.487 nat toan cuc = 0.2124 nat/chieu**
- quy ra do co lai cua do lech chuan co dieu kien: exp(-0.2124) = **0.8086**
- R^2 tuong duong: 1 - 0.8086^2 = **0.3461**

Doi chieu **tran cua probe tuyen tinh do truc tiep o Muc 43.6: R^2 = 0.2299.** Hai con so den tu hai
phep do hoan toan khac nhau (mot tu NLL cua flow trong luc huan luyen, mot tu probe tuyen tinh tren
embedding) nhung cung noi mot dieu: **dieu kien hoa theo tui chi giai thich duoc khoang mot phan ba
phuong sai tu the, va do la tran, khong phai san.** Bang chung doc lap thu hai cho ket luan Muc 43.7.

Luu y gioi han: 0.3461 la trung binh hinh hoc tren ca 7 chieu, gop chung `d/theta/phi` voi bon chieu
quaternion; no khong tach duoc chieu nao dong gop. Khong dung no thay cho phep do tu the truc tiep.

### 46.3 Ca bon loss **van dang giam** o epoch 18 -- va dieu do khong tran an

Token_Loss con giam ~0.03 nat/epoch giua epoch 17 va 18; Geo va Pocket cung chua phang. Theo le thuong
day la ly do de huan luyen lau hon 25 epoch.

**Nhung lan chay nay ket thuc voi 0/29 phan tu lien thong.** Nghia la trong suot 25 epoch, bon duong
loss deu di xuong trong khi **dai luong quyet dinh san pham cuoi -- phan tu co noi duoc hay khong --
dung yen o 0**. Day la bang chung truc tiep rang **loss huan luyen cua kien truc Rev 20 khong phai
proxy cho muc tieu cuoi**, dung voi phan tich Muc 43.7 (7D khong co kenh lien ket; sai so tu the pha
huy *danh tinh phan tu*, khong chi pha tu the).

He qua cho lan huan luyen sap toi:
1. **Khong lay "loss van giam" lam ly do keo dai epoch.** Truoc khi tang `--max-epochs`, phai co
   duong **val** va it nhat mot phep do cap phan tu (ty le lien thong tren valid).
2. Ba loss moi (`loss_parent`, `loss_attach_self`, `loss_attach_parent`) la nhung duong **co lien he
   truc tiep** toi tinh lien thong -- khac han bon duong tren. Do la ly do chung duoc theo doi rieng.

### 46.4 Int_Loss = 0.0001 co dinh

Dung nhu cau hinh: `lambda_int = 0.0`. Khong phai loi.

---

## Muc 47 -- Kiem lai hai phep kiem da "DAT": L4 va rmsd_spread

Ca `build_fragment_dict.py` lan `check_link_heads.py` deu tu tuyen bo DAT. Kiem lai thi mot cai
**dat chat hon** script tuong, con mot cai **giai thich sai** nguyen nhan.

### 47.1 L4 parent: ky vong ly thuyet bi tinh sai, mo hinh thuc ra khop gan nhu hoan hao

Script in `ly thuyet ~ ln(so lua chon) trung binh = 1.2763`, quan sat `loss_parent = 1.0153`, va
binh luan "gan". Lech 20% khong phai la "gan", nen toi tinh lai.

Vi tri `t` co **`t+1`** lua chon (cot 0 = GOC, cot `1..t` = fragment truoc do). Cross-entropy duoc
lay trung binh tren **moi** vi tri hop le, **ke ca `t=0`** -- vi tri `t=0` chi co dung mot lua chon
nen dong gop `ln(1) = 0`. Dong 128 cu co dieu kien `and t >= 1`, tuc **loai `t=0` khoi ky vong nhung
khong loai khoi CE**. Day la phep so sanh lech mau.

Voi `lens = [4, 4, 3, 5, 9, 5, 3, 7]` (40 vi tri):

| | gia tri |
|---|---|
| ky vong script in (loai `t=0`) | 1.2763 |
| **ky vong dung (ke ca `t=0`)** | **1.0210** |
| quan sat | 1.0153 |
| lech so voi ky vong dung | **0.56%** |

Nghia la head `parent` o khoi tao **khop phan bo dong deu gan nhu hoan hao** -- bang chung manh rang
mask nhan qua va muc tieu deu dung, manh hon nhieu so voi ket luan mo ho cua script.

Da sua: bo `and t >= 1`; them nguong tuyet doi **0,25 nat** cho ca ba head; va dua `ok4` vao ket luan
cuoi (truoc day L4 **khong** tham gia quyet dinh -- chi in ra cho dep, vi pham R5: mot tieu chi khong
the lam that bai thi khong phai tieu chi). Commit `512ba69`.

`attach_self = 2.1963` va `attach_parent = 2.0422` so voi `ln(8) = 2.0794`: lech +0.117 va -0.037,
deu trong pham vi sai so chuan voi 32 muc tieu hop le. Dat.

### 47.2 rmsd_spread: thong ke DUNG, loi giai thich SAI -- va toi da sai truoc do

Script liet ke `*O*` voi `rmsd = 6.250`, `n_atom = 3`, va giai thich la "hoan vi thu tu nguyen tu cua
fragment DOI XUNG".

Lap luan chan tren: fragment `*O*` co ba nguyen tu, lien ket ~1,4 A, nen sau khi tru trong tam moi
nguyen tu cach goc <= ~1,5 A. Hai bo toa do deu nam trong qua cau ban kinh 1,5 A
=> `||p_i - q_i|| <= 3 A` voi moi `i` => **RMSD <= ~3 A bat ke hoan vi nao**. Vay 6,250 A khong the
do hoan vi.

Toi da ket luan tu day rang **thong ke co loi**. Ket luan do **sai**, va bang chung bac bo nam ngay
trong chu thich dong 128 cua chinh script: *"Muc 39.3 -- co truong hop span > 12 A tren fragment 3
nguyen tu"*. Lap luan chan tren cua toi gia dinh **hoa hoc hop le**; du lieu nguon **khong** luon hop
le. Voi mot conformer hong co span 12 A thi RMSD 6,25 A hoan toan kha di.

Nen ket luan dung la: **thong ke chinh xac, cach doc cua script sai**, va con so 6,25 A la bang chung
rang **medoid dang lam dung viec no sinh ra de lam** (Muc 39.4) -- 60/3981 fragment co medoid khac
conformer dau > 0,5 A, va **ngoai lai hinh hoc trong template da chon = 0**.

Da sua loi giai thich in ra, commit `5435ed8`. Khong doi thong ke.

Bai hoc ghi lai: lap luan "bat kha thi ve hinh hoc" chi co gia tri khi du lieu dam bao hop le hoa hoc.
Tren du lieu thuc, mot gia tri "bat kha thi" la dau hieu cua **du lieu hong**, truoc khi la dau hieu
cua **code hong**.

### 47.3 Do phu tu dien tren valid: doc cho dung pham vi

- `do phu TU VUNG: 3981/3981 = 1.0000` -- **moi token mo hinh co the sinh ra deu co template.**
  Day la con so quyet dinh cho **sinh mau**: khong bao gio gap token thieu template.
- `do phu token valid = 0.9505`, `phan tu co DU moi fragment: 560/764 = 0.7330` -- hai con so nay chi
  rang buoc thi nghiem **tran xap xi template** tren ground truth, vi 26,7% phan tu valid chua
  fragment ngoai tu dien train. Khong rang buoc sinh mau.

Khong duoc dung 0.7330 nhu mot gioi han tren cua chat luong sinh.

---

## Muc 48 -- Kenh lien ket (Rev 21) HOAT DONG: ba head deu hoc, ke ca `parent`

### 48.1 Suy ra `loss_link` tu `epoch_losses.csv` ma khong can wandb

`log_losses` chi ghi 6 cot va khong co ba loss lien ket (da sua, commit `1cb042d`, co hieu luc tu lan
chay sau). Nhung o Rev 20 `Total_Loss` **bang dung** tong bon thanh phan (15.1499 so voi 15.1501), nen
o Rev 21 phan du chinh la `loss_link`:

| epoch | Total | 4 thanh phan | loss_link |
|---|---|---|---|
| khoi tao (L4) | -- | -- | 5.2538 |
| 0 | 16.9237 | 15.3534 | 1.5703 |
| 1 | 15.7426 | 14.4295 | 1.3131 |
| 4 | 14.9585 | 13.7350 | 1.2235 |
| 8 | 14.4867 | 13.3081 | 1.1786 |

Giam 70% ngay trong epoch dau. Nhung day la **tong** ba head; vi CE >= 0 va tong 1.1786 < ln(8) nen
suy ra duoc hai head attach da hoc, **khong** suy ra duoc gi ve `parent` (no co the van ket o 1.021
neu hai head kia cong lai chi 0.158).

### 48.2 Tach ba head tu checkpoint -- `parent` giam 94%

Do tren **valid** voi checkpoint giua chung (`--ckpt auto`), `check_link_heads.py`:

| head | muc dong deu | do duoc | giam | perplexity |
|---|---|---|---|---|
| **parent** | 1.0210 | **0.0639** | **93.7%** | 1.066 |
| attach_self | 2.0794 | 0.4820 | 76.8% | 1.619 |
| attach_parent | 2.0794 | 0.5867 | 71.8% | 1.798 |
| **loss_link** | 5.1798 | **1.1326** | 78.1% | -- |

**`parent` khong ket.** Perplexity 1.066 nghia la mo hinh gan nhu luon chon dung fragment cha. Day la
rui ro duy nhat con bo ngo o 48.1, nay da loai tru.

Quan trong hon: `loss_link` tren **valid** = 1.1326, thap hon ca tren **train** o epoch 8 (1.1786).
**Kenh lien ket khong overfit.** Ket hop voi K8 = 0.9975 (kenh lien ket xac dinh duoc phan tu ma khong
can hinh hoc lien-fragment), day la bang chung rang huong (b) o Muc 43.7 la dung huong.

### 48.3 CANH BAO: head token co ve overfit manh

Cung phep do do: `loss_token` tren **valid = 4.8236**, trong khi tren **train** o epoch 8 da la 2.8787
(va checkpoint nay muon hon epoch 8 nen train con thap hon nua). Khoang cach >= 1.95 nat, tuc
perplexity tren valid lon hon tren train it nhat **7 lan**.

Chua ket luan: day la **mot batch valid duy nhat**, nhieu lon. Can (a) trung binh nhieu batch va
(b) duong `val_total_loss` theo epoch -- co san trong **ten file checkpoint**
(`samg-opt-{epoch:03d}-{val_total_loss:.4f}`, step6_trainer dong 578). Neu val loss da cham day va di
len thi 25 epoch la **thua**, nguoc han voi suy doan o Muc 46.3.

## Muc 49 -- Hai loi cua chinh phep kiem

### 49.1 `--ckpt auto` tim sai thu muc

Tim trong `checkpoints/`; `ModelCheckpoint` ghi vao `saved_checkpoints_flow/` (dong 577). Da sua va
them loc bo file sua doi trong 60 giay qua, de khong nap checkpoint dang ghi do. Commit `b05a8ce`.

### 49.2 L4 bi ap dung cho mo hinh DA hoc -- ket luan "KHONG DAT" la vo nghia

L4 so loss voi muc doan dong deu de xac nhan **mask va muc tieu dung** o mo hinh **chua hoc**. Voi
`--ckpt`, lech xa muc dong deu la **dieu mong muon**, vay ma script van in "L4 KHONG DAT" va
"khong huan luyen lai" -- suyt lam dung mot lan chay 19 gio dang khoe manh.

Day la mot tieu chi **dung dau hieu, sai chieu**: cung mot con so (|lech| lon) la hong o che do nay va
la tot o che do kia. Da tach lam hai: `--ckpt` chuyen sang **L4'** (moi head phai THAP hon muc dong
deu it nhat 10%). Commit `83b183e`.

Bo sung vao R15b: mot phep kiem khong chi phai xac minh tien de cua no, ma con phai biet **no dang o
che do nao** -- neu cung mot nguong duoc dung cho hai che do co chieu nguoc nhau thi no se bao dong
gia o dung luc nguy hiem nhat.

---

## Muc 50 -- Chuan danh gia: tra lai tu bon PDF, va mot chi so KHONG co tien le

### 50.1 Bo chi so cua cac cong trinh truoc

| | Frag2Seq (4.1) | Apo2Mol (Eval) | 3DMolFormer (4.2) |
|---|---|---|---|
| Binding | Vina Score (QVina) | **Vina min** (AutoDock Vina, Eberhardt 2021) | Vina Score + Vina Dock (QVina 2) |
| High Affinity | co | co | -- |
| QED / SA | co | co | co |
| logP | -- | co | -- |
| Lipinski | co | co | -- |
| Diversity | co | -- | **co y khong bao cao** |
| Tong hop | Time | -- | Success Rate |
| So mau/hoc | 100 | -- | 100 (duy nhat) |
| mean & median | -- | **ca hai** | mean |

### 50.2 Ba quy uoc phai theo, neu khong so khong so sanh duoc

1. **SA.** Ban goc (Ertl & Schuffenhauer 2009) thuoc [1, 10], **THAP la tot**. Cac bai SBDD bao cao
   ban bien doi `SA = (10 - SA_goc)/9` thuoc [0, 1], **CAO la tot** (3DMolFormer Apx D, theo Guan et
   al. 2023b). `step7_generate_eval.py` nay tra ve **ca hai**: `SA` (chuan hoa) va `SA_raw` (goc).
   Da suyt mac bay: bang "so voi tran tren" doi chieu voi `check_assembly_ceiling.py`, file do ghi SA
   theo **ban goc**; neu khong doi sang `SA_raw` thi se tru hai thang do **nguoc chieu nhau**.
2. **Lipinski la SO QUY TAC thoa man** (0-5), khong phai bien nhi phan. Nam quy tac theo Apo2Mol
   Apx A: MW < 500 Da; HBD <= 5; HBA <= 10; logP trong **[-2, 5]**; so lien ket quay duoc <= 10.
   (Luu y: Apo2Mol con bao cao rieng rang logP "thuan loi" nam trong **[-0.4, 5.6]** theo Ghose et al.
   1999 -- day la **khoang khac** voi khoang trong quy tac Lipinski. Khong tron hai cai.)
3. **Bao cao ca mean va median** (Apo2Mol).

**Diversity:** Frag2Seq bao cao (1 - Tanimoto trung binh tung cap trong mot hoc). 3DMolFormer **co y
khong**, voi lap luan dang chu y: moi chi so da dang hien co deu dua tren **do thi 2D**, trong khi bai
toan la sinh phan tu **3D** (Apx D). Ta bao cao nhung ghi kem bat dong nay.

### 50.3 Tinh LIEN THONG khong co trong bai nao -- va vi sao

Khong bai nao trong bon bai do "ty le phan tu lien thong". Ly do khac nhau:

- **Sinh theo nguyen tu** (Pocket2Mol, GraphBP) va **khuech tan** (TargetDiff, DiffSBDD, Apo2Mol):
  lien ket duoc sinh hoac suy ra truc tiep, phan tu lien thong **theo cau truc**. Khong co gi de do.
- **Frag2Seq**: Dinh ly 3.4 khang dinh chuoi va phan tu 3D tuong ung mot-mot, tuc **gia dinh** tai
  tao duoc. Bai khong bao cao ty le that bai cua phep tai tao.

Muc 43 cho thay o mot mo hinh **da huan luyen that**, gia dinh do **sai**: 0/29 phan tu lien thong,
8.41 dummy mo coi. Dinh ly 3.4 dung cho chuoi **chinh xac**; no khong noi gi ve chuoi **du doan**.

Nen "ty le lien thong" la chi so **rieng cua du an**, khong co tien le de cite. Can cu cua no la
dac tinh du lieu va phep do 43.5-43.6, khong phai mot bai bao. **Phai ghi ro dieu nay khi viet bai**,
khong duoc trinh bay nhu mot chi so chuan.

### 50.4 S8: mot cho de tu lua minh, da tranh

Khi giai ma kenh lien ket, cam lua chon "GOC" o `t > 0` se khien moi phan tu chi co mot thanh phan,
tuc **ty le lien thong luon bang 100%**. Do la mot tieu chi **khong the that bai** -- vi pham R5, va
se bien S8 thanh mot phep do vo nghia.

`decode_links` **van cho phep GOC** o moi vi tri. Mask chi ap cho cac rang buoc **tinh dung dan**:
cha phai di truoc, va dummy duoc chi phai ton tai va chua bi chiem. Mo hinh duoc phep tra loi
"day la manh roi", va neu no tra loi nhu vay thi ta phai thay dieu do trong so lieu
(`n_goc trung binh` duoc in ra rieng).

Them `--assemble geometry` de chay lai duong lap rap cu tren **cung bo mau sinh ra**: chenh lech
giua hai che do do dung bang phan ma kenh lien ket mang lai, khong lan voi khac biet do lay mau.

---

## Muc 51 -- RUT LAI Muc 48.3; va mot bat nhat chua giai thich duoc

### 51.1 Muc 48.3 (head token overfit) -- RUT LAI HOAN TOAN

Do tren 10 batch valid (batch_size 8) voi checkpoint dang chay:

| | trung binh | **do lech chuan** |
|---|---|---|
| loss_parent | 0.1496 | 0.1178 |
| loss_attach_self | 0.6507 | 0.1464 |
| loss_attach_parent | 0.7733 | 0.1000 |
| loss_link | 1.5736 | 0.2183 |
| **loss_token** | **6.8130** | **1.3352** |
| loss_geo | 7.9488 | 0.2910 |
| loss_total | 17.3482 | 1.4779 |

`loss_token` co **sd = 1.3352** giua cac batch 8 phan tu. Hai so do mot-batch truoc day (4.8236 va
5.0254) deu nam duoi trung binh hon mot do lech chuan. **Khong ket luan nao ve khoang cach
train-valid co the rut ra tu mot batch**, va Muc 48.3 da lam dung dieu do. Rut lai.

Day la lan thu hai trong cung mot ngay toi xay lap luan tren mot phep do don le (lan truoc:
`rmsd_spread` 6,25 A, Muc 47.2). Ghi vao R15b: **truoc khi so sanh hai con so, phai biet do lech
chuan cua phep do sinh ra chung.** Mot con so khong kem sai so khong so sanh duoc voi bat cu gi.

Rieng ba head lien ket: sd nho (0.10-0.15) so voi muc giam (tu 1.021 -> 0.15 va 2.079 -> 0.65/0.77),
nen ket luan "ba head dang hoc" o Muc 48.2 **van dung**. Chi cap nhat con so: tren 10 batch,
`parent` giam 85,4% (khong phai 96,2% nhu mot batch may man cho thay).

### 51.2 Bat nhat CHUA giai thich: 17.35 so voi 14.65

`loss_total` do tren valid = **17.3482**, trong khi ten checkpoint ghi `val_total_loss=14.6493`
(epoch 13). Chenh **2,70**, lon hon mot sd (1.4779). Hai kha nang, chua phan dinh duoc:

**(a) Checkpoint nap khong day du.** `load_state_dict(..., strict=False)` cho phep bo qua khoa khong
khop. Neu mot phan trong so la ngau nhien thi moi so o 51.1 deu vo gia tri. Script co in dong
`khoa khop: N/M` nhung dong do bi `tail`/`grep` cat mat ca hai lan. **Phai xem dong do truoc khi
tin bang tren.** (R15b: phep kiem phai xac minh tien de cua no.)

**(b) Mo hinh da bat dau overfit sau epoch 13.** Bang chung gian tiep ung ho: `save_top_k=3` chi giu
ba checkpoint tot nhat, va luc 22:55 ba cai do la epoch 10 / 12 / 13. Neu bat ky epoch nao tu 14 tro
di tot hon 14.8257 thi no da thay cho epoch 10. **Khong cai nao thay** => moi epoch tu 14 den luc do
deu co val loss te hon epoch 10.

Neu (b) dung thi 25 epoch la **thua**, va checkpoint nen dung la epoch 13 chu khong phai `last`.
Dieu nay nguoc han voi suy doan o Muc 46.3 ("van dang giam, co the can nhieu hon 25 epoch") -- suy
doan do chi dua tren **train** loss, dung nhu canh bao o chinh Muc 46.3 rang train loss cua kien truc
nay khong phai proxy cho muc tieu cuoi.

Phan dinh: xem dong `khoa khop`, va xem `ls -t saved_checkpoints_flow/` co xuat hien
`samg-opt-epoch=0{14,...}` moi hay khong.

---

## Muc 52 -- Giai quyet bat nhat 51.2, va con so CHINH THUC cua ba head lien ket

### 52.1 Bat nhat da duoc giai thich het

Do tren **toan bo** valid (96 batch, da xao, checkpoint `periodic-epoch=014-v1`):

| | trung binh | sd | SEM = sd/sqrt(96) |
|---|---|---|---|
| loss_parent | **0.0835** | 0.1250 | 0.013 |
| loss_attach_self | **0.4544** | 0.1026 | 0.010 |
| loss_attach_parent | **0.6677** | 0.1371 | 0.014 |
| loss_link | 1.2056 | 0.2842 | 0.029 |
| loss_token | 3.8750 | 1.0673 | 0.109 |
| loss_geo | 6.9706 | 0.4537 | 0.046 |
| **loss_total** | **14.5576** | 1.6954 | **0.173** |

Lightning ghi `val_total_loss = 14.7169` o epoch 14 (ten checkpoint). Chenh **0.159**, nho hon mot
SEM (0.173). **Khop.** Bat nhat 2,70 o Muc 51.2 hoan toan do **10 batch dau la mot khuc lien tiep**
cua tap valid xep theo `pli_id`, khong phai mau ngau nhien. Ca hai gia thuyet (a) nap thieu trong so
va (b) overfit sau epoch 13 **deu bi loai**: `khoa khop 290/290`, va epoch 14 co val 14.7169 -- gan
nhu phang so voi epoch 13 (14.6493), khong di len.

**Ba lan trong mot ngay, cung mot dang loi, ba quy mo khac nhau:**
1. Muc 47.2 -- ket luan tu **mot gia tri** `rmsd_spread`.
2. Muc 48.3 -- ket luan tu **mot batch** `loss_token`.
3. Muc 51.2 -- ket luan tu **muoi batch dau** khong xao.

Bo sung vao R15b: **mot phep do tren tap con phai noi ro tap con do duoc chon the nao.** "N batch
dau cua loader" khong phai mau ngau nhien tru khi loader co `shuffle=True`. Mac dinh cua
`check_link_heads.py` da doi thanh xao, va `--batches 0` chay toan bo split.

### 52.2 Con so chinh thuc -- thay cho Muc 48.2 va 51.1

| head | muc dong deu | do duoc (toan valid) | giam | perplexity |
|---|---|---|---|---|
| **parent** | 1.0210 | **0.0835** | **91.8%** | 1.087 |
| attach_self | 2.0794 | 0.4544 | 78.1% | 1.575 |
| attach_parent | 2.0794 | 0.6677 | 67.9% | 1.950 |

Ket luan cua Muc 48.2 **giu nguyen va nay da vung**: ba head deu hoc, `parent` khong ket. Khac biet
so voi cac uoc luong truoc (0.0639, 0.0386) chi la nhieu chon mau; huong va do lon khong doi.

### 52.3 Head token CO khoang cach tong quat hoa, nhung nho hon con so da rut lai

Valid `loss_token = 3.8750 +/- 0.109`. Train o epoch 8 la 2.8787, ngoai suy toi epoch 14 khoang 2.6.
Khoang cach **~1.25 nat** (khong phai 1.95 nhu Muc 48.3 da noi), tuong duong perplexity tren valid
lon hon khoang **3,5 lan**. Voi SEM 0.109 thi khoang cach nay la **that**, khong phai nhieu.

Luu y ve phep so sanh: "train loss" la trung binh epoch **trong luc mo hinh dang thay doi**, khong
phai danh gia sach tren tap train. Nen 1,25 la uoc luong tho, co xu huong **phong dai** khoang cach.

Doi chieu quan trong: `loss_geo` tren valid = 6.9706, trong khi train epoch 8 = 7.2817 (ngoai suy
epoch 14 ~ 7.05). **Kenh hinh hoc tong quat hoa tot, khong co khoang cach; chi head token co.**
Day la mot quan sat co ich cho Rev 22: neu can regularize thi nham vao nhanh token, khong phai
nhanh hinh hoc.

Chua phai van de cap bach: `val_total_loss` van phang/giam nhe o epoch 14, chua co dau hieu overfit
o muc tong.

---

## Muc 53 -- Hai lan chay S8 dau tien VO HIEU: head lien ket la ngau nhien

### 53.1 Loi

`step7_generate_eval.py` chon `samg-opt-epoch=022-val_total_loss=13.2157.ckpt`, va Lightning canh
bao giua log:

```
Found keys that are in the model state dict but not in the checkpoint:
['generator.root_score', 'generator.parent_q.weight', 'generator.parent_q.bias',
 'generator.parent_k.weight', 'generator.parent_k.bias', 'generator.attach_self_head.weight',
 'generator.attach_self_head.bias', 'generator.attach_parent_head.weight',
 'generator.attach_parent_head.bias']
```

Chin tham so cua ba head lien ket **khong co trong checkpoint** -> ngau nhien. `strict=False` cho qua
im lang.

**Nguyen nhan: checkpoint do la cua lan chay Rev 20**, kien truc khong co head lien ket.

**Vi sao bo chon lai chon no:** toi vua sua `--ckpt auto` thanh "chon `val_total_loss` thap nhat".
Nhung Rev 20 **khong co so hang `loss_link`**, nen `val_total_loss` cua no thieu han ~1.2-1.4 so voi
Rev 21 o cung chat luong:

| | val_total_loss | co loss_link? |
|---|---|---|
| Rev 20, epoch 22 | 13.2157 | KHONG |
| Rev 21, epoch 13 | 14.6493 | CO |
| chenh | 1.43 | ~= loss_link do duoc (1.2056) |

Chon theo val loss thap nhat thi **luon** chon Rev 20. Bo chon "dung" theo mot nghia lai sai theo
nghia quan trong hon.

### 53.2 Day la dang loi toi da tu chi ra o Muc 46.1

Muc 46.1: khong duoc so `Token_Loss` giua hai bo tu vung khac nhau, vi tran doan ngau nhien khac nhau.
Muc 53: khong duoc so `val_total_loss` giua hai kien truc khac nhau, vi **so so hang trong tong khac
nhau**.

Cung mot nguyen tac -- **hai con so chi so sanh duoc khi dinh nghia cua chung giong nhau** -- va toi
tai pham no ngay trong doan code viet de tranh mot loi khac. Bo sung vao R15b: khi viet code chon
"cai tot nhat theo mot chi so", phai hoi chi so do co **cung dinh nghia** tren moi ung vien hay khong.

### 53.3 Da sua

1. `--ckpt auto` **loc theo kien truc truoc**, roi moi so val loss trong so cac checkpoint CO head
   lien ket. Kiem bang su ton tai cua khoa `generator.root_score` v.v., khong dua vao ten file
   (ten `samg-opt-*` cua hai lan chay khong phan biet duoc).
2. **Dung han** (`SystemExit`) neu `--assemble channel` ma checkpoint khong co head lien ket, thay vi
   im lang do mot mo hinh ngau nhien.
3. Them `--list-ckpt`: liet ke tung checkpoint kem val loss va co head lien ket hay khong.

### 53.4 Mot dieu dang gia van con lai: SAN

Hai lan chay tren deu dung head lien ket **ngau nhien**. Nhung chung khong vo ich hoan toan -- chung
do mot cai **san**:

| | channel (head NGAU NHIEN) | geometry |
|---|---|---|
| Validity (sanitize) | 1189/1275 = 0.9325 | 701/1275 = 0.5498 |
| **Lien thong** | **101/1275 = 0.0792** | **3/1275 = 0.0024** |
| so GOC trung binh | 3.19 | -- |
| dummy mo coi | -- | 6.47 |

Voi head **ngau nhien**, channel da hon geometry **33 lan** ve lien thong va gan **gap doi** ve
sanitize. Phan hon do **khong** den tu mo hinh hoc duoc gi, ma tu **masking** trong `decode_links`:
diem gan bat buoc phai ton tai va chua bi chiem. Tuc ngay ca mot bo lien ket ngau nhien nhung HOP LE
cung tot hon suy lien ket tu hinh hoc sai.

`link that bai trung binh = 0.00` xac nhan masking hoat dong: khong lan nao chi vao dummy khong ton
tai.

`so GOC trung binh = 3.19` la dau hieu cua head ngau nhien: parent gan nhu dong deu nen nhieu fragment
roi vao "goc". Voi head da hoc (perplexity parent **1.087**, Muc 52.2) con so nay phai gan 1.

**Du doan truoc khi do** (ghi lai de phep do sau co the lam that bai no, R5): voi head da huan luyen,
`so GOC trung binh` se giam ve khoang 1,0-1,5 va ty le lien thong se vuot **0,50**. Neu no van duoi
0,20 thi gia thuyet "kenh lien ket du de cuu tinh lien thong" (huong b, Muc 43.7) **sai**, va phai
tim nguyen nhan khac.
