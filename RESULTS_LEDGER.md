# SAMG — Sổ kết quả (RESULTS_LEDGER)

> Mọi con số dưới đây **đo trên server GPU**. Nguồn đầy đủ ở `RESEARCH_CONTEXT.md` mục tương ứng.
> Phần B là các kết luận **đã bị rút lại** — đọc để không đi lại.

---

## A. ĐÃ XÁC LẬP

### A1. Dữ liệu và nhãn
| | giá trị | nguồn |
|---|---|---|
| train | 23.052 entry → **17.200** sau `frame_stable` (101.791 token) | M9 |
| valid | 1.071 → **764** (4.869 token) | M9 |
| test | 478 → **255** | M9 |
| rò rỉ train/valid/test | **KHÔNG** — giao rỗng ở cả mức `pli_id` và mức **PDB** | Mục 27.1 |
| nhãn: lệch cùng PDB (G1) | 9,157° → **0,0155°** | Mục 18 |
| nhãn: đối chứng \|d\| (G3) | 0,0239 → **0,0001 Å** | Mục 18 |
| tỉ lệ bất định | 48% → **0,01%** | Mục 18 |
| canonicalization quaternion | còn nguyên; `min(q_i) ≈ −1/√2` ở cả 4 chiều | Mục 21.3 |
| phân phối `qy` | **không** thoái hoá — đồng nhất với `qw/qx/qz` ở mọi thống kê | Mục 21.1–21.2 |

**Biểu diễn 7D: `[d, theta, phi, qw, qx, qy, qz]`.**
`d, theta, phi` = **tịnh tiến** (khoảng cách + hai góc cầu chỉ hướng dịch chuyển).
`qw..qz` = **phép quay**. ⚠️ Mục 19–24 có chỗ gọi `theta/phi` là "chiều quay" — **sai**, đã sửa ở
Mục 25.4.

`scaler_7d.pt` (mean / std): `d` 5,161378/2,478297 · `theta` 1,531624/0,664945 ·
`phi` 0,391346/1,759804 · `qw` 0,200377/0,456459 · `qx` 0,204276/0,458135 ·
`qy` 0,201783/0,457270 · `qz` 0,185659/0,464332.

### A2. Công thức R² — dùng sai là lỗi đã xảy ra
```
R² = 1 − σ² · E[z²]        với z = (target − μ)/σ ;  KHÔNG phải 1 − σ²
```
`1 − σ²` chỉ đo **độ không chắc chắn mô hình tự khai báo**. Xáo trộn hốc làm `μ` sai mà hầu như
không đổi `σ`, nên `1 − σ²` **mù** với hiệu ứng cần đo (Mục 25.2).

### A3. Pipeline hiện tại (MAF, checkpoint `periodic-epoch=024.ckpt`), R² đúng trên valid
`d` 0,4195 · `theta` 0,4286 · `phi` 0,3608 · `qw` 0,5318 · `qx` 0,1745 · `qy` 0,7938 · `qz` 0,2923
(Mục 30.1. Đã **tái lập** độc lập qua hai run — lần đầu trong dự án.)

### A4. Thang đại số `‖q‖=1` — mô hình học ràng buộc chuẩn, không học hướng từ hốc
Thứ tự tự hồi quy thật của ARN, phục hồi từ **mask** trong `state_dict`:
`qx(0) → d(1) → phi(2) → theta(3) → qz(4) → qw(5) → qy(6)`; permutation `[4,0,2,1,6,3,5]` (M3).

Nguồn: **M5, `split=train`** (Mục 24.1–24.2). **Không** cùng mẫu với A3 (valid) — `qy` 86,0% ở đây vs
0,7938 ở A3 là hai mẫu khác nhau, không so trực tiếp (R2).

| chiều | số quaternion đi trước | R² đúng (train, M5) |
|---|---|---|
| `qx` | 0 | 3,5% |
| `qz` | 1 | 26,0% |
| `qw` | 2 | 44,4% |
| `qy` | 3 | **86,0%** |

Đơn điệu tăng theo lượng ràng buộc đại số sẵn có (Mục 24.2, 25.6). **Không** phải hiệu ứng "vị trí
càng sau càng học nhiều" — `d` ở vị trí 1 có R² cao nhất trong `d/phi/theta` (Mục 24.3).
Hai dấu vân tay độc lập cho cơ chế dấu của `qy`: nhóm `|qy|` trội có `log σ` thấp hơn rõ (83% dưới
−1,0 so với 3,4%, Mục 23.2), và `E[z²](qy) = 0,675` trong khi 6 chiều kia 0,94–1,01 (Mục 24.5).

### A5. Hốc protein CÓ được dùng — chỉ cho tịnh tiến (xáo trộn lúc suy luận, pipeline)
`Δ R²` = `R²(full) − R²(shuffled)` trên valid; **sàn nhiễu đo được = 0,0008–0,0011**:

| chiều | Δ R² | bội số sàn nhiễu |
|---|---|---|
| **`d`** | **0,0824** | ~100× |
| **`phi`** | **0,0208** | ~20× |
| `qw` | 0,0110 | ~10× |
| `qz` | 0,0098 | ~9× |
| `qy` | 0,0026 | ~3× |
| `theta` | 0,0015 | ~1,5× (không tách được) |
| `qx` | −0,0007 | — |

Đối chứng **cùng kích cỡ hốc** giữ **83%** hiệu ứng của `d` (0,0681/0,0824) ⇒ mô hình đọc thông
tin hốc **đặc thù**, không chỉ "hốc to/nhỏ" (Mục 29.3, 29.5, 30.2).
Sàn nhiễu không về 0 vì **`atomicAdd` của PyG scatter không tất định**, không phải lỗi logic (29.2).

⚠️ **Tiêu chí M7-a (`d > 0,10` VÀ `theta > 0,015` VÀ `phi > 0,015`) KHÔNG ĐẠT.** Kết quả chính
thức của M7-a là **không kết luận**. Không được hạ ngưỡng (Mục 29.4).

### A6. TIỀN ĐỀ ĐƯỢC XÁC NHẬN — thông tin tư thế có sẵn trong encoder đóng băng
Phép thăm dò tối thiểu (M11-a): encoder **đóng băng** + `h_target` cache + head mới, MSE thuần,
không autoregressive, không flow, không đưa pose fragment trước vào input.
Sàn tuyệt đối `tokenmean` (trung bình 7D theo `frag_id`): `d` 0,0670 · `theta` 0,0811 · `phi` 0,1022.

| | Δ R²(`d`) | Δ R²(`theta`) | Δ R²(`phi`) | đạt |
|---|---|---|---|---|
| **P1** ridge + target encoding OOF | 0,1040 | 0,2050 | 0,0488 | 2/3 |
| **P2** MLP + early stopping | **0,0999** | **0,0674** | **0,0690** | 3/3 |

**Hai giao thức độc lập cùng kết luận ⇒ xác nhận** (Mục 33.1).
**Con số công bố lấy từ P2** (baseline mạnh hơn nên bảo thủ hơn): `d ≈ 0,100`, `theta ≈ 0,067`,
`phi ≈ 0,069`. `d` vững nhất: 0,1040 và 0,0999 gần như trùng (Mục 33.3).
**`theta` và `phi` đều phụ thuộc giao thức** — không trình bày ngang hàng với `d`. `theta` lệch nặng hơn:
0,2050 (P1) vs 0,0674 (P2), ~3×; `phi` 0,0488 vs 0,0690 (Mục 33.4, 34-pre.5).
`Δ R²(qz) = −0,0049` < 0,05 ⇒ kết luận "phép quay không chịu chi phối của hốc" **giữ nguyên** (33.2).

### A7. Hạ tầng đã sửa
- **Seed**: `pl.seed_everything(1234)`, `deterministic=True` (commit `7ce0f06`). Trước đó **không có
  seed nào** → permutation ARN ngẫu nhiên mỗi process (Mục 20.3).
- **`build_h_target`** đã tách khỏi `forward` (commit `cd604cd`), kiểm không-đổi đạt (lệch ≤ 0,0012).
- **Cache probe**: `dataset/cache_probe/{train,valid}` — 733,4 MB / 33,3 MB, 17.200 / 764 complex.
  **Dùng lại, không build lại** (mất 2 giờ 24 phút).
- **`step7_evaluation.py`**: truyền `config=` tường minh vào `load_from_checkpoint` là cách khôi
  phục đúng `geo_head` (Mục 24.6).

### A7b. Kết luận cấu trúc C4 (đọc code tĩnh, HEAD `eb32820`, Mục 34-pre.2)
Mọi đường `h_target → geometric_head` đi qua `v_context.detach()` (`step4:99`). Gradient `Geo_Loss` dừng
tại `geo_in_proj`/`geometric_head` ⇒ **attention hub chỉ được huấn luyện bởi token loss**; encoder bởi
token + pocket loss. Input của `ar_decoder` chỉ có danh tính token, **không** có pose fragment trước.
Đây là kết luận **cấu trúc**, chưa nói gì về hiệu năng.

### A8. Bẫy kỹ thuật đã biết — đọc trước khi viết script đo
1. **Nhiễu dequant vẽ lại mỗi forward** (`step4_ligand_generator.py:123`, `DEQUANT_SIGMA=0.1`) và
   `target_scaled` là **input của ARN**. So sánh ghép đôi **phải** khoá seed trước mỗi forward:
   `torch.manual_seed(1234+batch_idx)` + `torch.cuda.manual_seed_all(...)` **bên trong** hàm chạy
   nhánh, ngay trước forward (Mục 27.2, 28.1).
2. **`arn.permutation` KHÔNG nằm trong `state_dict`** (là attribute thường). Khi load checkpoint,
   con số in ra là của lần dựng mới. Phải đọc **mask** của `MaskedLinear` (buffer) — xem
   `src/recover_arn_permutation.py`, đã kiểm 5/5 bằng mask tổng hợp (Mục 25.1).
3. **Pyro `create_mask` xếp cột CONTEXT TRƯỚC, input là D cột CUỐI.** Lấy `[:, :7]` là sai.
4. **Padding chiếm 51,45% số hàng.** `_ls_sum`/`_pin_sum` trong `step2_mdn_module.py` **chưa** mask
   padding, trong khi `Geo_Loss_Unclamped` (`step5:44-45`) thì có ⇒ `flow_scale_trace.csv` sai
   **17%–88%** tuỳ chiều. **Chưa sửa** (P3a-3).
5. `measure_pocket_shuffle.py` còn in verdict gây nhầm vì vế `max|Δ| < 1e-4` đã bị loại (Mục 30.3).

---

## B. ĐÃ RÚT LẠI — không đi lại

| kết luận | tại sao sai | mục |
|---|---|---|
| "Vanishing gradient do log-clamp" | hết sau khi chuẩn hoá + `DEQUANT_SIGMA=0.1`; `pinfrac ≈ 0` | 17–18 |
| "Hốc mang thông tin về hướng xoay" (19.4) | dựa trên ánh xạ tên cột sai (`qx` ↔ `qy`) | 22.4 |
| "Lượng học được phụ thuộc VỊ TRÍ tự hồi quy" (19.3) | sai **ngược chiều**; `d` ở vị trí 1 cao nhất | 24.3 |
| "5/7 chiều đứng ở marginal" (20.4) | dùng `logscale` bị nhiễm padding 17–88% | 24.1 |
| "Ảnh hưởng padding là nhỏ" (20.6) | padding tụt **chọn lọc** theo thứ tự AR, không đều | 24.4 |
| "Không đủ suy ra ⇒ không học gì" (22.2) | đọc nhị phân; `‖q‖=1` với tiền tố thiếu vẫn cho **chặn** | 24.2 |
| "M6/M7 bị nhiễm nhiễu dequant" (27.3) | vòng đo chính nằm **trong** vùng zero-hoá; phép kiểm thì không | 28.1 |
| "Rò rỉ train/valid" | giao rỗng ở cả `pli_id` và PDB | 27.1 |
| "`qy` thoái hoá trong dữ liệu" | đồng nhất với `qw/qx/qz` ở mọi thống kê | 21.1 |
| "Pipeline đang làm mất tín hiệu" | **chưa có căn cứ** — đang so hai estimator khác nhau. ⚠️ Cơ chế mà 33.5 viện dẫn cũng sai: cùng một mô hình thì Δ inference-shuffle ≥ Δ train-ablation (34-pre.3, **chờ duyệt**) | **33.5**, 34-pre.3 |

## C. ĐÃ HUỶ — không hồi sinh
- **G5** (3 nhánh huấn luyện, ~15–22 giờ): M6/M7 trả lời bằng forward-only. Huỷ từ Rev 7.
- **M11-b** (end-to-end, ~30 giờ): thông tin đã chứng minh có sẵn trong encoder đóng băng. Rev 15.
- **M10 `--split train`** và việc giải nghịch lý valid>train: giá trị thông tin thấp. Rev 11.
- **`torch.use_deterministic_algorithms(True)`**: sàn nhiễu 0,0011 đã đủ nhỏ so với 0,081.
- **Guard "R²(d) < 0,30 thì dừng"**: sai nguyên tắc — pipeline có context tự hồi quy, probe không.

## D. ĐANG HOÃN — có thể quay lại nhưng KHÔNG phải bước tiếp theo
`apo→holo` · `anti-target` · **6D continuous rotation** (lý do còn đúng nhưng chỉ ảnh hưởng **sinh
mẫu**: double cover `q ~ −q` + ranh giới canonicalization ở `|q_max| ≈ 1/√2`) · so sánh
`DiagonalGaussianHead` vs MAF · **P3a-1** (cố định `permutation=torch.arange`) · **P3a-3** (mask
padding cho `_ls_sum`/`_pin_sum`) · `tau` tuning (hiện 0,30 Å) · **N1** (token sắp theo alphabet
SMILES ⇒ phân rã tự hồi quy không có nghĩa hoá học) · **N2** (`updated_protein_pos` được supervise
rồi vứt) · **C4** (bỏ `v_context.detach()`) · **B7** (viết lại Step 7).
