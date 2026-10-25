# SAMG — ĐỌC FILE NÀY TRƯỚC

Trạng thái chốt ngày **2026-10-25** (cập nhật sau Mục 34-pre / PLAN Rev 17). File này là **điểm vào duy nhất**. Nó có quyền cao hơn
`RESEARCH_CONTEXT.md` và `plan.md`, vì hai file đó chứa nhiều mục **đã bị rút lại** mà vẫn đọc như
thể còn hiệu lực.

## Thứ tự đọc
1. **File này** — trạng thái, môi trường, việc kế tiếp.
2. **`RESULTS_LEDGER.md`** — mọi con số đã xác lập (phần A), đã rút lại (B), đã huỷ (C), đang hoãn (D).
3. **`WORKING_RULES.md`** — 15 quy tắc rút ra từ 9 lần đặt tiêu chí sai. Đọc hết **trước khi** viết
   tiêu chí đầu tiên.
4. **`plan.md`** — chỉ đọc **PLAN Rev 16** ở cuối file, trừ khi cần truy nguồn.
5. **`RESEARCH_CONTEXT.md`** — 2.582 dòng. **Không đọc tuần tự.** Chỉ tra khi cần chi tiết một mục
   cụ thể mà `RESULTS_LEDGER.md` đã trỏ tới.

## Môi trường

| | |
|---|---|
| Repo | `https://github.com/Hepha4438/SAMG.git`, nhánh **`main3`** |
| Local (macOS, máy người dùng) | `/Users/HIEU/Workspaces/Projects/SAMG` |
| Server GPU | `/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG`. Lệnh chạy dùng `~/SAMG` — quan hệ (symlink?) **CHƯA XÁC MINH**; chạy `readlink -f ~/SAMG` trên server |
| conda env (server) | `samg` |
| GPU | 3× RTX A5000 24 GB, **dùng chung với người khác** |
| HEAD | **`eb32820`**, đã push lên `origin/main3` |
| Checkpoint tham chiếu | `saved_checkpoints_flow/periodic-epoch=024.ckpt` (MAF, 25 epoch) |
| Cache probe | `dataset/cache_probe/{train,valid}` — 733,4 MB / 33,3 MB. **ĐÃ CÓ, không dựng lại** (mất 2 giờ 24 phút) |

⚠️ **`RESEARCH_CONTEXT.md` và `plan.md` KHÔNG nằm trong git** (untracked). Chúng chỉ tồn tại trên
máy local. **Mục 34-pre và PLAN Rev 17 đã ghi** (2026-10-25, ghi cứng trước M12). Kết quả M12 ghi vào
**Mục 34**; kế hoạch sau đó là **PLAN Rev 18**.

## Trạng thái, năm dòng

1. Nhãn 7D đã sạch, ghép dữ liệu đã đúng, rò rỉ ligand đã bịt, **không có rò rỉ train/valid/test**,
   huấn luyện ổn định và đã tái lập được.
2. Mô hình hiện tại học **ràng buộc `‖q‖=1`** của quaternion chứ không học hướng xoay từ hốc — thang
   đại số đơn điệu, hai dấu vân tay độc lập (`RESULTS_LEDGER` A4).
3. Hốc protein **có** được dùng, nhưng chỉ cho **tịnh tiến**: `d` thật và rõ (~100× sàn nhiễu), `phi`
   nhỏ nhưng thật, `theta` không tách được, phép quay không có gì (A5).
4. **Tiền đề cốt lõi ĐƯỢC XÁC NHẬN** (A6): thông tin tư thế **có sẵn** trong đầu ra encoder đóng
   băng — hai giao thức độc lập, trên valid chưa dùng, có sàn tuyệt đối. Con số công bố: `d ≈ 0,100`,
   `theta ≈ 0,067`, `phi ≈ 0,069`.
5. Câu **"pipeline đang làm mất tín hiệu" CHƯA có căn cứ** — đang so hai estimator khác nhau.
   **M12-a tồn tại để phân xử đúng việc này.** Cơ chế 33.5 viện dẫn cũng sai (34-pre.3, **chờ duyệt**).
6. **C4 xác nhận tĩnh** (34-pre.2): đường điều kiện hoá chỉ được huấn luyện bởi token loss
   (`v_context.detach()`, `step4:99`). C4 là nghi phạm **#1**, Tanh xuống #2.

## VIỆC KẾ TIẾP: M12 — CHƯA CHẠY. Tiêu chí và lệnh ĐÃ THAY bởi PLAN Rev 17

> ⚠️ **Phần dưới (đến hết mục M12-b) là tiêu chí Rev 16, đã bị Rev 17 thay thế.** Trước khi chạy,
> Claude Code phải làm T1–T3 của Rev 17 (K=20 derangement, tách `t=0`/`t≥1`, gắn nhãn verdict cũ).
> Tiêu chí đọc, lệnh chạy và tên log mới: **`plan.md` → PLAN Rev 17 mục 2–4**. Giữ phần dưới để truy nguồn.

### ⚠️ Cảnh báo bắt buộc trước khi đọc bất cứ gì
**`logs/m12b_tanh.log` (84 byte, 3 dòng, `device: cpu`, 13/10 11:52) — đã thấy ở máy LOCAL; trên server
CHƯA KIỂM** (Rev 17 có lệnh `ls` kiểm). Đó là một lần chạy **lỡ ở local**, **KHÔNG phải kết quả**. Xoá hoặc đổi tên nó trước khi chạy lại, và **không bao
giờ** đọc nó như số liệu. (Quy tắc R1 trong `WORKING_RULES.md`.)
Không có `m12a_probe_rev4.log` — M12-a chưa từng chạy.

### Lệnh
```bash
cd ~/SAMG && git pull && git log --oneline -3 && conda activate samg
rm -f logs/m12b_tanh.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/probe_pose_regression.py 2>&1 | tee logs/m12a_probe_rev4.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python -u src/measure_tanh_saturation.py 2>&1 | tee logs/m12b_tanh.log
```
Dòng đầu mỗi log **phải** là `device: cuda`. Nếu là `cpu` thì dừng, không đọc.

### M12-a — làm khớp estimator. Tiêu chí ghi cứng (plan Rev 16), KHÔNG được sửa
Nhánh mới **`pocket_shuf_eval`** dùng **đúng mô hình `pocket` đã huấn luyện** (P1: ma trận ridge đã
fit; P2: checkpoint tại `best_epoch`), chỉ xáo trộn `h_target` + `target_mask` (cùng hoán vị,
derangement seed 1234) **lúc đánh giá trên valid**. Không huấn luyện lại.

Hai cột, cả 7 chiều:
- **cột A** = `R²(pocket) − R²(blind)` → **train-ablation**
- **cột B** = `R²(pocket) − R²(pocket_shuf_eval)` → **inference-shuffle, CÙNG LOẠI với M10**

**Ngưỡng 0,081 CHỈ áp cho cột B, tại chiều `d`:**
- **B(d) > 0,081** ⇒ **XÁC NHẬN** pipeline khai thác kém hơn probe (cùng estimator). Khi đó mới đi
  theo hướng "tìm chỗ pipeline làm mất tín hiệu".
- **B(d) ≤ 0,081** ⇒ **BÁC** kết luận 32.3/33.5. Chênh lệch ở Mục 32–33 chỉ là hiệu ứng đổi
  estimator. Hướng đi **không** phải sửa generator, và phải thiết kế lại bước tiếp theo.

### M12-b — bão hoà `nn.Tanh()`, nghi phạm số 1. Tiêu chí ghi cứng
Cả hai head kết thúc `context_proj` bằng `nn.Tanh()`. Nếu bão hoà thì nghẽn đường từ `h_target` vào
head — và vì **cả hai head dùng chung**, phép so sánh head không thể phát hiện nó.
- `|tanh_out| > 0,99` **vượt 30%** ⇒ Tanh là nghẽn thật, phải bỏ **trước** mọi so sánh head.
- **dưới 5%** ⇒ loại giả thuyết (Mục 20.8), chuyển sang nghi phạm #2.
- trung gian ⇒ báo cáo, **không kết luận**.

### Thứ tự đọc log M12-a
1. Giao thức #1: `R²(tokenmean)` cả 7 chiều.
2. Giao thức #3: `R²(blind)` phải **≥ `R²(tokenmean)` − 0,03** ở mọi chiều tịnh tiến. Không đạt ⇒
   `blind` chưa hội tụ ⇒ **không** đọc tiêu chí chính.
3. Cột A và B cạnh nhau; áp ngưỡng 0,081 **chỉ** cho **B(d)**.
4. M12-b độc lập với M12-a.

## Nếu M12-a dương tính: ba nghi phạm
> ⚠️ Rev 17: thứ tự mới là **#1 C4, #2 Tanh, #3 null-sink**. C4 đã xác nhận tĩnh là có thật về cấu trúc
> (34-pre.2); Tanh không thể chặn gradient tới encoder vì detach chặn trước. Danh sách dưới là thứ tự cũ (Rev 16).
1. **`nn.Tanh()`** — M12-b đã đo luôn.
2. **`v_context.detach()`** (hạng mục C4, hoãn từ plan Rev 4) — nếu gradient hình học không chạy
   ngược vào attention hub và encoder thì đường điều kiện hoá **không được huấn luyện** cho nhiệm vụ
   hình học.
3. **Attention hub một lớp + null-sink** — số key = `R_max + 1`; nếu khối lượng attention dồn vào
   null-sink thì `h_target` bị bỏ qua. Kiểm bằng phân phối trọng số attention.

## Quyết định dự án (2026-10-01, người dùng đồng ý): KHÔNG viết lại từ đầu
Mọi bug đã sửa là bug **quy ước miền** (frame, thứ tự quaternion, ghép dữ liệu, mask padding, nhiễu
trong forward) — viết lại không kế thừa tri thức về quy ước nào đúng, phải trả giá lại. Tài sản đắt
nhất là nhãn đã sạch, rò rỉ đã bịt, hội tụ ổn định, và hồ sơ 33 mục — thứ **không nằm trong code**.

**Tiêu chí bỏ đã ghi cứng:** nếu phép thăm dò tối thiểu thất bại ⇒ tiền đề sai ⇒ chuyển sang kiến
trúc tương biến (viết lại **tầng encoder + biểu diễn**, không phải viết lại repo).
**Phép thăm dò đó đã chạy và ĐÃ XÁC NHẬN tiền đề.** Nên tiêu chí bỏ **không** được kích hoạt, và
kiến trúc tương biến **không** phải việc tiếp theo.
