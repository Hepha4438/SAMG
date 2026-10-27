# SAMG — ĐỌC FILE NÀY TRƯỚC

Trạng thái chốt ngày **2026-10-25** (cập nhật sau Mục 34-pre / PLAN Rev 17). File này là **điểm vào duy nhất**. Nó có quyền cao hơn
`RESEARCH_CONTEXT.md` và `plan.md`, vì hai file đó chứa nhiều mục **đã bị rút lại** mà vẫn đọc như
thể còn hiệu lực.

## Thứ tự đọc
1. **File này** — trạng thái, môi trường, việc kế tiếp.
2. **`RESULTS_LEDGER.md`** — mọi con số đã xác lập (phần A), đã rút lại (B), đã huỷ (C), đang hoãn (D).
3. **`WORKING_RULES.md`** — 16 quy tắc (R16: mọi tiêu chí phải có cơ sở học thuật, nguồn ở mục G). Đọc hết **trước khi** viết
   tiêu chí đầu tiên.
4. **`plan.md`** — chỉ đọc **PLAN Rev mới nhất** ở cuối file (hiện **Rev 18**), trừ khi cần truy nguồn.
5. **`RESEARCH_CONTEXT.md`** — ~2.800 dòng. **Không đọc tuần tự.** Chỉ tra khi cần chi tiết một mục
   cụ thể mà `RESULTS_LEDGER.md` đã trỏ tới.

## Môi trường

| | |
|---|---|
| Repo | `https://github.com/Hepha4438/SAMG.git`, nhánh **`main3`** |
| Local (macOS, máy người dùng) | `/Users/HIEU/Workspaces/Projects/SAMG` |
| Server GPU | `/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG`. Lệnh chạy dùng `~/SAMG` — quan hệ (symlink?) **CHƯA XÁC MINH**; chạy `readlink -f ~/SAMG` trên server |
| conda env (server) | `samg` |
| GPU | 3× RTX A5000 24 GB, **dùng chung với người khác** |
| HEAD | xem `git log --oneline -3` trên `main3`; M12 chạy ở `53b4778` |
| Checkpoint tham chiếu | `saved_checkpoints_flow/periodic-epoch=024.ckpt` (MAF, 25 epoch) |
| Cache probe | `dataset/cache_probe/{train,valid}` — 733,4 MB / 33,3 MB. **ĐÃ CÓ, không dựng lại** (mất 2 giờ 24 phút) |

⚠️ **Từ 2026-10-25, `RESEARCH_CONTEXT.md`, `plan.md` và ba file hướng dẫn ĐÃ được commit vào git**
(trước đó untracked, chỉ có ở local). **Bản gốc là file local**; bản trong Project chỉ là snapshot. **Mục 34-pre và PLAN Rev 17 đã ghi** (2026-10-25, ghi cứng trước M12). Kết quả M12 ghi vào
**Mục 34**; kế hoạch sau đó là **PLAN Rev 18**.

## Trạng thái (2026-10-26, sau Mục 34)

1. Nhãn 7D sạch, ghép dữ liệu đúng, rò rỉ ligand đã bịt, **không rò rỉ train/valid/test**, huấn luyện tái lập được.
2. Mô hình học **ràng buộc `‖q‖=1`** của quaternion (thang đại số, A4).
3. Hốc protein **có** được dùng. Độ lớn **phụ thuộc cách xáo**: M10 (loader không xáo) `d` ≈ 0,08 là **cận dưới**;
   xáo loader cho `d` **0,216**, và cả `qz` 0,063 / `qw` 0,050 ⇒ "phép quay không có gì" đang **XEM LẠI** (34.3).
4. **Tiền đề XÁC NHẬN** (A6): thông tin tư thế có sẵn trong encoder đóng băng. Số công bố (P2, tái lập từng chữ số):
   `d ≈ 0,100`, `theta ≈ 0,067`, `phi ≈ 0,069`.
5. **"Pipeline khai thác hốc kém hơn probe": KHÔNG KẾT LUẬN** (M12-a, 34.2). Hai probe nằm hai phía pipeline.
   Đẳng thức `B − A = Var(m)` được dữ liệu ủng hộ ⇒ B không sạch; đo **lợi ích MSE G** (Rev 18 b).
6. **Tanh bị loại** (34.1). **C4** xác nhận cấu trúc (tĩnh + khi chạy) nhưng **chưa có bằng chứng gây hại**.
   Danh sách nghi phạm (C4 / null-sink) **tạm dừng** cho tới khi (b) cho thấy pipeline thực sự thu kém.

## VIỆC KẾ TIẾP: PLAN Rev 18 — (a) và (b)
Code đã viết bởi trợ lý (WORKING_RULES F). Tiêu chí, lệnh chạy, tên log: **`plan.md` → PLAN Rev 18**. Kết quả → **Mục 35**.
- **(a)** `check_valid_adjacency.py` — không GPU: các complex liền kề trong valid có chung hốc không (giải thích M10 thấp?).
- **(b)** chạy lại hai script M12 (K=20) với `Var(m)` và `G = B − Var(m)`; so `G_pipe(d)` với `G_probe(d)`.
- Còn thiếu từ M12: output `readlink -f ~/SAMG` và `git status --short src/` trên server.

Lịch sử M12 (tiêu chí Rev 16→17, log `m12b_tanh.log` lỡ chạy local 13/10) xem `plan.md` Rev 16–17 và Mục 34-pre/34.

## Quyết định dự án (2026-10-01, người dùng đồng ý): KHÔNG viết lại từ đầu
Mọi bug đã sửa là bug **quy ước miền** (frame, thứ tự quaternion, ghép dữ liệu, mask padding, nhiễu
trong forward) — viết lại không kế thừa tri thức về quy ước nào đúng, phải trả giá lại. Tài sản đắt
nhất là nhãn đã sạch, rò rỉ đã bịt, hội tụ ổn định, và hồ sơ 33 mục — thứ **không nằm trong code**.

**Tiêu chí bỏ đã ghi cứng:** nếu phép thăm dò tối thiểu thất bại ⇒ tiền đề sai ⇒ chuyển sang kiến
trúc tương biến (viết lại **tầng encoder + biểu diễn**, không phải viết lại repo).
**Phép thăm dò đó đã chạy và ĐÃ XÁC NHẬN tiền đề.** Nên tiêu chí bỏ **không** được kích hoạt, và
kiến trúc tương biến **không** phải việc tiếp theo.
