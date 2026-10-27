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

## Trạng thái (2026-10-27, sau Mục 35)

1. Dữ liệu sạch, không rò rỉ, huấn luyện tái lập được.
2. **Tiền đề XÁC NHẬN** (A6): thông tin tư thế có sẵn trong encoder đóng băng (P2: `d ≈ 0,100`).
3. Pipeline **có** dùng hốc, cho cả tịnh tiến và một phần phép quay (`qz`: lợi ích MSE 0,030 > 0).
4. **R² của pipeline dùng từ Mục 25 KHÔNG phải R² chuẩn** (A2 rút lại, 35.3). R² chuẩn cho `d` = **0,099**.
5. "Pipeline thu lợi ích từ hốc kém probe": **KHÔNG KẾT LUẬN** (M12-a, Rev 18 b). Quan sát: kém hẳn ở t ≥ 1.
6. Tanh bị loại. C4 xác nhận cấu trúc, chưa có bằng chứng gây hại. Nghi phạm tạm dừng.
7. **R16** (2026-10-27): mọi tiêu chí phải có cơ sở học thuật — nguồn ở `WORKING_RULES.md` mục G.
8. ⚠️ **Mọi số Vina/QED/SA đã có đều VÔ NGHĨA** (Mục 37-pre.2): Step 7 đánh giá hình học của ETKDG,
   không phải tư thế mô hình dự đoán. Chưa có baseline nào đáng tin.

## VIỆC KẾ TIẾP: PLAN Rev 20 — baseline TRƯỚC, huấn luyện lại SAU

⚠️ **Mục 37-pre: `step7_evaluation.py` có 3 lỗi nặng ⇒ MỌI số Vina/QED từ trước đến nay vô nghĩa.**
Tư thế từng fragment bị vứt (ETKDG dựng hình, chỉ áp 1 tịnh tiến + 1 quaternion trung bình cho cả phân tử);
trung bình quaternion theo thành phần sai toán học; khung quy chiếu không phải frame hốc.
⇒ **Không huấn luyện lại trước khi sửa Step 7** — sẽ không đọc được là tốt lên hay xấu đi.

- **A** (Rev 20): từ điển fragment (`build_fragment_dict.py`) + viết lại đường đặt pose của Step 7 (hạng mục B7)
  + đánh giá chuẩn Apo2Mol ⇒ **baseline**. Kiểm không-đổi bắt buộc: tái dựng từ **nhãn thật** phải cho
  RMSD < 0,5 Å ở ≥ 18/20 complex, nếu không thì dừng.
- **B** (Rev 20): cluster bootstrap (131 cluster) cho A6, (b2), quan sát t1 — Field & Welsh 2007 (mục G3b).
- **C**: chốt danh sách sửa step 0–6 rồi huấn luyện lại — **chỉ sau khi có A3 và B**. Rev 21.

Phép nghịch đảo đã xác định đầy đủ (37-pre.3); mắt xích thiếu duy nhất là toạ độ cục bộ của fragment —
dựng lại được bằng script duyệt SDF, **không** phải chạy lại step0 2,5 giờ.
Còn thiếu: `readlink -f ~/SAMG`; kiểm L4 (lệnh trong Rev 20).

## Quyết định dự án (2026-10-01, người dùng đồng ý): KHÔNG viết lại từ đầu
Mọi bug đã sửa là bug **quy ước miền** (frame, thứ tự quaternion, ghép dữ liệu, mask padding, nhiễu
trong forward) — viết lại không kế thừa tri thức về quy ước nào đúng, phải trả giá lại. Tài sản đắt
nhất là nhãn đã sạch, rò rỉ đã bịt, hội tụ ổn định, và hồ sơ 33 mục — thứ **không nằm trong code**.

**Tiêu chí bỏ đã ghi cứng:** nếu phép thăm dò tối thiểu thất bại ⇒ tiền đề sai ⇒ chuyển sang kiến
trúc tương biến (viết lại **tầng encoder + biểu diễn**, không phải viết lại repo).
**Phép thăm dò đó đã chạy và ĐÃ XÁC NHẬN tiền đề.** Nên tiêu chí bỏ **không** được kích hoạt, và
kiến trúc tương biến **không** phải việc tiếp theo.
