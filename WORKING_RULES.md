# SAMG — Nguyên tắc làm việc (WORKING_RULES)

> 15 quy tắc này (R1–R15, thêm R10b) rút ra từ **9 lần đặt tiêu chí/đặc tả đo sai** trong dự án, ghi ở
> `RESEARCH_CONTEXT.md` các mục 20.11, 23.6, 26.5, 29.8, 31.4, 32.4, 33.5.
> Vi phạm chúng đã làm mất nhiều ngày và **đình chỉ oan** hai kết luận đúng (Mục 25, M7).
> Đọc hết trước khi viết tiêu chí đầu tiên.

## A. Nguồn số liệu

**R1 — Chỉ server GPU mới ra kết quả.** Máy local chỉ dùng để **đọc code** và **ghi tài liệu**.
Mọi phép đo phải là script người dùng chạy trên server. Dấu hiệu nhận biết log local: dòng
`device: cpu`, hoặc không có checkpoint. Ví dụ thật: `logs/m12b_tanh.log` (13/10) chỉ có 3 dòng
header vì chạy ở local — **không phải kết quả**.

**R2 — Mọi log phải in `git rev-parse HEAD`, `SEED`, `device` ở dòng đầu, và cấu hình loader**
(`split`, `shuffle`, `batch_size`, `n_batches`, `n_token_thật`). Hai log có số token khác nhau là
hai **mẫu** khác nhau, không so trực tiếp được (Mục 25.1).

**R3 — Mọi script đo phải in tiến độ mỗi 20 batch với `flush=True`,** và lệnh chạy luôn dùng
`python -u ... | tee`. Không có `-u` thì stdout bị buffer 4 KB qua pipe và không phân biệt được
"đang chạy" với "treo" (đã mất 30 phút vì việc này).

## B. Thiết kế tiêu chí — nơi 9 lỗi đã xảy ra

**R4 — Tiêu chí phải đo ĐÚNG đại lượng mà giả thuyết nói về.** Giả thuyết về *hình dạng phân phối*
thì tiêu chí phải là độ đo hình dạng, không phải moment bậc một. (Lỗi #2: dùng hiệu `mean` cho một
giả thuyết về tính lưỡng cực, trong khi hai nhóm tách gần hoàn toàn — AUC ≈ 0,95.)

**R5 — Trước khi ghi cứng một tiêu chí, phải trả lời được: "nếu mọi thứ ĐÚNG thì đại lượng này
bằng bao nhiêu, và nó CÓ THỂ đạt giá trị đó không?"** (Lỗi #4: yêu cầu `torch.equal` bit-identical
cho một phép biến đổi không bảo toàn bit trên GPU — tiêu chí không có khả năng PASS.)

**R6 — Tiêu chí phải ở ĐÚNG ĐƠN VỊ của đại lượng sẽ quyết định.** `max|Δ|` từng phần tử không trả
lời được câu hỏi về `Δ R²` trung bình. Mọi phép kiểm "không-đổi" phải báo cáo **sàn nhiễu trong đơn
vị của tiêu chí**. (Lỗi #7, và là lần **lặp lại** một quy tắc vừa ghi một mục trước đó.)

**R7 — Mọi phép kiểm phải chạy TRONG cùng môi trường (seed, patch, train/eval mode) với phép đo mà
nó bảo vệ.** Nếu phép đo nằm trong một context manager hay patch, phép kiểm phải nằm trong đó.
(Lỗi #6: phép kiểm no-op chạy ngoài vùng zero-hoá nhiễu → đo một đường chạy khác → đình chỉ oan
Mục 25 và M7 trong hai lượt.)

**R8 — Kế hoạch phải đặc tả GIAO THỨC HUẤN LUYỆN, không chỉ tiêu chí đọc.** "100 epoch" mà không
model selection, không early stopping, không weight decay là bảo đảm overfit. (Lỗi #8.)

**R9 — Không được hạ ngưỡng sau khi thấy số.** Nếu một tiêu chí đã ghi cứng không đạt, kết quả
chính thức là **không đạt**. Được phép **loại** một vế của tiêu chí chỉ khi nguyên tắc làm nó vô
hiệu đã được ghi vào tài liệu **trước khi** thấy dữ liệu — và phải ghi rõ đó là sửa lỗi, không
phải retrofit.

**R10 — Phân biệt hai câu hỏi và không thay thế cho nhau:**
- *"Hiệu ứng có đủ LỚN?"* → ngưỡng tuyệt đối.
- *"Hiệu ứng có KHÁC KHÔNG?"* → so với sàn nhiễu đo được.
Trả lời được câu thứ hai không cho phép tuyên bố câu thứ nhất (Mục 28.4).

**R10b — Hai estimator khác nhau không so trực tiếp.** "Xáo trộn lúc suy luận" đo *hàm đã học dùng
X bao nhiêu*; "huấn luyện có X vs không X" đo *có X thì được thêm bao nhiêu*. Cái sau lớn hơn một
cách hệ thống kể cả khi không có gì bị làm mất (Mục 33.5).

## C. Tài liệu

**R11 — Không tin bản copy; stage file THẬT, append, commit, rồi ĐỌC LẠI từ máy người dùng để xác
nhận.** Commit nhiều file trong một lệnh có thể báo thành công mà không ghi — **commit từng file
một và xác minh từng file**. Đã mất Mục 29 và Rev 12 hai lần vì việc này (Mục 29.9).

**R12 — Mọi kết luận ghi vào `RESEARCH_CONTEXT.md` phải kèm: số liệu, log nguồn, commit HEAD, và
tiêu chí đã ghi cứng trước khi chạy.** Khi rút lại một kết luận, **giữ nguyên văn bản gốc** và đánh
dấu `[RÚT LẠI — xem mục X]`. Không xoá dấu vết sai sót.

## D. Thực thi

**R13 — Ước lượng chi phí của trợ lý trong dự án này LUÔN thấp hơn thực tế** (9/9 lần). Cache ước
"vài chục phút" → 2 giờ 24; epoch ước "vài giây" → 25 giây. Đừng dùng ước lượng làm căn cứ để phán
script treo.

**R14 — Không bao giờ dựng lại một đoạn logic đã có bằng code riêng.** Dùng lại hàm/method gốc
(ví dụ `SAMGLightningModule.build_h_target`). Tính lại bằng code khác là nguồn lệch định nghĩa đã
hại dự án nhiều lần (Mục 16).

**R15 — Mọi refactor phải có phép kiểm không-đổi dùng công cụ đã có**, so với số đã công bố, trong
sai số bằng sàn nhiễu. Ví dụ đã làm: sau khi tách `build_h_target`, chạy lại
`measure_pocket_shuffle.py --split valid` và đối chiếu `R²(full)` 7 chiều, lệch tối đa 0,0012 < 0,002.

## E. Mẫu lệnh chuẩn (server)

```bash
cd ~/SAMG && git pull && git log --oneline -3 && conda activate samg \
  && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
     python -u src/<script>.py 2>&1 | tee logs/<ten>.log
```

- `CUDA_DEVICE_ORDER=PCI_BUS_ID` bắt buộc: không có nó, `CUDA_VISIBLE_DEVICES=1` **không** trỏ tới
  GPU 1 của `nvidia-smi`.
- Luôn `git log --oneline -3` sau `pull`: đã có lần server trễ **4 commit** mà không ai biết, khiến
  một run 25 epoch chạy bằng MAF trong khi tưởng là diagonal head (Mục 20.9).
- Kiểm tiến trình: `ps -o pid,etime,time,stat,cmd -C python | grep <script>`. Tỉ lệ `TIME/ELAPSED`
  ≈ 1 là đang chạy trên GPU; ≈ 10 là đang chạy CPU bằng nhiều thread (sai device).
- Kiểm GPU: `nvidia-smi` — PID của script **phải** xuất hiện trong mục Processes.

## F. Chia việc — ĐỔI từ 2026-10-25 (người dùng quyết định)

**Trợ lý làm luôn phần code và commit.** Không còn bước viết prompt cho Claude Code.
- Chỉ code **đúng** phần plan yêu cầu, không refactor, không dọn dẹp ngoài phạm vi.
- Kiểm tra **chỉ ở mức cú pháp** (`python -m py_compile`) trên máy local. **Không chạy script, không chạy
  training, không đo dữ liệu ở local** (R1 vẫn nguyên hiệu lực).
- **Commit từng việc riêng**, xác minh blob đã commit khớp file (R11). Push khi người dùng cho phép.
- Ngưỡng vẫn phải ghi cứng trong `plan.md` **trước** khi code và trước khi chạy.
- Người dùng chạy mọi thứ trên server và dán log.

*(Quy tắc cũ, giữ để truy nguồn: Claude Code đọc `plan.md` + `RESEARCH_CONTEXT.md` rồi code; prompt phải
chỉ rõ mục cần đọc, file được sửa, không chạy script, bẫy đã biết, commit từng việc; không tự đặt ngưỡng.)*
