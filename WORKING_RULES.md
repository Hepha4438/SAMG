# SAMG — Nguyên tắc làm việc (WORKING_RULES)

> 16 quy tắc này (R1–R16, thêm R10b) rút ra từ **9 lần đặt tiêu chí/đặc tả đo sai** trong dự án, ghi ở
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

**R16 — Mọi tiêu chí, định nghĩa, ngưỡng và quyết định phải có cơ sở học thuật** (người dùng quy định
2026-10-27). Chấp nhận theo thứ tự ưu tiên:
1. **Định nghĩa/quy ước đã dùng trong ngành**, kèm cite: bài báo (PDF trong Project: Apo2Mol, Frag2Seq,
   3DMolFormer, báo cáo SBDD; hoặc bài đã xuất bản có link), hoặc **code của công trình trước** lưu ở local
   (`Apo2Mol/`, `nummolformer/`, `posebusters/`) — ghi rõ file và dòng.
2. **Đặc tính đo được của chính dữ liệu** — ghi con số, log nguồn, commit.
3. Nếu không có (1) hoặc (2): ghi rõ **"quy ước của dự án, chưa có cơ sở ngành"** kèm lập luận R5, và coi
   đó là điểm yếu cần thay khi tìm được nguồn.
Ví dụ vi phạm đã xảy ra: Rev 18 (a) định nghĩa "cùng hốc" = cùng PDB & tâm < 2 Å — không có nguồn nào
dùng định nghĩa này; ngành dùng độ trùng chuỗi 30% (MMseqs2) hoặc TM-score hốc (xem G4).

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

⚠️ **`~/SAMG` KHÔNG tồn tại** (xác minh 2026-10-29). Đường dẫn thật là
`/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG`. Luôn đặt biến ở đầu phiên:

```bash
export SAMG=/vol/grid-solar/sgeusers/longnd/molecular_rl/SAMG
cd $SAMG && git pull && git log --oneline -3 && conda activate samg \
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

## G. Cơ sở học thuật đã kiểm chứng (R16) — cập nhật khi thêm nguồn

**G1. Hai cách đo tầm quan trọng của một đầu vào (hốc protein)**
- *Xáo trộn lúc suy luận* (cột B, M10, M12) = **permutation importance / model reliance**: Breiman (2001),
  *Random Forests*, Machine Learning 45; Fisher, Rudin & Dominici (2019), *All Models are Wrong, but Many are
  Useful*, JMLR 20(177) — https://www.jmlr.org/papers/v20/18-760.html
- *Huấn luyện có/không có đầu vào* (cột A, M11) = **LOCO (leave-one-covariate-out)**: Lei et al. (2018),
  *Distribution-Free Predictive Inference for Regression*, JASA 113(523) — https://arxiv.org/abs/1604.04173
- **Hai cách này khác nhau một cách hệ thống**: Hooker, Mentch & Zhou, *Unrestricted Permutation Forces
  Extrapolation* (arXiv:1905.03151; Statistics and Computing 2021). Định lý 1: với mô hình tuyến tính,
  permutation importance kỳ vọng = 2·β̂²·Σ(x − x̄)², tức **gấp đôi** LOCO khi biến độc lập. Đẳng thức
  `B − A = Var(m)` ở Mục 34-pre.3 là dạng tổng quát của kết quả này. — https://arxiv.org/abs/1905.03151

**G2. Probe và đối chứng**
- Linear probe: Alain & Bengio (2016), *Understanding intermediate layers using linear classifier probes*.
- Đối chứng để biết probe đọc biểu diễn hay tự học thuộc: Hewitt & Liang (2019), *Designing and Interpreting
  Probes with Control Tasks*, EMNLP — https://arxiv.org/abs/1909.03368 (nhánh `blind`/`tokenmean` cùng tinh thần).

**G3. R²**
- Hệ số xác định chuẩn: `R² = 1 − SS_res / SS_tot` (MSE-based). Công thức `1 − σ̄²·E[z²]` của dự án (A2)
  **không phải định nghĩa chuẩn**; nó chỉ trùng R² khi σ không đổi theo token (dữ liệu: log `m13b_pipe.log`, chờ ghi
  Mục 35; `R2_mse(d)` 0,099 vs 0,41).

**G3b. Bootstrap khi dữ liệu phân cụm**
- Field & Welsh (2007), *Bootstrapping clustered data*, JRSS-B 69(3) 369–390: phải lấy mẫu lại **toàn bộ cụm**;
  lấy mẫu lại quan sát lẻ cho khoảng tin cậy quá hẹp. Cụm ở SAMG = cluster MMseqs2 30% (Mục 36.1: valid có 131).

**G3c. Trung bình quaternion**
- Markley, Cheng, Crassidis & Oshman (2007), *Averaging Quaternions*, J. Guidance Control Dynamics 30(4):
  trung bình đúng = vector riêng ứng với trị riêng lớn nhất của `Σ qᵢqᵢᵀ`. Trung bình số học theo thành phần
  (step7 cũ, Mục 37-pre.2 L2) sai, và bị triệt tiêu do double cover `q ~ −q`.

**G4. Định nghĩa hốc và độ tương đồng protein trong ngành SBDD**
- Hốc = mọi residue trong **10 Å** quanh ligand; split theo thời gian của PLINDER; test pocket có
  **TM-score < 0,35** với mọi train pocket (Zhang & Skolnick 2004): Apo2Mol (PDF trong Project, mục 5 và Apx. A).
- Split theo **30% độ trùng chuỗi bằng MMseqs2** (CrossDocked2020): Frag2Seq (PDF, phụ lục dữ liệu),
  3DMolFormer (PDF, "< 30%" độ tương đồng chuỗi).

**G5. Đánh giá chất lượng phân tử sinh ra (chuẩn SBDD)**
- Sáu chỉ số của Apo2Mol (PDF, "Evaluation Metrics"): Vina min, QED (Bickerton 2012), SA (Ertl & Schuffenhauer
  2009), logP, Lipinski, High Affinity; báo cả mean và median. Phụ lục C.1 thêm Validity, Novelty
  (Tanimoto ≤ 0,4, Morgan fingerprint).
- Thực thi (code `Apo2Mol/eval_split.py`): Vina `mode='minimize'`, `--exhaustiveness 128` (dòng 55, 204);
  `--num_samples 5` mỗi hốc (`sample_split.py:266`). Tiện ích dùng lại: `utils/evaluation/scoring_func.py`,
  `docking_vina.py`, `similarity.py`.

**G6. Tái dựng phân tử 3D từ biểu diễn fragment + pose (không cần ETKDG)**
- Frag2Seq mục 3.3.1 + phụ lục (chứng minh surjective): lưu từ điển {canonical SMILES fragment → atom types +
  toạ độ cục bộ trong frame fragment}; từ chuỗi `[d, θ, φ, quaternion]` tái dựng **duy nhất** phân tử 3D, sai khác
  một phép biến đổi SE(3). Phép nghịch đảo cụ thể cho SAMG: Mục 37-pre.3.

