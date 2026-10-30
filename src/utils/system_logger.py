import os
import csv
import time
from collections import defaultdict
import pytorch_lightning as pl

class SAMGLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.loss_file = os.path.join(self.log_dir, "epoch_losses.csv")
        self.hw_file = os.path.join(self.log_dir, "hardware_profiler.csv")
        self.err_file = os.path.join(self.log_dir, "error_reports.txt")
        self.flow_scale_file = os.path.join(self.log_dir, "flow_scale_trace.csv")
        self.error_buffer = defaultdict(int)

        # Muc 48: ba loss cua kenh lien ket PHAI co trong CSV, neu khong thi khong ve duoc
        # duong cua chung theo epoch -- ma do chinh la dai luong quyet dinh cua Rev 21.
        self.LOSS_HEADER = ["Epoch", "Total_Loss", "Token_Loss", "Geo_Loss", "Pocket_Loss",
                            "Int_Loss", "Geo_Loss_Unclamped",
                            "Parent_Loss", "AttachSelf_Loss", "AttachParent_Loss"]
        # Khởi tạo Header cho CSV nếu file chưa tồn tại, hoặc XOAY file cu neu header khac
        # (them cot ma giu nguyen file cu se lam moi dong sau lech cot so voi header).
        self._init_csv(self.loss_file, self.LOSS_HEADER)

        if not os.path.exists(self.hw_file):
            with open(self.hw_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Data_Load_Time_sec", "Model_Compute_Time_sec"])

        # P1-1: thứ tự dim cố định d, theta, phi, qx, qy, qz, qw
        if not os.path.exists(self.flow_scale_file):
            with open(self.flow_scale_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                dim_names = ["d", "theta", "phi", "qx", "qy", "qz", "qw"]
                header = (["Epoch"]
                          + [f"logscale_mean_{d}" for d in dim_names]
                          + [f"pinfrac_{d}" for d in dim_names]
                          + ["nll_median", "nll_p99", "lr"])
                writer.writerow(header)

    @staticmethod
    def _init_csv(path, header):
        """Tao file voi header; neu da ton tai voi header KHAC thi doi ten file cu."""
        if os.path.exists(path):
            try:
                with open(path, newline='', encoding='utf-8') as f:
                    old = next(csv.reader(f), [])
            except Exception:
                old = []
            if old == header:
                return
            os.rename(path, path + "." + time.strftime("%Y%m%d_%H%M%S") + ".bak")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(header)

    def log_losses(self, epoch, loss_dict):
        keys = ["loss_total", "loss_token", "loss_geo", "loss_pocket", "loss_int",
                "loss_geo_unclamped", "loss_parent", "loss_attach_self", "loss_attach_parent"]
        with open(self.loss_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(
                [epoch] + [f"{float(loss_dict.get(k, 0)):.4f}" for k in keys])

    def log_flow_scale_stats(self, epoch, stats):
        dim_names = ["d", "theta", "phi", "qx", "qy", "qz", "qw"]
        if stats is None:
            stats = {}
        with open(self.flow_scale_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [epoch]
            row += [stats.get(f"logscale_mean_{d}", "") for d in dim_names]
            row += [stats.get(f"pinfrac_{d}", "") for d in dim_names]
            row += [stats.get("nll_median", ""), stats.get("nll_p99", ""), stats.get("lr", "")]
            writer.writerow(row)

    def log_hardware(self, epoch, cpu_time, compute_time):
        with open(self.hw_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{cpu_time:.2f}", f"{compute_time:.2f}"])

    def log_error(self, category, msg):
        # Cắt ngắn thông báo lỗi (lấy dòng đầu tiên) để gom nhóm các lỗi giống nhau
        clean_msg = str(msg).split('\n')[0][:100]
        key = f"[{category}] {clean_msg}"
        self.error_buffer[key] += 1

    def flush_errors(self, phase="Unknown"):
        if not self.error_buffer: return
        with open(self.err_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n[BÁO CÁO LỖI] GIAI ĐOẠN: {phase}\n{'='*60}\n")
            sorted_errors = sorted(self.error_buffer.items(), key=lambda x: x[1], reverse=True)
            for err_msg, count in sorted_errors:
                f.write(f"❌ Xảy ra {count:5d} lần | {err_msg}\n")
        self.error_buffer.clear()

# --- PLUGIN DÀNH RIÊNG CHO PYTORCH LIGHTNING (DÙNG CHO STEP 6) ---
class SAMGLoggingCallback(pl.Callback):
    def __init__(self, log_dir="logs"):
        super().__init__()
        self.sys_logger = SAMGLogger(log_dir)
        self.epoch_cpu_time = 0
        self.epoch_compute_time = 0
        self.last_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_cpu_time = 0
        self.epoch_compute_time = 0
        self.last_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.time()
        # Thời gian từ cuối batch trước đến đầu batch này chính là I/O CPU Data Loading
        self.epoch_cpu_time += (now - self.last_time)
        self.last_time = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        # Thời gian trong nội bộ batch chính là GPU Computation
        self.epoch_compute_time += (now - self.last_time)
        self.last_time = now

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        
        # Các metric là tensor (epoch-mean do Lightning tự reduce), gọi .item() tại đây
        def _to_scalar(v):
            return v.item() if hasattr(v, 'item') else v

        loss_dict = {
            'loss_total': _to_scalar(metrics.get('loss_total', 0)),
            'loss_token': _to_scalar(metrics.get('loss_token', 0)),
            'loss_geo': _to_scalar(metrics.get('loss_geo', 0)),
            'loss_pocket': _to_scalar(metrics.get('loss_pocket', 0)),
            'loss_int': _to_scalar(metrics.get('loss_int', 0)),
            'loss_geo_unclamped': _to_scalar(metrics.get('loss_geo_unclamped', 0)),
            # Muc 48: ba head lien ket cua Rev 21. training_step day toan bo loss_dict vao
            # callback_metrics nen cac khoa nay co san; thieu thi mac dinh 0.
            'loss_parent': _to_scalar(metrics.get('loss_parent', 0)),
            'loss_attach_self': _to_scalar(metrics.get('loss_attach_self', 0)),
            'loss_attach_parent': _to_scalar(metrics.get('loss_attach_parent', 0)),
        }
        
        # Ghi ra ổ cứng (Chỉ ghi 1 lần duy nhất mỗi epoch)
        self.sys_logger.log_losses(epoch, loss_dict)
        self.sys_logger.log_hardware(epoch, self.epoch_cpu_time, self.epoch_compute_time)

        # P1-1: instrumentation log_scale/pinfrac/NLL của AutoregressiveFlowLayer
        geometric_head = pl_module.generator.geometric_head
        scale_stats = geometric_head.scale_stats() or {}
        scale_stats["lr"] = trainer.optimizers[0].param_groups[0]["lr"]
        self.sys_logger.log_flow_scale_stats(epoch, scale_stats)
        geometric_head.reset_scale_stats()