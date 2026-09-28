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

        # Khởi tạo Header cho CSV nếu file chưa tồn tại
        if not os.path.exists(self.loss_file):
            with open(self.loss_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "Total_Loss", "Token_Loss", "Geo_Loss", "Pocket_Loss", "Int_Loss", "Geo_Loss_Unclamped"])

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
                          + ["nll_median", "nll_p99"])
                writer.writerow(header)

    def log_losses(self, epoch, loss_dict):
        with open(self.loss_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f"{loss_dict.get('loss_total', 0):.4f}", 
                f"{loss_dict.get('loss_token', 0):.4f}", 
                f"{loss_dict.get('loss_geo', 0):.4f}", 
                f"{loss_dict.get('loss_pocket', 0):.4f}",
                f"{loss_dict.get('loss_int', 0):.4f}",
                f"{loss_dict.get('loss_geo_unclamped', 0):.4f}"
            ])

    def log_flow_scale_stats(self, epoch, stats):
        dim_names = ["d", "theta", "phi", "qx", "qy", "qz", "qw"]
        if stats is None:
            stats = {}
        with open(self.flow_scale_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            row = [epoch]
            row += [stats.get(f"logscale_mean_{d}", "") for d in dim_names]
            row += [stats.get(f"pinfrac_{d}", "") for d in dim_names]
            row += [stats.get("nll_median", ""), stats.get("nll_p99", "")]
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
            'loss_geo_unclamped': _to_scalar(metrics.get('loss_geo_unclamped', 0))
        }
        
        # Ghi ra ổ cứng (Chỉ ghi 1 lần duy nhất mỗi epoch)
        self.sys_logger.log_losses(epoch, loss_dict)
        self.sys_logger.log_hardware(epoch, self.epoch_cpu_time, self.epoch_compute_time)

        # P1-1: instrumentation log_scale/pinfrac/NLL của AutoregressiveFlowLayer
        geometric_head = pl_module.generator.geometric_head
        scale_stats = geometric_head.scale_stats()
        self.sys_logger.log_flow_scale_stats(epoch, scale_stats)
        geometric_head.reset_scale_stats()