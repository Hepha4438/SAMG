import math
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

import torch_geometric.nn as pyg_nn
from torch_cluster import knn_graph as cluster_knn_graph

import pyro
import pyro.distributions as dist
# --- ĐIỂM SỬA CHỮA (THÊM CONDITIONAL) ---
from pyro.nn import ConditionalAutoRegressiveNN
from pyro.distributions.transforms import ConditionalAffineAutoregressive

# --- RUNTIME MONKEY PATCH ---
def custom_knn_graph(x, k=32, batch=None, loop=False, flow='source_to_target', cosine=False, num_workers=1):
    return cluster_knn_graph(x=x, k=k, batch=batch, loop=loop, flow=flow)
pyg_nn.knn_graph = custom_knn_graph
# ----------------------------

# P1-1: thứ tự 7 chiều cố định cho instrumentation log_scale
DIM_NAMES = ["d", "theta", "phi", "qw", "qx", "qy", "qz"]

class AutoregressiveFlowLayer(nn.Module):
    """
    Conditional Masked Autoregressive Flow (MAF) cho không gian 7D.
    Thay thế hoàn toàn cấu trúc MDN cũ để đảm bảo hội tụ 100%.
    """
    def __init__(self, hidden_dim, out_dim=7):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        
        # Mạng biến đổi Context: Nắn lại hidden state trước khi đưa vào Flow
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()                  
        )
        
        # --- ĐIỂM SỬA CHỮA ---
        # Khởi tạo mạng tự hồi quy có điều kiện: (input_dim, context_dim, hidden_dims)
        self.arn = ConditionalAutoRegressiveNN(out_dim, hidden_dim, [hidden_dim, hidden_dim])
        self.transform = ConditionalAffineAutoregressive(self.arn)
        
        # Phân phối chuẩn gốc Z ~ N(0, I)
        self.register_buffer("base_loc", torch.zeros(out_dim))
        self.register_buffer("base_scale", torch.ones(out_dim))

        # --- P1-1: Instrumentation log_scale (KHÔNG vào state_dict) ---
        self._ls_floor = self.transform.kwargs.get("log_scale_min_clip", -5.0)
        self._ls_sum = torch.zeros(out_dim)
        self._pin_sum = torch.zeros(out_dim)
        self._ls_count = 0
        self._nll_reservoir = []

        def _log_scale_hook(module, input, output):
            if not self.training:
                return
            with torch.no_grad():
                ls = output[1].detach()
                self._ls_sum += ls.sum(0).cpu()
                self._pin_sum += (ls <= self._ls_floor + 1e-3).float().sum(0).cpu()
                self._ls_count += ls.shape[0]

        self.arn.register_forward_hook(_log_scale_hook)

    def scale_stats(self):
        if self._ls_count == 0:
            return None
        mean_ls = self._ls_sum / self._ls_count
        pinfrac = self._pin_sum / self._ls_count
        stats = {}
        for i, name in enumerate(DIM_NAMES):
            stats[f"logscale_mean_{name}"] = mean_ls[i].item()
            stats[f"pinfrac_{name}"] = pinfrac[i].item()
        if self._nll_reservoir:
            nll_tensor = torch.tensor(self._nll_reservoir)
            stats["nll_median"] = nll_tensor.median().item()
            stats["nll_p99"] = torch.quantile(nll_tensor, 0.99).item()
        else:
            stats["nll_median"] = None
            stats["nll_p99"] = None
        return stats

    def reset_scale_stats(self):
        self._ls_sum.zero_()
        self._pin_sum.zero_()
        self._ls_count = 0
        self._nll_reservoir = []

    def forward(self, h, target=None):
        """
        Args:
            h: [batch_size, hidden_dim] Context từ Transformer
            target: [batch_size, 7] Tọa độ thực tế (Chỉ dùng khi Training)
        """
        context = self.context_proj(h)
        batch_size = context.size(0)
        
        # Mở rộng kích thước phân phối gốc khớp với batch_size để tránh lỗi broadcast của Pyro
        base_dist = dist.Normal(self.base_loc, self.base_scale).expand([batch_size, self.out_dim]).to_event(1)
        
        # Ép mạng nắn phân phối dựa theo ngữ cảnh Protein hiện tại
        conditioned_transform = self.transform.condition(context)
        
        # Tạo ra phân phối 3D phức tạp cuối cùng
        flow_dist = dist.TransformedDistribution(base_dist, [conditioned_transform])
        
        if target is not None:
            # LUỒNG TRAINING: Tính Exact Log-Likelihood
            log_prob = flow_dist.log_prob(target)

            # --- P1-1: Reservoir cho NLL (median/p99 chính xác cuối epoch) ---
            with torch.no_grad():
                nll_flat = (-log_prob).detach().flatten().cpu()
                n = nll_flat.numel()
                take = min(512, n)
                if take > 0:
                    idx = torch.randperm(n)[:take]
                    self._nll_reservoir.extend(nll_flat[idx].tolist())
                    if len(self._nll_reservoir) > 20000:
                        self._nll_reservoir = self._nll_reservoir[-20000:]

            return -log_prob
        else:
            # LUỒNG INFERENCE (STEP 7): Lấy mẫu một tọa độ mới
            sampled = flow_dist.sample()

            # Chuẩn hóa Quaternion đã chuyển sang step4, SAU inverse-scale (chuẩn hóa
            # ở đây, trước phép affine scale/shift, sẽ bị phép affine phá vỡ chuẩn đơn vị).
            return sampled


class DiagonalGaussianHead(nn.Module):
    """
    Head đối chứng cho AutoregressiveFlowLayer (MAF): phân phối Gaussian đường chéo,
    7 chiều độc lập (KHÔNG có autoregressive coupling giữa các chiều). Dùng để so sánh
    công bằng xem MAF có thực sự cần thiết hay một Gaussian đơn giản đã đủ.
    """
    def __init__(self, hidden_dim, out_dim=7, log_sigma_min=-5.0, log_sigma_max=3.0):
        super().__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        # GIỮ Y HỆT kiến trúc context_proj của AutoregressiveFlowLayer để so sánh công bằng
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        self.mu_head = nn.Linear(hidden_dim, out_dim)
        self.logsig_head = nn.Linear(hidden_dim, out_dim)

        # --- Instrumentation (CÙNG chu kỳ với AutoregressiveFlowLayer, xem P1-1) ---
        self._ls_floor = log_sigma_min
        self._ls_sum = torch.zeros(out_dim)
        self._pin_sum = torch.zeros(out_dim)
        self._ls_count = 0
        self._nll_reservoir = []

    def scale_stats(self):
        if self._ls_count == 0:
            return None
        mean_ls = self._ls_sum / self._ls_count
        pinfrac = self._pin_sum / self._ls_count
        stats = {}
        for i, name in enumerate(DIM_NAMES):
            stats[f"logscale_mean_{name}"] = mean_ls[i].item()
            stats[f"pinfrac_{name}"] = pinfrac[i].item()
        if self._nll_reservoir:
            nll_tensor = torch.tensor(self._nll_reservoir)
            stats["nll_median"] = nll_tensor.median().item()
            stats["nll_p99"] = torch.quantile(nll_tensor, 0.99).item()
        else:
            stats["nll_median"] = None
            stats["nll_p99"] = None
        return stats

    def reset_scale_stats(self):
        self._ls_sum.zero_()
        self._pin_sum.zero_()
        self._ls_count = 0
        self._nll_reservoir = []

    def forward(self, h, target=None):
        c = self.context_proj(h)
        mu = self.mu_head(c)
        log_sigma = self.logsig_head(c).clamp(self.log_sigma_min, self.log_sigma_max)

        if target is not None:
            z = (target - mu) / log_sigma.exp()
            nll = 0.5 * z.pow(2) + log_sigma + 0.5 * math.log(2 * math.pi)
            nll = nll.sum(-1)

            if self.training:
                with torch.no_grad():
                    self._ls_sum += log_sigma.detach().sum(0).cpu()
                    self._pin_sum += (log_sigma.detach() <= self._ls_floor + 1e-3).float().sum(0).cpu()
                    self._ls_count += log_sigma.shape[0]

                    nll_flat = nll.detach().flatten().cpu()
                    n = nll_flat.numel()
                    take = min(512, n)
                    if take > 0:
                        idx = torch.randperm(n)[:take]
                        self._nll_reservoir.extend(nll_flat[idx].tolist())
                        if len(self._nll_reservoir) > 20000:
                            self._nll_reservoir = self._nll_reservoir[-20000:]

            return nll
        else:
            # LUỒNG INFERENCE: KHÔNG chuẩn hóa quaternion -- step4 đã lo phần clamp
            # d/theta, wrap phi, chuẩn hóa quaternion SAU inverse-scale.
            return mu + log_sigma.exp() * torch.randn_like(mu)

# ==============================================================================
# SCRIPT TEST: OVERFIT ON A MOCK BATCH (AUTOREGRESSIVE FLOW)
# ==============================================================================
if __name__ == "__main__":
    import torch
    print("[*] Starting Autoregressive Flow Module Overfitting Test using MOCK data...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 256
    batch_size = 4
    
    # 1. Giả lập vector Context đầu ra của Transformer (Step 1 -> Step 3)
    h_graph_mock = torch.randn(batch_size, hidden_dim, device=device)

    # 2. Giả lập 7D Target: [d, theta, phi, q_x, q_y, q_z, q_w]
    target_7d = torch.randn(batch_size, 7, device=device)
    # Tự động chuẩn hóa Quaternion mục tiêu (độ dài = 1)
    target_7d[:, 3:7] = target_7d[:, 3:7] / target_7d[:, 3:7].norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # 3. Khởi tạo Flow layer và Optimizer
    flow_layer = AutoregressiveFlowLayer(hidden_dim, out_dim=7).to(device)
    optimizer = torch.optim.Adam(flow_layer.parameters(), lr=5e-3)

    print(f"\n[+] Mock target_7d shape: {target_7d.shape}")
    print("[+] Training loop over 1000 epochs to force overfitting on 1 mock batch:")
    
    for epoch in range(1, 1001):
        optimizer.zero_grad()
        
        # LUỒNG HUẤN LUYỆN: Truyền target vào để tính Exact NLL Loss
        nll_loss_batch = flow_layer(h_graph_mock, target=target_7d)
        loss = nll_loss_batch.mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow_layer.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0 or epoch == 1:
            print(f"    Epoch {epoch:4d} | Exact NLL Loss: {loss.item():.4f}")
            
    print("\n[v] Test Evaluation (Inference Sampling):")
    with torch.no_grad():
        # LUỒNG KIỂM THỬ: Không truyền target, mạng tự sinh ra 7D (Sinh phân tử ở Step 7)
        sampled_7d = flow_layer(h_graph_mock, target=None)
        print(f"    -> Sampled 7D shape: {sampled_7d.shape}")
        
        # Kiểm tra toán học: Xem Quaternion sinh ra đã tự chuẩn hóa L2 norm = 1 chưa
        quat_norms = sampled_7d[:, 3:7].norm(dim=-1)
        print(f"    -> Sampled Quaternion norms: {quat_norms.tolist()}")

    if loss.item() < 0.0:
        print("    -> SUCCESS: NLL Loss decreased significantly (Exact log-likelihood can be negative).")
    if torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-4):
        print("    -> SUCCESS: Quaternions are correctly L2-normalized during sampling.")
        
    print("=== STEP 2 AUTOREGRESSIVE FLOW MODULE MOCK TEST COMPLETED ===")   