import math
import torch
import torch.nn as nn

# Import Flow Module
from step2_mdn_module import AutoregressiveFlowLayer, DiagonalGaussianHead
from step3_attention_hub import MultiDifferentialCrossAttention

# Dẫn xuất giá trị 0.1 (không phải con số tùy ý): trên một hướng suy biến, MLE
# sẽ đẩy sigma về đúng bề dày thật của dữ liệu, tức DEQUANT_SIGMA. Độ cứng của
# số hạng bậc hai khi đó là 1/sigma^2:
# - sigma_n = 1e-2 -> log_scale tối ưu ≈ ln(0.01) = -4.61, sát sàn clip -5.0,
#   độ cứng 1e4. Vẫn đủ để vòng phản hồi dương chạy. Clip vẫn là thứ chịu lực.
# - sigma_n = 0.1  -> log_scale tối ưu ≈ -2.30, độ cứng 1e2. Giảm 100 lần,
#   và điểm tối ưu nằm HẲN trên sàn clip nên clip thôi chịu lực.
#
# Ý nghĩa vật lý: 0.1 trong không gian chuẩn hóa tương đương 0.1 * 3.035 = 0.30 Å
# nhiễu vị trí cho d, và ~5 độ sai số góc quay. Cả hai đều cùng cỡ với sai số tọa
# độ tinh thể học, nên đây không phải bóp méo dữ liệu mà là nhiễu trung thực.
#
# Đây là sàn mật độ chống likelihood phân kỳ trên tập suy biến (Mục 4.2), không
# phải regularizer tùy ý; có thể anneal giảm dần sau khi huấn luyện đã ổn định.
DEQUANT_SIGMA = 0.1

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

class DualStreamLigandGenerator(nn.Module):
    # --- Đã thêm tham số shift và scale vào __init__ ---
    def __init__(self, vocab_size, hidden_dim=256, num_heads=4, num_layers=3, shift_factors=None, scale_factors=None,
                 geo_head="diag_gauss"):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, batch_first=True, norm_first=True
        )
        self.ar_decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.attention_hub = MultiDifferentialCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads, beta=1.5)
        
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        # Co chon head doi chung: "diag_gauss" (mac dinh) vs "maf" (kien truc goc)
        if geo_head == "diag_gauss":
            self.geometric_head = DiagonalGaussianHead(hidden_dim=hidden_dim, out_dim=7)
        elif geo_head == "maf":
            self.geometric_head = AutoregressiveFlowLayer(hidden_dim=hidden_dim, out_dim=7)
        else:
            raise ValueError(f"geo_head không hợp lệ: {geo_head!r} (chỉ 'diag_gauss' hoặc 'maf')")

        # P2c: nối token_embedding của token ĐANG ĐƯỢC SINH (không phải token < t) vào
        # context của geometric head, rồi chiếu 2*hidden_dim -> hidden_dim.
        self.geo_in_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Đăng ký Tensor vật lý được trích xuất từ Dataset
        if shift_factors is None: shift_factors = torch.zeros(7)
        if scale_factors is None: scale_factors = torch.ones(7)
        self.register_buffer('shift_factors', shift_factors.clone().detach().view(1, 7))
        self.register_buffer('scale_factors', scale_factors.clone().detach().view(1, 7))

    def forward(self, input_ids, h_target, list_h_anti, target_7d=None, target_mask=None, target_ids=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        x = self.token_embedding(input_ids) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)

        h_ar = self.ar_decoder(x, mask=causal_mask, is_causal=True)
        # P2a-3: h_target giờ là [B, R_max, H+3] (feature + tọa độ residue); target_mask
        # [B, R_max] đánh dấu residue thật để loại padding khỏi softmax của attention hub.
        v_context, attn_weights = self.attention_hub(h_ar, h_target, list_h_anti, target_mask=target_mask)
        
        # Luồng 1: Semantic
        logits_vocab = self.semantic_head(v_context)
        
        # Luồng 2: Geometric Flow (Kèm Detach bảo vệ SA Score)
        v_context_detached = v_context.detach()

        # P2c: v_context[t] chỉ mang thông tin từ token < t ("đặt cái này ở đâu"), Flow
        # không được cho biết "cái này là gì". Nối embedding của CHÍNH token đang được
        # sinh vào context trước khi chiếu về hidden_dim. Teacher forcing dùng target_ids
        # (nhãn thật); sampling dùng token vừa được lấy mẫu từ logits_vocab của chính lượt
        # forward này (chưa có nhãn thật). tok_emb.detach() để không tạo đường gradient
        # thứ hai vào cùng bảng embedding (đã có đường chính qua self.token_embedding(input_ids)).
        if target_7d is not None:
            tok_emb = self.token_embedding(target_ids)
        else:
            sampled_ids = torch.multinomial(
                torch.softmax(logits_vocab.detach().view(-1, logits_vocab.size(-1)), dim=-1), 1
            ).view(batch_size, seq_len)
            tok_emb = self.token_embedding(sampled_ids)

        geo_in = self.geo_in_proj(torch.cat([v_context_detached, tok_emb.detach()], dim=-1))
        geo_in_flat = geo_in.view(batch_size * seq_len, self.hidden_dim)

        if target_7d is not None:
            target_7d_flat = target_7d.view(batch_size * seq_len, 7)
            # Standardization: Áp dụng Mean/Std thực tế
            target_scaled = (target_7d_flat - self.shift_factors) / self.scale_factors
            # Dequantization: Làm dày mặt cầu quaternion (trong không gian đã chuẩn hóa)
            noise = torch.randn_like(target_scaled) * DEQUANT_SIGMA
            target_scaled = target_scaled + noise

            # Chi nap token THAT vao reservoir NLL cua geometric_head -- dong bo tong the
            # voi Geo_Loss_Unclamped (step5), von duoc tinh SAU khi mask padding.
            pad_mask_flat = (target_ids != 0).view(-1)
            loss_geo_flat = self.geometric_head(geo_in_flat, target=target_scaled, pad_mask=pad_mask_flat)
            loss_geo = loss_geo_flat.view(batch_size, seq_len)
            sampled_7d = None
        else:
            loss_geo = None
            sampled_scaled_flat = self.geometric_head(geo_in_flat, target=None)
            # Inverse Scale: Lôi từ phân phối chuẩn N(0,1) ra kích thước vật lý thật
            sampled_real_flat = (sampled_scaled_flat * self.scale_factors) + self.shift_factors

            # Chuẩn hóa quaternion và ràng buộc miền giá trị (d, theta, phi) SAU inverse-scale
            q = sampled_real_flat[:, 3:7]
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            d = sampled_real_flat[:, 0:1].clamp_min(0.0)
            theta = sampled_real_flat[:, 1:2].clamp(0.0, math.pi)
            phi = torch.atan2(torch.sin(sampled_real_flat[:, 2:3]), torch.cos(sampled_real_flat[:, 2:3]))
            sampled_real_flat = torch.cat([d, theta, phi, q], dim=-1)

            sampled_7d = sampled_real_flat.view(batch_size, seq_len, 7)
            
        return logits_vocab, loss_geo, sampled_7d, attn_weights