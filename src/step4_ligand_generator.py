import math
import torch
import torch.nn as nn

# Import Flow Module
from step2_mdn_module import AutoregressiveFlowLayer
from step3_attention_hub import MultiDifferentialCrossAttention

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
    def __init__(self, vocab_size, hidden_dim=256, num_heads=4, num_layers=3, shift_factors=None, scale_factors=None):
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
        self.geometric_head = AutoregressiveFlowLayer(hidden_dim=hidden_dim, out_dim=7)
        
        # Đăng ký Tensor vật lý được trích xuất từ Dataset
        if shift_factors is None: shift_factors = torch.zeros(7)
        if scale_factors is None: scale_factors = torch.ones(7)
        self.register_buffer('shift_factors', shift_factors.clone().detach().view(1, 7))
        self.register_buffer('scale_factors', scale_factors.clone().detach().view(1, 7))

    def forward(self, input_ids, h_target, list_h_anti, target_7d=None):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        x = self.token_embedding(input_ids) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        
        h_ar = self.ar_decoder(x, mask=causal_mask, is_causal=True)
        v_context, attn_weights = self.attention_hub(h_ar, h_target, list_h_anti)
        
        # Luồng 1: Semantic
        logits_vocab = self.semantic_head(v_context)
        
        # Luồng 2: Geometric Flow (Kèm Detach bảo vệ SA Score)
        v_context_detached = v_context.detach()
        v_context_flat = v_context_detached.view(batch_size * seq_len, self.hidden_dim)
        
        if target_7d is not None:
            target_7d_flat = target_7d.view(batch_size * seq_len, 7)
            # Dequantization: Làm dày mặt cầu quaternion
            noise = torch.randn_like(target_7d_flat) * 1e-3
            target_noisy = target_7d_flat + noise
            # Standardization: Áp dụng Mean/Std thực tế
            target_scaled = (target_noisy - self.shift_factors) / self.scale_factors
            
            loss_geo_flat = self.geometric_head(v_context_flat, target=target_scaled)
            loss_geo = loss_geo_flat.view(batch_size, seq_len)
            sampled_7d = None
        else:
            loss_geo = None
            sampled_scaled_flat = self.geometric_head(v_context_flat, target=None)
            # Inverse Scale: Lôi từ phân phối chuẩn N(0,1) ra kích thước vật lý thật
            sampled_real_flat = (sampled_scaled_flat * self.scale_factors) + self.shift_factors
            sampled_7d = sampled_real_flat.view(batch_size, seq_len, 7)
            
        return logits_vocab, loss_geo, sampled_7d, attn_weights