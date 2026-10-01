import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDifferentialCrossAttention(nn.Module):
    """
    Multi-Differential Cross-Attention Hub (Step 3).
    Computes positive attention against the target and differential repulsion against M anti-targets.
    """
    def __init__(self, hidden_dim, num_heads=4, beta=1.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.beta = beta

        # P2a: h_target giờ mang hình học (feature bất biến + tọa độ residue trong frame học
        # canonical) → [.., hidden_dim + 3]. Chiếu về hidden_dim trước khi làm K/V.
        self.target_in_proj = nn.Linear(hidden_dim + 3, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_pos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_pos_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_neg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Null-sink token to absorb probability mass when penalized by negative attention
        self.null_key = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.null_val = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, query, h_target, list_h_anti, target_mask=None):
        """
        Args:
            query: [batch_size, seq_len, hidden_dim]
            h_target: [batch_size, R_max, hidden_dim + 3] -- feature bất biến nối tọa độ
                residue trong frame học canonical (P2a). Được chiếu về hidden_dim trước K/V.
            list_h_anti: List of tensors [batch_size, N_neg_m, hidden_dim] for m in 1..M
            target_mask: [batch_size, R_max] Bool, True = residue thật (không phải padding).
                None => coi như mọi residue đều hợp lệ (tương thích ngược).
        Returns:
            v_context: [batch_size, seq_len, hidden_dim]
            attn_weights: [batch_size, num_heads, seq_len, R_max + 1]
        """
        batch_size, seq_len, _ = query.size()
        h_target = self.target_in_proj(h_target)

        if target_mask is None:
            target_mask = torch.ones(batch_size, h_target.size(1), dtype=torch.bool, device=query.device)
        # Null-sink luôn hợp lệ, bất kể mask của các residue thật.
        full_mask = torch.cat([
            torch.ones(batch_size, 1, dtype=torch.bool, device=query.device), target_mask
        ], dim=1)

        # Project queries
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Append null-sink to positive keys and values
        target_with_null = torch.cat([self.null_key.expand(batch_size, -1, -1), h_target], dim=1)
        val_with_null = torch.cat([self.null_val.expand(batch_size, -1, -1), h_target], dim=1)

        K_pos = self.k_pos_proj(target_with_null).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V_pos = self.v_pos_proj(val_with_null).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute A_pos: [batch_size, num_heads, seq_len, N_pos + 1]
        A_pos = torch.matmul(Q, K_pos.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Compute A_neg across M anti-targets and max-pool along M axis
        max_A_neg = torch.zeros(batch_size, self.num_heads, seq_len, 1, device=query.device)
        if len(list_h_anti) > 0:
            for h_anti in list_h_anti:
                K_neg = self.k_neg_proj(h_anti).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                # A_neg_m: [batch_size, num_heads, seq_len, N_neg_m]
                A_neg_m = torch.matmul(Q, K_neg.transpose(-2, -1)) / math.sqrt(self.head_dim)
                # Max-pooling along the residue dimension of the anti-target
                max_m, _ = torch.max(A_neg_m, dim=-1, keepdim=True)
                # Max-pooling across the M anti-targets dimension
                max_A_neg = torch.max(max_A_neg, max_m)

        # Apply differential attention: Softmax(A^+ - beta * A^-_{agg}) * V^+
        A_diff = A_pos.clone()
        penalty = self.beta * F.relu(max_A_neg)
        # Do not penalize the null-sink column (index 0)
        A_diff[:, :, :, 1:] = A_diff[:, :, :, 1:] - penalty

        # Áp mask trước softmax (True = residue thật); null-sink (cột 0) luôn hợp lệ.
        A_diff = A_diff.masked_fill(~full_mask[:, None, None, :], float('-inf'))

        attn_weights = F.softmax(A_diff, dim=-1)
        v_context = torch.matmul(attn_weights, V_pos)
        v_context = v_context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)

        return self.out_proj(v_context), attn_weights

if __name__ == "__main__":
    print("[*] Testing Step 3: Multi-Differential Cross-Attention Hub with 1 Query, "
          "3 Positive Residue Keys (1 padding) + Null-sink, 3 Negative Keys...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    seq_len = 1
    hidden_dim = 64
    num_heads = 2

    hub = MultiDifferentialCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads, beta=2.0).to(device)

    # 1 Query
    query = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    # Positive Key: 3 residue (H+3, kèm tọa độ) + null sink -> 4 keys total, residue cuối là padding
    r_max = 3
    h_target = torch.randn(batch_size, r_max, hidden_dim + 3, device=device)
    target_mask = torch.tensor([[True, True, False]], device=device)
    # 3 Negative Keys (Anti-targets with varying residue lengths)
    h_anti_1 = torch.randn(batch_size, 2, hidden_dim, device=device)
    h_anti_2 = torch.randn(batch_size, 3, hidden_dim, device=device)
    h_anti_3 = torch.randn(batch_size, 1, hidden_dim, device=device)
    list_h_anti = [h_anti_1, h_anti_2, h_anti_3]

    # Forward pass
    v_context, attn_weights = hub(query, h_target, list_h_anti, target_mask=target_mask)

    print("\n[v] Test Results:")
    print(f"    -> Query shape: {query.shape}")
    print(f"    -> V_context output shape: {v_context.shape}")
    print(f"    -> Attention weights shape (batch, heads, seq, keys): {attn_weights.shape}")
    print(f"    -> Trong so tai residue padding (cot cuoi, phai ~0): "
          f"{attn_weights[0, 0, 0, -1].item():.6f}")
    print(f"    -> Attention Weights (Head 0):\n{attn_weights[0, 0].detach().cpu().numpy()}")
    print("    [v] Step 3 Multi-Differential Cross-Attention verification passed successfully.")