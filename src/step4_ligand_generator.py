import math
import torch
import torch.nn as nn

# Import core modules from previous steps
from step2_mdn_module import MDNLayer
from step3_attention_hub import MultiDifferentialCrossAttention

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for the autoregressive sequence.
    """
    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, hidden_dim]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class DualStreamLigandGenerator(nn.Module):
    """
    Dual-Stream Autoregressive Ligand Generator.
    At each step t, it predicts the next semantic token (Classification) 
    and the 7D geometric spatial parameters via MDN (Regression).
    """
    def __init__(self, vocab_size, hidden_dim=256, num_heads=4, num_layers=3, num_gaussians=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        
        # 1. Input Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        
        # 2. Autoregressive Transformer Decoder (Causal)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True,
            norm_first=True
        )
        # Added enable_nested_tensor=False to prevent PyTorch UserWarning
        self.ar_decoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False
        )
        
        # 3. Cross-Modal Routing Hub (Step 3)
        self.attention_hub = MultiDifferentialCrossAttention(
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            beta=1.5
        )
        
        # 4. Dual-Stream Output Heads
        # Stream 1: Semantic (Token classification)
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Stream 2: Geometric (MDN 7D Regression)
        self.geometric_head = MDNLayer(hidden_dim, num_gaussians=num_gaussians, out_dim=7)

    def forward(self, input_ids, h_target, list_h_anti):
        """
        Args:
            input_ids: [batch_size, seq_len] (Tokens generated so far)
            h_target: [batch_size, N_pos, hidden_dim] (Target pocket)
            list_h_anti: List of [batch_size, N_neg, hidden_dim] (Anti-targets)
        Returns:
            logits_vocab: [batch_size, seq_len, vocab_size]
            pi_logits: [batch_size, seq_len, num_gaussians]
            mu: [batch_size, seq_len, num_gaussians, 7]
            sigma: [batch_size, seq_len, num_gaussians, 7]
            attn_weights: Cross-attention routing weights
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Generate Causal Mask to prevent looking ahead
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # 1. Embed and encode sequence
        x = self.token_embedding(input_ids) * math.sqrt(self.hidden_dim)
        x = self.positional_encoding(x)
        
        # h_ar shape: [batch_size, seq_len, hidden_dim]
        h_ar = self.ar_decoder(x, mask=causal_mask, is_causal=True)
        
        # 2. Route through Differential Attention Hub
        v_context, attn_weights = self.attention_hub(h_ar, h_target, list_h_anti)
        
        # 3. Dual-Stream Predictions
        logits_vocab = self.semantic_head(v_context)
        
        # Flatten temporal dimension for MDN compatibility: [batch_size * seq_len, hidden_dim]
        v_context_flat = v_context.view(batch_size * seq_len, self.hidden_dim)
        pi_logits_flat, mu_flat, sigma_flat = self.geometric_head(v_context_flat)
        
        # Reshape back to 3D and 4D tensors
        pi_logits = pi_logits_flat.view(batch_size, seq_len, self.num_gaussians)
        mu = mu_flat.view(batch_size, seq_len, self.num_gaussians, 7)
        sigma = sigma_flat.view(batch_size, seq_len, self.num_gaussians, 7)
        
        return logits_vocab, pi_logits, mu, sigma, attn_weights


# ==============================================================================
# SCRIPT TEST: 5-STEP AUTOREGRESSIVE GENERATION LOOP
# ==============================================================================
def test_step4_autoregressive_loop():
    print("[*] Starting Step 4: Dual-Stream Autoregressive Generator Test...")
    
    device = torch.device("cpu")
    batch_size = 1
    hidden_dim = 256
    vocab_size = 50  # Mock vocabulary size
    
    # Initialize Model
    generator = DualStreamLigandGenerator(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    generator.eval() # Set to evaluation mode for inference
    
    # Mock Protein Contexts
    h_target = torch.randn(batch_size, 20, hidden_dim, device=device)
    
    h_anti_1 = torch.randn(batch_size, 15, hidden_dim, device=device)
    h_anti_2 = torch.randn(batch_size, 25, hidden_dim, device=device)
    list_h_anti = [h_anti_1, h_anti_2]
    
    # Initial token: [SOS] token (assuming index 0)
    input_ids = torch.tensor([[0]], dtype=torch.long, device=device) # Shape: [1, 1]
    
    print("\n[+] Entering 5-Step Autoregressive Loop:")
    
    with torch.no_grad():
        for step in range(1, 6):
            print(f"\n    --- Step {step} ---")
            print(f"    Input sequence shape: {input_ids.shape}")
            
            # Forward pass
            logits, pi, mu, sigma, attn = generator(input_ids, h_target, list_h_anti)
            
            # Extract features of the LAST generated step
            next_token_logits = logits[:, -1, :] # Shape: [batch_size, vocab_size]
            next_token_mu = mu[:, -1, :, :]      # Shape: [batch_size, num_gaussians, 7]
            
            print(f"    -> Output Logits (last step): {next_token_logits.shape}")
            print(f"    -> Output MDN Mu (last step): {next_token_mu.shape}")
            print(f"    -> Attention Weights (last step): {attn[:, :, -1, :].shape}")
            
            # Greedy Search: Pick the token with the highest probability
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True) # Shape: [batch_size, 1]
            print(f"    -> Predicted Token ID: {next_token_id.item()}")
            
            # Append the predicted token to the input sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
    print("\n[v] Final sequence tensor shape:", input_ids.shape)
    print("=== STEP 4 AR GENERATOR TEST COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    test_step4_autoregressive_loop()