import mlx.core as mx
import mlx.nn as nn
from .minicpm import MiniCPMModel
from .config import EncoderConfig, LMConfig

class VoxCPMLocEnc(nn.Module):
    def __init__(self, config: LMConfig, input_dim: int = 64):
        super().__init__()
        self.config = config
        
        self.special_token = mx.random.normal((1, 1, 1, config.hidden_size))
        self.in_proj = nn.Linear(input_dim, config.hidden_size, bias=True)
        
        if config.vocab_size != 0:
            # warn or force?
            pass
            
        self.encoder = MiniCPMModel(config)

    def __call__(self, x):
        # x: (B, T, P, D)
        B, T, P, D = x.shape
        
        x = self.in_proj(x) # (B, T, P, H)
        
        # Expand special token: (1, 1, 1, H) -> (B, T, 1, H)
        # MLX broadcast works automatically for elementwise, but checked via concat.
        special_tokens = mx.broadcast_to(self.special_token, (B, T, 1, self.config.hidden_size))
        
        # concat along P dimension (axis 2)
        x = mx.concatenate([special_tokens, x], axis=2) # (B, T, P+1, H)
        
        # Flatten B, T -> (B*T, P+1, H)
        x = x.reshape(B * T, P + 1, -1)
        
        # Run encoder with bidirectional attention (is_causal=False)
        outputs, _ = self.encoder(inputs_embeds=x, is_causal=False) # (B*T, P+1, H)
        
        # Take CLS token (index 0)
        cls_output = outputs[:, 0, :] # (B*T, H)
        
        # Reshape back to (B, T, H)
        return cls_output.reshape(B, T, -1)
