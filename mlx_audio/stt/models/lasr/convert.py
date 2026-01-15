
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoModelForCTC, AutoConfig

from mlx_audio.stt.models.lasr.config import ModelConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_weights(key: str, value: Any) -> Dict[str, Any]:
    # Remap key names from HF to MLX
    
    # 1. Subsampling
    if "subsampler" in key:
        # HF: subsampler.dense_0 -> MLX: subsampler.dense_0
        pass 
        
    # 2. Rotary embedding (no weights properly, just buffers if persistent, but we calc on fly)
    if "rotary_emb.inv_freq" in key:
        return {} # Don't save inv_freq buffer
        
    # 3. Handle 1D Conv transposes
    # HF Conv1d weights: (out_channels, in_channels, kernel_size)
    # MLX Conv1d weights: (out_channels, kernel_size, in_channels) -> Wait, MLX is (N, W, C)?
    # Let's check MLX Conv1d weight shape: (output_channels, kernel_size, input_channels)
    
    # Check if this is a conv weight
    if "conv" in key and "weight" in key:
        # Value is numpy array or tensor
        if isinstance(value, torch.Tensor):
            value = value.numpy()
            
        # If it's a 3D conv weight
        if len(value.shape) == 3:
            # HF: (Out, In, K)
            # MLX: (Out, K, In)
            value = value.transpose(0, 2, 1)
            
    # 4. Encoder Layers
    # HF: encoder.layers.0...
    # MLX: encoder.layers is list, so encoder.layers.0... matches
    
    # HF: self_attn.q_proj -> MLX: self_attn.q_proj (matches)
    
    # HF: conv.depthwise_conv
    # HF: depthwise is groups=channels
    # HF weight: (Out, 1, K) (because groups=In=Out)
    # MLX weight: (Out, K, 1) ? since input channels is 1 per group?
    # MLX Conv1d for depthwise:
    # If we use weight of shape (Out, K, 1)?
    # Actually for MLX `nn.Conv1d`, weights are [out_channels, kernel_size, in_channels/groups]
    # Here groups = in_channels = out_channels. So last dim is 1.
    # So HF (C, 1, K) -> MLX (C, K, 1). Transpose (0, 2, 1) still holds.
    
    # 5. Output Norm
    # HF: out_norm -> MLX: out_norm
    if "out_norm" in key:
        pass

    # 6. CTC Head
    # HF: ctc_head (Conv1d: kernel 1)
    # MLX: ctc_head (Linear)
    # HF Conv1d (1x1) weight: (Vocab, Hidden, 1)
    # MLX Linear weight: (Out, In) = (Vocab, Hidden)
    if "ctc_head.weight" in key:
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        # Squeeze the kernel dimension
        # (Vocab, Hidden, 1) -> (Vocab, Hidden)
        value = value.squeeze(-1)
        
    if "ctc_head.bias" in key:
        # (Vocab,) -> matches
        pass

    # Remove .weight from LayerNorm if MLX uses .weight (it does)
    # Remove .bias (it does)
    
    # Rename keys?
    # HF: encoder.layers.X
    # MLX: encoder.layers.X
    
    # Some naming differences?
    # HF: norm_feed_forward1 (LayerNorm)
    # MLX: norm_feed_forward1
    
    return {key: value}

def convert(model_id: str, output_dir: str):
    logger.info(f"Loading model from {model_id}")
    
    # Load config
    hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Create MLX config
    # We need to extract encoder config
    if hasattr(hf_config, "encoder_config"):
        encoder_config_dict = hf_config.encoder_config.to_dict()
    else:
        encoder_config_dict = hf_config.to_dict()
        
    # Map HF config to LasrEncoderConfig
    # We can pass the dict and let from_dict handle filtering
    
    mlx_config = ModelConfig.from_dict({
        "vocab_size": hf_config.vocab_size,
        "encoder_config": encoder_config_dict,
        "pad_token_id": hf_config.pad_token_id,
        "ctc_loss_reduction": getattr(hf_config, "ctc_loss_reduction", "mean"),
        "ctc_zero_infinity": getattr(hf_config, "ctc_zero_infinity", True),
    })
    
    # Load HF Model
    # Note: Requires the specific transformers fork installed
    try:
        hf_model = AutoModelForCTC.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load HF model: {e}")
        logger.error("Ensure you have the compatible transformers version installed.")
        return

    hf_model.eval()
    state_dict = hf_model.state_dict()
    
    weights = {}
    for k, v in state_dict.items():
        # Map weights
        mapped = map_weights(k, v)
        for mk, mv in mapped.items():
            if isinstance(mv, torch.Tensor):
                mv = mv.numpy()
            weights[mk] = mv
            
    # Save weights and config
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save config.json
    import dataclasses
    # Helper to convert nested dataclasses to dict
    def asdict_factory(data):
        def convert(obj):
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            return obj
        return convert(data)

    # We can just use the __dict__ or construct a clean dict
    # But since we have a from_dict, let's just make sure we capture everything
    # The config structure in MLX is:
    # {
    #   "vocab_size": ...,
    #   "encoder_config": { ... }
    # }
    
    config_dict = dataclasses.asdict(mlx_config)
    
    with open(output_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
        
    # Save weights
    np.savez(output_path / "weights.npz", **weights)
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MedASR weights to MLX")
    parser.add_argument("--model_id", type=str, default="google/medasr", help="HF Model ID")
    parser.add_argument("--output_dir", type=str, default="medasr_mlx", help="Output directory")
    
    args = parser.parse_args()
    convert(args.model_id, args.output_dir)
