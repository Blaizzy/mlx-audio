import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .config import HiggsAudioConfig
from .dac import AcousticDecoder, AcousticEncoder, ResidualVectorQuantizer


class HiggsAudioTokenizer(nn.Module):
    """
    HiggsAudioV2 acoustic tokenizer.

    Decode path (tokens → waveform): quantizer → fc2 → acoustic_decoder  [MLX]
    Encode path (waveform → tokens): HiggsAudioV2TokenizerModel  [PyTorch CPU]

    The encode path requires the HuBERT semantic model which is only available
    via the transformers PyTorch implementation. We bridge via numpy.
    """

    def __init__(self, config: HiggsAudioConfig):
        super().__init__()
        self.config = config
        self.acoustic_encoder = AcousticEncoder()
        self.quantizer = ResidualVectorQuantizer()
        self.acoustic_decoder = AcousticDecoder()
        # Decode path: quantizer (1024-dim) → fc2 → decoder (256-dim)
        self.fc2 = nn.Linear(1024, 256, bias=True)
        # PyTorch tokenizer for encoding (set by from_pretrained if available)
        self._pt_tokenizer = None

    def decode(self, tokens: mx.array) -> mx.array:
        """
        tokens: [T, 8] or [B, T, 8] int32
        Returns: [T*960] (1D) if 2D input, or [B, T*960, 1] if 3D input
        """
        squeeze = tokens.ndim == 2
        if squeeze:
            tokens = tokens[None]  # [1, T, 8]
        z = self.quantizer.decode(tokens)  # [B, T, 1024]
        z = self.fc2(z)  # [B, T, 256]
        wav = self.acoustic_decoder(z)  # [B, T*960, 1]
        if squeeze:
            return wav[0, :, 0]  # [T*960]
        return wav  # [B, T*960, 1]

    def encode(self, waveform: mx.array) -> mx.array:
        """
        waveform: [B, T, 1] float32 at 24kHz
        Returns: [B, T//960, 8] int32 codebook tokens

        Uses PyTorch HiggsAudioV2TokenizerModel (CPU) when loaded via from_pretrained().
        The full encode path requires HuBERT semantic features; it cannot be replicated
        with acoustic encoder alone.
        """
        if self._pt_tokenizer is None:
            raise RuntimeError(
                "Encode requires the PyTorch HiggsAudioV2TokenizerModel. "
                "Load via HiggsAudioTokenizer.from_pretrained() with a valid model_path."
            )
        import numpy as np
        import torch

        # MLX [B, T, 1] → numpy → PyTorch [B, 1, T]
        wav_np = np.array(waveform.astype(mx.float32))  # [B, T, 1]
        wav_pt = torch.from_numpy(wav_np).permute(0, 2, 1)  # [B, 1, T]

        with torch.no_grad():
            codes = self._pt_tokenizer.encode(
                wav_pt, return_dict=False
            )  # [B, 8, T//960]

        # PyTorch [B, 8, T//960] → MLX [B, T//960, 8]
        codes_np = codes.numpy().astype("int32")  # [B, 8, T//960]
        codes_mx = mx.array(codes_np).transpose(0, 2, 1)  # [B, T//960, 8]
        return codes_mx

    def sanitize(self, weights: dict) -> dict:
        """Filter checkpoint keys to acoustic path only and fix weight layouts.

        PyTorch conv weights are [C_out, C_in, K]; MLX expects [C_out, K, C_in].
        PyTorch ConvTranspose1d weights are [C_in, C_out, K]; MLX expects [C_in, K, C_out].
        Both require the same (0, 2, 1) transpose.
        """
        keep_prefixes = ("acoustic_encoder.", "acoustic_decoder.", "quantizer.", "fc2.")
        drop_suffixes = (".embed_avg", ".cluster_size", ".inited")
        result = {}
        for k, v in weights.items():
            if not any(k.startswith(p) for p in keep_prefixes):
                continue
            if any(k.endswith(s) for s in drop_suffixes):
                continue
            # Remap embedding key: checkpoint uses .embed, MLX nn.Embedding uses .weight
            if k.endswith(".codebook.embed"):
                k = k[: -len("embed")] + "weight"
            # Transpose snake activation alpha: PyTorch [1, C, 1] → MLX [1, 1, C]
            if k.endswith(".alpha") and v.ndim == 3:
                v = v.transpose(0, 2, 1)
            # Transpose 3-D weights from PyTorch layout → MLX layout
            elif v.ndim == 3 and k.endswith(".weight"):
                # ConvTranspose1d: PyTorch [C_in, C_out, K] → MLX [C_out, K, C_in]
                # Conv1d:          PyTorch [C_out, C_in, K] → MLX [C_out, K, C_in]
                if "conv_t" in k:
                    v = v.transpose(1, 2, 0)
                else:
                    v = v.transpose(0, 2, 1)
            result[k] = v
        return result

    @classmethod
    def from_pretrained(cls, model_path: str) -> "HiggsAudioTokenizer":
        """
        Load from k2-fsa/OmniVoice local directory.
        Expects: <model_path>/audio_tokenizer/config.json
                 <model_path>/audio_tokenizer/model.safetensors

        Also loads PyTorch HiggsAudioV2TokenizerModel (CPU) for voice-clone encoding.
        """
        config_path = Path(model_path) / "audio_tokenizer" / "config.json"
        weights_path = Path(model_path) / "audio_tokenizer" / "model.safetensors"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        config = HiggsAudioConfig.from_dict(json.loads(config_path.read_text()))
        inst = cls(config)
        raw = mx.load(str(weights_path))
        sanitized = inst.sanitize(raw)
        inst.load_weights(list(sanitized.items()))
        mx.eval(inst.parameters())

        # Load PyTorch tokenizer for encoding (requires torchaudio)
        try:
            from transformers import HiggsAudioV2TokenizerModel

            pt_tok = HiggsAudioV2TokenizerModel.from_pretrained(
                str(Path(model_path) / "audio_tokenizer")
            )
            pt_tok.train(False)  # set to eval mode without using eval()
            inst._pt_tokenizer = pt_tok
        except Exception as e:
            import warnings

            warnings.warn(
                f"Could not load PyTorch tokenizer for encoding: {e}. "
                "Voice cloning (encode) will not work.",
                stacklevel=2,
            )

        return inst
