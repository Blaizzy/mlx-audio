from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MossAudioTokenizerConfig:
    model_type: str = "moss_audio_tokenizer"

    sampling_rate: int = 24000
    downsample_rate: int = 1920

    code_dim: int = 768
    num_quantizers: int = 32
    codebook_size: int = 1024
    codebook_dim: int = 8
    rvq_dim: int = 512
    quantizer_input_dim: int = 768
    quantizer_output_dim: int = 768
    quantizer_type: str = "rlfq"

    encoder_kwargs: list[dict[str, Any]] | None = None
    decoder_kwargs: list[dict[str, Any]] | None = None
    reversed_decoder_kwargs: list[dict[str, Any]] | None = None

    causal_transformer_context_duration: float = 10.0
    dtype: str | None = None
    version: str | None = None

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_rate

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "MossAudioTokenizerConfig":
        sampling_rate = params.get("sampling_rate", params.get("sample_rate", 24000))

        quantizer_kwargs = params.get("quantizer_kwargs") or {}
        num_quantizers = int(quantizer_kwargs.get("num_quantizers", 32))
        codebook_size = int(quantizer_kwargs.get("codebook_size", 1024))
        codebook_dim = int(quantizer_kwargs.get("codebook_dim", 8))
        rvq_dim = int(quantizer_kwargs.get("rvq_dim", 512))
        quantizer_input_dim = int(
            quantizer_kwargs.get("input_dim", params.get("code_dim", 768))
        )
        quantizer_output_dim = int(
            quantizer_kwargs.get("output_dim", params.get("code_dim", 768))
        )
        quantizer_type = (
            quantizer_kwargs.get("quantizer_type")
            or params.get("quantizer_type")
            or "rlfq"
        )
        encoder_kwargs = params.get("encoder_kwargs")
        decoder_kwargs = params.get("decoder_kwargs")
        reversed_decoder_kwargs = params.get("reversed_decoder_kwargs")

        return cls(
            model_type="moss_audio_tokenizer",
            sampling_rate=int(sampling_rate),
            downsample_rate=int(params.get("downsample_rate", 1920)),
            code_dim=int(params.get("code_dim", 768)),
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            rvq_dim=rvq_dim,
            quantizer_input_dim=quantizer_input_dim,
            quantizer_output_dim=quantizer_output_dim,
            quantizer_type=str(quantizer_type),
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            reversed_decoder_kwargs=reversed_decoder_kwargs,
            causal_transformer_context_duration=float(
                params.get("causal_transformer_context_duration", 10.0)
            ),
            dtype=params.get("dtype"),
            version=params.get("version"),
        )


__all__ = ["MossAudioTokenizerConfig"]
