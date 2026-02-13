"""Decoder module construction for the MOSS audio tokenizer."""

from __future__ import annotations

import mlx.nn as nn

from .config import MossAudioTokenizerConfig
from .modules import (
    MossAudioTokenizerPatchedPretransform,
    MossAudioTokenizerProjectedTransformer,
    MossAudioTokenizerTransformerConfig,
)


def build_moss_audio_tokenizer_decoder_modules(
    config: MossAudioTokenizerConfig,
) -> list[nn.Module]:
    current_frame_rate = float(config.sampling_rate) / float(config.downsample_rate)
    modules: list[nn.Module] = []

    for module_config in config.decoder_modules:
        if module_config.module_type == "PatchedPretransform":
            if module_config.patch_size is None:
                raise ValueError(
                    "PatchedPretransform in decoder is missing patch_size"
                )
            module = MossAudioTokenizerPatchedPretransform(
                patch_size=int(module_config.patch_size),
                is_downsample=False,
                module_type=module_config.module_type,
            )
        elif module_config.module_type == "Transformer":
            transformer_config = MossAudioTokenizerTransformerConfig.from_module_config(
                module_config,
                context=int(
                    current_frame_rate * config.causal_transformer_context_duration
                ),
            )
            module = MossAudioTokenizerProjectedTransformer(
                transformer_config, module_type=module_config.module_type
            )
        else:
            raise ValueError(
                f"Unsupported decoder module_type: {module_config.module_type}"
            )

        modules.append(module)
        current_frame_rate *= int(getattr(module, "downsample_ratio", 1))

    return modules

