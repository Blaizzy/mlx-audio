"""Parakeet Redux: HF tensor names and lossless ternary-to-MLX repacking.

The source ``thrush-ternary-v2`` export stores five base-3 digits per byte.
MLX's affine 2-bit format represents the same weights exactly with codes
0/1/2, the original scales, and affine biases equal to minus those scales.
"""

import json
import re
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_audio.stt.models.nemo.alignment import (
    AlignedToken,
    sentences_to_result,
    tokens_to_sentences,
)
from mlx_audio.utils import from_dict

from . import tokenizer
from .parakeet import ParakeetTDT, ParakeetTDTArgs

_VAD_KEYS = {
    f"vad_head.{layer}.{parameter}"
    for layer in ("proj", "ctx", "out")
    for parameter in ("weight", "bias")
}


def repack_ternary(qweight: mx.array, in_features: int) -> mx.array:
    """Repack row-padded base-3 bytes into little-endian uint32 2-bit words."""
    if (
        qweight.dtype != mx.uint8
        or qweight.ndim != 2
        or in_features % 16
        or qweight.shape[1] != (in_features + 4) // 5
    ):
        raise ValueError("Expected uint8 ternary rows of ceil(in_features / 5) bytes.")
    if bool(mx.any(qweight > 242)):
        raise ValueError("Invalid ternary byte: five base-3 digits must be <= 242.")
    powers = mx.array([1, 3, 9, 27, 81], dtype=mx.uint32)
    codes = (qweight.astype(mx.uint32)[..., None] // powers) % 3
    codes = codes.reshape(qweight.shape[0], -1)[:, :in_features]
    shifts = 2 * mx.arange(16, dtype=mx.uint32)
    packed = mx.sum(codes.reshape(qweight.shape[0], -1, 16) << shifts, axis=-1)
    # Do not retain unpacked encoder-sized intermediates until the final load.
    mx.eval(packed)
    return packed


def map_weight_name(name: str) -> str:
    """Translate HF Parakeet tensor/module names to the existing MLX family."""
    for old, new in (
        ("encoder.subsampling.layers.", "encoder.pre_encode.conv."),
        ("encoder.subsampling.linear", "encoder.pre_encode.out"),
        (".conv.norm.", ".conv.batch_norm."),
        (".self_attn.q_proj", ".self_attn.linear_q"),
        (".self_attn.k_proj", ".self_attn.linear_k"),
        (".self_attn.v_proj", ".self_attn.linear_v"),
        (".self_attn.o_proj", ".self_attn.linear_out"),
        (".self_attn.relative_k_proj", ".self_attn.linear_pos"),
        (".self_attn.bias_u", ".self_attn.pos_bias_u"),
        (".self_attn.bias_v", ".self_attn.pos_bias_v"),
        ("decoder.embedding.", "decoder.prediction.embed."),
        ("decoder.decoder_projector.", "joint.pred."),
        ("encoder_projector.", "joint.enc."),
        ("joint.head.", "joint.joint_net.2."),
    ):
        name = name.replace(old, new)
    match = re.fullmatch(r"decoder.lstm.weight_(ih|hh)_l(\d+)", name)
    if match:
        weight = "Wx" if match[1] == "ih" else "Wh"
        name = f"decoder.prediction.dec_rnn.lstm.{match[2]}.{weight}"
    return name


def _vocabulary(path: Path, blank_id: int) -> list[str]:
    data = json.loads((path / "tokenizer.json").read_text())
    vocab = data["model"]["vocab"]
    if not isinstance(vocab, dict) or set(vocab.values()) != set(range(blank_id)):
        raise ValueError("Redux requires a contiguous BPE vocabulary before blank.")
    added = {token["content"]: token["id"] for token in data["added_tokens"]}
    if added.get("<blank>") != blank_id:
        raise ValueError("Redux tokenizer and config disagree on the blank token.")
    return [piece for piece, _ in sorted(vocab.items(), key=lambda pair: pair[1])]


def _model_args(config: dict, vocabulary: list[str]) -> ParakeetTDTArgs:
    enc = config["encoder_config"]
    hidden = enc["hidden_size"]
    factor = enc["subsampling_factor"]
    if (
        enc["hidden_act"] != "silu"
        or enc.get("attention_bias", False)
        or enc.get("convolution_bias", False)
        or enc["num_attention_heads"] != enc["num_key_value_heads"]
        or hidden % enc["num_attention_heads"]
        or enc["intermediate_size"] % hidden
        or factor not in (2, 4, 8)
        or enc["num_mel_bins"] % factor
        or enc["subsampling_conv_kernel_size"] != 3
        or enc["subsampling_conv_stride"] != 2
        or config["hidden_act"] != "relu"
    ):
        raise ValueError("Unsupported Parakeet Redux architecture.")
    durations = config["durations"]
    if (
        not durations
        or durations[0] != 0
        or any(type(d) is not int or d < 0 for d in durations)
        or any(a >= b for a, b in zip(durations, durations[1:]))
        or config["max_symbols_per_step"] <= 0
        or config["blank_token_id"] != config["vocab_size"] - 1
    ):
        raise ValueError("Invalid Redux vocabulary or TDT decoding configuration.")
    decoder_hidden = config["decoder_hidden_size"]
    return from_dict(
        ParakeetTDTArgs,
        {
            "preprocessor": {
                "sample_rate": 16000,
                "normalize": "per_feature",
                "window_size": 0.025,
                "window_stride": 0.01,
                "window": "hann",
                "features": enc["num_mel_bins"],
                "n_fft": 512,
                "dither": 0.0,
                "normalize_valid_frames": True,
            },
            "encoder": {
                "feat_in": enc["num_mel_bins"],
                "n_layers": enc["num_hidden_layers"],
                "d_model": hidden,
                "n_heads": enc["num_attention_heads"],
                "ff_expansion_factor": enc["intermediate_size"] // hidden,
                "subsampling_factor": factor,
                "self_attention_model": "rel_pos",
                "subsampling": "dw_striding",
                "conv_kernel_size": enc["conv_kernel_size"],
                "subsampling_conv_channels": enc["subsampling_conv_channels"],
                "pos_emb_max_len": enc["max_position_embeddings"],
                "use_bias": False,
                "xscaling": enc.get("scale_input", False),
                "mask_padding": True,
            },
            "decoder": {
                "blank_as_pad": True,
                "vocab_size": len(vocabulary),
                "prednet": {
                    "pred_hidden": decoder_hidden,
                    "pred_rnn_layers": config["num_decoder_layers"],
                },
            },
            "joint": {
                "num_classes": len(vocabulary),
                "vocabulary": vocabulary,
                "num_extra_outputs": len(durations),
                "jointnet": {
                    "joint_hidden": decoder_hidden,
                    "activation": "relu",
                    "encoder_hidden": hidden,
                    "pred_hidden": decoder_hidden,
                },
            },
            "decoding": {
                "model_type": "tdt",
                "durations": durations,
                "greedy": {"max_symbols": config["max_symbols_per_step"]},
            },
        },
    )


class ParakeetRedux(ParakeetTDT):
    """Ternary Parakeet with the standard mlx-audio transcription interface.

    The auxiliary VAD head is not used. Long audio uses the existing explicit
    chunking/streaming interface, rather than Photon's VAD segmentation.
    """

    supports_vad = False

    def __init__(self, config: dict):
        if hasattr(self, "preprocessor_config"):
            return
        path = Path(config["model_path"])
        self._source_config = dict(config)
        self._weight_format = config.get("redux_weight_format", "thrush-ternary-v2")
        if self._weight_format not in ("thrush-ternary-v2", "mlx-2bit"):
            raise ValueError(f"Unsupported Redux weight format: {self._weight_format}")
        vocabulary = _vocabulary(path, config["blank_token_id"])
        super().__init__(_model_args(config, vocabulary))

        names = config["ternary_modules"]
        if len(names) != len(set(names)) or config["ternary_group_size"] != 128:
            raise ValueError(
                "Redux requires unique ternary modules with group size 128."
            )
        manifest = None
        if self._weight_format == "thrush-ternary-v2":
            manifest = json.loads((path / "ternary.json").read_text())
            if (
                manifest.get("format") != "thrush-ternary-v2"
                or manifest.get("names") != "hf"
                or manifest.get("quant") != {"mode": "ternary", "group_size": 128}
                or manifest["packing"].get("base") != 3
                or manifest["packing"].get("elements_per_byte") != 5
                or manifest["packing"].get("code_offset") != 1
            ):
                raise ValueError("Unsupported Redux ternary manifest.")
            entries = manifest["quantized_modules"]
            manifest = {entry["name"]: entry for entry in entries}
            if len(manifest) != len(entries) or set(manifest) != set(names):
                raise ValueError(
                    "Redux config and ternary manifest disagree on modules."
                )

        modules = dict(self.named_modules())
        replacements = []
        self._ternary_shapes = {}
        for name in names:
            target = map_weight_name(name)
            module = modules.get(target)
            is_conv = isinstance(module, nn.Conv1d)
            if not target.startswith("encoder.layers.") or not (
                isinstance(module, nn.Linear)
                or (is_conv and module.weight.shape[1] == 1 and module.groups == 1)
            ):
                raise ValueError(f"Unsupported ternary module: {name}")
            out_features, in_features = module.weight.shape[0], module.weight.shape[-1]
            if in_features % 128 or "bias" in module:
                raise ValueError(f"Unsupported ternary shape or bias: {name}")
            if manifest is not None:
                entry = manifest[name]
                expected = {
                    "in_features": in_features,
                    "out_features": out_features,
                    "group_size": 128,
                    "has_bias": False,
                    "as_conv1d": is_conv,
                }
                if any(entry.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Ternary manifest shape/type mismatch: {name}")
            self._ternary_shapes[name] = (out_features, in_features)
            replacements.append(
                (
                    target,
                    nn.QuantizedLinear(
                        in_features, out_features, bias=False, group_size=128, bits=2
                    ),
                )
            )
        self.update_modules(tree_unflatten(replacements))

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        if self._weight_format == "mlx-2bit":
            converted = dict(weights)
        else:
            converted = {}
            ignored = _VAD_KEYS | {
                f"encoder.layers.{i}.conv.norm.num_batches_tracked"
                for i in range(len(self.encoder.layers))
            }
            for name, value in weights.items():
                if name in ignored or re.fullmatch(
                    r"decoder.lstm.bias_(ih|hh)_l\d+", name
                ):
                    continue
                target = map_weight_name(name)
                if name.endswith(".qweight"):
                    module_name = name.removesuffix(".qweight")
                    if module_name not in self._ternary_shapes:
                        raise ValueError(f"Unexpected ternary weight: {name}")
                    rows, columns = self._ternary_shapes[module_name]
                    if value.shape[0] != rows:
                        raise ValueError(f"Invalid ternary row count: {name}")
                    value = repack_ternary(value, columns)
                    target = map_weight_name(module_name) + ".weight"
                elif name.endswith(".scales"):
                    module_name = name.removesuffix(".scales")
                    if module_name not in self._ternary_shapes or not bool(
                        mx.all(mx.isfinite(value) & (value >= 0))
                    ):
                        raise ValueError(f"Invalid ternary scales: {name}")
                    converted[map_weight_name(module_name) + ".biases"] = -value
                elif name.endswith(".weight") and value.ndim in (3, 4):
                    value = (
                        value.transpose(0, 2, 1)
                        if value.ndim == 3
                        else value.transpose(0, 2, 3, 1)
                    )
                if target in converted:
                    raise ValueError(f"Duplicate Redux weight: {target}")
                converted[target] = value
            for i in range(self.decoder.prediction["dec_rnn"].num_layers):
                ih, hh = f"decoder.lstm.bias_ih_l{i}", f"decoder.lstm.bias_hh_l{i}"
                if ih not in weights or hh not in weights:
                    raise ValueError(f"Missing Redux LSTM biases for layer {i}.")
                converted[f"decoder.prediction.dec_rnn.lstm.{i}.bias"] = (
                    weights[ih] + weights[hh]
                )

        # The shared loader defaults to strict=False. A partial ternary export
        # must never leave randomly initialized encoder or decoder parameters.
        expected = dict(tree_flatten(self.parameters()))
        missing, extra = (
            expected.keys() - converted.keys(),
            converted.keys() - expected.keys(),
        )
        if missing or extra:
            raise ValueError(
                f"Redux weight mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        for name, value in converted.items():
            if value.shape != expected[name].shape:
                raise ValueError(
                    f"Redux weight shape mismatch for {name}: {value.shape} != {expected[name].shape}"
                )
            if expected[name].dtype == mx.uint32 and value.dtype != mx.uint32:
                raise ValueError(f"Packed Redux weights must remain uint32: {name}")
        return converted

    def generate(self, audio, *, dtype=mx.float32, **kwargs):
        return super().generate(audio, dtype=dtype, **kwargs)

    def stream_generate(self, audio, *, dtype=mx.float32, **kwargs):
        return super().stream_generate(audio, dtype=dtype, **kwargs)

    def decode(self, mel: mx.array):
        if mel.ndim == 2:
            mel = mel[None]
        features, lengths = self.encoder(mel)
        mx.eval(features, lengths)
        results = []
        frame_seconds = (
            self.encoder_config.subsampling_factor
            * self.preprocessor_config.hop_length
            / self.preprocessor_config.sample_rate
        )
        for b in range(features.shape[0]):
            length = int(lengths[b])
            hidden, cell = self._make_initial_decoder_state(1, features.dtype)
            last_token, frame = self.blank_id, 0
            hypothesis = []
            # Photon bounds total decode work, and advances blank/zero-duration
            # emissions immediately instead of retrying them at the same frame.
            for _ in range(self.max_symbols * length):
                if frame >= length:
                    break
                token, duration, next_hidden, next_cell = self._compiled_tdt_step(
                    features[b : b + 1, frame : frame + 1],
                    mx.array([[last_token]], dtype=mx.int32),
                    hidden,
                    cell,
                )
                mx.eval(token, duration, next_hidden, next_cell)
                token = int(token)
                duration = self.durations[int(duration)]
                if token == self.blank_id:
                    duration = max(duration, 1)
                else:
                    last_token, hidden, cell = token, next_hidden, next_cell
                    if not tokenizer.is_special_token(token, self.vocabulary):
                        hypothesis.append(
                            AlignedToken(
                                token,
                                start=frame * frame_seconds,
                                duration=duration * frame_seconds,
                                text=tokenizer.decode([token], self.vocabulary),
                            )
                        )
                frame += duration
            results.append(sentences_to_result(tokens_to_sentences(hypothesis)))
        return results
