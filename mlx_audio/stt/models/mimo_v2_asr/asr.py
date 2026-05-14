import json
import time
from glob import glob
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.stt.models.base import STTOutput
from mlx_audio.utils import get_model_path, load_weights

from .audio_encoder import AudioEncoder, AudioEncoderConfig
from .config import MiMoAudioConfig
from .mel import SAMPLE_RATE as AUDIO_SAMPLE_RATE, log_mel_spectrogram
from .model import MiMoAudioMLX, MiMoSampler
from .prompt import build_asr_prompt


class Model(nn.Module):
    """End-to-end MiMo-V2.5-ASR pipeline for MLX-Audio."""

    def __init__(self, config: MiMoAudioConfig):
        super().__init__()
        self.config = config
        self.audio_encoder = AudioEncoder(AudioEncoderConfig())
        self.model = MiMoAudioMLX(config)

        self._tokenizer = None
        self._eos_token_id = None
        self._pad_token_id = None

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        return dict(weights)

    def load_weights_from_path(
        self,
        model_path: Path,
        *,
        config: dict,
        strict: bool = False,
    ) -> None:
        quantization = config.get("quantization_config") or config.get("quantization")
        if quantization:
            from mlx_lm.utils import quantize_model

            self.model, _ = quantize_model(
                self.model,
                config,
                group_size=quantization.get("group_size"),
                bits=quantization.get("bits"),
                mode=quantization.get("mode", "affine"),
            )

        for weight_file in sorted(glob(str(model_path / "model*.safetensors"))):
            self.model.load_weights(
                list(self.sanitize(mx.load(weight_file)).items()),
                strict=strict,
            )
            mx.eval(self.model.parameters())

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        from transformers import AutoTokenizer

        model._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        model._eos_token_id = model._tokenizer.eos_token_id
        model._pad_token_id = model._tokenizer.pad_token_id or model._eos_token_id

        audio_tokenizer_dir = cls._resolve_audio_tokenizer_dir(model_path)
        with open(audio_tokenizer_dir / "config.json") as f:
            tokenizer_config = json.load(f)

        model.audio_encoder = AudioEncoder(
            AudioEncoderConfig.from_dict(tokenizer_config)
        )
        audio_encoder_weights = model._sanitize_audio_encoder_weights(
            load_weights(audio_tokenizer_dir)
        )
        model.audio_encoder.load_weights(
            list(audio_encoder_weights.items()),
            strict=False,
        )
        mx.eval(model.audio_encoder.parameters())
        return model

    @staticmethod
    def _resolve_audio_tokenizer_dir(model_path: Path) -> Path:
        def _has_weights(path: Path) -> bool:
            return any(path.glob("*.safetensors"))

        manifest_path = model_path / "mlx_manifest.json"
        manifest = None
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)

        if manifest:
            rel_dir = manifest.get("audio_tokenizer_dir")
            if rel_dir:
                candidate = (model_path / rel_dir).resolve()
                if candidate.exists():
                    return candidate

            repo = manifest.get("audio_tokenizer_repo")
            if repo:
                repo_path = get_model_path(repo)
                if not _has_weights(repo_path):
                    repo_path = get_model_path(repo, force_download=True)
                return repo_path

        sibling = (model_path.parent / "MiMo-Audio-Tokenizer").resolve()
        if sibling.exists():
            return sibling

        raise FileNotFoundError(
            "Unable to resolve MiMo audio tokenizer directory from "
            "mlx_manifest.json or a sibling MiMo-Audio-Tokenizer directory."
        )

    @staticmethod
    def _sanitize_audio_encoder_weights(
        hf_weights: Dict[str, mx.array],
    ) -> Dict[str, mx.array]:
        out = {}

        for key, tensor in hf_weights.items():
            new_key = key
            new_tensor = tensor

            if key.startswith("decoder."):
                continue

            if new_key.startswith("encoder."):
                new_key = new_key[len("encoder.") :]

            if new_key.endswith(".weight") and tensor.ndim == 3:
                if tensor.shape[-1] <= 8 and tensor.shape[1] > tensor.shape[-1]:
                    new_tensor = tensor.transpose(0, 2, 1)

            if "down_sample_layer.0" in new_key:
                new_key = new_key.replace("down_sample_layer.0", "down_sample")

            if "_codebook." in new_key:
                new_key = new_key.replace("_codebook.", "codebook.")

            if any(x in new_key for x in (".cluster_size", ".embed_avg", ".inited")):
                continue

            out[new_key] = new_tensor

        for key in list(out.keys()):
            if ".self_attn.k_proj.weight" in key:
                bias_key = key.replace(".weight", ".bias")
                if bias_key not in out:
                    out[bias_key] = mx.zeros((out[key].shape[0],))

        return out

    def _normalize_language(self, language: Optional[str]) -> str:
        if language is None:
            return "auto"
        lowered = language.lower()
        if lowered in {"zh", "zh-cn", "chinese", "mandarin"}:
            return "zh"
        if lowered in {"en", "en-us", "english"}:
            return "en"
        return "auto"

    def _clean_text(self, text: str) -> str:
        return (
            text.replace("<|empty|>", "")
            .replace("<|eot|>", "")
            .replace("<|eostm|>", "")
            .replace("<chinese>", "")
            .replace("<english>", "")
            .strip()
        )

    def _encode_audio_codes(self, audio) -> mx.array:
        mel = log_mel_spectrogram(audio)
        codes = self.audio_encoder.encode(mel, n_q=self.config.audio_channels)
        mx.eval(codes)

        audio_codes = codes.transpose(1, 0).reshape(-1)
        total_needed = self.config.group_size * self.config.audio_channels
        remainder = audio_codes.shape[0] % total_needed
        if remainder != 0:
            pad_len = total_needed - remainder
            last_frame = audio_codes[-self.config.audio_channels :]
            padding = mx.tile(last_frame, (pad_len // self.config.audio_channels,))
            audio_codes = mx.concatenate([audio_codes, padding[:pad_len]])
        return audio_codes

    def generate(
        self,
        audio,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        language: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> STTOutput:
        del kwargs
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized. Call post_load_hook first.")

        if isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            audio = load_audio(audio, sr=AUDIO_SAMPLE_RATE)
        elif not isinstance(audio, mx.array):
            audio = mx.array(audio)

        start_time = time.time()
        resolved_language = self._normalize_language(language)
        audio_codes = self._encode_audio_codes(audio)
        prompt = build_asr_prompt(
            audio_codes=audio_codes,
            text_token_ids=[],
            config=self.config,
            tokenizer=self._tokenizer,
            language=resolved_language,
        )

        global_sampler = MiMoSampler(
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5) if temperature > 0 else 1.0,
            top_k=top_k,
            top_p=top_p,
        )
        local_sampler = MiMoSampler(do_sample=False)

        generated = self.model.generate(
            prompt[None],
            max_new_tokens=max_tokens,
            global_sampler=global_sampler,
            local_sampler=local_sampler,
            stop_tokens=[self._eos_token_id, self.config.eot_idx],
        )[0]

        prompt_len = prompt.shape[1]
        text_tokens = generated[0, prompt_len :: self.config.group_size]
        token_list = text_tokens.tolist()
        while token_list and token_list[-1] in {
            self._eos_token_id,
            self.config.eot_idx,
            self.config.eostm_idx,
        }:
            token_list.pop()

        text = self._clean_text(
            self._tokenizer.decode(token_list, skip_special_tokens=False)
        )
        total_time = time.time() - start_time
        audio_duration = float(audio.shape[0]) / float(AUDIO_SAMPLE_RATE)

        return STTOutput(
            text=text,
            segments=[{"text": text, "start": 0.0, "end": audio_duration}],
            language=resolved_language if resolved_language != "auto" else None,
            prompt_tokens=int(prompt.shape[1]),
            generation_tokens=len(token_list),
            total_tokens=int(prompt.shape[1]) + len(token_list),
            total_time=total_time,
            prompt_tps=(float(prompt.shape[1]) / total_time) if total_time > 0 else 0.0,
            generation_tps=(len(token_list) / total_time) if total_time > 0 else 0.0,
        )


MiMoASR = Model
