import json
import time
from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx

from mlx_audio.stt.models.base import STTOutput
from mlx_audio.utils import get_model_path

from .audio_encoder import AudioEncoder, AudioEncoderConfig
from .config import MiMoAudioConfig
from .mel import SAMPLE_RATE as AUDIO_SAMPLE_RATE, log_mel_spectrogram
from .model import MiMoAudioMLX, MiMoSampler
from .prompt import build_asr_prompt
from .weight_loader import (
    load_hf_weights,
    sanitize_asr_weights,
    sanitize_audio_encoder_weights,
)


class MiMoASR:
    """End-to-end MiMo-V2.5-ASR pipeline for MLX-Audio."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        audio_tokenizer_dir: Optional[Union[str, Path]] = None,
        *,
        model_path_hint: Optional[str] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
    ):
        self.model_dir = Path(model_dir)
        self.model_path_hint = model_path_hint or str(model_dir)
        self.revision = revision
        self.force_download = force_download

        self.manifest = self._load_manifest()
        self.audio_tokenizer_dir = self._resolve_audio_tokenizer_dir(audio_tokenizer_dir)

        with open(self.model_dir / "tokenizer_config.json") as f:
            self.tokenizer_config = json.load(f)

        self._init_tokenizer()
        self._build_audio_encoder()
        self._build_asr_model()

    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        *,
        audio_tokenizer_dir: Optional[Union[str, Path]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
    ) -> "MiMoASR":
        if isinstance(model_path, Path):
            model_dir = model_path
            model_hint = str(model_path)
        elif Path(model_path).exists():
            model_dir = Path(model_path)
            model_hint = model_path
        else:
            model_dir = get_model_path(
                model_path,
                revision=revision,
                force_download=force_download,
            )
            model_hint = model_path

        return cls(
            model_dir=model_dir,
            audio_tokenizer_dir=audio_tokenizer_dir,
            model_path_hint=model_hint,
            revision=revision,
            force_download=force_download,
        )

    def _load_manifest(self) -> Optional[dict[str, Any]]:
        manifest_path = self.model_dir / "mlx_manifest.json"
        if not manifest_path.exists():
            return None
        with open(manifest_path) as f:
            return json.load(f)

    def _resolve_audio_tokenizer_dir(
        self,
        explicit_dir: Optional[Union[str, Path]],
    ) -> Path:
        def _has_weights(path: Path) -> bool:
            return any(path.glob("*.safetensors"))

        if explicit_dir is not None:
            return Path(explicit_dir).expanduser().resolve()

        if self.manifest:
            rel_dir = self.manifest.get("audio_tokenizer_dir")
            if rel_dir:
                candidate = (self.model_dir / rel_dir).resolve()
                if candidate.exists():
                    return candidate

            repo = self.manifest.get("audio_tokenizer_repo")
            if repo:
                repo_path = get_model_path(
                    repo,
                    revision=self.revision,
                    force_download=self.force_download,
                )
                if not _has_weights(repo_path) and not self.force_download:
                    repo_path = get_model_path(
                        repo,
                        revision=self.revision,
                        force_download=True,
                    )
                return repo_path

        sibling = (self.model_dir.parent / "MiMo-Audio-Tokenizer").resolve()
        if sibling.exists():
            return sibling

        raise FileNotFoundError(
            "Unable to resolve MiMo audio tokenizer directory from explicit path, "
            "mlx_manifest.json, or sibling MiMo-Audio-Tokenizer directory."
        )

    def _init_tokenizer(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    def _build_audio_encoder(self):
        with open(self.audio_tokenizer_dir / "config.json") as f:
            tokenizer_config = json.load(f)

        encoder_config = AudioEncoderConfig.from_dict(tokenizer_config)
        self.audio_encoder = AudioEncoder(encoder_config)

        raw_weights = load_hf_weights(self.audio_tokenizer_dir)
        sanitized = sanitize_audio_encoder_weights(raw_weights)
        self._load_into(self.audio_encoder, sanitized)

    def _build_asr_model(self):
        with open(self.model_dir / "config.json") as f:
            raw_config = json.load(f)

        self.config = MiMoAudioConfig.from_dict(raw_config)
        self.model = MiMoAudioMLX(self.config)
        self._apply_quantization_if_needed(raw_config)

        for weight_file in sorted(self.model_dir.glob("model*.safetensors")):
            sanitized = sanitize_asr_weights(mx.load(str(weight_file)))
            self._load_into(self.model, sanitized)

    def _apply_quantization_if_needed(self, raw_config: dict):
        quantization = raw_config.get("quantization_config") or raw_config.get("quantization")
        if not quantization:
            return
        if not isinstance(quantization, dict) or "bits" not in quantization:
            raise ValueError(f"Unsupported quantization config: {quantization!r}")

        from mlx_lm.utils import quantize_model

        self.model, _ = quantize_model(
            self.model,
            raw_config,
            group_size=quantization.get("group_size"),
            bits=quantization.get("bits"),
            mode=quantization.get("mode", "affine"),
        )

    @staticmethod
    def _load_into(model, weights):
        from mlx.nn.utils import tree_flatten, tree_unflatten

        flat = tree_flatten(model.parameters())
        merged = []
        for key, value in flat:
            merged.append((key, weights.get(key, value)))
        model.update(tree_unflatten(merged))
        mx.eval(model.parameters())

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
            tokenizer=self.tokenizer,
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
            stop_tokens=[self.eos_token_id, self.config.eot_idx],
        )[0]

        prompt_len = prompt.shape[1]
        text_tokens = generated[0, prompt_len :: self.config.group_size]
        token_list = text_tokens.tolist()
        while token_list and token_list[-1] in {
            self.eos_token_id,
            self.config.eot_idx,
            self.config.eostm_idx,
        }:
            token_list.pop()

        text = self._clean_text(
            self.tokenizer.decode(token_list, skip_special_tokens=False)
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
