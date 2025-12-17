import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult


@dataclass
class ModelConfig(BaseModelArgs):
    model_path: Path = None
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    n_positions: int = 8196
    layer_norm_epsilon: float = 1e-5
    text_tokens_dict_size: int = 50276
    speech_tokens_dict_size: int = 6563
    start_speech_token: int = 6561
    stop_speech_token: int = 6562
    speech_cond_prompt_len: int = 375
    speaker_embed_size: int = 256
    sample_rate: int = 24000


@dataclass
class T3Cond:
    speaker_emb: mx.array
    cond_prompt_speech_tokens: Optional[mx.array] = None
    cond_prompt_speech_emb: Optional[mx.array] = None


class T3CondEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spkr_enc = nn.Linear(
            config.speaker_embed_size, config.hidden_size, bias=True
        )

    def __call__(self, cond: T3Cond):
        cond_spkr = self.spkr_enc(
            cond.speaker_emb.reshape(-1, cond.speaker_emb.shape[-1])
        )[:, None, :]
        cond_prompt = cond.cond_prompt_speech_emb
        if cond_prompt is None:
            cond_prompt = mx.zeros_like(cond_spkr[:, :0])
        return mx.concatenate([cond_spkr, cond_prompt], axis=1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embd = config.hidden_size
        self.n_head = config.num_attention_heads
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim**-0.5
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def __call__(self, x: mx.array, mask=None, cache=None):
        B, L, _ = x.shape
        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)
        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.c_proj = nn.Linear(4 * config.hidden_size, config.hidden_size)

    def __call__(self, x):
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def __call__(self, x, mask=None, cache=None):
        h = x + self.attn(self.ln_1(x), mask, cache)
        return h + self.mlp(self.ln_2(h))


class GPT2Backbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.wpe = nn.Embedding(config.n_positions, config.hidden_size)
        self.h = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def __call__(self, inputs_embeds: mx.array, cache=None):
        _, L, _ = inputs_embeds.shape
        if cache is None:
            cache = [None] * len(self.h)

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        offset = mx.array(offset)
        position_ids = mx.arange(L) + offset[..., None]

        hidden_states = inputs_embeds + self.wpe(position_ids)
        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_dir = Path(config.model_path) if config.model_path else None

        self.cond_enc = T3CondEncoder(config)
        self.text_emb = nn.Embedding(config.text_tokens_dict_size, config.hidden_size)
        self.speech_emb = nn.Embedding(config.speech_tokens_dict_size, config.hidden_size)
        self.tfmr = GPT2Backbone(config)
        self.speech_head = nn.Linear(config.hidden_size, config.speech_tokens_dict_size, bias=True)

        self.start_speech_token = config.start_speech_token
        self.stop_speech_token = config.stop_speech_token

        self.tokenizer = None
        self._torch_modules = None

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def model_type(self):
        return "chatterbox"

    def sanitize(self, weights: dict):
        sanitized = {}
        for k, v in weights.items():
            if not (
                k.startswith("tfmr.")
                or k.startswith("text_emb.")
                or k.startswith("speech_emb.")
                or k.startswith("speech_head.")
                or k.startswith("cond_enc.")
            ):
                continue

            needs_transpose = any(
                tag in k
                for tag in [
                    "c_attn.weight",
                    "c_proj.weight",
                    "c_fc.weight",
                ]
            )
            if needs_transpose:
                v = mx.transpose(v)
            sanitized[k] = v
        return sanitized

    def load_weights(self, weights, strict=True):
        if isinstance(weights, dict):
            weights = list(weights.items())

        filtered = {}
        for k, v in weights:
            if k.startswith("tfmr.wte"):
                continue
            if k.startswith(
                ("tfmr.", "text_emb.", "speech_emb.", "speech_head.", "cond_enc.")
            ):
                filtered[k] = v
        sample_key = next(
            (k for k in filtered if "tfmr.h.0.attn.c_attn.weight" in k), None
        )
        needs_sanitize = True
        expected_shape = (3 * self.config.hidden_size, self.config.hidden_size)
        if sample_key is not None and filtered[sample_key].shape == expected_shape:
            needs_sanitize = False
        if needs_sanitize:
            filtered = self.sanitize(filtered)
        super().load_weights(list(filtered.items()), strict=strict)

    def _encode_text(self, text: str) -> mx.array:
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.model_dir)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.tokenizer = tok

        tokenized = self.tokenizer(
            text,
            return_tensors="np",
            padding=True,
            truncation=True,
        )
        return mx.array(tokenized.input_ids, dtype=mx.int32)

    def _prepare_cond(self, cond: T3Cond) -> Tuple[mx.array, int]:
        cond_emb = self.cond_enc(cond)
        if (
            cond.cond_prompt_speech_tokens is not None
            and cond.cond_prompt_speech_emb is None
        ):
            cond.cond_prompt_speech_emb = self.speech_emb(
                cond.cond_prompt_speech_tokens
            )
            cond_emb = self.cond_enc(cond)
        return cond_emb, cond_emb.shape[1]

    def _prepare_input_embeds(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        speech_tokens: mx.array,
    ):
        cond_emb, _ = self._prepare_cond(t3_cond)
        text_emb = self.text_emb(text_tokens)
        speech_emb = self.speech_emb(speech_tokens)
        if cond_emb.shape[0] != text_emb.shape[0]:
            cond_emb = mx.repeat(cond_emb, text_emb.shape[0], axis=0)

        embeds = mx.concatenate([cond_emb, text_emb, speech_emb], axis=1)
        return embeds

    def inference_turbo(
        self,
        t3_cond: T3Cond,
        text_tokens: mx.array,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = 400,
    ) -> mx.array:
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=min(200, max_gen_len),
        )
        sampler = make_sampler(
            temp=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        speech_start = mx.full(
            (text_tokens.shape[0], 1), self.start_speech_token, dtype=mx.int32
        )
        embeds = self._prepare_input_embeds(
            t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=speech_start
        )

        generated_tokens = []
        history_tokens = []

        for _ in range(max_gen_len):
            hidden_states = self.tfmr(embeds)
            speech_hidden = hidden_states[:, -1:, :]
            logits = self.speech_head(speech_hidden).astype(mx.float32)[:, -1, :]
            tokens_array = (
                mx.array(history_tokens, dtype=mx.int32)
                if len(history_tokens) > 0
                else mx.array([], dtype=mx.int32)
            )
            for proc in logits_processors:
                logits = proc(tokens_array, logits)
            logprobs = nn.log_softmax(logits, axis=-1)
            next_token = sampler(logprobs)[:, None]

            if int(next_token[0, 0].item()) == self.stop_speech_token:
                break

            generated_tokens.append(next_token)
            history_tokens.append(int(next_token[0, 0].item()))

            next_embed = self.speech_emb(next_token)
            embeds = mx.concatenate([embeds, next_embed], axis=1)

        if not generated_tokens:
            return mx.zeros((1, 0), dtype=mx.int32)

        return mx.concatenate(generated_tokens, axis=1)

    def _ensure_torch_modules(self):
        if self._torch_modules is not None:
            return self._torch_modules
        try:
            import torch
            import librosa
            from safetensors.torch import load_file as load_torch_file

            from chatterbox.models.s3gen import S3Gen
            from chatterbox.models.s3gen.const import S3GEN_SR, S3GEN_SIL
            from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
            from chatterbox.models.voice_encoder import VoiceEncoder
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "Chatterbox-Turbo requires optional torch dependencies. "
                "Install with: pip install git+https://github.com/resemble-ai/chatterbox.git"
            ) from exc

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32

        ve = VoiceEncoder().to(device).eval()
        ve_state = load_torch_file(self.model_dir / "ve.safetensors")
        ve.load_state_dict(ve_state)

        s3gen = S3Gen(meanflow=True).to(device).eval()
        s3_state = load_torch_file(self.model_dir / "s3gen_meanflow.safetensors")
        s3gen.load_state_dict(s3_state, strict=True)

        tokenizer = S3Tokenizer().to(device)

        self._torch_modules = dict(
            torch=torch,
            librosa=librosa,
            ve=ve,
            s3gen=s3gen,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            S3_SR=S3_SR,
            S3GEN_SR=S3GEN_SR,
            S3GEN_SIL=S3GEN_SIL,
        )
        return self._torch_modules

    def _prepare_conditionals(
        self,
        modules: dict,
        ref_audio: Path,
        norm_loudness: bool = True,
    ):
        torch = modules["torch"]
        librosa = modules["librosa"]
        s3gen = modules["s3gen"]
        tokenizer = modules["tokenizer"]
        ve = modules["ve"]
        device = modules["device"]
        S3_SR = modules["S3_SR"]
        S3GEN_SR = modules["S3GEN_SR"]

        if isinstance(ref_audio, (str, Path)):
            ref_wav_24, _ = librosa.load(ref_audio, sr=S3GEN_SR)
        else:
            ref_wav_24 = np.array(ref_audio, dtype=np.float32)
            if ref_wav_24.ndim > 1:
                ref_wav_24 = ref_wav_24.reshape(-1)
            if self.sample_rate != S3GEN_SR:
                ref_wav_24 = librosa.resample(
                    ref_wav_24, orig_sr=self.sample_rate, target_sr=S3GEN_SR
                )
        if len(ref_wav_24) / S3GEN_SR < 5.0:
            raise ValueError("Reference audio must be at least 5 seconds.")

        if norm_loudness:
            rms = np.sqrt(np.mean(ref_wav_24**2))
            if rms > 0:
                ref_wav_24 = ref_wav_24 / max(rms, 1e-5) * 0.2

        ref_wav_16 = librosa.resample(ref_wav_24, orig_sr=S3GEN_SR, target_sr=S3_SR)

        ref_tensor_24 = torch.from_numpy(ref_wav_24).to(device=device, dtype=modules["dtype"])
        ref_dict = s3gen.embed_ref(ref_tensor_24, S3GEN_SR, device=device)

        cond_tokens, _ = tokenizer.forward(
            [torch.from_numpy(ref_wav_16[: 15 * S3_SR]).to(device=device)],
            max_len=self.config.speech_cond_prompt_len,
        )

        ve_embed = torch.from_numpy(
            ve.embeds_from_wavs([ref_wav_16], sample_rate=S3_SR)
        ).to(device=device, dtype=modules["dtype"])
        ve_embed = ve_embed.mean(dim=0, keepdim=True)

        cond = T3Cond(
            speaker_emb=mx.array(ve_embed.cpu().numpy(), dtype=mx.float32),
            cond_prompt_speech_tokens=mx.array(
                cond_tokens.cpu().numpy(), dtype=mx.int32
            ),
        )
        return cond, ref_dict

    def generate(
        self,
        text: str,
        ref_audio: Optional[Path] = None,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_tokens: int = 400,
        **_,
    ) -> Generator[GenerationResult, None, None]:
        if ref_audio is None:
            raise ValueError("Chatterbox-Turbo requires --ref_audio for voice cloning.")

        start_time = time.perf_counter()
        modules = self._ensure_torch_modules()
        t3_cond, ref_dict = self._prepare_conditionals(modules, ref_audio)

        text_tokens = self._encode_text(text)
        speech_tokens = self.inference_turbo(
            t3_cond,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_gen_len=max_tokens,
        )

        torch = modules["torch"]
        s3gen = modules["s3gen"]
        device = modules["device"]
        silence_token = modules["S3GEN_SIL"]
        if speech_tokens.size == 0:
            raise RuntimeError("No speech tokens were generated.")

        speech_tokens_np = np.array(speech_tokens.tolist(), dtype=np.int64)
        speech_tokens_np = speech_tokens_np[speech_tokens_np < self.start_speech_token]
        if speech_tokens_np.size == 0:
            raise RuntimeError("No valid speech tokens were generated.")
        speech_tokens_np = speech_tokens_np.reshape(1, -1)
        silence = np.array([silence_token, silence_token, silence_token], dtype=np.int64)
        speech_tokens_np = np.concatenate([speech_tokens_np, silence[None, :]], axis=1)
        speech_tokens_torch = torch.from_numpy(speech_tokens_np).to(device=device)

        wav, _ = s3gen.inference(
            speech_tokens=speech_tokens_torch,
            ref_dict=ref_dict,
            n_cfm_timesteps=2,
        )

        audio = wav.squeeze(0).detach().cpu().numpy()
        samples = audio.shape[0]
        sample_rate = self.sample_rate
        audio_duration_seconds = samples / sample_rate
        elapsed_time = time.perf_counter() - start_time
        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        result = GenerationResult(
            audio=mx.array(audio, dtype=mx.float32),
            samples=samples,
            sample_rate=sample_rate,
            segment_idx=0,
            token_count=int(speech_tokens.shape[1]),
            audio_duration=duration_str,
            real_time_factor=audio_duration_seconds / max(elapsed_time, 1e-5),
            prompt={
                "tokens": int(text_tokens.shape[1]),
                "tokens-per-sec": (
                    round(text_tokens.shape[1] / elapsed_time, 2)
                    if elapsed_time > 0
                    else 0
                ),
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": (
                    round(samples / elapsed_time, 2) if elapsed_time > 0 else 0
                ),
            },
            processing_time_seconds=elapsed_time,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )
        yield result

    def model_quant_predicate(self, p, _m):
        return p.startswith("tfmr") or p.startswith("text_emb") or p.startswith("speech_emb") or p.startswith("speech_head") or p.startswith("cond_enc")
