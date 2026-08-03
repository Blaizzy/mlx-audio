"""CosyVoice3 text-to-speech model.

The model generates speech tokens from text, converts them to mel spectrograms
with a flow decoder, then synthesizes the final waveform with HiFT.
"""

import math
import time
from pathlib import Path
from typing import Generator, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import GenerationResult
from .config import ModelConfig
from .flow import CausalMaskedDiffWithDiT
from .frontend import CosyVoice3FrontEnd
from .hift import CausalHiFTGenerator
from .llm import CosyVoice3LM


class CosyVoice3(nn.Module):
    """CosyVoice3 text-to-speech."""

    # The frontend internally resamples ref_audio to two different rates
    # (16kHz for the speech tokenizer/speaker encoder, 24kHz for the prompt
    # mel) — see frontend.py's S3_SAMPLE_RATE vs self.sample_rate. generate.py
    # otherwise pre-loads ref_audio to a single fixed sample rate as an
    # mx.array before calling generate(); mlx_audio.utils.load_audio() then
    # returns that array unchanged (ignoring the requested sample_rate) on
    # every subsequent call, silently feeding wrong-rate audio into whichever
    # stage didn't get the rate it was loaded at. Preserving the raw path lets
    # the frontend re-load it correctly at each rate it actually needs.
    preserve_ref_audio_path = True

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.llm = CosyVoice3LM(config.llm)
        self.flow = CausalMaskedDiffWithDiT(config.flow)
        self.hift = CausalHiFTGenerator(config.hift)

        # populated by post_load_hook
        self.frontend: Optional[CosyVoice3FrontEnd] = None

    # ------------------------------------------------------------------ #
    # mlx-audio contract
    # ------------------------------------------------------------------ #
    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def sanitize(self, weights: dict) -> dict:
        """Route checkpoint weights to their matching submodules."""
        buckets: dict[str, dict] = {"llm": {}, "flow": {}, "hift": {}}
        passthrough: dict = {}
        for k, v in weights.items():
            for name in buckets:
                if k.startswith(name + "."):
                    buckets[name][k[len(name) + 1 :]] = v
                    break
            else:
                passthrough[k] = v

        out = dict(passthrough)
        for name, sub in (
            ("llm", self.llm),
            ("flow", self.flow),
            ("hift", self.hift),
        ):
            sub_weights = sub.sanitize(buckets[name]) if buckets[name] else {}
            for k, v in sub_weights.items():
                out[f"{name}.{k}"] = v
        return out

    @classmethod
    def post_load_hook(cls, model: "CosyVoice3", model_path: Path) -> "CosyVoice3":
        """Load tokenizer / speaker encoder after weights are in place.

        Speech tokenizer (S3TokenizerV2) is loaded lazily on first use inside
        the frontend, since it ships from a separate HF repo. Optional
        spk2info (preset speakers for SFT mode) is left as a TODO.
        """
        model.frontend = CosyVoice3FrontEnd.from_model_path(
            model_path,
            sample_rate=model.config.sample_rate,
            endofprompt_token=model.config.llm.endofprompt_token,
            use_onnx_speech_tokenizer=model.config.use_onnx_speech_tokenizer,
        )
        return model

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def _drop_overlong_silence(self, speech_ids: list) -> list:
        """Limit consecutive silent or breath tokens before flow decoding."""
        silent = set(self.config.silent_tokens)
        max_run = self.config.max_silent_token_num
        out = []
        run = 0
        for tok in speech_ids:
            if tok in silent:
                run += 1
                if run > max_run:
                    continue
            else:
                run = 0
            out.append(tok)
        return out

    def _tame_tail_hiss(
        self,
        audio: mx.array,
        window_ms: float = 100.0,
        hf_cutoff_hz: float = 4000.0,
        rms_db_threshold: float = -45.0,
        hf_ratio_threshold: float = 0.20,
        fade_ms: float = 60.0,
    ) -> mx.array:
        """Fade out an anomalous high-frequency tail burst.

        LLM speech-token sampling is stochastic; rare trailing token
        sequences make the flow/HiFT decode a loud, hissy ("snake-like")
        burst in the final ~100ms instead of a clean tail. Rather than
        chase the exact bad token pattern, detect the acoustic signature
        directly — a tail window that is both audible (RMS above
        ``rms_db_threshold``) AND unusually bright (energy above
        ``hf_cutoff_hz`` exceeding ``hf_ratio_threshold`` of total FFT
        energy, versus ~1-5% for a normal voiced/silent tail) — and fade it
        to silence instead of leaving the hiss in the output.
        """
        n = int(window_ms / 1000 * self.config.sample_rate)
        if n <= 0 or audio.shape[0] <= n:
            return audio
        seg = audio[-n:]
        rms = mx.sqrt(mx.mean(seg**2) + 1e-12)
        rms_db = 20 * mx.log10(rms + 1e-12)
        if rms_db.item() <= rms_db_threshold:
            return audio

        window = mx.array(
            [0.5 * (1 - math.cos(2 * math.pi * i / n)) for i in range(n)]
        )
        spectrum = mx.abs(mx.fft.rfft(seg * window)) ** 2
        freqs = mx.fft.rfftfreq(n, 1 / self.config.sample_rate)
        hf_ratio = mx.sum(spectrum * (freqs > hf_cutoff_hz)) / (
            mx.sum(spectrum) + 1e-12
        )
        if hf_ratio.item() <= hf_ratio_threshold:
            return audio

        fade_n = min(int(fade_ms / 1000 * self.config.sample_rate), audio.shape[0] - n)
        fade_n = max(fade_n, 0)
        fade_curve = mx.concatenate(
            [
                mx.ones(audio.shape[0] - n - fade_n),
                mx.linspace(1.0, 0.0, fade_n),
                mx.zeros(n),
            ]
        )
        return audio * fade_curve

    # ---- spk2info convenience methods (delegate to frontend) ----

    def add_zero_shot_spk(
        self, prompt_text: str, prompt_wav, spk_id: str
    ) -> bool:
        """Cache the acoustic prompt for *spk_id* so it can be reused."""
        if self.frontend is None:
            raise RuntimeError("frontend not initialized")
        return self.frontend.add_zero_shot_spk(prompt_text, prompt_wav, spk_id)

    def list_spks(self) -> list:
        """Return cached zero-shot speaker ids."""
        if self.frontend is None:
            return []
        return self.frontend.list_spks()

    def save_spkinfo(self, path: str) -> None:
        """Persist cached speaker presets to *path* (safetensors)."""
        if self.frontend is None:
            raise RuntimeError("frontend not initialized")
        self.frontend.save_spkinfo(path)

    def load_spkinfo(self, path: str) -> int:
        """Load speaker presets from a safetensors file saved by save_spkinfo."""
        if self.frontend is None:
            raise RuntimeError("frontend not initialized")
        return self.frontend.load_spkinfo(path)

    # ---- inference ----

    def generate(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        n_timesteps: int = 10,
        sampling: int = 25,
        verbose: bool = False,
        stream: bool = False,
        cross_lingual: bool = False,
        instruct: Optional[str] = None,
        text_frontend: bool = True,
        spk_id: str = "",
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Yield ``GenerationResult`` segments for ``text`` (zero-shot cloning).

        Pipeline:
          1. frontend: text_normalize → tokenize; extract prompt speech tokens,
             prompt mel, and speaker x-vector from ref_audio.
          2. llm.inference(...) -> speech token ids.
          3. flow.inference(tokens, prompt...) -> mel.
          4. hift.inference(mel) -> waveform.
          5. wrap in GenerationResult.

        ``cross_lingual`` excludes reference text and speech tokens from the LLM
        prompt while retaining reference-audio conditioning in later stages.

        ``instruct`` uses instruction text as the LLM prompt and is mutually
        exclusive with ``cross_lingual``.

        ``text_frontend``: when ``True`` (default), ``text``, ``ref_text`` and
        ``instruct`` are normalised before synthesis (Chinese: blank removal,
        bracket replacement; English: number spelling). Set to ``False`` to
        skip normalisation (e.g. when text already contains SSML-style tokens).
        """
        if self.frontend is None:
            raise RuntimeError(
                "frontend not initialized — load via the mlx_audio loader "
                "(post_load_hook) first"
            )
        if ref_audio is None and not (spk_id and spk_id in self.frontend.spk2info):
            raise ValueError(
                "CosyVoice3 requires ref_audio for voice cloning "
                "(or use --spk_id with a previously cached speaker)"
            )
        if instruct is not None and cross_lingual:
            raise ValueError(
                "instruct and cross_lingual are mutually exclusive"
            )

        # Normalize synthesis inputs before constructing frontend features.
        _norm = lambda t: self.frontend.text_normalize(
            t, split=False, text_frontend=text_frontend
        )
        text = _norm(text)
        ref_text = _norm(ref_text or "")
        if instruct is not None:
            instruct = _norm(instruct)

        t0 = time.perf_counter()

        # 1. frontend — pick the right mode
        if instruct is not None:
            model_input = self.frontend.frontend_instruct2(
                tts_text=text,
                instruct=instruct,
                prompt_wav=ref_audio,
            )
        else:
            model_input = self.frontend.frontend_zero_shot(
                tts_text=text,
                prompt_text=ref_text,
                prompt_wav=ref_audio,
                spk_id=spk_id,
            )

        # 2. LLM: text (+prompt) -> speech tokens
        if instruct is not None:
            # Instruction mode excludes reference speech tokens from the LLM prompt.
            llm_prompt_text = model_input["prompt_text"]
            llm_prompt_speech_token = model_input["prompt_speech_token"][:, :0]
        elif cross_lingual:
            llm_prompt_text = model_input["prompt_text"][:, :0]
            llm_prompt_speech_token = model_input["prompt_speech_token"][:, :0]
        else:
            llm_prompt_text = model_input["prompt_text"]
            llm_prompt_speech_token = model_input["prompt_speech_token"]
        speech_ids = self.llm.inference(
            text=model_input["text"],
            prompt_text=llm_prompt_text,
            prompt_speech_token=llm_prompt_speech_token,
            sampling=sampling,
        )
        speech_ids = self._drop_overlong_silence(speech_ids)
        speech_token = mx.array([speech_ids], dtype=mx.int32)
        token_len = mx.array([speech_token.shape[1]], dtype=mx.int32)

        # 3. flow: speech tokens -> mel
        mel = self.flow.inference(
            token=speech_token,
            token_len=token_len,
            prompt_token=model_input["prompt_speech_token"],
            prompt_token_len=mx.array(
                [model_input["prompt_speech_token"].shape[1]], dtype=mx.int32
            ),
            prompt_feat=model_input["prompt_feat"],
            prompt_feat_len=None,
            embedding=model_input["embedding"],
            n_timesteps=n_timesteps,
        )

        # Rescale the mel time axis before vocoding to change playback speed.
        if speed != 1.0:
            from .hift import _linear_interpolate_align_false

            new_t = int(mel.shape[2] / speed)
            mel = _linear_interpolate_align_false(mel, new_t)

        # 4. hift: mel -> waveform
        wav, _ = self.hift.inference(speech_feat=mel)
        mx.eval(wav)
        audio = wav[0] if wav.ndim == 2 else wav
        audio = self._tame_tail_hiss(audio)
        mx.eval(audio)

        # 5. GenerationResult
        t1 = time.perf_counter()
        audio_samples = int(audio.shape[0])
        dur = audio_samples / self.config.sample_rate
        h = int(dur // 3600)
        m = int((dur % 3600) // 60)
        s = int(dur % 60)
        ms = int((dur % 1) * 1000)
        duration_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

        yield GenerationResult(
            audio=audio,
            sample_rate=self.config.sample_rate,
            samples=audio_samples,
            segment_idx=0,
            token_count=len(speech_ids),
            audio_samples={
                "samples": audio_samples,
                "samples-per-sec": round(audio_samples / dur, 2) if dur > 0 else 0,
            },
            audio_duration=duration_str,
            real_time_factor=round(dur / (t1 - t0), 2) if (t1 - t0) > 0 else 0,
            prompt={
                "tokens": len(speech_ids),
                "tokens-per-sec": round(len(speech_ids) / dur, 2) if dur > 0 else 0,
            },
            processing_time_seconds=t1 - t0,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )


# Alias required by the mlx-audio loader (base_load_model expects ``Model``).
Model = CosyVoice3
