"""Ming Audio's separately packaged Qwen2/flow-matching speech decoder."""

import json
from pathlib import Path

import mlx.core as mx

from mlx_audio.lm.models.cache import KVCache
from mlx_audio.tts.models.bailingmm.bailingmm import Model as BailingModel
from mlx_audio.tts.models.bailingmm.bailingmm import (
    get_epss_timesteps,
    ming3_instruction,
)
from mlx_audio.tts.models.lingv3_tts.flow import FlowLoss
from mlx_audio.tts.models.lingv3_tts.speaker import SPEAKER_SUBPATH, SpeakerEncoder


class TalkerFlowLoss(FlowLoss):
    """The bundled Talker uses device-dtype SDE steps, including terminal noise."""

    def sample(
        self,
        hidden,
        history,
        *,
        cfg=2.0,
        patch_size=1,
        sigma=0.25,
        temperature=0.0,
        steps=10,
    ):
        if cfg < 0 or sigma < 0 or temperature < 0 or steps < 1:
            raise ValueError("Invalid Talker flow sampling parameters")
        dtype = hidden.dtype
        state = mx.random.normal(
            (hidden.shape[0], patch_size, self.latent_dim), dtype=dtype
        )
        times = get_epss_timesteps(steps, dtype)
        times = times - (mx.cos(mx.array(mx.pi / 2, dtype=dtype) * times) - 1 + times)
        history = history.astype(dtype)
        for index in range(steps):
            # Talker's CFG always evaluates both branches, even at CFG zero/one.
            outputs = self.cfm.model.forward_with_cfg(
                state, times[index], hidden, 2.0, history, patch_size
            )
            conditional, unconditional = mx.split(outputs, 2, axis=0)
            prediction = conditional + (conditional - unconditional) * cfg
            dt = times[index + 1] - times[index]
            state = state + prediction * dt
            noise = mx.random.normal(state.shape, dtype=dtype)
            state = state + sigma * temperature**0.5 * mx.sqrt(mx.abs(dt)) * noise
        return state


class Talker(BailingModel):
    def _speaker_embeddings(self, spk_emb, use_zero_spk_emb):
        if use_zero_spk_emb or spk_emb is None:
            return super()._speaker_embeddings(spk_emb, use_zero_spk_emb)
        embedding = mx.array(spk_emb, dtype=mx.float32)
        if embedding.size != 192 or not bool(mx.all(mx.isfinite(embedding))):
            raise ValueError("Talker requires one finite 192-value speaker embedding")
        return [
            self.spk_head(embedding.reshape(1, 192).astype(self.spk_head.weight.dtype))[
                0
            ]
        ]

    @classmethod
    def from_pretrained(cls, path):
        from transformers import PreTrainedTokenizerFast

        path = Path(path)
        settings = json.loads((path / "config.json").read_text())
        llm = json.loads((path / "llm/config.json").read_text())
        vae = json.loads((path / "vae/config.json").read_text())
        model = cls(
            {
                "model_type": "dense",
                "llm_config": llm,
                "audio_tokenizer_config": vae,
                "ditar_config": {
                    **{
                        key: value
                        for key, value in settings["flowmodel"].items()
                        if key not in {"in_channels", "llm_cond_dim", "z_channels"}
                    },
                    "patch_size": settings["patch_size"],
                    "history_patch_size": settings["history_patch_size"],
                },
                "aggregator_config": {
                    key: value
                    for key, value in settings["aggregator"].items()
                    if key not in {"in_channels", "llm_input_dim"}
                },
            }
        )
        model.flowloss = TalkerFlowLoss(
            model.latent_dim, model.llm_args.hidden_size, settings["flowmodel"]
        )
        weights = {}
        for key, value in mx.load(str(path / "model.safetensors")).items():
            if key.startswith("model."):
                key = "model.model." + key[len("model.") :]
            elif key.startswith("aggregator."):
                key = "linear_proj_audio." + key[len("aggregator.") :]
            elif key.startswith("cfm."):
                key = "flowloss." + key
            weights[key] = value
        weights.update(
            {
                "audio." + key: value
                for key, value in mx.load(str(path / "vae/model.safetensors")).items()
            }
        )
        weights = model.sanitize(weights)
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())
        model.eval()
        model.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(path / "llm"), trust_remote_code=False, local_files_only=True
        )
        model._speaker_encoder = None
        model._speaker_path = path / SPEAKER_SUBPATH
        return model

    def sample(
        self,
        prompt,
        text,
        instruction=None,
        prompt_text=None,
        max_decode_steps=1000,
        cfg=2.0,
        sigma=0.25,
        temperature=0.0,
        flow_steps=10,
        prompt_latent=None,
        spk_emb=None,
    ):
        _, embeds = self._prepare_input_embed(
            prompt, text, instruction, prompt_latent, prompt_text, spk_emb
        )
        if embeds.shape[1] + max_decode_steps > self.llm_args.max_position_embeddings:
            raise ValueError("Talker prompt and generation exceed the context budget")
        cache = [KVCache() for _ in self.layers]
        if max_decode_steps < 1:
            raise ValueError("max_decode_steps must be positive")
        history = mx.zeros((1, self.history_patch_size, self.latent_dim), embeds.dtype)
        if prompt_latent is not None:
            available = min(history.shape[1], prompt_latent.shape[1])
            history[:, -available:] = prompt_latent[:, -available:]
        for step in range(max_decode_steps):
            hidden = self._forward_llm_hidden(embeds, cache)[:, -1:]
            latent = self.flowloss.sample(
                hidden,
                history,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                steps=flow_steps,
                patch_size=self.patch_size,
            )
            stop = mx.softmax(self.stop_head(hidden).astype(mx.float32), axis=-1)[
                0, 0, 1
            ]
            last = step > 3 and bool(stop > 0.5)
            yield latent, last
            if last:
                break
            embeds = self.linear_proj_audio(latent.astype(hidden.dtype))
            history = mx.concatenate([history[:, self.patch_size :], latent], axis=1)

    def generate(
        self,
        text,
        voice="US_James",
        ref_audio=None,
        ref_text=None,
        spk_emb=None,
        **kwargs,
    ):
        if ref_text is not None and ref_audio is None:
            raise ValueError("Talker reference text requires reference audio")
        if isinstance(ref_audio, Path):
            ref_audio = str(ref_audio)
        instruction = None
        if ref_audio is None and spk_emb is None:
            instruction = ming3_instruction(ip=voice or "US_James")
        elif spk_emb is None:
            if self._speaker_encoder is None:
                self._speaker_encoder = SpeakerEncoder(self._speaker_path)
            spk_emb = self._speaker_encoder(ref_audio, self.sample_rate)
        return super().generate(
            text=text,
            voice=None,
            instruct=instruction,
            ref_audio=ref_audio,
            ref_text=ref_text,
            spk_emb=spk_emb,
            **kwargs,
        )
