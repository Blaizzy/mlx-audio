import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.qwen2 import Qwen2Model, TransformerBlock

from .config import ModelConfig
from .generation import GenerationResult, Sampler, generate_tokens, split_reasoning
from .processor import MiMoAudioProcessor


class LocalTransformer(nn.Module):
    def __init__(self, args, full_attention=False):
        super().__init__()
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.full_attention = full_attention

    def __call__(self, x, cache=None):
        cache = [None] * len(self.layers) if cache is None else cache
        mask = None if self.full_attention else create_attention_mask(x, cache[0])
        for layer, state in zip(self.layers, cache):
            x = layer(x, mask, state)
        return self.norm(x)


@dataclass
class ModelOutput:
    text_logits: mx.array
    local_hidden_states: mx.array


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = "mimo_audio"
        self.model = Qwen2Model(config.qwen_args())
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.input_local_transformer = LocalTransformer(
            config.qwen_args("input"), config.input_full_attention
        )
        self.local_transformer = LocalTransformer(config.qwen_args("output"))
        self.speech_embeddings = [
            nn.Embedding(size, config.input_local_dim)
            for size in config.speech_vocab_sizes
        ]
        self.local_transformer_lm_heads = [
            nn.Linear(config.local_dim, size, bias=False)
            for size in config.speech_vocab_sizes
        ]
        self.speech_embeddings_to_local = (
            nn.Linear(config.input_local_dim, config.local_dim, bias=False)
            if config.input_local_dim != config.local_dim
            else None
        )
        self.speech_group_downcast = nn.Linear(
            config.input_local_dim * config.group_size, config.hidden_size, bias=False
        )
        self.hidden_states_downcast = nn.Linear(
            config.hidden_size, config.local_dim, bias=False
        )
        self._processor = None

    @property
    def sample_rate(self):
        return 24000

    @property
    def layers(self):
        return self.model.layers

    @property
    def processor(self):
        if self._processor is None:
            raise RuntimeError(
                "Load tokenizer metadata with Model.from_pretrained() or the STS loader"
            )
        return self._processor

    @staticmethod
    def post_load_hook(model, model_path):
        from transformers import AutoTokenizer, Qwen2Config

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), config=Qwen2Config())
        processor = MiMoAudioProcessor(tokenizer, model.config)
        codec_path = Path(model_path) / model.config.audio_tokenizer_path
        if codec_path.exists():
            processor.audio_tokenizer_path = str(codec_path)
            processor.audio_tokenizer_revision = None
        model._processor = processor
        return model

    @classmethod
    def from_pretrained(cls, path, *, audio_tokenizer_path=None, **kwargs):
        from mlx_audio.sts.utils import load

        model = load(path, strict=True, **kwargs)
        if audio_tokenizer_path is not None:
            model.processor.audio_tokenizer_path = str(audio_tokenizer_path)
            model.processor.audio_tokenizer_revision = None
        return model

    def make_cache(self):
        return [KVCache() for _ in self.model.layers]

    def sanitize(self, weights):
        return {
            k: v for k, v in weights.items() if not k.endswith("rotary_emb.inv_freq")
        }

    def model_quant_predicate(self, path, module):
        # Establish acoustic parity before introducing quantization into the
        # patch transformers, speech embeddings, or speech heads.
        return path.startswith("model.") or path == "lm_head"

    def _prepare_input_embeds(self, input_ids):
        c = self.config
        batch, channels, length = input_ids.shape
        if channels != c.audio_channels + 1 or length % c.group_size:
            raise ValueError("Expected complete text/audio patches")
        groups = length // c.group_size
        text_ids = input_ids[:, 0, :: c.group_size]
        is_speech = text_ids == c.empty_token_id
        speech = (
            input_ids[:, 1:]
            .reshape(batch, c.audio_channels, groups, c.group_size)
            .transpose(0, 2, 3, 1)
        )
        embeds = None
        for i, embedding in enumerate(self.speech_embeddings):
            value = embedding(speech[..., i])
            value = mx.where(
                (speech[..., i] != c.speech_empty_ids[i])[..., None], value, 0
            )
            embeds = value if embeds is None else embeds + value
        embeds = mx.where(is_speech[..., None, None], embeds, 0)
        embeds = self.input_local_transformer(
            embeds.reshape(batch * groups, c.group_size, c.input_local_dim)
        )
        embeds = embeds.reshape(batch, groups, c.group_size * c.input_local_dim)
        embeds = mx.where(is_speech[..., None], embeds, 0)
        text = self.model.embed_tokens(text_ids)
        text = mx.where(is_speech[..., None], 0, text)
        return text + self.speech_group_downcast(embeds)

    def __call__(self, input_ids, cache=None):
        hidden = self.model(
            None, cache=cache, input_embeddings=self._prepare_input_embeds(input_ids)
        )
        return ModelOutput(
            self.lm_head(hidden[:, -1:]), self.hidden_states_downcast(hidden[:, -1:])
        )

    def local_forward(self, local_embeds, sampler=None):
        c = self.config
        sampler = sampler or Sampler(temperature=0.9)
        cache = [KVCache() for _ in self.local_transformer.layers]
        tokens = [[None] * c.audio_channels for _ in range(c.group_size)]
        for t in range(c.group_size + max(c.delays)):
            hidden = self.local_transformer(local_embeds, cache)
            local_embeds = mx.zeros_like(local_embeds)
            for i, delay in enumerate(c.delays):
                if delay <= t < delay + c.group_size:
                    logits = self.local_transformer_lm_heads[i](hidden[:, -1])
                    token = sampler(logits, [c.speech_empty_ids[i]])
                    tokens[t - delay][i] = token
                    embed = self.speech_embeddings[i](token[:, None])
                    if self.speech_embeddings_to_local is not None:
                        embed = self.speech_embeddings_to_local(embed)
                    local_embeds = local_embeds + embed
            mx.async_eval(local_embeds, [layer.state for layer in cache])
        return mx.stack([mx.stack(row, axis=-1) for row in tokens], axis=1)

    def generate(
        self,
        audio=None,
        *,
        text=None,
        task="spoken_dialogue",
        sample_rate=None,
        messages=None,
        segments=None,
        prompt_examples=None,
        instruction=None,
        instruct=None,
        thinking=False,
        read_text_only=True,
        ref_audio=None,
        ref_sample_rate=None,
        system_prompt=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        temperature=None,
        audio_temperature=0.9,
        top_k=0,
        top_p=None,
        audio_top_k=0,
        audio_top_p=0.95,
        prefill_step_size=256,
        audio_tokenizer_path=None,
        stream=False,
    ):
        """Generate one offline response. Raw audio arrays need sample_rate."""
        if stream:
            raise ValueError(
                "MiMo-Audio streaming is not implemented; use offline generation"
            )
        start = time.perf_counter()
        p = self.processor
        if (
            audio_tokenizer_path is not None
            and str(audio_tokenizer_path) != p.audio_tokenizer_path
        ):
            p.audio_tokenizer_path = str(audio_tokenizer_path)
            p.audio_tokenizer_revision = None
            p._audio_tokenizer = None
        speech_output = task in {
            "tts",
            "spoken_dialogue",
            "few_shot",
            "continuation",
            "completion",
        }
        if task == "tts":
            prompt = p.tts(
                text,
                instruct=instruct,
                read_text_only=read_text_only,
                ref_audio=ref_audio,
                ref_sample_rate=ref_sample_rate,
            )
        elif task in {"asr", "audio_understanding"}:
            prompt = p.understanding(
                audio,
                text or "Please transcribe this audio file",
                sample_rate=sample_rate,
                thinking=thinking,
            )
        elif task == "few_shot":
            if not instruction:
                raise ValueError("Few-shot generation requires an instruction")
            prompt = p.few_shot(instruction, prompt_examples or [], audio, sample_rate)
        elif task == "completion":
            prompt = p.segments(segments or [])
        elif task == "continuation":
            prompt = p.segments(
                [{"audio": audio, "sample_rate": sample_rate, "boundaries": False}]
            )
        elif task in {"spoken_dialogue", "speech_to_text", "text"}:
            if messages is None:
                content = text if task == "text" else audio
                messages = [
                    {"role": "user", "content": content, "sample_rate": sample_rate}
                ]
            prompt = p.dialogue(
                messages,
                speech_input=task != "text",
                speech_output=speech_output,
                thinking=thinking,
                system_prompt=system_prompt,
                ref_audio=ref_audio,
                ref_sample_rate=ref_sample_rate,
            )
        else:
            raise ValueError(f"Unsupported MiMo task: {task}")
        defaults = {
            "asr": (0.0, 1.0),
            "tts": (0.6, 1.0),
            "audio_understanding": (0.3, 0.95),
            "text": (0.4, 0.95),
            "few_shot": (0.0, 1.0),
        }
        temp, nucleus = defaults.get(task, (0.6, 0.95))
        stops = {p.ids["im_end"], p.tokenizer.eos_token_id}
        if speech_output:
            stops.add(p.ids["eostm"])
        tokens, reason = generate_tokens(
            self,
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            stop_tokens=stops,
            global_sampler=Sampler(
                temp if temperature is None else temperature,
                top_k,
                nucleus if top_p is None else top_p,
            ),
            local_sampler=Sampler(audio_temperature, audio_top_k, audio_top_p),
            prefill_step_size=prefill_step_size,
        )
        text_ids = tokens[0, :: self.config.group_size].tolist()
        decoded = p.tokenizer.decode(
            [
                v
                for v in text_ids
                if v
                not in {
                    p.ids["empty"],
                    p.ids["sosp"],
                    p.ids["eosp"],
                    p.ids["sostm"],
                    p.ids["eostm"],
                    p.ids["eot"],
                    *stops,
                }
            ],
            skip_special_tokens=False,
        ).strip()
        think_prefix = p.tokenizer.encode("<think>\n", add_special_tokens=False)
        thinking_prefix = (
            prompt[
                0,
                0,
                -len(think_prefix) * self.config.group_size :: self.config.group_size,
            ].tolist()
            == think_prefix
        )
        response, reasoning = split_reasoning(decoded, thinking_prefix=thinking_prefix)
        mask = tokens[0] == self.config.empty_token_id
        # MLX has no dynamic boolean indexing; determine output frame positions once.
        positions = [i for i, valid in enumerate(mask.tolist()) if valid]
        codes = tokens[1:, mx.array(positions, dtype=mx.int32)]
        waveform = (
            p.audio_tokenizer.decode(codes) if speech_output and positions else None
        )
        return GenerationResult(
            response,
            waveform,
            codes,
            self.sample_rate,
            tokens,
            prompt.shape[-1] // self.config.group_size,
            len(text_ids),
            reason,
            time.perf_counter() - start,
            reasoning,
            decoded,
        )
