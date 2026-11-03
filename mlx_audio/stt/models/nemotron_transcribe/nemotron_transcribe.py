"""Nemotron 3.5 Transcribe: speech/speaker experts and a Nemotron-H decoder."""

import re
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten

from mlx_audio.lm.generate import StreamingDetokenizer, generate_step
from mlx_audio.lm.sample_utils import make_repetition_penalty, make_sampler
from mlx_audio.stt.models.base import STTOutput
from mlx_audio.stt.utils import load_audio, merge_hotwords, wired_limit

from .config import ModelConfig
from .language_model import LanguageModel
from .perception import Perception
from .speculative import speculative_generate_step
from .timestamps import CTCTimestamp, align_words, load_alignment_tokenizer

ASR_PROMPT = (
    "Produce a verbatim transcript of the audio. Preserve named entities, "
    "abbreviations, numbers, dates, measurements, acronyms, and technical terms "
    "as clearly as possible. Keep the same language as spoken and do not translate."
)
SPEAKER_PROMPT = (
    "Transcribe this audio with punctuation and capitalization. Include all "
    "spoken words and verbal fillers. Write numbers in their standard written form. "
    "Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> <spk:4> <spk:5> <spk:6> "
    "<spk:7> on the transcription."
)
RICH_PROMPT = (
    "Transcribe this audio with punctuation and capitalization. Include all "
    "spoken words and verbal fillers. Write numbers in their standard written form. "
    "Use <ev:*> and <sty:*> special tokens to recognize all non-speech verbatims "
    "and speaking style."
)


def resolve_prompt(task, prompt, language, hotwords, audio_tag):
    if task not in {"asr", "transcribe", "translate", "saa", "rich", "qa"}:
        raise ValueError(f"Unsupported task: {task}")
    if prompt is None:
        if task == "translate":
            if not language:
                raise ValueError("Translation requires a target language")
            prompt = f"Translate the speech in this audio into {language}."
        elif task == "saa":
            prompt = SPEAKER_PROMPT
        elif task == "rich":
            prompt = RICH_PROMPT
        elif task == "qa":
            prompt = audio_tag
        else:
            prompt = ASR_PROMPT
    if hotwords:
        words = merge_hotwords(None, hotwords)
        if words:
            prompt += f"\nContextual vocabulary: {words}"
    count = prompt.count(audio_tag)
    if count > 1:
        raise ValueError("Pass exactly one audio placeholder for one recording")
    return prompt if count else f"{prompt} {audio_tag}"


def speaker_segments(text):
    """Keep native markup in text; expose speaker turns without inventing times."""
    matches = list(re.finditer(r"<spk:([0-7])>", text))
    return [
        {
            "speaker": int(match[1]),
            "text": text[
                match.end() : (
                    matches[i + 1].start() if i + 1 < len(matches) else len(text)
                )
            ].strip(),
        }
        for i, match in enumerate(matches)
    ] or None


@dataclass
class StreamingResult:
    text: str
    is_final: bool = False
    start_time: float | None = None
    end_time: float | None = None
    language: str | None = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    segments: list[dict] | None = None
    speculative_stats: dict | None = None


@dataclass
class TranscriptionOutput(STTOutput):
    speculative_stats: dict | None = None


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.llm = LanguageModel(config.text_config, config.mtp)
        self.perception = Perception(config)
        self.ctc_timestamp = (
            CTCTimestamp(config.ctc_timestamps) if config.ctc_timestamps else None
        )
        self._tokenizer = None
        self._alignment_tokenizer = None

    @property
    def sample_rate(self):
        return self.config.processor_config.sampling_rate

    @property
    def layers(self):
        return self.llm.layers

    def make_cache(self):
        return self.llm.make_cache()

    def __call__(self, inputs=None, cache=None, input_embeddings=None):
        return self.llm.lm_head(
            self.llm.backbone(inputs, cache=cache, input_embeddings=input_embeddings)
        )

    def model_quant_predicate(self, path, module):
        return path.startswith("llm.")

    @property
    def cast_predicate(self):
        return lambda key: not key.endswith(
            (".A_log", ".D", ".dt_bias", ".e_score_correction_bias", ".gate.weight")
        )

    def sanitize(self, weights):
        sanitized = {}
        for key, value in weights.items():
            if key.endswith("._extra_state") or key in {
                "ctc_timestamp.tokenizer.model_proto",
                "ctc_timestamp.decoder.upsample_factor",
            }:
                continue
            if key.startswith(("backbone.", "lm_head.")):
                key = "llm." + key
            key = key.replace("llm.model.embed_tokens.", "llm.backbone.embeddings.")
            key = key.replace("llm.model.norm.", "llm.backbone.norm_f.")
            key = key.replace("llm.model.layers.", "llm.backbone.layers.")
            key = key.replace("._fp32_params.", ".")
            key = key.replace(".ffn.net.0.", ".ffn.linear1.")
            key = key.replace(".ffn.net.3.", ".ffn.linear2.")
            key = key.replace(".preprocessor.featurizer.", ".preprocessor.")
            if key.endswith(".experts.gate_and_up_projs"):
                key = key.replace(
                    ".experts.gate_and_up_projs", ".switch_mlp.fc1.weight"
                )
                value = value.transpose(0, 2, 1)
            elif key.endswith(".experts.down_projs"):
                key = key.replace(".experts.down_projs", ".switch_mlp.fc2.weight")
                value = value.transpose(0, 2, 1)
            elif (
                key.endswith(".conv1d.weight")
                and value.shape[1] == 1
                or key.endswith(".subpixel_upsample.weight")
                and value.shape[-1] == 3
            ):
                value = value.transpose(0, 2, 1)
            elif key == "ctc_timestamp.decoder.upsample.weight" and value.shape[
                -1
            ] == self.config.ctc_timestamps["decoder_config"].get(
                "upsampling_factor", 1
            ):
                value = value.transpose(1, 2, 0)
            elif (
                key == "ctc_timestamp.decoder.decoder_layers.0.weight"
                and value.ndim == 3
            ):
                value = value.squeeze(-1)
            if key.startswith("llm.") and key.endswith(".gate.weight"):
                # Automodel computes routing in FP32 even with BF16 weights.
                # This small FP32 matrix promotes the shared backbone's matmul.
                value = value.astype(mx.float32)
            sanitized[key] = value
        # ModelOpt exports store separate up/down matrices for each expert.
        # Their NVFP4 scales and global multipliers must follow the same stack.
        for key in list(sanitized):
            match = re.match(
                r"(.+)\.experts\.0\.(up_proj|down_proj)\.(weight|scales|biases|bias|weight_scale_2|pre_quant_scale)$",
                key,
            )
            if match is None:
                continue
            prefix, projection, suffix = match.groups()
            keys = [
                f"{prefix}.experts.{i}.{projection}.{suffix}"
                for i in range(self.llm.args.n_routed_experts)
            ]
            missing = [name for name in keys if name not in sanitized]
            if missing:
                raise ValueError(f"Incomplete ModelOpt expert weights: {missing}")
            target = "fc1" if projection == "up_proj" else "fc2"
            sanitized[f"{prefix}.switch_mlp.{target}.{suffix}"] = mx.stack(
                [sanitized.pop(name) for name in keys]
            )
        # Earlier conversions discarded MTP, and some exports have no CTC head.
        # Keep ordinary transcription usable without retaining random heads.
        for prefix, owner, name in [
            ("llm.mtp.", self.llm, "mtp"),
            ("ctc_timestamp.", self, "ctc_timestamp"),
        ]:
            present = {key for key in sanitized if key.startswith(prefix)}
            module = getattr(owner, name, None)
            if not present:
                setattr(owner, name, None)
            elif module is None:
                raise ValueError(
                    f"Checkpoint has {prefix} weights but no matching configuration"
                )
            else:
                missing = {
                    prefix + key for key, _ in tree_flatten(module.parameters())
                } - present
                if missing:
                    raise ValueError(
                        f"Incomplete {prefix} checkpoint: missing {sorted(missing)}"
                    )
        return sanitized

    @classmethod
    def post_load_hook(cls, model, model_path):
        from transformers import PreTrainedTokenizerFast

        model._tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_path, fix_mistral_regex=False
        )
        if model.ctc_timestamp is not None:
            model._alignment_tokenizer = load_alignment_tokenizer(model_path)
        tag = model.config.audio_locator_tag
        ids = model._tokenizer.encode(tag, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"The checkpoint tokenizer must contain {tag} as one token"
            )
        return model

    def _prepare_inputs(
        self, audio, prompt, system_prompt, enable_thinking, word_timestamps=False
    ):
        if self._tokenizer is None:
            raise RuntimeError(
                "Load the model with mlx_audio.stt.load to initialize its tokenizer"
            )
        if isinstance(audio, (str, Path)):
            audio = load_audio(str(audio), sr=self.sample_rate)
        else:
            audio = mx.array(audio, dtype=mx.float32)
        if audio.ndim != 1 or audio.size < self.config.processor_config.hop_length:
            raise ValueError("Expected nonempty mono audio at the model sample rate")
        if not mx.all(mx.isfinite(audio)).item():
            raise ValueError("Audio contains non-finite samples")
        # NeMo tokenizes each complete chat turn independently (no BOS).
        thinking = "<think>\n" if enable_thinking else "<think></think>"
        turns = [
            f"<|im_start|>system\n{system_prompt or ''}<|im_end|>\n",
            f"<|im_start|>user\n{prompt}<|im_end|>\n",
            f"<|im_start|>assistant\n{thinking}",
        ]
        ids = [
            token
            for turn in turns
            for token in self._tokenizer.encode(turn, add_special_tokens=False)
        ]
        placeholder = self._tokenizer.convert_tokens_to_ids(
            self.config.audio_locator_tag
        )
        positions = [i for i, token in enumerate(ids) if token == placeholder]
        if len(positions) != 1:
            raise ValueError("Expected exactly one audio token in the formatted prompt")
        if word_timestamps:
            audio_embeds, states, probabilities = self.perception(
                audio, return_alignment=True
            )
            alignment = (states, probabilities, audio.shape[0] / self.sample_rate)
        else:
            audio_embeds = self.perception(audio)
        index = positions[0]
        # Zero placeholders are ignored by the decoder, which consumes the
        # replacement audio embeddings. Keep token/embedding lengths aligned for
        # chunked prefill and the standard logits processors.
        expanded = (
            ids[:index]
            + [self.config.pad_token_id] * audio_embeds.shape[1]
            + ids[index + 1 :]
        )
        tokens = mx.array(expanded)
        embeddings = self.llm.backbone.embeddings(tokens[None])
        embeddings = mx.concatenate(
            [
                embeddings[:, :index],
                audio_embeds.astype(embeddings.dtype),
                embeddings[:, index + audio_embeds.shape[1] :],
            ],
            axis=1,
        )
        mx.eval(embeddings)
        if word_timestamps:
            return tokens, embeddings[0], alignment
        return tokens, embeddings[0]

    def _generate_stream(
        self,
        audio,
        *,
        prompt,
        system_prompt,
        enable_thinking,
        max_tokens,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        repetition_context_size,
        prefill_step_size,
        verbose,
        word_timestamps,
        language,
        speculative,
        num_draft_tokens,
    ):
        with wired_limit(self):
            if verbose:
                print("Encoding audio...")
            prepared = self._prepare_inputs(
                audio, prompt, system_prompt, enable_thinking, word_timestamps
            )
            tokens, embeddings = prepared[:2]
            processors = None
            if repetition_penalty is not None:
                processors = [
                    make_repetition_penalty(repetition_penalty, repetition_context_size)
                ]
            decoder = StreamingDetokenizer(self._tokenizer, skip_special_tokens=False)
            count = 0
            stats = {} if speculative else None
            generation = speculative_generate_step if speculative else generate_step
            extra = (
                {"num_draft_tokens": num_draft_tokens, "stats": stats}
                if speculative
                else {
                    "sampler": make_sampler(
                        temperature, top_p=top_p, top_k=top_k, min_p=min_p
                    )
                }
            )
            for token, _ in generation(
                tokens,
                self,
                input_embeddings=embeddings,
                max_tokens=max_tokens,
                logits_processors=processors,
                prefill_step_size=prefill_step_size,
                **extra,
            ):
                if token in self.config.eos_token_id:
                    break
                count += 1
                decoder.add_token(token)
                if text := decoder.last_segment:
                    yield StreamingResult(
                        text, prompt_tokens=len(tokens), generation_tokens=count
                    )
            decoder.finalize()
            segments = None
            if word_timestamps:
                states, probabilities, duration = prepared[2]
                segments = align_words(
                    decoder.text,
                    self._alignment_tokenizer,
                    self.ctc_timestamp,
                    states,
                    np.array(probabilities[0].astype(mx.float32)),
                    duration,
                    self.config.pe_encoder_config,
                    language=language,
                    special_tokens=self._tokenizer.all_special_tokens,
                    speaker_frame_seconds=self.config.processor_config.hop_length
                    / self.sample_rate,
                )
            yield StreamingResult(
                decoder.last_segment,
                is_final=True,
                prompt_tokens=len(tokens),
                generation_tokens=count,
                segments=segments,
                speculative_stats=stats,
            )

    def generate(
        self,
        audio,
        *,
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repetition_penalty=None,
        repetition_context_size=64,
        task="asr",
        prompt=None,
        language=None,
        hotwords=None,
        system_prompt=None,
        enable_thinking=False,
        prefill_step_size=512,
        verbose=False,
        stream=False,
        word_timestamps=False,
        speculative=False,
        num_draft_tokens=4,
        **kwargs,
    ):
        """Transcribe, translate, attribute speakers, or answer an audio prompt.

        Arrays must be mono at ``sample_rate``; file inputs are resampled.
        ``stream`` yields decoded text tokens after the complete audio is encoded.
        ``word_timestamps`` returns word segments (seconds) with speaker IDs.
        Streaming timings arrive on the final result, after CTC alignment.
        ``speculative`` enables greedy MTP decoding; ``num_draft_tokens`` controls
        the repeated draft depth (1 to 8). It requires retained MTP weights.
        """
        if max_tokens < 0 or prefill_step_size < 1:
            raise ValueError(
                "max_tokens must be nonnegative and prefill_step_size positive"
            )
        if speculative:
            if self.llm.mtp is None:
                raise ValueError(
                    "This checkpoint has no MTP weights; reconvert the original model to enable speculative decoding"
                )
            if temperature != 0:
                raise ValueError(
                    "MTP speculative decoding currently requires temperature=0"
                )
            if type(num_draft_tokens) is not int or not 1 <= num_draft_tokens <= 8:
                raise ValueError("num_draft_tokens must be an integer from 1 to 8")
        if word_timestamps:
            if task not in {"asr", "transcribe", "saa", "rich"}:
                raise ValueError("Word timestamps require a transcription task")
            if self.ctc_timestamp is None or self._alignment_tokenizer is None:
                raise ValueError(
                    "Word timestamps require CTC weights and the alignment_tokenizer directory from the original checkpoint"
                )
        prompt = resolve_prompt(
            task, prompt, language, hotwords, self.config.audio_locator_tag
        )
        start = time.perf_counter()
        results = self._generate_stream(
            audio,
            prompt=prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
            prefill_step_size=prefill_step_size,
            verbose=verbose,
            word_timestamps=word_timestamps,
            language=language,
            speculative=speculative,
            num_draft_tokens=num_draft_tokens,
        )
        if stream:
            return results
        text = ""
        for result in results:
            text += result.text
        elapsed = time.perf_counter() - start
        return TranscriptionOutput(
            text=text.strip(),
            segments=result.segments if word_timestamps else speaker_segments(text),
            prompt_tokens=result.prompt_tokens,
            generation_tokens=result.generation_tokens,
            total_tokens=result.prompt_tokens + result.generation_tokens,
            total_time=elapsed,
            generation_tps=result.generation_tokens / elapsed,
            speculative_stats=result.speculative_stats,
        )
