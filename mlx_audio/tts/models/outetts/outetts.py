import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import stream_generate
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelConfig
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelConfig
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelConfig
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import AutoTokenizer

from ..base import GenerationResult
from .audio_processor import AudioProcessor
from .dac_interface import DacInterface
from .prompt_processor import PromptProcessor


@dataclass
class ModelConfig(LlamaModelConfig, Qwen2ModelConfig, Qwen3ModelConfig):
    tokenizer_name: str = "OuteAI/Llama-OuteTTS-1.0-1B"
    sample_rate: int = 24000


class Model(nn.Module):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.model = self._initialize_model(config, **kwargs)

    def _initialize_model(self, config: ModelConfig, **kwargs) -> nn.Module:

        model_map = {"llama": LlamaModel, "qwen2": Qwen2Model, "qwen3": Qwen3Model}

        if config.model_type not in model_map:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        return model_map[config.model_type](config, **kwargs)

    def sanitize(self, weights):
        weights = self.model.sanitize(weights)
        return {
            (
                f"model.{k}"
                if not k.startswith("model.model.")
                and not k.startswith("model.lm_head")
                else k
            ): v
            for k, v in weights.items()
        }

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_speaker(self, voice: Optional[str], ref_audio: Optional[str]) -> dict:
        if voice is None and ref_audio is None:
            voice = f"{Path(__file__).parent}/default_speaker.json"
            return self.audio_processor.load_speaker(voice)

        if voice is not None:
            return self.audio_processor.load_speaker(voice)

        speaker = self.audio_processor.create_speaker_from_whisper(ref_audio)
        file_id = str(uuid.uuid4())
        save_path = f"~/.cache/mlx_audio/voices/outetts_{file_id}.json"
        self.audio_processor.save_speaker(speaker, save_path)
        return speaker

    def chunk_text(self, text: str, max_words: int = 30) -> List[str]:
        sentences = re.split(r"[.!?。！？︕︖]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            words = sentence.split()
            if current_length + len(words) > max_words:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.extend(words)
            current_length += len(words)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def generate_result(
        self, audio, start_time: float, token_count: int, segment_idx: int, **kwargs
    ) -> GenerationResult:
        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0, "No audio generated"

        sample_rate = (
            self.config.sample_rate
            if kwargs.get("sample_rate") is None
            else kwargs.get("sample_rate")
        )
        audio_duration_seconds = samples / sample_rate

        elapsed_time = time.perf_counter() - start_time
        rtf = audio_duration_seconds / elapsed_time

        duration_mins = int(audio_duration_seconds // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_hours = int(audio_duration_seconds // 3600)
        duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

        return GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=sample_rate,
            segment_idx=segment_idx,
            token_count=token_count,
            audio_duration=duration_str,
            real_time_factor=rtf,
            prompt={
                "tokens": token_count,
                "tokens-per-sec": (
                    round(token_count / elapsed_time, 2) if elapsed_time > 0 else 0
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

    def generate(
        self,
        text,
        voice: Optional[str] = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        split_pattern: str = "\n",
        max_tokens: int = 1200,
        verbose: bool = False,
        ref_audio: Optional[str] = None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ):

        prompts = self.chunk_text(text)

        self.prompt_processor = PromptProcessor(self.tokenizer)
        self.audio_processor = AudioProcessor()

        speaker = self.get_speaker(voice, ref_audio)

        sampler = make_sampler(
            temperature,
            top_p,
            min_p=kwargs.get("min_p", 0.05),
            top_k=kwargs.get("top_k", 40),
        )
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.1),
            kwargs.get("repetition_context_size", 64),
        )

        for prompt in prompts:
            completion_prompt = self.prompt_processor.get_completion_prompt(
                prompt, speaker
            )
            input_ids = self.tokenizer.encode(
                completion_prompt, add_special_tokens=False, return_tensors="mlx"
            )
            input_length = input_ids.shape[1]

            generated_token_count = 0
            yielded_token_count = 0
            streaming_token_interval = int(streaming_interval * 137.5)
            yielded_frame_count = 0

            time_start = time.perf_counter()

            for i, response in enumerate(
                tqdm(
                    stream_generate(
                        self.model,
                        tokenizer=self.tokenizer,
                        prompt=input_ids.squeeze(0),
                        max_tokens=max_tokens,
                        sampler=sampler,
                        logits_processors=logits_processors,
                    ),
                    total=max_tokens,
                    disable=not verbose,
                )
            ):
                next_token = mx.array([response.token])
                input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
                generated_token_count += 1

                # send a partial result in streaming mode
                if stream and generated_token_count % streaming_token_interval == 0:
                    output_ids = input_ids[:, input_length:].tolist()[0]
                    output = self.prompt_processor.extract_audio_from_tokens(output_ids)
                    audio = self.audio_processor.audio_codec.decode(mx.array([output]))[
                        -1, -1, :
                    ]

                    yield self.generate_result(
                        audio=audio[yielded_frame_count:],
                        start_time=time_start,
                        token_count=len(output_ids) - yielded_token_count,
                        segment_idx=i,
                        **kwargs,
                    )
                    yielded_token_count = len(output_ids)
                    yielded_frame_count = audio.shape[0]
                    time_start = time.perf_counter()

            output_ids = input_ids[:, input_length:].tolist()[0]
            output = self.prompt_processor.extract_audio_from_tokens(output_ids)

            audio = self.audio_processor.audio_codec.decode(mx.array([output]))[
                -1, -1, :
            ]
            if audio.shape[0] > yielded_frame_count:
                yield self.generate_result(
                    audio=audio[yielded_frame_count:],
                    start_time=time_start,
                    token_count=len(output_ids) - yielded_token_count,
                    segment_idx=i,
                    **kwargs,
                )

            # Clear cache after each segment to avoid memory leaks
            mx.clear_cache()
