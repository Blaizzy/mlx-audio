import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
        return {f"model.{k}": v for k, v in weights.items()}

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def generate(
        self,
        text,
        voice: Optional[str] = None,
        temperature: float = 0.4,
        top_p: float = 0.9,
        split_pattern: str = "\n",
        max_tokens: int = 1200,
        verbose: bool = False,
        **kwargs,
    ):
        prompt = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = prompt.split(split_pattern)

        self.prompt_processor = PromptProcessor(self.tokenizer)
        self.audio_codec = DacInterface()

        if voice is None:
            voice = f"{Path(__file__).parent}/default_speaker.json"

        with open(voice, "r") as f:
            speaker = json.load(f)

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

        all_audio = []

        for prompt in prompts:
            completion_prompt = self.prompt_processor.get_completion_prompt(
                prompt, speaker
            )
            input_ids = self.tokenizer.encode(
                completion_prompt, add_special_tokens=False, return_tensors="mlx"
            )
            input_length = input_ids.shape[1]

            time_start = time.time()

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

            output_ids = input_ids[:, input_length:].tolist()[0]
            output = self.prompt_processor.extract_audio_from_tokens(output_ids)
            audio = self.audio_codec.decode(mx.array([output])).squeeze(0)
            all_audio.append(audio)

        time_end = time.time()

        for i in range(len(all_audio)):
            audio = all_audio[i][0]

            samples = audio.shape[0] if audio is not None else 0
            assert samples > 0, "No audio generated"

            token_count = input_ids.shape[1] if input_ids is not None else 0

            sample_rate = (
                self.config.sample_rate
                if kwargs.get("sample_rate") is None
                else kwargs.get("sample_rate")
            )
            audio_duration_seconds = samples / sample_rate

            elapsed_time = time_end - time_start
            rtf = audio_duration_seconds / elapsed_time

            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=sample_rate,
                segment_idx=i,
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
                processing_time_seconds=time_end - time_start,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            # Clear cache after each segment to avoid memory leaks
            mx.clear_cache()
