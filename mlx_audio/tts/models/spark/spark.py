import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import stream_generate
from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import load_tokenizer
from tqdm import tqdm

from mlx_audio.tts.models.base import BaseModelArgs, GenerationResult
from mlx_audio.tts.utils import get_model_path

from .audio_tokenizer import BiCodecTokenizer
from .utils.token_parser import GENDER_MAP, LEVELS_MAP, TASK_TOKEN_MAP

PITCH_MAP = SPEED_MAP = {
    0.0: "very_low",
    0.5: "low",
    1.0: "moderate",
    1.5: "high",
    2.0: "very_high",
}


@dataclass
class ModelConfig(BaseModelArgs):
    model_path: Path = None
    sample_rate: int = 16000
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    hidden_act: str = "silu"
    hidden_size: int = 896
    initializer_range: float = 0.02
    intermediate_size: int = 4864
    max_position_embeddings: int = 32768
    max_window_layers: int = 21
    model_type: str = "qwen2"
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1000000.0
    sliding_window: int = 32768
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.43.1"
    use_sliding_window: bool = False
    vocab_size: int = 166000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None


class Model(nn.Module):
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, config: ModelConfig):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            config (ModelConfig): The configuration for the model.
        """
        self.config = config

        model_dir = config.model_path

        self.model = Qwen2Model(config)
        self.tokenizer = load_tokenizer(model_dir, eos_token_ids=config.eos_token_id)

        self._audio_tokenizer = BiCodecTokenizer(model_dir)

    def load_weights(self, weights, strict=True):
        self.model.load_weights(weights, strict=strict)

    def parameters(self):
        return self.model.parameters()

    def model_type(self):
        return "spark"

    def sanitize(self, weights):
        return self.model.sanitize(weights)

    @property
    def sample_rate(self):
        return self.config.sample_rate

    @property
    def layers(self):
        return self.model.layers

    def model_quant_predicate(self, p, m, config):
        """
        Model modules to skip during quantization
        """
        return not p.startswith("_audio_tokenizer")

    def process_prompt(
        self,
        text: str,
        ref_audio: Path,
        ref_text: str,
    ) -> Tuple[str, mx.array]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            ref_audio (Path): Path to the audio file used as a reference.
            ref_text (str, optional): Transcript of the reference audio.

        Return:
            Tuple[str, mx.array]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self._audio_tokenizer.tokenize(ref_audio)
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if ref_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                ref_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)

    def generate(
        self,
        text: str,
        ref_audio: Path = None,
        ref_text: str = None,
        gender: str = "male",
        pitch: float = 1.0,
        speed: float = 1.0,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        max_tokens: int = 3000,
        verbose: bool = False,
        split_pattern: str = "\n",
        **kwargs,
    ) -> GenerationResult:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            ref_audio (Path): Path to the audio file used as a reference.
            ref_text (str, optional): Transcript of the reference audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            GenerationResult: Generated waveform as a tensor.
        """

        speed_factor = SPEED_MAP[speed]
        pitch_factor = PITCH_MAP[pitch]

        if ref_audio is not None:  # voice cloning
            gender = None

        text_splits = text.split(split_pattern)

        for text_split in text_splits:
            if gender is not None:
                prompt = self.process_prompt_control(
                    gender, pitch_factor, speed_factor, text_split
                )

            else:
                prompt, global_token_ids = self.process_prompt(
                    text_split, ref_audio, ref_text
                )

            inputs = self.tokenizer._tokenizer([prompt], return_tensors="pt")

            input_ids = mx.array(inputs.input_ids)

            sampler = make_sampler(temperature, top_p=top_p, top_k=top_k)
            logits_processors = make_logits_processors(
                kwargs.get("logit_bias", None),
                kwargs.get("repetition_penalty", 1.3),
                kwargs.get("repetition_context_size", 20),
            )

            time_start = time.time()

            generated_ids = []

            # Generate speech using the model
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
                if i % 50 == 0:
                    mx.clear_cache()

                if next_token == 128258:
                    break

            time_end = time.time()
            # Trim the output tokens to remove the input tokens
            generated_ids = mx.array(
                [
                    output[len(input) :]
                    for input, output in zip(inputs.input_ids, input_ids)
                ]
            ).tolist()

            # Decode the generated tokens into text
            predicts = self.tokenizer._tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Extract semantic token IDs from the generated text
            pred_semantic_ids = mx.array(
                [
                    int(token)
                    for token in re.findall(r"bicodec_semantic_(\d+)", predicts)
                ]
            )[None, ...]

            if gender is not None:
                global_token_ids = mx.array(
                    [
                        int(token)
                        for token in re.findall(r"bicodec_global_(\d+)", predicts)
                    ]
                )[None, ...]

            # Convert semantic tokens back to waveform
            audio = self._audio_tokenizer.detokenize(
                global_token_ids.astype(mx.int32),
                pred_semantic_ids.astype(mx.int32),
            )

            # Clear cache
            mx.clear_cache()

            audio_samples = len(audio)
            audio_duration_seconds = audio_samples / self.config.sample_rate

            # Format duration as HH:MM:SS.mmm
            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=audio,
                sample_rate=self.config.sample_rate,
                samples=audio_samples,
                segment_idx=0,  # Default segment index
                token_count=len(pred_semantic_ids.squeeze()),
                audio_samples={
                    "samples": audio_samples,
                    "samples-per-sec": (
                        round(audio_samples / audio_duration_seconds, 2)
                        if audio_duration_seconds > 0
                        else 0
                    ),
                },
                audio_duration=duration_str,
                real_time_factor=(
                    audio_duration_seconds / (time_end - time_start)
                    if (time_end - time_start) > 0
                    else 0
                ),
                prompt={
                    "tokens": len(pred_semantic_ids.squeeze()),
                    "tokens-per-sec": (
                        round(
                            len(pred_semantic_ids.squeeze()) / audio_duration_seconds, 2
                        )
                        if audio_duration_seconds > 0
                        else 0
                    ),
                },
                processing_time_seconds=time_end - time_start,
                peak_memory_usage=mx.get_peak_memory() / 1e9,
            )

            # Clear cache after each segment to avoid memory leaks
            mx.clear_cache()
