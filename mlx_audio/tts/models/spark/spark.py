import re
import time
from tqdm import tqdm
import mlx.core as mx
import mlx.nn as nn
from typing import Tuple
from pathlib import Path
from dataclasses import dataclass
# from transformers import AutoTokenizer

from .audio_tokenizer import BiCodecTokenizer
from .utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP

import torch
# from mlx_lm.models.qwen2 import Model as Qwen2Model
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_audio.tts.utils import get_model_path
from mlx_audio.tts.models.base import GenerationResult

@dataclass
class ModelConfig:
    model_repo: str = "SparkAudio/Spark-TTS-0.5B"
    sample_rate: int = 16000


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
        self.configs = config
        self.model_dir = get_model_path(config.model_repo)
        self.device = "cpu"

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "LLM")
        self.model, self.tokenizer = load(self.model_dir / "LLM") #Qwen2Model()
        print("Model loaded successfully")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir)


    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, mx.array]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, mx.array]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
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
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
        max_tokens: int = 3000,
        verbose: bool = False,
        **kwargs,
    ) -> GenerationResult:
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            GenerationResult: Generated waveform as a tensor.
        """
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )

        inputs = self.tokenizer._tokenizer([prompt], return_tensors="pt")

        input_ids = mx.array(inputs.input_ids)

        sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", -1))
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
        generated_ids = mx.array([
            output[len(input) :]
            for input, output in zip(inputs.input_ids, input_ids)
        ]).tolist()


        # Decode the generated tokens into text
        predicts = self.tokenizer._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract semantic token IDs from the generated text
        pred_semantic_ids = (
            torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
            .long()
            .unsqueeze(0)
        )

        if gender is not None:
            global_token_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
            )

        # Convert semantic tokens back to waveform
        audio = self.audio_tokenizer.detokenize(
            global_token_ids.to(self.device).squeeze(0),
            pred_semantic_ids.to(self.device),
        )

        yield GenerationResult(
            audio=audio,
            sample_rate=self.configs.sample_rate,
            processing_time_seconds=time_end - time_start,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )