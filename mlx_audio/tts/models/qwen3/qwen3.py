import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelConfig
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm

from mlx_audio.codec.models.snac import SNAC

from ..base import GenerationResult

# VyvoTTS special token IDs (Qwen3-based tokenizer)
TOKENIZER_LENGTH = 151669
START_OF_TEXT = 151643
END_OF_TEXT = 151645
START_OF_SPEECH = TOKENIZER_LENGTH + 1  # 151670
END_OF_SPEECH = TOKENIZER_LENGTH + 2  # 151671
START_OF_HUMAN = TOKENIZER_LENGTH + 3  # 151672
END_OF_HUMAN = TOKENIZER_LENGTH + 4  # 151673
START_OF_AI = TOKENIZER_LENGTH + 5  # 151674
END_OF_AI = TOKENIZER_LENGTH + 6  # 151675
PAD_TOKEN = TOKENIZER_LENGTH + 7  # 151676
AUDIO_TOKENS_START = TOKENIZER_LENGTH + 10  # 151679


@dataclass
class ModelConfig(Qwen3ModelConfig):
    tokenizer_name: str = None
    sample_rate: int = 24000


snac_model = SNAC.from_pretrained("mlx-community/snac_24khz").eval()


def decode_audio_from_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))
    codes = [
        mx.expand_dims(mx.array(layer_1), 0),
        mx.expand_dims(mx.array(layer_2), 0),
        mx.expand_dims(mx.array(layer_3), 0),
    ]
    audio_hat = snac_model.decode(codes).squeeze(-1)
    return audio_hat


def encode_audio_to_codes(audio):
    audio = audio[None, None, :]

    codes = snac_model.encode(audio)

    layer_1 = codes[0].squeeze(0).tolist()
    layer_2 = codes[1].squeeze(0).tolist()
    layer_3 = codes[2].squeeze(0).tolist()

    code_list = []
    num_groups = len(layer_1)
    for i in range(num_groups):
        code_list.append(layer_1[i])
        code_list.append(layer_2[2 * i] + 4096)
        code_list.append(layer_3[4 * i] + 2 * 4096)
        code_list.append(layer_3[4 * i + 1] + 3 * 4096)
        code_list.append(layer_2[2 * i + 1] + 4 * 4096)
        code_list.append(layer_3[4 * i + 2] + 5 * 4096)
        code_list.append(layer_3[4 * i + 3] + 6 * 4096)

    return mx.array(code_list)[None, :]


class Model(Qwen3Model):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.model_type = config.model_type
        self.tokenizer = None

    @property
    def layers(self):
        return self.model.layers

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path):
        """
        Hook called after model weights are loaded.
        Used to initialize the tokenizer which is required for text input.
        """
        from transformers import AutoTokenizer

        if model.tokenizer is None:
            model.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        return model

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def parse_output(self, input_ids):
        token_to_find = START_OF_SPEECH  # 151670
        token_to_remove = END_OF_SPEECH  # 151671

        # MLX doesn't have nonzero, so we need to create indices manually
        mask = input_ids == token_to_find
        indices = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    indices.append((i, j))
        token_indices = [[], []]
        for i, j in indices:
            token_indices[0].append(i)
            token_indices[1].append(j)

        # Check if we found any tokens BEFORE converting to MLX array
        # to avoid "Cannot do a non-empty take from an array with zero elements" error
        if len(token_indices[1]) > 0:
            token_indices = mx.array(token_indices)
            last_occurrence_idx = int(token_indices[1][-1])
            cropped_tensor = input_ids[:, last_occurrence_idx + 1 :]
        else:
            cropped_tensor = input_ids

        mask = cropped_tensor != token_to_remove

        processed_rows = []

        for row in cropped_tensor:
            # Create a mask and filter manually since boolean indexing isn't supported
            row_list = row.tolist()
            masked_row = mx.array([val for val in row_list if val != token_to_remove])
            processed_rows.append(masked_row)

        code_lists = []

        for row in processed_rows:
            row_length = row.shape[0]
            new_length = (row_length // 7) * 7
            trimmed_row = row[:new_length]
            trimmed_row = [t - AUDIO_TOKENS_START for t in trimmed_row]  # 151679
            code_lists.append(trimmed_row)

        return code_lists

    def prepare_zeroprompt(
        self,
        ref_audio: mx.array,
        ref_text: str,
    ):
        """Prepare the reference audio context (zeroprompt) for voice cloning."""
        print(
            "\033[93mWARNING: Audio cloning doesn't work reliably on this model.\033[0m"
        )
        audio_input_ids = encode_audio_to_codes(ref_audio) + AUDIO_TOKENS_START
        audio_transcript_ids = self.tokenizer(ref_text, return_tensors="mlx").input_ids

        start_token = mx.array([[START_OF_HUMAN]], dtype=mx.int64)  # 151672
        end_tokens = mx.array(
            [[END_OF_TEXT, END_OF_HUMAN]], dtype=mx.int64
        )  # 151645, 151673
        audio_start_tokens = mx.array(
            [[START_OF_AI, START_OF_SPEECH]], dtype=mx.int64
        )  # 151674, 151670
        audio_end_tokens = mx.array(
            [[END_OF_SPEECH, END_OF_AI]], dtype=mx.int64
        )  # 151671, 151675

        # [SOH] [transcript] [EOT EOH] [SOA SOS] [audio_tokens] [EOS EOA]
        zeroprompt = mx.concatenate(
            [
                start_token,
                audio_transcript_ids,
                end_tokens,
                audio_start_tokens,
                audio_input_ids,
                audio_end_tokens,
            ],
            axis=1,
        )
        return zeroprompt

    def prepare_input_ids(
        self,
        prompt: Union[str, List[str]],
        voice: Optional[str] = None,
        zeroprompt: Optional[mx.array] = None,
    ):
        """Prepare input ids for a single prompt or batch of prompts, optionally with zeroprompt prefix."""
        # Handle single prompt vs batch
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt

        all_input_ids = []
        for p in prompts:
            if voice is not None and zeroprompt is None:
                p = f"{voice}: {p}"

            start_token = mx.array([[START_OF_HUMAN]], dtype=mx.int64)  # 151672
            end_tokens = mx.array(
                [[END_OF_TEXT, END_OF_HUMAN]], dtype=mx.int64
            )  # 151645, 151673

            input_ids = self.tokenizer(p, return_tensors="mlx").input_ids
            prompt_input_ids = mx.concatenate(
                [start_token, input_ids, end_tokens], axis=1
            )  # [SOH] [text] [EOT EOH]

            if zeroprompt is not None:
                prompt_input_ids = mx.concatenate(
                    [zeroprompt, prompt_input_ids], axis=1
                )

            all_input_ids.append(prompt_input_ids)

        # If only one prompt, return as-is (keeps original behavior)
        if len(all_input_ids) == 1:
            return all_input_ids[0]

        # Pad to same length and stack for batch
        max_len = max(ids.shape[1] for ids in all_input_ids)
        padded = []
        for ids in all_input_ids:
            if ids.shape[1] < max_len:
                padding = mx.zeros((1, max_len - ids.shape[1]), dtype=mx.int64)
                ids = mx.concatenate([ids, padding], axis=1)
            padded.append(ids)

        return mx.concatenate(padded, axis=0)

    def generate_result(
        self, audio, start_time: float, token_count: int, segment_idx: int, **kwargs
    ) -> GenerationResult:
        """Helper to create a GenerationResult from audio."""
        samples = audio.shape[0] if audio is not None else 0
        assert samples > 0, "No audio generated"

        sample_rate = self.config.sample_rate
        audio_duration_seconds = samples / sample_rate

        elapsed_time = time.perf_counter() - start_time
        rtf = audio_duration_seconds / elapsed_time if elapsed_time > 0 else 0

        duration_hours = int(audio_duration_seconds // 3600)
        duration_mins = int((audio_duration_seconds % 3600) // 60)
        duration_secs = int(audio_duration_seconds % 60)
        duration_ms = int((audio_duration_seconds % 1) * 1000)
        duration_str = (
            f"{duration_hours:02d}:{duration_mins:02d}:"
            f"{duration_secs:02d}.{duration_ms:03d}"
        )

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
        voice: str,
        temperature: float = 0.6,
        top_p: float = 0.8,
        split_pattern: str = "\n",
        max_tokens: int = 1200,
        verbose: bool = False,
        ref_audio: mx.array = None,
        ref_text: Optional[str] = None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        **kwargs,
    ):
        prompt_text = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = [p for p in prompt_text.split(split_pattern) if p.strip()]

        # Prepare zeroprompt once if voice cloning is requested
        zeroprompt = None
        if ref_audio is not None and ref_text is not None:
            zeroprompt = self.prepare_zeroprompt(ref_audio, ref_text)

        sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", -1))
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.3),
            kwargs.get("repetition_context_size", 20),
        )

        streaming_token_interval = int(streaming_interval * 137.5)

        # Process each prompt segment individually for memory efficiency
        for segment_idx, segment_prompt in enumerate(prompts):
            time_start = time.perf_counter()

            # Prepare input_ids for this segment (with zeroprompt if voice cloning)
            input_ids = self.prepare_input_ids(
                segment_prompt,
                voice,
                zeroprompt,
            )

            generated_token_count = 0
            yielded_token_count = 0
            yielded_frame_count = 0

            # Generate tokens for this segment
            for i, response in enumerate(
                tqdm(
                    stream_generate(
                        self,
                        tokenizer=self.tokenizer,
                        prompt=input_ids.squeeze(0),
                        max_tokens=max_tokens,
                        sampler=sampler,
                        logits_processors=logits_processors,
                    ),
                    total=max_tokens,
                    disable=not verbose,
                    desc=f"Segment {segment_idx + 1}/{len(prompts)}",
                )
            ):
                next_token = mx.array([response.token])
                input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
                generated_token_count += 1

                if i % 50 == 0:
                    mx.clear_cache()

                # Stream partial audio at intervals
                if stream and generated_token_count % streaming_token_interval == 0:
                    code_lists = self.parse_output(input_ids)
                    if code_lists and len(code_lists[0]) > 0:
                        audio = decode_audio_from_codes(code_lists[0])[0]
                        if audio.shape[0] > yielded_frame_count:
                            yield self.generate_result(
                                audio=audio[yielded_frame_count:],
                                start_time=time_start,
                                token_count=generated_token_count - yielded_token_count,
                                segment_idx=segment_idx,
                            )
                            yielded_token_count = generated_token_count
                            yielded_frame_count = audio.shape[0]
                            time_start = time.perf_counter()

                if next_token == END_OF_SPEECH:  # 151671
                    break

            # Decode and yield remaining audio for this segment
            code_lists = self.parse_output(input_ids)

            for code_list in code_lists:
                if len(code_list) == 0:
                    continue
                audio = decode_audio_from_codes(code_list)[0]

                if audio.shape[0] > yielded_frame_count:
                    yield self.generate_result(
                        audio=audio[yielded_frame_count:],
                        start_time=time_start,
                        token_count=generated_token_count - yielded_token_count,
                        segment_idx=segment_idx,
                    )

            # Clear cache after each segment to avoid memory buildup
            mx.clear_cache()
