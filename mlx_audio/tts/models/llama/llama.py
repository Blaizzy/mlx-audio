import time
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs as LlamaModelConfig
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import AutoTokenizer

from mlx_audio.codec.models.snac import SNAC

from ..base import GenerationResult


@dataclass
class ModelConfig(LlamaModelConfig):
    tokenizer_name: str = "mlx-community/orpheus-3b-0.1-ft-bf16"
    sample_rate: int = 24000

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads


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


class Model(LlamaModel):
    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.model_type = config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    @property
    def layers(self):
        return self.model.layers

    @property
    def sample_rate(self):
        return self.config.sample_rate

    def parse_output(self, input_ids):
        token_to_find = 128257
        token_to_remove = 128258

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

        token_indices = mx.array(token_indices)

        if len(token_indices[1]) > 0:
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
            trimmed_row = [t - 128266 for t in trimmed_row]
            code_lists.append(trimmed_row)

        return code_lists

    def prepare_input_ids(
        self,
        prompts: List[str],
        voice: Optional[str] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
    ):
        audio_input_ids = None
        if ref_audio is not None and ref_text is not None:
            print(
                "\033[93mWARNING: Audio cloning doesn't work reliably on Orpheus.\033[0m \nA known issue affecting Torch and MLX versions. \nWill be fixed once the Canopy labs repo update their code or the model."
            )
            audio_input_ids = encode_audio_to_codes(ref_audio) + 128266
            audio_transcript_ids = self.tokenizer(
                ref_text, return_tensors="mlx"
            ).input_ids
        elif voice is not None:
            prompts = [f"{voice}: " + p for p in prompts]

        start_token = mx.array([[128259]], dtype=mx.int64)  # Start of human
        end_tokens = mx.array(
            [[128009, 128260]], dtype=mx.int64
        )  # End of text, End of human

        prompt_input_ids = []
        for prompt in prompts:
            prompt_input_ids.append(
                self.tokenizer(prompt, return_tensors="mlx").input_ids
            )

        batch_input_ids = []
        pad_token = mx.array([128263], dtype=mx.int64)
        max_len = max([p.shape[1] for p in prompt_input_ids])

        for input_ids in prompt_input_ids:
            modified_input_ids = []

            padding_len = max_len - input_ids.shape[1]
            if padding_len > 0:
                modified_input_ids.append(mx.repeat(pad_token, padding_len)[None, :])

            # reference audio and transcript
            if audio_input_ids is not None:
                audio_start_tokens = mx.array([[128261, 128257]], dtype=mx.int64)
                audio_end_tokens = mx.array([[128258, 128262]], dtype=mx.int64)
                ref_input_ids = mx.concatenate(
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
                modified_input_ids.append(ref_input_ids)

            # prompt
            one_prompt_input_ids = mx.concatenate(
                [start_token, input_ids, end_tokens], axis=1
            )  # SOH SOT Text EOT EOH
            modified_input_ids.append(one_prompt_input_ids)

            batch_input_ids.append(mx.concatenate(modified_input_ids, axis=1))

        batch_input_ids = mx.concatenate(batch_input_ids, axis=0)
        batch_mask = mx.where(batch_input_ids == pad_token, False, True)

        return batch_input_ids, batch_mask

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
        **kwargs,
    ):
        prompt = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = prompt.split(split_pattern)

        input_ids, _ = self.prepare_input_ids(
            prompts,
            voice,
            ref_audio,
            ref_text,
        )

        sampler = make_sampler(temperature, top_p, top_k=kwargs.get("top_k", -1))
        logits_processors = make_logits_processors(
            kwargs.get("logit_bias", None),
            kwargs.get("repetition_penalty", 1.3),
            kwargs.get("repetition_context_size", 20),
        )

        time_start = time.time()
        # TODO: Support batch processing as in the Colab: https://github.com/canopyai/Orpheus-TTS
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
            )
        ):
            next_token = mx.array([response.token])
            input_ids = mx.concatenate([input_ids, next_token[None, :]], axis=1)
            if i % 50 == 0:
                mx.clear_cache()

            if next_token == 128258:
                break

        code_lists = self.parse_output(input_ids)

        my_samples = []
        for code_list in code_lists:
            samples = decode_audio_from_codes(code_list)
            my_samples.append(samples)

        time_end = time.time()

        if len(prompts) != len(my_samples):
            raise Exception("Number of prompts and samples do not match")
        else:
            for i in range(len(my_samples)):
                audio = my_samples[i][0]

                samples = audio.shape[0] if audio is not None else 0
                assert samples > 0, "No audio generated"

                # Calculate token count
                token_count = input_ids.shape[1] if input_ids is not None else 0

                # Calculate audio duration in seconds
                sample_rate = self.config.sample_rate
                audio_duration_seconds = samples / sample_rate

                # Calculate real-time factor (RTF)
                rtf = audio_duration_seconds / (time_end - time_start)

                # Format duration as HH:MM:SS.mmm
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
                            round(token_count / audio_duration_seconds, 2)
                            if audio_duration_seconds > 0
                            else 0
                        ),
                    },
                    audio_samples={
                        "samples": samples,
                        "samples-per-sec": (
                            round(samples / audio_duration_seconds, 2)
                            if audio_duration_seconds > 0
                            else 0
                        ),
                    },
                    processing_time_seconds=time_end - time_start,
                    peak_memory_usage=mx.get_peak_memory() / 1e9,
                )

                # Clear cache after each segment to avoid memory leaks
                mx.clear_cache()
