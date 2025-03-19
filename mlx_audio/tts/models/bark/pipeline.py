import mlx.nn as nn
import mlx.core as mx
import tqdm
import math
from .isftnet import codec_decode
from ..base import adjust_speed
from dataclasses import dataclass
from typing import Optional

TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129595
SEMANTIC_INFER_TOKEN = 129_599

CONTEXT_WINDOW_SIZE = 1024

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75
COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050
SAMPLE_RATE = 24_000

@dataclass
class Result:
    audio: mx.array
    tokens: mx.array

    ### MARK: BEGIN BACKWARD COMPAT ###
    def __iter__(self):
        yield self.audio
        yield self.tokens

    def __getitem__(self, index):
        return [self.audio, self.tokens][index]

    def __len__(self):
        return 2

class Pipeline:
    def __init__(self, model: nn.Module, tokenizer: any):
        self.model = model
        self.tokenizer = tokenizer

    def generate_text_semantic(
        self,
        text: str,
        temperature: float = 0.7,
        use_kv_caching: bool = False,
    ):
        """Generate semantic tokens from text."""
        print("Generating semantic tokens...")
        encoded_text = (
            mx.array(self.tokenizer.encode(text, add_special_tokens=False))
            + TEXT_ENCODING_OFFSET
        )
        if len(encoded_text) > 256:
            p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
            encoded_text = encoded_text[:256]
        encoded_text = mx.pad(
            encoded_text,
            (0, 256 - len(encoded_text)),
            constant_values=TEXT_PAD_TOKEN,
        )
        semantic_history = mx.array([SEMANTIC_PAD_TOKEN] * 256)
        x = (
            mx.concatenate(
                [encoded_text, semantic_history, mx.array([SEMANTIC_INFER_TOKEN])]
            )
            .reshape(1, -1)
            .astype(mx.int64)
        )
        n_tot_steps = 768
        kv_cache = None
        for i in tqdm.tqdm(range(n_tot_steps)):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, -1:]
            else:
                x_input = x
            logits, kv_cache = self.model.semantic(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            # Early stop
            relevant_logits = mx.concatenate(
                [relevant_logits, logits[0, 0, SEMANTIC_PAD_TOKEN].reshape(1)], axis=-1
            )
            next_token = mx.random.categorical(
                relevant_logits * 1 / (temperature), num_samples=1
            ).astype(mx.int32)

            if next_token == SEMANTIC_VOCAB_SIZE:
                print(f"Early stop at step {i} with token {next_token}")
                break
            x = mx.concatenate([x, next_token.reshape(1, -1)], axis=1)
            if i == n_tot_steps - 1:
                break
        out = x.squeeze()[256 + 256 + 1 :]
        return out, encoded_text


    def generate_coarse(
        self,
        x_semantic: mx.array,
        temperature: float = 0.7,
        silent: bool = False,
        max_coarse_history: int = 60,  # min 60 (faster), max 630 (more context)
        sliding_window_len: int = 60,
        use_kv_caching: bool = False,
    ):
        """Generate coarse tokens from semantic tokens."""
        print("Generating coarse tokens...")
        semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
        max_semantic_history = int(
            math.floor(max_coarse_history / semantic_to_coarse_ratio)
        )
        x_semantic_history = mx.array([], dtype=mx.int32)
        x_coarse_history = mx.array([], dtype=mx.int32)
        n_steps = int(
            round(
                math.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
                * N_COARSE_CODEBOOKS
            )
        )
        x_semantic = mx.concatenate([x_semantic_history, x_semantic]).astype(mx.int32)
        x_coarse = x_coarse_history.astype(mx.int32)
        base_semantic_idx = len(x_semantic_history)
        # Inference
        x_semantic_in = x_semantic.reshape(1, -1)
        x_coarse_in = x_coarse.reshape(1, -1)
        n_window_steps = int(round(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            x_in = x_semantic_in[:, max(0, semantic_idx - max_semantic_history) :]
            x_in = x_in[:, :256]
            x_in = mx.pad(
                x_in,
                ((0, 0), (0, 256 - x_in.shape[-1])),
                constant_values=COARSE_SEMANTIC_PAD_TOKEN,
            )
            x_in = mx.concatenate(
                [
                    x_in,
                    mx.array([COARSE_INFER_TOKEN]).reshape(1, -1),
                    x_coarse_in[:, -max_coarse_history:],
                ],
                axis=1,
            )
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0
                x_input = x_in[:, -1:] if use_kv_caching and kv_cache is not None else x_in
                logits, kv_cache = self.model.coarse_acoustics(
                    x_input, use_cache=use_kv_caching, past_kv=kv_cache
                )
                logit_start_idx = (
                    SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = (
                    SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = min(logit_end_idx, logits.shape[-1])
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                item_next = mx.random.categorical(
                    relevant_logits * (1 / temperature), num_samples=1
                ).astype(mx.int32)

                item_next += logit_start_idx
                x_coarse_in = mx.concatenate([x_coarse_in, item_next.reshape(1, 1)], axis=1)
                x_in = mx.concatenate([x_in, item_next.reshape(1, 1)], axis=1)
                n_step += 1

        gen_coarse_arr = x_coarse_in[0, len(x_coarse_history) :]
        gen_coarse_audio_arr = (
            gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
        )
        for n in range(1, N_COARSE_CODEBOOKS):
            gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE

        return gen_coarse_audio_arr


    def generate_fine(
        self,
        x_coarse_gen: mx.array,
        temperature: float = 0.5,
    ):

        """Generate fine tokens from coarse tokens."""
        print("Generating fine tokens...")
        x_fine_history = None
        n_coarse = x_coarse_gen.shape[0]
        in_arr = mx.concatenate(
            [
                x_coarse_gen,
                mx.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
                + CODEBOOK_SIZE,  # padding
            ],
            axis=0,
        )
        n_history = 0
        n_remove_from_end = 0
        # need to pad if too short (since non-causal model)
        if in_arr.shape[1] < 1024:
            n_remove_from_end = 1024 - in_arr.shape[1]
            in_arr = mx.concatenate(
                [
                    in_arr,
                    mx.zeros((N_FINE_CODEBOOKS, n_remove_from_end)) + CODEBOOK_SIZE,
                ],
                axis=1,
            )
        # Inference
        n_loops = (
            max(0, int(math.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))) + 1
        )
        in_arr = in_arr.T
        for n in tqdm.tqdm(range(n_loops)):
            start_idx = mx.min(mx.array([n * 512, in_arr.shape[0] - 1024])).item()
            start_fill_idx = mx.min(
                mx.array([n_history + n * 512, in_arr.shape[0] - 512])
            ).item()
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx : start_idx + 1024, :][None]
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = self.model.fine_acoustics(nn, in_buffer)
                if temperature is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = mx.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temperature
                    codebook_preds = (
                        mx.random.categorical(
                            relevant_logits[rel_start_fill_idx:1024], num_samples=1
                        )
                        .reshape(-1)
                        .astype(mx.int32)
                    )
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                in_arr[
                    start_fill_idx : start_fill_idx + (1024 - rel_start_fill_idx), nn
                ] = in_buffer[0, rel_start_fill_idx:, nn]
        gen_fine_arr = in_arr.squeeze().T
        gen_fine_arr = gen_fine_arr[:, n_history:]
        if n_remove_from_end > 0:
            gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
        assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
        return gen_fine_arr

    def __call__(self, text: str, temperature: float = 0.1, silent: bool = False, speed: float = 1.0, use_kv_caching: bool = False, **kwargs):
        semantic_tokens, tokens = self.generate_text_semantic(text, temperature, use_kv_caching)
        coarse_tokens = self.generate_coarse(semantic_tokens, temperature, silent, use_kv_caching)
        fine_tokens = self.generate_fine(coarse_tokens, temperature)[None, ...]
        # TODO: adjust speed
        # audio_arr = adjust_speed(fine_tokens, speed)
        audio_arr = codec_decode(self.model.codec_model, fine_tokens)[None, ...]

        yield Result(audio=audio_arr, tokens=tokens)
