import math
import os
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tqdm

from mlx_audio.codec.models.encodec.encodec import Encodec

from ..base import adjust_speed
from .isftnet import codec_decode

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

CUR_PATH = os.path.dirname(os.path.abspath(__file__))


SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

ALLOWED_PROMPTS = {"announcer"}
for _, lang in SUPPORTED_LANGS:
    for prefix in ("", f"v2{os.path.sep}"):
        for n in range(10):
            ALLOWED_PROMPTS.add(f"{prefix}{lang}_speaker_{n}")


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


def _load_voice_prompt(voice_prompt_input):
    if isinstance(voice_prompt_input, str) and voice_prompt_input.endswith(".npz"):
        voice_prompt = np.load(voice_prompt_input)
    elif isinstance(voice_prompt_input, str):
        # make sure this works on non-ubuntu
        voice_prompt_input = os.path.join(*voice_prompt_input.split("/"))
        if voice_prompt_input not in ALLOWED_PROMPTS:
            raise ValueError("voice prompt not found")

        path = f"{voice_prompt_input}.npz"

        # TODO: Get the path from the Hugging Face cache directory
        # TODO: If not found, download the voice from Hugging Face
        # TODO: If still not found, raise an error

        if not os.path.exists(path):
            raise ValueError("voice prompt not found")
        voice_prompt = np.load(path)
    elif isinstance(voice_prompt_input, dict):
        assert "semantic_prompt" in voice_prompt_input
        assert "coarse_prompt" in voice_prompt_input
        assert "fine_prompt" in voice_prompt_input
        voice_prompt = voice_prompt_input
    else:
        raise ValueError("voice prompt format unrecognized")
    return voice_prompt


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    assert len(arr.shape) == 2
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    # MLX doesn't have ravel with order parameter, so we transpose and reshape
    # to achieve the same effect as numpy's ravel('F')
    flat_arr = arr.transpose().reshape(-1)
    return flat_arr


class Pipeline:
    def __init__(self, model: nn.Module, tokenizer: any, config: any):
        self.model = model
        self.tokenizer = tokenizer
        self.codec_model, _ = Encodec.from_pretrained(config.codec_path)

    def generate_text_semantic(
        self,
        text: str,
        voice: str = "announcer",
        temperature: float = 0.7,
        use_kv_caching: bool = False,
        allow_early_stop: bool = True,
        **kwargs,
    ):
        """Generate semantic tokens from text."""
        verbose = kwargs.get("verbose", False)
        if verbose:
            print("Generating semantic tokens...")
        if voice is not None:
            voice_prompt = _load_voice_prompt(voice)
            semantic_history = mx.array(voice_prompt["semantic_prompt"])
            assert (
                isinstance(semantic_history, mx.array)
                and len(semantic_history.shape) == 1
                and len(semantic_history) > 0
                and semantic_history.min() >= 0
                and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            )
        else:
            semantic_history = None

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
        if semantic_history is not None:
            semantic_history = semantic_history.astype(mx.int64)
            # lop off if history is too long, pad if needed
            semantic_history = semantic_history[-256:]
            semantic_history = mx.pad(
                semantic_history,
                (0, 256 - len(semantic_history)),
                constant_values=SEMANTIC_PAD_TOKEN,
                mode="constant",
            )
        else:
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
        for i in tqdm.tqdm(range(n_tot_steps), disable=not verbose):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, -1:]
            else:
                x_input = x
            logits, kv_cache = self.model.semantic(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                # Early stop
                relevant_logits = mx.concatenate(
                    [relevant_logits, logits[0, 0, SEMANTIC_PAD_TOKEN].reshape(1)],
                    axis=-1,
                )
            next_token = mx.random.categorical(
                relevant_logits * 1 / (temperature), num_samples=1
            ).astype(mx.int32)

            if next_token == SEMANTIC_VOCAB_SIZE:
                print(f"Early stop at step {i} with token {next_token.tolist()}")
                break
            x = mx.concatenate([x, next_token.reshape(1, -1)], axis=1)
            if i == n_tot_steps - 1:
                break
        out = x.squeeze()[256 + 256 + 1 :]
        return out, encoded_text

    def generate_coarse(
        self,
        x_semantic: mx.array,
        voice: str = "announcer",
        temperature: float = 0.7,
        max_coarse_history: int = 60,  # min 60 (faster), max 630 (more context)
        sliding_window_len: int = 60,
        use_kv_caching: bool = False,
        **kwargs,
    ):
        """Generate coarse tokens from semantic tokens."""
        verbose = kwargs.get("verbose", False)
        if verbose:
            print("Generating coarse tokens...")
        semantic_to_coarse_ratio = (
            COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
        )
        max_semantic_history = int(
            math.floor(max_coarse_history / semantic_to_coarse_ratio)
        )
        if voice is not None:
            voice_prompt = _load_voice_prompt(voice)
            x_semantic_history = mx.array(voice_prompt["semantic_prompt"])
            x_coarse_history = mx.array(voice_prompt["coarse_prompt"])
            assert (
                isinstance(x_semantic_history, mx.array)
                and len(x_semantic_history.shape) == 1
                and len(x_semantic_history) > 0
                and x_semantic_history.min() >= 0
                and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
                and isinstance(x_coarse_history, mx.array)
                and len(x_coarse_history.shape) == 2
                and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
                and x_coarse_history.shape[-1] >= 0
                and x_coarse_history.min() >= 0
                and x_coarse_history.max() <= CODEBOOK_SIZE - 1
                and (
                    round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                    == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
                )
            )
            x_coarse_history = (
                _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
            )
            # trim histories correctly
            n_semantic_hist_provided = min(
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(math.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            )
            n_coarse_hist_provided = int(
                round(n_semantic_hist_provided * semantic_to_coarse_ratio)
            )
            x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(
                mx.int32
            )
            x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(
                mx.int32
            )
            # TODO: bit of a hack for time alignment (sounds better)
            x_coarse_history = x_coarse_history[:-2]
        else:
            x_semantic_history = mx.array([], dtype=mx.int32)
            x_coarse_history = mx.array([], dtype=mx.int32)

        n_steps = int(
            round(
                math.floor(
                    len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS
                )
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
        for _ in tqdm.tqdm(
            range(n_window_steps), total=n_window_steps, disable=not verbose
        ):
            semantic_idx = base_semantic_idx + int(
                round(n_step / semantic_to_coarse_ratio)
            )
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
                x_input = (
                    x_in[:, -1:] if use_kv_caching and kv_cache is not None else x_in
                )
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
                x_coarse_in = mx.concatenate(
                    [x_coarse_in, item_next.reshape(1, 1)], axis=1
                )
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
        temperature: float = 0.7,
        **kwargs,
    ):
        verbose = kwargs.get("verbose", False)
        """Generate fine tokens from coarse tokens."""
        if verbose:
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
            max(0, int(math.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512)))
            + 1
        )
        in_arr = in_arr.T
        for n in tqdm.tqdm(range(n_loops), disable=not verbose):
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

    def __call__(
        self,
        text: str,
        voice: str = None,
        temperature: float = 0.7,
        speed: float = 1.0,
        use_kv_caching: bool = False,
        **kwargs,
    ):
        semantic_tokens, tokens = self.generate_text_semantic(
            text, voice, temperature, use_kv_caching, **kwargs
        )
        coarse_tokens = self.generate_coarse(
            semantic_tokens, voice, temperature, use_kv_caching, **kwargs
        )
        fine_tokens = self.generate_fine(coarse_tokens, temperature, **kwargs)
        # TODO: adjust speed
        # audio_arr = adjust_speed(fine_tokens, speed)
        audio_arr = codec_decode(self.codec_model, fine_tokens)

        yield Result(audio=audio_arr, tokens=tokens)
