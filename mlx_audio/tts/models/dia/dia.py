import re
import time
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download
from mlx_lm.sample_utils import make_sampler
from tqdm import trange

from mlx_audio.codec.models import DAC

from ..base import GenerationResult
from .audio import audio_to_codebook, codebook_to_audio
from .config import DiaConfig
from .layers import DiaModel, KVCache


def _sample_next_token(
    logits_BCxV: mx.array,
    temperature: float,
    sampler: callable,
) -> mx.array:
    if temperature == 0.0:
        return mx.argmax(logits_BCxV, axis=-1)

    sampled = sampler(logits_BCxV)
    return sampled


class Model(nn.Module):
    def __init__(self, config: dict):
        """Initializes the Dia model.

        Args:
            config: The configuration object for the model.

        Raises:
            RuntimeError: If there is an error loading the DAC model.
        """
        super().__init__()
        self.config = DiaConfig.load_dict(config)
        self.model = DiaModel(self.config)
        self.dac_model = DAC.from_pretrained("mlx-community/descript-audio-codec-44khz")

    @classmethod
    def from_local(cls, config_path: str, checkpoint_path: str) -> "Dia":
        """Loads the Dia model from local configuration and checkpoint files.

        Args:
            config_path: Path to the configuration JSON file.
            checkpoint_path: Path to the model checkpoint (.pth) file.

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If the config or checkpoint file is not found.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config = DiaConfig.load(config_path)
        if config is None:
            raise FileNotFoundError(f"Config file not found at {config_path}")

        dia = cls(config)

        try:
            weights = mx.load(checkpoint_path)
            dia.model.load_weights(list(weights.items()))
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(
                f"Error loading checkpoint from {checkpoint_path}"
            ) from e

        dia.dac_model = DAC.from_pretrained("mlx-community/descript-audio-codec-44khz")

        return dia

    @classmethod
    def from_pretrained(cls, model_name: str = "mlx-community/Dia-1.6B") -> "Dia":
        """Loads the Dia model from a Hugging Face Hub repository.

        Downloads the configuration and checkpoint files from the specified
        repository ID and then loads the model.

        Args:
            model_name: The Hugging Face Hub repository ID (e.g., "NariLabs/Dia-1.6B").

        Returns:
            An instance of the Dia model loaded with weights and set to eval mode.

        Raises:
            FileNotFoundError: If config or checkpoint download/loading fails.
            RuntimeError: If there is an error loading the checkpoint.
        """
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        checkpoint_path = hf_hub_download(
            repo_id=model_name, filename="model.safetensors"
        )
        return cls.from_local(config_path, checkpoint_path)

    def load_weights(self, weights, strict: bool = True):
        self.model.load_weights(weights, strict=strict)

    def sanitize(self, weights):
        return weights

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()

    @property
    def sample_rate(self):
        return self.config.model.sample_rate

    def _create_attn_mask(
        self,
        q_padding_mask_1d: mx.array,
        k_padding_mask_1d: mx.array,
        is_causal: bool = False,
    ) -> mx.array:
        """
        Creates the attention mask (self or cross) mimicking JAX segment ID logic.
        """
        B1, Tq = q_padding_mask_1d.shape
        B2, Tk = k_padding_mask_1d.shape
        assert B1 == B2, "Query and key batch dimensions must match"

        p_mask_q = mx.expand_dims(q_padding_mask_1d, 2)  # Shape [B, Tq, 1]
        p_mask_k = mx.expand_dims(k_padding_mask_1d, 1)  # Shape [B, 1, Tk]

        # Condition A: Non-padding query attends to non-padding key
        non_pad_attends_non_pad = mx.logical_and(
            p_mask_q, p_mask_k
        )  # Shape [B, Tq, Tk]

        # Condition B: Padding query attends to padding key
        pad_attends_pad = mx.logical_and(
            mx.logical_not(p_mask_q), mx.logical_not(p_mask_k)
        )  # Shape [B, Tq, Tk]

        # Combine: True if padding status is compatible (both non-pad OR both pad)
        # This implementation follows Jax TPU splash attention kernel
        mask = mx.logical_or(
            non_pad_attends_non_pad, pad_attends_pad
        )  # Shape [B, Tq, Tk]

        if is_causal:
            # Ensure causality for self-attention (Tq == Tk)
            assert (
                Tq == Tk
            ), "Causal mask requires query and key sequence lengths to be equal"
            # Standard lower-triangular causal mask (True means allow)
            causal_mask_2d = mx.tril(
                mx.ones((Tq, Tk), dtype=mx.bool_)
            )  # Shape [Tq, Tk]
            causal_mask = mx.logical_and(mask, causal_mask_2d)  # Shape [B, Tq, Tk]
            return mx.expand_dims(
                causal_mask, 1
            )  # Shape [B, 1, Tq, Tk] for broadcasting across heads
        else:
            # For cross-attention or non-causal self-attention
            return mx.expand_dims(
                mask, 1
            )  # Shape [B, 1, Tq, Tk] for broadcasting across heads

    def _prepare_text_input(
        self, text: str
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Encodes text prompt, pads, and creates attention mask and positions."""
        text_pad_value = self.config.data.text_pad_value
        max_len = self.config.data.text_length

        byte_text = text.encode("utf-8")
        replaced_bytes = byte_text.replace(b"[S1]", b"\x01").replace(b"[S2]", b"\x02")
        text_tokens = list(replaced_bytes)

        current_len = len(text_tokens)
        padding_needed = max_len - current_len
        if padding_needed <= 0:
            text_tokens = text_tokens[:max_len]
            padded_text_np = np.array(text_tokens, dtype=np.uint8)
        else:
            padded_text_np = np.pad(
                text_tokens,
                (0, padding_needed),
                mode="constant",
                constant_values=text_pad_value,
            ).astype(np.uint8)

        src_tokens = mx.array(padded_text_np, dtype=mx.int32)
        src_tokens = mx.expand_dims(src_tokens, 0)  # [1, S]
        src_positions = mx.expand_dims(mx.arange(max_len, dtype=mx.int32), 0)  # [1, S]

        src_padding_mask = src_tokens != text_pad_value  # [1, S]

        enc_self_attn_mask = self._create_attn_mask(
            src_padding_mask, src_padding_mask, is_causal=False
        )  # [1, S, S]

        return src_tokens, src_positions, src_padding_mask, enc_self_attn_mask

    def _split_turns(self, text: str) -> List[str]:
        """
        Splits a conversation text into segments each containing a maximum of two [S1]/[S2] chunks.
        """
        pattern = re.compile(
            r"\[S1\]\s*(.*?)\s*\[S2\]\s*(.*?)(?=(?:\[S1\])|$)", re.DOTALL
        )
        segments = []
        for s1_chunk, s2_chunk in pattern.findall(text):
            segments.append(f"[S1] {s1_chunk.strip()} [S2] {s2_chunk.strip()}")

        if len(segments) > 1:
            merged_segments = []
            for i in range(0, len(segments), 2):
                if i + 1 < len(segments):
                    merged_segments.append(f"{segments[i]} {segments[i + 1]}")
                else:
                    merged_segments.append(segments[i])
            segments = merged_segments

        return segments

    def generate(
        self,
        text,
        voice: Optional[str] = None,
        temperature: float = 1.3,
        top_p: float = 0.95,
        split_pattern: str = "\n",
        max_tokens: int | None = None,
        verbose: bool = False,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        **kwargs,
    ):
        prompt = text.replace("\\n", "\n").replace("\\t", "\t")
        prompts = prompt.split(split_pattern)

        segments = []
        for p in prompts:
            if "[S1]" in p and "[S2]" in p:
                segments.extend(self._split_turns(p))
            else:
                segments.append(p)

        for segment_index, segment in enumerate(segments):
            time_start = time.perf_counter()

            audio, token_count = self._generate(
                segment,
                max_tokens=max_tokens,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )

            time_end = time.perf_counter()

            samples = audio.shape[0] if audio is not None else 0
            assert samples > 0, "No audio generated"

            sample_rate = self.config.model.sample_rate
            audio_duration_seconds = samples / sample_rate

            elapsed_time = time_end - time_start
            rtf = (
                elapsed_time / audio_duration_seconds
                if audio_duration_seconds > 0
                else 0
            )

            duration_mins = int(audio_duration_seconds // 60)
            duration_secs = int(audio_duration_seconds % 60)
            duration_ms = int((audio_duration_seconds % 1) * 1000)
            duration_hours = int(audio_duration_seconds // 3600)
            duration_str = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}.{duration_ms:03d}"

            yield GenerationResult(
                audio=audio,
                samples=samples,
                sample_rate=sample_rate,
                segment_idx=segment_index,
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

    def _generate(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        use_cfg_filter: bool = True,
        cfg_filter_top_k: int = 35,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generates audio from a text prompt (and optional audio prompt) using the Dia model.

        Returns:
            A numpy array of generated audio samples.
        """
        num_channels = self.config.data.channels
        audio_bos_value = int(self.config.data.audio_bos_value)
        audio_eos_value = int(self.config.data.audio_eos_value)
        audio_pad_value = int(self.config.data.audio_pad_value)
        delay_pattern = self.config.data.delay_pattern
        max_tokens = self.config.data.audio_length if max_tokens is None else max_tokens
        delay_tensor = mx.array(delay_pattern, dtype=mx.int32)
        max_delay_pattern = max(delay_pattern)

        if ref_text is not None:
            text = ref_text.strip() + " " + text

        (
            cond_src_BxS,
            cond_src_positions_BxS,
            cond_src_padding_mask_BxS,
            cond_enc_self_attn_mask_Bx1xSxS,
        ) = self._prepare_text_input(text)

        unc_src_BxS = mx.zeros_like(cond_src_BxS)
        src_BxS = mx.concatenate([unc_src_BxS, cond_src_BxS], axis=0)
        src_positions_BxS = mx.concatenate(
            [cond_src_positions_BxS, cond_src_positions_BxS], axis=0
        )
        src_padding_mask_BxS = mx.concatenate(
            [cond_src_padding_mask_BxS, cond_src_padding_mask_BxS], axis=0
        )
        enc_self_attn_mask_Bx1xSxS = mx.concatenate(
            [cond_enc_self_attn_mask_Bx1xSxS, cond_enc_self_attn_mask_Bx1xSxS], axis=0
        )

        # 2. Encoder Pass
        encoder_out = self.model.encoder(
            x_ids=src_BxS,
            src_positions=src_positions_BxS,
            deterministic=True,
            attn_mask=enc_self_attn_mask_Bx1xSxS,
        )  # Shape: (B, S, E)

        # 3. Prepare Decoder Inputs
        # 3-1. Allocate KV Cache (Static)
        decoder_cross_attention_cache: list[KVCache] = (
            self.model.decoder.precompute_cross_attention_kv(
                max_tokens, encoder_out, src_positions_BxS
            )
        )

        decoder_self_attention_cache: list[KVCache] = []
        for _ in range(self.model.decoder.num_layers):
            decoder_self_attention_cache.append(
                KVCache(
                    self.config.model.decoder.gqa_query_heads,
                    max_tokens,
                    self.config.model.decoder.gqa_head_dim,
                )
            )

        # 3-2. Initialize Decoder Inputs
        generated_BxTxC = mx.full(
            (2, 1, num_channels),
            vals=audio_bos_value,
            dtype=mx.int32,
        )

        current_step = 0
        prompt_len_inc_bos = 1  # Start with BOS length

        # 3-3. Load Audio Prompt (if provided)
        if ref_audio is not None:
            audio_prompt = mx.array(ref_audio)[None, None, ...]  # 1, C, T

            audio_prompt_codebook = audio_to_codebook(
                self.dac_model, audio_prompt, data_config=self.config.data
            )
            audio_prompt_codebook = mx.concatenate(
                [audio_prompt_codebook, audio_prompt_codebook], axis=0
            )
            generated_BxTxC = mx.concatenate(
                [generated_BxTxC, audio_prompt_codebook], axis=1
            )

            prefill_len = generated_BxTxC.shape[1]
            prompt_len_inc_bos = prefill_len
            prefill_tgt_pos = mx.broadcast_to(
                mx.expand_dims(mx.arange(prefill_len), 0), (2, prefill_len)
            )
            prefill_tgt_padding_mask = mx.any(
                generated_BxTxC != audio_pad_value, axis=2
            )

            prefill_self_attn_mask = self._create_attn_mask(
                prefill_tgt_padding_mask,
                prefill_tgt_padding_mask,
                is_causal=True,
            )
            prefill_cross_attn_mask = self._create_attn_mask(
                prefill_tgt_padding_mask,
                src_padding_mask_BxS,
                is_causal=False,
            )

            _ = self.model.decoder(
                tgt_ids_BxTxC=generated_BxTxC,
                encoder_out=encoder_out,
                tgt_positions=prefill_tgt_pos,
                src_positions=src_positions_BxS,
                deterministic=True,
                self_attn_mask=prefill_self_attn_mask,
                cross_attn_mask=prefill_cross_attn_mask,
                self_attention_cache=decoder_self_attention_cache,
                cross_attention_cache=decoder_cross_attention_cache,
            )

            current_step = prefill_len - 1

        # 4. Autoregressive Generation Loop
        eos_detected_channel_0 = False
        eos_countdown = -1
        extra_steps_after_eos = 30

        # Make generated_BxTxC a fixed size tensor
        # Length is either 1 + max tokens or 1 + prompt len + max tokens
        padding = mx.full(
            (2, max_tokens, num_channels),
            vals=-1,
            dtype=mx.int32,
        )
        generated_BxTxC = mx.concatenate([generated_BxTxC, padding], axis=1)

        decode_step = self.model.decoder.decode_step

        tgt_padding_mask = mx.any(
            mx.expand_dims(generated_BxTxC[:, -1, :], 1) != audio_pad_value, axis=2
        )  # [B, 1]

        # Generated tokens are never PAD, so we use fixed mask
        decoder_cross_attn_mask = self._create_attn_mask(
            tgt_padding_mask,  # Query mask [B, 1]
            src_padding_mask_BxS,  # Key mask [B, S]
            is_causal=False,
        )  # [B, 1, 1, S]

        top_k = -1
        if use_cfg_filter and cfg_filter_top_k is not None:
            top_k = cfg_filter_top_k
        sampler = make_sampler(temperature, top_p, top_k=top_k)

        for step in trange(current_step, current_step + max_tokens):
            tgt_ids_Bx1xC = mx.expand_dims(generated_BxTxC[:, step, :], 1)
            tgt_pos_Bx1 = mx.full(
                (2, 1),
                vals=step,
                dtype=mx.int32,
            )

            logits_Bx1xCxV = decode_step(
                tgt_ids_Bx1xC=tgt_ids_Bx1xC,
                tgt_pos_Bx1=tgt_pos_Bx1,
                encoder_out=encoder_out,
                self_attn_mask=None,
                cross_attn_mask=decoder_cross_attn_mask,
                self_attention_cache=decoder_self_attention_cache,
                cross_attention_cache=decoder_cross_attention_cache,
            )

            V = self.config.model.tgt_vocab_size
            logits_last_BxCxV = logits_Bx1xCxV[:, -1, :, :]  # B, C, V
            uncond_logits_CxV = logits_last_BxCxV[0, :, :]
            cond_logits_CxV = logits_last_BxCxV[1, :, :]

            cfg_logits_CxV = cond_logits_CxV + cfg_scale * (
                cond_logits_CxV - uncond_logits_CxV
            )

            logits_CxV = mx.reshape(cfg_logits_CxV, (-1, V))  # C, V

            # Create a mask for setting tokens beyond 1025 to -inf
            inf_mask = mx.full(logits_CxV.shape, -float("inf"), dtype=logits_CxV.dtype)
            keep_mask = mx.concatenate(
                [
                    mx.ones((logits_CxV.shape[0], 1025)),
                    mx.zeros((logits_CxV.shape[0], logits_CxV.shape[1] - 1025)),
                ],
                axis=1,
            )
            logits_CxV = mx.where(keep_mask == 1, logits_CxV, inf_mask)

            # Sample next token
            pred_C = _sample_next_token(
                logits_CxV,
                temperature=temperature,
                sampler=sampler,
            )

            generation_step_index = step - current_step
            if ref_audio is None:
                pred_C = mx.where(
                    generation_step_index >= delay_tensor,
                    pred_C,
                    mx.full(pred_C.shape, audio_bos_value, dtype=pred_C.dtype),
                )

            # Update generated tokens for next step
            pred_C_expanded = mx.broadcast_to(
                mx.expand_dims(pred_C, 0), (2, num_channels)
            )

            # Split the tensor into parts: before the update, the update itself, and after the update
            before_update = generated_BxTxC[:, : step + 1, :]
            new_token = mx.expand_dims(
                pred_C_expanded, 1
            )  # Shape: (2, 1, num_channels)
            after_update = generated_BxTxC[:, step + 2 :, :]
            generated_BxTxC = mx.concatenate(
                [before_update, new_token, after_update], axis=1
            )

            if not eos_detected_channel_0 and pred_C[0] == audio_eos_value:
                print(f"EOS detected at step {step} for channel 0")
                eos_detected_channel_0 = True
                eos_countdown = extra_steps_after_eos

            if eos_countdown > 0:
                step_after_eos = max_delay_pattern - eos_countdown
                for i, d in enumerate(delay_pattern):
                    if step_after_eos == d:
                        # Update EOS token
                        # Create new array with updated value at position i in the current sequence
                        eos_values = mx.zeros((2, num_channels), dtype=mx.int32)
                        eos_values = eos_values.at[:, i].add(audio_eos_value)
                        # Replace the values at step+1
                        generated_BxTxC = generated_BxTxC.astype(mx.int32)
                        generated_BxTxC = generated_BxTxC.at[:, step + 1, :].add(
                            eos_values
                        )
                    elif step_after_eos > d:
                        # Update PAD token
                        # Create new array with updated value at position i in the current sequence
                        pad_values = mx.zeros((2, num_channels), dtype=mx.int32)
                        pad_values = pad_values.at[:, i].add(audio_pad_value)
                        # Replace the values at step+1
                        generated_BxTxC = generated_BxTxC.astype(mx.int32)
                        generated_BxTxC = generated_BxTxC.at[:, step + 1, :].add(
                            pad_values
                        )

                eos_countdown -= 1
                if eos_countdown == 0:
                    break

            generation_step_index = step - current_step + 1

        output_codes = generated_BxTxC[:, prompt_len_inc_bos : step + 1, :]
        generated_codes = output_codes[0]

        audio = codebook_to_audio(
            generated_codes.transpose(1, 0),
            self.dac_model,
            delay_pattern,
            B=1,
            T=max_tokens,
            C=num_channels,
        )
        return audio.squeeze(), generation_step_index
