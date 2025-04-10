import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import re

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import stream_generate
from mlx_lm.models import cache as cache_utils
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import AutoTokenizer

from mlx_audio.codec.models.snac import SNAC

from ..base import GenerationResult


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    tokeniser_name: str = "mlx-community/orpheus-3b-0.1-ft-bf16"

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


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class LlamaModel(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelConfig, **kwargs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokeniser_name)
        self.model = LlamaModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        # Remove unused precomputed rotary freqs
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

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
        # Use the smart text chunking instead of simple split
        text_chunks = self._smart_text_chunking(text)

        # Process one chunk at a time to avoid the squeeze(0) error
        all_samples = []
        time_start = time.time()

        for idx, chunk in enumerate(text_chunks):
            if verbose:
                print(f"Processing chunk {idx+1}/{len(text_chunks)}: {chunk[:50]}...")

            # Process each chunk separately
            single_input_ids, _ = self.prepare_input_ids(
                [chunk],
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

            # Process this single chunk
            for i, response in enumerate(
                tqdm(
                    stream_generate(
                        self,
                        tokenizer=self.tokenizer,
                        prompt=single_input_ids.squeeze(0),
                        max_tokens=max_tokens,
                        sampler=sampler,
                        logits_processors=logits_processors,
                    ),
                    total=max_tokens,
                    disable=not verbose,
                )
            ):
                next_token = mx.array([response.token])
                single_input_ids = mx.concatenate(
                    [single_input_ids, next_token[None, :]], axis=1
                )
                if i % 50 == 0:
                    mx.clear_cache()

                if next_token == 128258:
                    break

            # Process output for this chunk
            single_code_lists = self.parse_output(single_input_ids)

            for code_list in single_code_lists:
                samples = decode_audio_from_codes(code_list)
                all_samples.append(samples)

        time_end = time.time()

        # Yield results for each sample
        for i in range(len(all_samples)):
            audio = all_samples[i][0]

            samples = audio.shape[0] if audio is not None else 0
            assert samples > 0, "No audio generated"

            # For token count, we don't have a single input_ids anymore
            token_count = 1000  # Use an estimate

            # Calculate audio duration in seconds
            sample_rate = 24000  # Assuming 24kHz sample rate, adjust if different
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

    def _smart_text_chunking(self, text, max_chunk_chars=800):
        """
        Intelligently split text into chunks for better TTS processing.

        Steps:
        1. Split by paragraphs (newlines)
        2. For paragraphs longer than max_chunk_chars, split by sentences
        3. Ensure no chunk exceeds max_chunk_chars

        Args:
            text (str): Input text to split
            max_chunk_chars (int): Maximum characters per chunk

        Returns:
            List[str]: List of text chunks to process
        """
        # Handle empty text case
        if not text or not text.strip():
            return [""]

        # Normalize line endings and remove extra whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Normalize multiple newlines

        # First split by paragraphs
        paragraphs = text.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Process each paragraph, potentially splitting long ones
        chunks = []
        for para in paragraphs:
            # If paragraph is short enough, keep it as is
            if len(para) <= max_chunk_chars:
                chunks.append(para)
                continue

            # Split long paragraph into sentences
            # Look for sentence boundaries (period, question mark, exclamation mark)
            # followed by space or end of string
            sentences = re.split(r"([.!?])\s+", para)

            # Re-join the punctuation that was separated in the split
            complete_sentences = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    complete_sentences.append(sentences[i] + sentences[i + 1])
                else:
                    complete_sentences.append(sentences[i])

            # If we couldn't split into sentences properly, fall back to character limit
            if not complete_sentences:
                complete_sentences = [para]

            # Build chunks from sentences, trying not to exceed max_chunk_chars
            current_chunk = ""
            for sentence in complete_sentences:
                # If individual sentence exceeds max length, split it
                if len(sentence) > max_chunk_chars:
                    # Add any accumulated chunk first
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""

                    # Split long sentence by clauses (commas)
                    clauses = sentence.split(", ")
                    sub_chunk = ""

                    for clause in clauses:
                        if (
                            len(sub_chunk) + len(clause) + 2 > max_chunk_chars
                        ):  # +2 for ", "
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = clause
                        else:
                            sub_chunk = (
                                sub_chunk + ", " + clause if sub_chunk else clause
                            )

                    if sub_chunk:
                        chunks.append(sub_chunk)

                # Normal case - check if adding this sentence would exceed the limit
                elif (
                    len(current_chunk) + len(sentence) + 1 > max_chunk_chars
                ):  # +1 for space
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk = (
                        current_chunk + " " + sentence if current_chunk else sentence
                    )

            # Add any remaining chunk
            if current_chunk:
                chunks.append(current_chunk)

        # Verify and clean the chunks
        final_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:  # Only add non-empty chunks
                final_chunks.append(chunk)

        # If we somehow got no chunks, return the original text
        if not final_chunks:
            return [text]

        return final_chunks
