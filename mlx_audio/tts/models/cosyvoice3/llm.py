"""CosyVoice3 speech-token language model (Qwen2 backbone).

Architecture / token layout:
  * A Qwen2 decoder-only backbone (via mlx_lm) consumes INPUT EMBEDDINGS, not
    token ids — text is embedded by Qwen2's own embed_tokens, speech tokens by
    a separate speech_embedding table.
  * sos / eos / task_id / fill special tokens live in the speech_embedding table
    at indices speech_token_size + {0, 1, 2, 3}.
  * llm_decoder projects hidden states to (speech_token_size + 200) logits.
  * the text stream MUST contain the <|endofprompt|> token (151646).

Inference sequence:

    lm_input = [ sos_emb, text_emb(prompt+target), task_id_emb, prompt_speech_emb ]

then autoregressively:
    y = qwen2(inputs_embeds=lm_input, cache=cache)[:, -1]
    logits = llm_decoder(y)
    next_id = ras_sampling(logits, decoded, top_p=0.8, top_k=25, win_size=10, tau_r=0.1)
    stop when next_id in stop_token_ids (eos == speech_token_size)
    append speech_embedding[next_id] to the running input (KV-cached).
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .config import LLMConfig
from .sampling import ras_sampling


class CosyVoice3LM(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config
        self.speech_token_size = config.speech_token_size
        self.llm_input_size = config.llm_input_size
        self.llm_output_size = config.llm_output_size

        # special tokens (offsets into the speech_embedding table)
        self.sos = config.speech_token_size + 0
        self.eos_token = config.speech_token_size + 1
        self.task_id = config.speech_token_size + 2
        self.fill_token = config.speech_token_size + 3
        # any id >= speech_token_size terminates decoding
        self.stop_token_ids = list(
            range(config.speech_token_size, config.speech_token_size + config.speech_vocab_extra)
        )

        # Qwen2 backbone (mlx_lm), built eagerly: the mlx-audio loader calls
        # model.load_weights(...) right after construction, so the Qwen2
        # parameter tree must already exist for its weights to load. See
        # build_backbone() below.
        self.llm: Optional[nn.Module] = None
        self.build_backbone()

        out_dim = config.speech_token_size + config.speech_vocab_extra
        self.llm_decoder = nn.Linear(config.llm_output_size, out_dim, bias=False)
        self.speech_embedding = nn.Embedding(out_dim, config.llm_input_size)

    # ------------------------------------------------------------------ #
    def build_backbone(self):
        """Instantiate the mlx_lm Qwen2 backbone from the LLM config.

        Called from ``__init__``. Idempotent — safe to call again (e.g. after
        changing ``self.config``) if the backbone needs rebuilding.
        """
        from mlx_lm.models.qwen2 import Model as Qwen2ForCausalLM
        from mlx_lm.models.qwen2 import ModelArgs as Qwen2Args

        args = Qwen2Args(
            model_type="qwen2",
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            intermediate_size=self.config.intermediate_size,
            num_attention_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            rms_norm_eps=self.config.rms_norm_eps,
            vocab_size=self.config.vocab_size,
            rope_theta=self.config.rope_theta,
            tie_word_embeddings=self.config.tie_word_embeddings,
            max_position_embeddings=self.config.max_position_embeddings,
        )
        self.llm = Qwen2ForCausalLM(args)

    def text_embed(self, text_tokens: mx.array) -> mx.array:
        """Embed text ids with the Qwen2 embed_tokens table."""
        if self.llm is None:
            raise RuntimeError("call build_backbone() first")
        return self.llm.model.embed_tokens(text_tokens)

    def _special_emb(self, idx: int) -> mx.array:
        return self.speech_embedding(mx.array([[idx]], dtype=mx.int32))  # (1,1,D)

    def inference(
        self,
        text: mx.array,
        prompt_text: mx.array,
        prompt_speech_token: mx.array,
        sampling: int = 25,
        max_token_text_ratio: float = 20.0,
        min_token_text_ratio: float = 2.0,
        top_p: float = 0.8,
        top_k: int = 25,
        win_size: int = 10,
        tau_r: float = 0.1,
    ) -> List[int]:
        """Autoregressively generate speech tokens for one utterance.

        text / prompt_text: (1, L) int ids (Qwen2 vocab; must contain 151646).
        prompt_speech_token: (1, P) int speech token ids (0 if none).
        Returns list of generated speech token ids (eos excluded).
        """
        from mlx_lm.models.cache import make_prompt_cache

        if self.llm is None:
            raise RuntimeError("call build_backbone() first")

        # 1. concat prompt_text + text and embed via Qwen2
        text = mx.concatenate([prompt_text, text], axis=1)
        if self.config.endofprompt_token not in text.tolist()[0]:
            raise ValueError(
                f"<|endofprompt|> ({self.config.endofprompt_token}) not detected "
                "in CosyVoice3 text or prompt_text, check your input! (zero_shot "
                "mode embeds it automatically via prompt_text; cross_lingual / "
                "instruct usage must include it explicitly in `text`, e.g. "
                "'You are a helpful assistant.<|endofprompt|>...')"
            )
        text_emb = self.text_embed(text)  # (1, L, D)

        # 2. build lm_input = [sos, text_emb, task_id, prompt_speech_emb]
        sos_emb = self._special_emb(self.sos)
        task_id_emb = self._special_emb(self.task_id)
        if prompt_speech_token is not None and prompt_speech_token.shape[1] > 0:
            prompt_speech_emb = self.speech_embedding(prompt_speech_token)
        else:
            prompt_speech_emb = mx.zeros((1, 0, self.llm_input_size), dtype=text_emb.dtype)
        lm_input = mx.concatenate(
            [sos_emb, text_emb, task_id_emb, prompt_speech_emb], axis=1
        )

        # 3. length bounds relative to target text length
        text_len = text.shape[1] - prompt_text.shape[1]
        min_len = int(text_len * min_token_text_ratio)
        max_len = int(text_len * max_token_text_ratio)

        # 4. step-by-step decode with KV cache
        cache = make_prompt_cache(self.llm)
        out_tokens: List[int] = []
        cur = lm_input
        for i in range(max_len):
            # Qwen2Model uses input_embeddings when provided; inputs is ignored
            # but must be a valid array for shape bookkeeping.
            dummy_ids = mx.zeros((cur.shape[0], cur.shape[1]), dtype=mx.int32)
            h = self.llm.model(inputs=dummy_ids, cache=cache, input_embeddings=cur)
            logits = self.llm_decoder(h[:, -1])  # (1, out_dim)
            logits = logits[0]
            ignore_eos = i < min_len
            if ignore_eos:
                # forbid all stop tokens until min_len reached
                mask = mx.zeros_like(logits)
                idx = mx.array(self.stop_token_ids, dtype=mx.int32)
                mask = mask.at[idx].add(mx.array(float("-inf")))
                logits = logits + mask
            top_id = ras_sampling(
                logits, out_tokens, sampling, top_p=top_p, top_k=top_k,
                win_size=win_size, tau_r=tau_r,
            )
            if top_id in self.stop_token_ids:
                break
            out_tokens.append(top_id)
            cur = self.speech_embedding(mx.array([[top_id]], dtype=mx.int32))
        return out_tokens

    def sanitize(self, weights: dict) -> dict:
        """Map checkpoint keys to this module's tree."""
        from .convert import convert_llm_weights

        return convert_llm_weights(weights, tie_word_embeddings=self.config.tie_word_embeddings)
