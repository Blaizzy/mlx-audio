# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)

"""5Hz Language Model for ACE-Step music generation.

This module provides the ACEStepLM class which generates audio codes from text prompts
using a Qwen3-based language model. The audio codes are then decoded to latent hints
that guide the diffusion process for improved generation quality.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LMConfig:
    """Configuration for the 5Hz Language Model.

    Available models:
        - "0.6B": Smallest, fastest (default, recommended for most use cases)
        - "4B": Largest, highest quality but slower

    Note: The 1.7B bundled model requires manual weight conversion and is not
    currently supported. Use 0.6B or 4B instead.
    """

    model_size: str = "0.6B"  # "0.6B" or "4B"
    max_new_tokens: int = 3000
    temperature: float = 0.8
    top_k: int = 200
    top_p: float = 0.95
    repetition_penalty: float = 1.05

    # Base model path (set when loading from ACE-Step1.5)
    base_model_path: Optional[str] = None

    # Model ID mapping for standalone models (HuggingFace repos)
    _STANDALONE_MODEL_IDS = {
        "0.6B": "ACE-Step/acestep-5Hz-lm-0.6B",
        "4B": "ACE-Step/acestep-5Hz-lm-4B",
    }

    @property
    def model_id(self) -> str:
        """Get the HuggingFace model ID."""
        # Use standalone HuggingFace model
        if self.model_size in self._STANDALONE_MODEL_IDS:
            return self._STANDALONE_MODEL_IDS[self.model_size]

        # Fallback for custom model paths
        return self.model_size


class ACEStepLM:
    """5Hz Language Model for generating audio codes.

    This class wraps a Qwen3-based LM that generates audio codes from text/lyrics.
    The generation happens in two phases:
    1. Phase 1: Generate Chain-of-Thought metadata in <think>...</think> tags
    2. Phase 2: Generate audio codes <|audio_code_XXXXX|>

    The audio codes are then decoded to 25Hz latent hints using the quantizer
    and detokenizer from the main ACE-Step model.
    """

    # FSQ levels for decoding audio codes
    FSQ_LEVELS = [8, 8, 8, 5, 5, 5]  # Product = 64000
    MAX_AUDIO_CODE = 63999

    def __init__(self, config: Optional[LMConfig] = None):
        """Initialize the LM.

        Args:
            config: LM configuration. If None, uses defaults.
        """
        self.config = config or LMConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the Qwen3 model and tokenizer."""
        if self._loaded:
            return

        try:
            import mlx_lm
        except ImportError:
            raise ImportError(
                "mlx_lm is required for 5Hz LM support. "
                "Install it with: pip install mlx-lm"
            )

        print(f"Loading 5Hz LM: {self.config.model_id}...")
        self.model, self.tokenizer = mlx_lm.load(self.config.model_id)
        self._loaded = True
        print("5Hz LM loaded successfully.")

    def offload(self) -> None:
        """Offload the model to free memory."""
        if not self._loaded:
            return

        self.model = None
        self.tokenizer = None
        self._loaded = False
        mx.clear_cache()

    def _format_prompt(
        self,
        caption: str,
        lyrics: str,
        duration: int,
        language: str = "en",
    ) -> str:
        """Format the prompt for the LM.

        Args:
            caption: Music description/prompt
            lyrics: Song lyrics
            duration: Duration in seconds
            language: Language code

        Returns:
            Formatted prompt string
        """
        instruction = "Generate audio semantic tokens based on the given conditions"

        # Format lyrics section
        if lyrics and lyrics.strip():
            lyrics_section = f"# Lyrics\n{lyrics}"
        else:
            lyrics_section = "# Lyrics\n[instrumental]"

        prompt = f"""# Instruction
{instruction}

# Caption
{caption}

{lyrics_section}

# Metas
- duration: {duration} seconds
<|endoftext|>
"""
        return prompt

    def _apply_chat_template(self, prompt: str, enable_thinking: bool = True) -> str:
        """Apply the Qwen3 chat template.

        Args:
            prompt: The user prompt
            enable_thinking: Whether to enable CoT thinking

        Returns:
            Formatted prompt with chat template
        """
        # Qwen3 chat template
        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        else:
            # Fallback for tokenizers without apply_chat_template
            return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def generate_audio_codes(
        self,
        caption: str,
        lyrics: str = "",
        duration: int = 30,
        language: str = "en",
        seed: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate audio codes from text/lyrics.

        Args:
            caption: Music description/prompt
            lyrics: Song lyrics (empty string for instrumental)
            duration: Duration in seconds
            language: Language code
            seed: Random seed for reproducibility

        Returns:
            Tuple of (audio_codes_string, metadata_dict)
        """
        if not self._loaded:
            self.load()

        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        # Format prompt
        prompt = self._format_prompt(caption, lyrics, duration, language)
        formatted_prompt = self._apply_chat_template(prompt, enable_thinking=True)

        # Set seed if provided
        if seed is not None:
            mx.random.seed(seed)

        # Create sampler with temperature and top_p
        sampler = make_sampler(
            temp=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )

        # Generate with the LM
        output = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=self.config.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )

        # Parse output
        metadata, audio_codes = self.parse_output(output)

        return audio_codes, metadata

    def parse_output(self, output_text: str) -> Tuple[Dict[str, Any], str]:
        """Parse LM output to extract metadata and audio codes.

        Expected format:
        <think>
        bpm: 120
        caption: A calm piano melody
        duration: 30
        genres: Lo-fi
        keyscale: C major
        language: en
        timesignature: 4/4
        </think>

        <|audio_code_56535|><|audio_code_62918|>...

        Args:
            output_text: Raw LM output

        Returns:
            Tuple of (metadata_dict, audio_codes_string)
        """
        metadata = {}
        audio_codes = ""

        # Extract audio codes - find all <|audio_code_XXX|> patterns
        code_pattern = r"<\|audio_code_\d+\|>"
        code_matches = re.findall(code_pattern, output_text)
        if code_matches:
            audio_codes = "".join(code_matches)

        # Extract metadata from <think>...</think> section
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, output_text, re.DOTALL)

        if think_match:
            reasoning_text = think_match.group(1).strip()
            metadata = self._parse_metadata(reasoning_text)

        return metadata, audio_codes

    def _parse_metadata(self, text: str) -> Dict[str, Any]:
        """Parse metadata from the reasoning text.

        Args:
            text: Text from inside <think> tags

        Returns:
            Dictionary of metadata
        """
        metadata = {}

        # Define fields to extract
        field_patterns = {
            "bpm": r"bpm:\s*(\d+)",
            "caption": r"caption:\s*(.+?)(?:\n|$)",
            "duration": r"duration:\s*(\d+)",
            "genres": r"genres:\s*(.+?)(?:\n|$)",
            "keyscale": r"keyscale:\s*(.+?)(?:\n|$)",
            "language": r"language:\s*(\w+)",
            "timesignature": r"timesignature:\s*(.+?)(?:\n|$)",
        }

        for field, pattern in field_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Convert numeric fields
                if field in ["bpm", "duration"]:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                metadata[field] = value

        return metadata

    def parse_audio_codes(self, code_string: str) -> List[int]:
        """Extract integer audio codes from the code string.

        Args:
            code_string: String of audio codes like "<|audio_code_123|><|audio_code_456|>"

        Returns:
            List of integer code values, clamped to valid range [0, 63999]
        """
        if not code_string:
            return []

        pattern = r"<\|audio_code_(\d+)\|>"
        matches = re.findall(pattern, code_string)

        codes = []
        for match in matches:
            code_value = int(match)
            # Clamp to valid range
            clamped = max(0, min(code_value, self.MAX_AUDIO_CODE))
            codes.append(clamped)

        return codes

    def codes_to_fsq_indices(self, codes: List[int]) -> mx.array:
        """Convert flat audio codes to FSQ indices.

        The audio codes are single integers that encode multiple FSQ levels.
        This function converts them back to per-level indices.

        Args:
            codes: List of audio code integers (0-63999)

        Returns:
            mx.array of shape [1, num_codes, num_quantizers] where num_quantizers=1
            for the 5Hz LM (it uses a single codebook)
        """
        if not codes:
            return mx.zeros((1, 0, 1), dtype=mx.int32)

        # For the 5Hz LM, each code is a single index into a 64000-entry codebook
        # The quantizer expects shape [batch, time, num_quantizers]
        indices = mx.array(codes, dtype=mx.int32)
        indices = indices[None, :, None]  # [1, T, 1]

        return indices

    def decode_codes_to_latents(
        self,
        code_string: str,
        quantizer: nn.Module,
        detokenizer: nn.Module,
        target_len: int,
    ) -> mx.array:
        """Decode audio codes to 25Hz latent hints.

        Args:
            code_string: String of audio codes
            quantizer: ResidualFSQ quantizer from the main model
            detokenizer: AudioTokenDetokenizer from the main model
            target_len: Target length in 25Hz frames

        Returns:
            Latent hints of shape [1, target_len, audio_dim]
        """
        # Parse codes
        codes = self.parse_audio_codes(code_string)
        if not codes:
            return None

        # Convert to indices
        indices = self.codes_to_fsq_indices(codes)

        # Get 5Hz latents from quantizer
        latents_5hz = quantizer.get_output_from_indices(indices)

        # Upsample to 25Hz using detokenizer
        latents_25hz = detokenizer(latents_5hz)

        # Adjust length to match target
        current_len = latents_25hz.shape[1]
        if current_len < target_len:
            # Pad with last frame
            pad_len = target_len - current_len
            padding = mx.broadcast_to(
                latents_25hz[:, -1:, :], (1, pad_len, latents_25hz.shape[-1])
            )
            latents_25hz = mx.concatenate([latents_25hz, padding], axis=1)
        elif current_len > target_len:
            # Truncate
            latents_25hz = latents_25hz[:, :target_len, :]

        return latents_25hz
