# Qwen2-Audio-7B-Instruct MLX Port

## Goal

Port `Qwen/Qwen2-Audio-7B-Instruct` to mlx-audio, supporting full multimodal audio understanding: transcription, translation, emotion detection, audio captioning, sound classification, and voice chat.

## Architecture

```
Audio (16kHz wav)
  -> 128-channel mel spectrogram (Whisper-style)
  -> Qwen2AudioEncoder (2 Conv1d stems + 32 transformer layers + AvgPool1d)
  -> MultiModalProjector (Linear 1280 -> 4096)
  -> Replace <|AUDIO|> token embeddings in input sequence
  -> Qwen2 LLM (from mlx_lm, 28 layers, 3584 hidden)
  -> Text output
```

Source model: `Qwen/Qwen2-Audio-7B-Instruct` (~8.2B params)

## Components

### Audio Encoder (~120 lines)

Whisper-large-v3 encoder architecture with one addition:

- **Conv stem**: Conv1d(128, 1280, kernel=3, padding=1) + GELU, then Conv1d(1280, 1280, kernel=3, stride=2, padding=1) + GELU. The stride=2 conv halves the sequence length.
- **Positional embeddings**: Sinusoidal (fixed, not learned), max 1500 positions.
- **Transformer layers**: 32 pre-norm encoder layers. Each has multi-head self-attention (20 heads, head_dim=64) and FFN (hidden 1280, intermediate 5120, GELU activation).
- **AvgPool1d(kernel=2, stride=2)**: Halves sequence length again. This is the key difference from standard Whisper.
- **Final LayerNorm**.

Output: (batch, seq_len, 1280) where each frame is ~40ms of audio.

### Multi-Modal Projector (~5 lines)

Single `nn.Linear(1280, 4096, bias=True)`. Projects encoder output to LLM hidden size.

### Language Model (imported)

`from mlx_lm.models.qwen2 import Model as Qwen2LM`. Standard Qwen2-7B:
- 32 layers, 32 attention heads, 32 KV heads (no GQA)
- hidden_size=4096, intermediate_size=11008
- vocab_size=156032, RoPE theta=10000
- RMSNorm eps=1e-5

### Audio-Text Merging

During forward pass:
1. Text `input_ids` are embedded by `language_model.model.embed_tokens`
2. Audio features replace positions where `input_ids == audio_token_id` (token 151646, `<|AUDIO|>`)
3. Combined embeddings are fed to the LLM

## File Structure

```
mlx_audio/stt/models/qwen2_audio/
  __init__.py          # exports Model, ModelConfig
  config.py            # EncoderConfig, TextConfig, ModelConfig dataclasses
  qwen2_audio.py       # Encoder, Projector, Model
```

Plus registration in `mlx_audio/stt/utils.py` MODEL_REMAPPING.

## Model Class API

Follows granite_speech pattern:

```python
class Model(nn.Module):
    def __init__(self, config: ModelConfig)
    def __call__(self, input_ids, cache, input_embeddings) -> logits
    def get_audio_features(self, input_features) -> projected_features
    def generate(self, audio, *, max_tokens, temperature, prompt, ...) -> STTOutput
    def sanitize(weights) -> weights        # Conv1d transposition
    def post_load_hook(model, model_path)   # Load tokenizer
    def model_quant_predicate(p, m) -> bool # Only quantize LLM
```

### generate() flow

1. Load audio -> resample to 16kHz if needed
2. Compute 128-channel log-mel spectrogram
3. Encode with Qwen2AudioEncoder
4. Project with linear layer
5. Build prompt: `<|audio_bos|><|AUDIO|>...<|AUDIO|><|audio_eos|>` + optional text instruction
6. Replace `<|AUDIO|>` embeddings with projected audio features
7. Run `mlx_lm.generate.generate_step` for autoregressive generation
8. Return STTOutput

### Feature extraction

128-channel mel spectrogram at 16kHz, matching Whisper-large-v3 preprocessing:
- n_fft=400, hop_length=160, window=hann(400)
- 128 mel filterbanks
- Log-mel with padding to 30s (or dynamic length)

### Prompt construction

Uses the Qwen2-Audio chat template via `AutoProcessor.apply_chat_template`. Special tokens:
- `<|audio_bos|>` (151647): start of audio
- `<|AUDIO|>` (151646): audio placeholder (repeated N times for N audio frames)
- `<|audio_eos|>` (151648): end of audio

Number of audio tokens = encoder output sequence length (after both halvings).

## Weight Mapping

HuggingFace weight prefixes map directly:
- `audio_tower.*` -> `encoder.*`
- `multi_modal_projector.*` -> `projector.*`
- `language_model.*` -> `language_model.*`

### sanitize()

- Transpose Conv1d weights from PyTorch (out, in, kernel) to MLX (out, kernel, in) for the 2 encoder conv stems
- Skip quantization-related weights (`scales`) if already converted
- Rename `audio_tower.` -> `encoder.`
- Rename `multi_modal_projector.` -> `projector.`

### model_quant_predicate()

Only quantize the language model. Skip encoder and projector (same as granite_speech).

## Testing

- Transcription: Compare output against HuggingFace transformers for same audio input
- Audio understanding: Test with environmental sounds, music
- Voice chat: Audio-only input without text instruction
- Streaming: Verify token-by-token generation works
