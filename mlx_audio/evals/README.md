# MLX-Audio Evaluations

This module provides evaluation benchmarks for MLX-Audio models, enabling you to measure model performance against standardized datasets.

## Available Benchmarks

### InstructTTSEval

[InstructTTSEval](https://arxiv.org/abs/2506.16381) is a benchmark for evaluating TTS systems' ability to follow complex natural-language style instructions. It measures how well models can synthesize speech that matches specified acoustic properties, styles, and personas.

**Dataset**: [CaasiHUANG/InstructTTSEval](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval)

## Installation

Install mlx-audio with the evals dependencies:

```bash
pip install mlx-audio[evals]
```

Or install the dependencies manually:

```bash
pip install datasets google-generativeai openai
```

## Features

- **Three Instruction Types**:
  - **APS (Acoustic Property Specification)**: Low-level acoustic attribute descriptions (gender, pitch, speed, volume, age, emotion, etc.)
  - **DSD (Detailed Style Description)**: High-level natural language style instructions
  - **RP (Role-Play)**: Context-based scenario instructions for persona-driven synthesis

- **Multilingual Support**:
  - English (`en`): 1,000 samples
  - Chinese (`zh`): 1,000 samples

- **Flexible Evaluation**:
  - LLM-as-judge scoring with Gemini or OpenAI
  - Audio-only generation mode (skip scoring)
  - Configurable sampling parameters

- **Comprehensive Output**:
  - Generated audio files (WAV format)
  - Per-sample results (CSV)
  - Summary statistics (JSON)

## Usage

### Basic Usage - Audio Generation Only

Generate audio for all samples without LLM scoring:

```bash
python -m mlx_audio.evals.instruct_tts_eval \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
    --split en \
    --evaluator skip \
    --save-audio \
    --output-dir results/instruct_tts_eval
```

### With Gemini Scoring

Evaluate instruction-following with Gemini as the judge:

```bash
python -m mlx_audio.evals.instruct_tts_eval \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
    --split en \
    --evaluator gemini \
    --api-key $GOOGLE_API_KEY \
    --save-audio \
    --output-dir results/instruct_tts_eval
```

### With OpenAI Scoring

```bash
python -m mlx_audio.evals.instruct_tts_eval \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
    --split en \
    --evaluator openai \
    --api-key $OPENAI_API_KEY \
    --save-audio \
    --output-dir results/instruct_tts_eval
```

### Debugging with Limited Samples

Test on a small subset before running full evaluation:

```bash
python -m mlx_audio.evals.instruct_tts_eval \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
    --split en \
    --max-samples 10 \
    --evaluator skip \
    --save-audio \
    --verbose
```

### Evaluate Specific Instruction Types

```bash
python -m mlx_audio.evals.instruct_tts_eval \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
    --split en \
    --instruction-types APS DSD \
    --evaluator gemini \
    --api-key $GOOGLE_API_KEY
```

### Chinese Evaluation

```bash
python -m mlx_audio.evals.instruct_tts_eval \
    --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16 \
    --split zh \
    --evaluator gemini \
    --api-key $GOOGLE_API_KEY
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Path or HuggingFace repo ID of the TTS model | Required |
| `--dataset` | HuggingFace dataset name | `CaasiHUANG/InstructTTSEval` |
| `--split` | Dataset split (`en` or `zh`) | `en` |
| `--instruction-types` | Instruction types to evaluate (`APS`, `DSD`, `RP`) | All three |
| `--max-samples` | Maximum samples to evaluate (for debugging) | None (all) |
| `--output-dir` | Directory to save results | `results/instruct_tts_eval` |
| `--max-tokens` | Maximum tokens to generate | `2048` |
| `--temperature` | Sampling temperature | `0.7` |
| `--voice` | Voice/speaker name (model-specific) | Auto-detected |
| `--evaluator` | LLM evaluator (`gemini`, `openai`, `skip`) | `skip` |
| `--api-key` | API key for evaluator service | None |
| `--save-audio` | Save generated audio files | False |
| `--verbose` | Print detailed output | False |
| `--seed` | Random seed | `42` |

## Evaluation Metrics

### Scoring Methodology

InstructTTSEval uses an **LLM-as-judge** approach with binary scoring:

| Score | Criteria |
|-------|----------|
| **TRUE** | The sample's primary style attributes (gender, pitch, rate, emotion) align with the instruction without conflict |
| **FALSE** | At least one key style attribute clearly conflicts with the instruction, or the overall style deviates from the prompt |

The final score for each instruction type is the **percentage of TRUE responses** across all samples.

### Reported Scores

Reference scores for Qwen3-TTS models on InstructTTSEval:

#### Qwen3-TTS-12Hz-1.7B-CustomVoice

| Language | APS | DSD | RP |
|----------|-----|-----|-----|
| Chinese | 83.0 | 77.8 | 61.2 |
| English | 77.3 | 77.1 | 63.7 |

#### Qwen3-TTS-12Hz-1.7B-VoiceDesign

| Language | APS | DSD | RP |
|----------|-----|-----|-----|
| Chinese | 85.2 | 81.1 | 65.1 |
| English | 82.9 | 82.4 | 68.4 |

### Human-LLM Agreement

The benchmark authors validated Gemini's evaluation against human annotators:

| Instruction Type | Agreement Rate |
|------------------|----------------|
| APS | 87% |
| DSD | 79% |
| RP | 71% |
| **Average** | **79%** |

## Output Files

After running an evaluation, you'll find:

```
results/instruct_tts_eval/
├── {model_name}_InstructTTSEval_{split}.csv    # Per-sample results
├── {model_name}_InstructTTSEval_{split}.json   # Summary statistics
└── audio/                                       # Generated audio (if --save-audio)
    ├── en_0_APS.wav
    ├── en_0_DSD.wav
    ├── en_0_RP.wav
    └── ...
```

### CSV Format

| Column | Description |
|--------|-------------|
| `id` | Sample identifier (e.g., `en_0`) |
| `instruction_type` | Type of instruction (`APS`, `DSD`, `RP`) |
| `text` | Text that was synthesized |
| `instruction` | Style instruction given to the model |
| `generated` | Whether audio was successfully generated |
| `score` | Evaluation result (`True`, `False`, or empty if skipped) |

### JSON Summary

```json
{
  "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
  "dataset": "CaasiHUANG/InstructTTSEval",
  "split": "en",
  "instruction_types": ["APS", "DSD", "RP"],
  "evaluator": "gemini",
  "total_samples": 1000,
  "scores": {
    "APS": {"correct": 773, "total": 1000, "accuracy": 77.3},
    "DSD": {"correct": 771, "total": 1000, "accuracy": 77.1},
    "RP": {"correct": 637, "total": 1000, "accuracy": 63.7}
  },
  "average_score": 72.7
}
```

## Python API

You can also use the evaluation functions programmatically:

```python
from mlx_audio.evals.instruct_tts_eval import (
    load_dataset,
    run_inference,
    evaluate_with_llm,
    save_audio,
)
from mlx_audio.tts.utils import load as load_tts_model

# Load model
model = load_tts_model("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16")

# Load dataset
dataset = load_dataset(split="en", max_samples=10)

# Generate and evaluate
for sample in dataset:
    audio = run_inference(
        model=model,
        text=sample["text"],
        instruction=sample["APS"],
        voice="vivian",
        lang_code="en",
    )

    if audio is not None:
        save_audio(audio, "output.wav", sample_rate=model.sample_rate)
```

## References

- **Paper**: [InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems](https://arxiv.org/abs/2506.16381)
- **Dataset**: [CaasiHUANG/InstructTTSEval](https://huggingface.co/datasets/CaasiHUANG/InstructTTSEval)
- **GitHub**: [InstructTTSEval](https://github.com/KexinHUANG19/InstructTTSEval)
