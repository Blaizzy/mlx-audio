import argparse
import sys
import time

import mlx.core as mx
import numpy as np
from transformers import AutoProcessor

from mlx_audio.stt.utils import load as load_mlx

# Try to import load_audio from mlx_audio, fallback if needed
try:
    from mlx_audio.stt.utils import load_audio
except ImportError:
    # Fallback to simple librosa load if internal util not available
    import librosa

    def load_audio(file):
        audio, _ = librosa.load(file, sr=16000)
        return mx.array(audio)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio file using MedASR on MLX"
    )
    parser.add_argument(
        "audio_file", type=str, help="Path to the audio file to transcribe."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/medasr",
        help="Path to the model (HF repo or local path)",
    )

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    try:
        model = load_mlx(args.model_path)
    except Exception as e:
        print(f"Error loading MLX model: {e}")
        sys.exit(1)

    print("Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading processor: {e}")
        sys.exit(1)

    print(f"Loading audio: {args.audio_file}")
    try:
        audio = load_audio(args.audio_file)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        sys.exit(1)

    print("Transcribing...")
    start_time = time.time()

    # Preprocess
    if isinstance(audio, mx.array):
        audio = np.array(audio)

    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    input_features = mx.array(inputs.input_features)

    # Inference
    logits = model(input_features)

    # Decode
    predicted_ids = mx.argmax(logits, axis=-1)
    predicted_ids = np.array(predicted_ids)

    # 1. Get the pad token ID (blank token)
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(processor.tokenizer, "blank_token_id", 0)

    # 2. Extract the sequence for the first item in the batch
    seq = predicted_ids[0] if len(predicted_ids.shape) > 1 else predicted_ids

    # 3. Collapse CTC tokens manually
    collapsed = []
    prev = -1

    for tk_id in seq:
        val = int(tk_id) # Ensure native int/compatibility with arrays
        
        # Standard CTC rule: drop blanks, drop adjacent duplicates
        if val != prev and val != pad_id:
            collapsed.append(val)
            
        prev = val

    # 4. Decode the cleaned list of IDs
    transcription_raw = processor.tokenizer.decode(collapsed)

    # Cleanup end-of-sentence or special tokens commonly in MedASR
    transcription = transcription_raw.replace("</s>", "").strip()

    end_time = time.time()

    print("-" * 40)
    print("Transcription:")
    print(transcription)
    print(f"\nInference time: {end_time - start_time:.2f} seconds")
    print("-" * 40)


if __name__ == "__main__":
    main()
