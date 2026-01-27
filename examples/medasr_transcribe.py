
import argparse
import sys
import time
import numpy as np
import mlx.core as mx
from mlx_audio.stt.utils import load as load_mlx
from transformers import AutoProcessor

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
    parser = argparse.ArgumentParser(description="Transcribe audio file using MedASR on MLX")
    parser.add_argument("audio_file", type=str, help="Path to the audio file to transcribe.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="google/medasr", 
        help="Path to the model (HF repo or local path)"
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
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
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
    transcription = processor.batch_decode(np.array(predicted_ids))[0]
    
    end_time = time.time()
    
    print("-" * 40)
    print("Transcription:")
    print(transcription)
    print(f"\nInference time: {end_time - start_time:.2f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    main()
