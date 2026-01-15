
import argparse
import sys
import numpy as np
import mlx.core as mx

# Ensure we can find local mlx-audio if running from root without install
# Ensure we can find local mlx-audio if running from root without install
from mlx_audio.stt.utils import load as load_mlx
import torch
from transformers import AutoModelForCTC, AutoProcessor

def record_audio(duration=None, sr=16000):
    try:
        import sounddevice as sd
    except ImportError:
        print("Please install sounddevice: pip install sounddevice")
        sys.exit(1)

    print("-" * 40)
    print("Model loaded.")
    if duration:
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, blocking=True)
    else:
        print("Recording... Press Ctrl+C to stop.")
        try:
            # Record indefinitely until keyboard interrupt
            # Helper to collect data
            q = []
            def callback(indata, frames, time, status):
                if status:
                    print(status, file=sys.stderr)
                q.append(indata.copy())
            
            with sd.InputStream(samplerate=sr, channels=1, callback=callback):
                while True:
                    sd.sleep(100)
        except KeyboardInterrupt:
            print("\nRecording stopped.")
            if not q:
                return np.array([])
            audio = np.concatenate(q)
    
    print("-" * 40)
    return audio.flatten()

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using MedASR on MLX")
    parser.add_argument("audio_file", type=str, nargs="?", help="Path to the audio file to transcribe. If omitted, starts recording.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="medasr_mlx_converted", 
        help="Path to the converted MLX model"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Duration to record in seconds (optional, default: unlimited/Ctrl+C)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mlx",
        choices=["mlx", "transformers"],
        help="Backend to use for inference: 'mlx' (default) or 'transformers'"
    )
    
    args = parser.parse_args()
    
    model = None
    if args.backend == "mlx":
        print(f"Loading MLX model from {args.model_path}...")
        try:
            model = load_mlx(args.model_path)
        except Exception as e:
            print(f"Error loading MLX model: {e}")
            sys.exit(1)
    else:
        print("Loading Transformers model (google/medasr)...")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        try:
            model = AutoModelForCTC.from_pretrained("google/medasr", trust_remote_code=True)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading Transformers model: {e}")
            sys.exit(1)
        
    sr = 16000
    
    if args.audio_file:
        print(f"Transcribing {args.audio_file}...")
        import librosa
        try:
            speech, sample_rate = librosa.load(args.audio_file, sr=sr)
        except FileNotFoundError:
            print(f"Audio file not found: {args.audio_file}")
            sys.exit(1)
    else:
        # Recording mode
        speech = record_audio(duration=args.duration, sr=sr)
        if len(speech) == 0:
            print("No audio recorded.")
            sys.exit(0)

    try:
        from transformers import AutoProcessor
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        processor = AutoProcessor.from_pretrained("google/medasr", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor: {e}")
        print("Please ensure transformers is installed.")
        sys.exit(1)
        
    inputs = processor(speech, sampling_rate=sr, return_tensors="np")
    input_features = mx.array(inputs.input_features)
    
    # Run Inference
    print(f"Transcribing using {args.backend} backend...")
    
    import time
    start_time = time.time()
    
    if args.backend == "mlx":
        input_features = mx.array(inputs.input_features)
        logits = model(input_features)
        # Decode
        log_probs = mx.softmax(logits, axis=-1)
        tokens = mx.argmax(log_probs, axis=-1)
        predicted_ids = np.array(tokens)
    else:
        # Transformers backend
        device = next(model.parameters()).device
        input_features = torch.tensor(inputs.input_features).to(device)
        with torch.no_grad():
            logits = model(input_features).logits.cpu()
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids = predicted_ids.numpy()
    transcription = processor.batch_decode(predicted_ids)[0]
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("-" * 40)
    print("Transcription:")
    print(transcription)
    print(f"\nInference time: {duration:.2f} seconds")
    print("-" * 40)

if __name__ == "__main__":
    main()
