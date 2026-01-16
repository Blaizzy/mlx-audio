
import argparse
import sys
import numpy as np
import mlx.core as mx
import queue
import time
import torch
import sounddevice as sd
from collections import deque

# Ensure we can find local mlx-audio if running from root without install
from mlx_audio.stt.utils import load as load_mlx
from transformers import AutoModelForCTC, AutoProcessor
from ten_vad import TenVad

def main():
    parser = argparse.ArgumentParser(description="Live Transcribe using MedASR on MLX and TEN-VAD")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="medasr_mlx_converted", 
        help="Path to the converted MLX model"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mlx",
        choices=["mlx", "transformers"],
        help="Backend to use for inference: 'mlx' (default) or 'transformers'"
    )
    
    args = parser.parse_args()
    
    # Load Model
    model = None
    if args.backend == "mlx":
        print(f"Loading MLX model from {args.model_path}...")
        try:
            model = load_mlx(args.model_path)
            # Warmup?
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

    # Load Processor
    try:
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        processor = AutoProcessor.from_pretrained("google/medasr", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading processor: {e}")
        sys.exit(1)

    # Initialize VAD
    try:
        vad = TenVad()
        print("TEN-VAD initialized.")
    except Exception as e:
        print(f"Error initializing TEN-VAD: {e}")
        sys.exit(1)

    # Audio Parameters
    SR = 16000
    FRAME_SIZE = 256 # required by TEN-VAD for 16kHz
    
    # VAD Parameters
    PAUSE_THRESHOLD_SEC = 0.5
    MIN_DURATION_SEC = 0.5
    
    PAUSE_FRAMES = int(PAUSE_THRESHOLD_SEC * SR / FRAME_SIZE)
    MIN_FRAMES = int(MIN_DURATION_SEC * SR / FRAME_SIZE)
    
    # State
    audio_buffer = []
    silence_counter = 0
    is_speaking = False
    
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print("-" * 50)
    print("Listening... (Press Ctrl+C to stop)")
    print("-" * 50)

    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype='int16', 
                           blocksize=FRAME_SIZE, callback=callback):
            while True:
                frame = q.get()
                
                # Check VAD
                # frame is int16, shape (256, 1) or (256,)
                frame_flat = frame.flatten()
                
                # VAD process expects 1D array
                if frame_flat.size != 256:
                    continue # Should not happen with fixed blocksize
                
                prob, is_speech_flag = vad.process(frame_flat)
                # is_speech_flag seems to be 0 or 1.
                
                if is_speech_flag == 1:
                    is_speaking = True
                    silence_counter = 0
                    audio_buffer.append(frame_flat)
                else:
                    if is_speaking:
                        # verifying silence while previously speaking
                        silence_counter += 1
                        audio_buffer.append(frame_flat) # Keep silence tail
                        
                        if silence_counter >= PAUSE_FRAMES:
                            # End of utterance
                            is_speaking = False
                            
                            # Process Utterance
                            total_samples = len(audio_buffer) * FRAME_SIZE
                            duration = total_samples / SR
                            
                            if duration >= MIN_DURATION_SEC:
                                # Transcribe
                                raw_audio = np.concatenate(audio_buffer)
                                
                                # Normalize float32 for model
                                float_audio = raw_audio.astype(np.float32) / 32768.0
                                
                                # Peak normalize slightly to ensure good volume? 
                                # MedASR might expect specific levels, but let's just make sure it's not silent.
                                # Simple peak norm within reasonable bounds
                                max_val = np.max(np.abs(float_audio))
                                if max_val > 0:
                                    # Normalize to 0.9
                                    float_audio = float_audio / max_val * 0.9
                                
                                inputs = processor(float_audio, sampling_rate=SR, return_tensors="np")
                                
                                if args.backend == "mlx":
                                    input_features = mx.array(inputs.input_features)
                                    logits = model(input_features)
                                    log_probs = mx.softmax(logits, axis=-1)
                                    tokens = mx.argmax(log_probs, axis=-1)
                                    predicted_ids = np.array(tokens)
                                else:
                                    device_t = next(model.parameters()).device
                                    input_features = torch.tensor(inputs.input_features).to(device_t)
                                    with torch.no_grad():
                                        logits = model(input_features).logits.cpu()
                                    predicted_ids = torch.argmax(logits, dim=-1)
                                    predicted_ids = predicted_ids.numpy()
                                    
                                transcription = processor.batch_decode(predicted_ids)[0]
                                
                                clean_text = transcription.replace("</s>", "").strip()
                                if clean_text:
                                    print(f"\rUser: {clean_text}")
                                    sys.stdout.flush()
                            
                            # Reset
                            audio_buffer = []
                            silence_counter = 0
                    else:
                        # Just silence, do nothing or keep minimal context?
                        # For now, drop it.
                        pass
                        
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)

if __name__ == "__main__":
    main()
