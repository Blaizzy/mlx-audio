
import argparse
import sys
import numpy as np
import mlx.core as mx
import queue
import time
import sounddevice as sd
from collections import deque

# Ensure we can find local mlx-audio if running from root without install
from mlx_audio.stt.utils import load as load_mlx
from transformers import AutoProcessor
from ten_vad import TenVad

def main():
    parser = argparse.ArgumentParser(description="Live Transcribe using MedASR on MLX and TEN-VAD")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="medasr_mlx_converted", 
        help="Path to the converted MLX model"
    )
    
    args = parser.parse_args()
    
    # Load Model
    model = None
    print(f"Loading MLX model from {args.model_path}...")
    try:
        model = load_mlx(args.model_path)
    except Exception as e:
        print(f"Error loading MLX model: {e}")
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
    
    # Ring Buffer for Context (0.5s)
    BUFFER_DURATION = 0.5
    BUFFER_SIZE = int(BUFFER_DURATION * SR) # in samples?
    # Note: working with frames of 256 samples
    # We will buffer the raw float samples
    # BUFFER_SIZE samples
    
    # State
    audio_buffer = [] # Current utterance buffer
    context_buffer = deque(maxlen=int(BUFFER_DURATION * SR / 256) * 256) # Pre-speech context (approx)
    
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
                
                # Convert to float for processing/buffering
                frame_float = frame_flat.astype(np.float32) / 32768.0
                
                # Maintain context buffer
                if not is_speaking:
                    context_buffer.extend(frame_float)
                
                if is_speech_flag == 1:
                    if not is_speaking:
                        # Start of speech
                        is_speaking = True
                        silence_counter = 0
                        # Prepend context
                        audio_buffer = list(context_buffer)
                        audio_buffer.extend(frame_float)
                    else:
                        audio_buffer.extend(frame_float)
                else:
                    if is_speaking:
                        # verifying silence while previously speaking
                        silence_counter += 1
                        audio_buffer.extend(frame_float) # Keep silence tail
                        
                        if silence_counter >= PAUSE_FRAMES:
                            # End of utterance
                            is_speaking = False
                            
                            # Process Utterance
                            # audio_buffer is list of floats. (or list of list of floats? frame_float is array)
                            # Let's fix extension above: extend(frame_float) adds elements.
                            
                            raw_audio = np.array(audio_buffer)
                            duration = len(raw_audio) / SR
                            
                            if duration >= MIN_DURATION_SEC:
                                # Transcribe
                                
                                # Peak normalize slightly
                                max_val = np.max(np.abs(raw_audio))
                                if max_val > 0:
                                    raw_audio = raw_audio / max_val * 0.9
                                
                                inputs = processor(raw_audio, sampling_rate=SR, return_tensors="np")
                                
                                input_features = mx.array(inputs.input_features)
                                
                                logits = model(input_features)
                                log_probs = mx.softmax(logits, axis=-1)
                                tokens = mx.argmax(log_probs, axis=-1)
                                predicted_ids = np.array(tokens)
                                    
                                transcription = processor.batch_decode(predicted_ids)[0]
                                
                                clean_text = transcription.replace("</s>", "").strip()
                                if clean_text:
                                    print(f"User: {clean_text}")
                                    sys.stdout.flush()
                            
                            # Reset
                            audio_buffer = []
                            silence_counter = 0
                            context_buffer.clear()
                    else:
                        # Just silence
                        pass
                        
    except KeyboardInterrupt:
        print("\nStopping...")
        sys.exit(0)

if __name__ == "__main__":
    main()
