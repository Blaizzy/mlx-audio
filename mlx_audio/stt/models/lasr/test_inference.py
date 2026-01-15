
import argparse
import mlx.core as mx
import numpy as np

from mlx_audio.stt.utils import load_audio, load
from mlx_audio.stt.models.base import STTOutput

def verify(model_path: str, audio_file: str):
    print(f"Loading model from {model_path}")
    model = load(model_path)
    
    print(f"Loading audio from {audio_file}")
    # Load audio (assuming base utils have this or use mlx_audio.utils.load_audio)
    # The Lasr implementation expects input_features (Mel Spectrogram) if using raw forward
    # But often decode/generate handles raw audio.
    # In my implementation, LasrForCTC.__call__ takes input_features.
    # We need a processor/feature extractor.
    
    # Check if we implemented audio processing in LasrForCTC or LasrEncoder
    # In `lasr.py`, LasrEncoder expects input_features.
    # We need to replicate the feature extraction logic from HF processor.
    
    # For now, let's assume we can use the HF processor to get features and pass to MLX model to verify numeric correctness
    try:
        from transformers import AutoProcessor
        hf_processor = AutoProcessor.from_pretrained("google/medasr", trust_remote_code=True)
    except:
        print("Couldn't load HF processor. Using placeholder.")
        return

    import librosa
    speech, sample_rate = librosa.load(audio_file, sr=16000)
    
    inputs = hf_processor(speech, sampling_rate=16000, return_tensors="np")
    input_features = mx.array(inputs.input_features)
    
    # Run Inference
    print("Running inference...")
    logits = model(input_features)
    
    # Decode
    log_probs = mx.softmax(logits, axis=-1)
    tokens = mx.argmax(log_probs, axis=-1)
    
    # Decode tokens
    predicted_ids = np.array(tokens)
    transcription = hf_processor.batch_decode(predicted_ids)[0]
    
    print(f"Transcription: {transcription}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--audio", type=str, required=True)
    args = parser.parse_args()
    
    verify(args.model_path, args.audio)
