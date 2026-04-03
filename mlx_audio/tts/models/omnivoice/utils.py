import mlx.core as mx


def create_voice_clone_prompt(
    ref_audio_path: str,
    ref_text: str = "",
    tokenizer=None,
    max_duration_s: float = 10.0,
) -> mx.array:
    """
    Encode a reference audio file into acoustic token prefix for voice cloning.

    Args:
        ref_audio_path: Path to reference audio file (WAV, 24kHz preferred)
        ref_text: Unused, kept for backward compatibility
        tokenizer: Optional HiggsAudioTokenizer instance. If None, returns empty prefix.
        max_duration_s: Maximum reference audio duration in seconds (default 10.0)

    Returns:
        mx.array of shape [T_ref, 8] int32 — acoustic token prefix.
        Returns empty array [0, 8] when tokenizer is None.

    Raises:
        FileNotFoundError: If ref_audio_path does not exist.
    """
    if tokenizer is None:
        return mx.zeros((0, 8), dtype=mx.int32)

    from math import gcd
    from pathlib import Path

    import soundfile as sf
    from scipy.signal import resample_poly

    path = Path(ref_audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)  # [T]
    mono = mono[: int(max_duration_s * sr)]  # clip to max duration

    if sr != 24000:
        g = gcd(24000, sr)
        mono = resample_poly(mono, 24000 // g, sr // g).astype("float32")

    wav = mx.array(mono)[None, :, None]  # [1, T, 1]
    tokens = tokenizer.encode(wav)  # [1, T_ref, 8]
    return tokens[0]  # [T_ref, 8]
