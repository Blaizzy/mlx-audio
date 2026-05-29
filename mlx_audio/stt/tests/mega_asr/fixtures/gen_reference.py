#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingTypeArgument=false, reportOperatorIssue=false, reportPossiblyUnboundVariable=false, reportUndefinedVariable=false, reportCallIssue=false, reportArgumentType=false
from __future__ import annotations

import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from huggingface_hub import snapshot_download
from safetensors.torch import safe_open
from scipy.signal import fftconvolve, resample_poly

REPO_ROOT = Path("/mnt/d/Projects/Mega-ASR")
FIXTURE_DIR = REPO_ROOT / "fixtures_out"
CKPT_DIR = REPO_ROOT / "ckpt" / "Mega-ASR"
ROUTER_CKPT = CKPT_DIR / "audio_quality_router" / "best_acc_model.safetensors"
TARGET_SR = 16000
MAX_SECONDS = 5.0

sys.path.insert(0, str(REPO_ROOT / "src"))
from MegaASR.model.megaASR import MegaASR
from MegaASR.model.router import AudioQualityRouter


def ensure_checkpoint_layout() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    needed = [
        CKPT_DIR / "Qwen3-ASR-1.7B",
        CKPT_DIR / "mega-asr-merged",
        CKPT_DIR / "audio_quality_router",
    ]
    if not all(path.exists() for path in needed):
        print("Downloading zhifeixie/Mega-ASR into", CKPT_DIR, flush=True)
        snapshot_download(
            repo_id="zhifeixie/Mega-ASR",
            local_dir=str(CKPT_DIR),
            local_dir_use_symlinks=False,
        )
    for path in needed:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint path: {path}")


def to_mono_16k(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        gcd = math.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // gcd, sr // gcd).astype(np.float32)
    return audio.astype(np.float32)


def peak_normalize(audio: np.ndarray, peak: float = 0.8) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > 0:
        audio = audio * (peak / max_abs)
    return np.clip(audio, -0.999, 0.999).astype(np.float32)


def save_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(str(path), peak_normalize(audio), TARGET_SR, subtype="PCM_16")


def router_features(router: AudioQualityRouter, audio_path: Path) -> dict:
    waveform = router._load_audio(audio_path)
    with torch.no_grad():
        mel = router.mel_extractor(waveform)
        mel_t = mel.squeeze(0).transpose(0, 1).contiguous()
        logits = router.model(mel_t.unsqueeze(0), mask=None)[0].float()
        probs = torch.softmax(logits, dim=-1)
    return {
        "logmel": mel_t.float().cpu().numpy(),
        "logits": logits.cpu().tolist(),
        "degraded_prob": float(probs[1].item()),
    }


def select_clean_clip(router: AudioQualityRouter) -> tuple[Path, dict]:
    ds = load_dataset("google/fleurs", "en_us", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    best = None
    clean_path = FIXTURE_DIR / "clean.wav"
    for idx, sample in enumerate(ds):
        audio_info = sample["audio"]
        if audio_info.get("bytes") is not None:
            audio, sr = sf.read(BytesIO(audio_info["bytes"]), dtype="float32")
        else:
            audio, sr = sf.read(audio_info["path"], dtype="float32")
        audio = to_mono_16k(audio, int(sr))
        duration = len(audio) / TARGET_SR
        if duration <= 0.5 or duration > MAX_SECONDS:
            continue
        save_wav(clean_path, audio)
        info = router_features(router, clean_path)
        record = {
            "idx": idx,
            "duration": duration,
            "degraded_prob": info["degraded_prob"],
            "text": sample.get("transcription", ""),
            "info": info,
        }
        print(
            "clean candidate",
            idx,
            f"dur={duration:.2f}s",
            f"prob={info['degraded_prob']:.4f}",
            flush=True,
        )
        if best is None or record["degraded_prob"] < best["degraded_prob"]:
            best = record
        if info["degraded_prob"] < 0.5:
            return clean_path, info
    if best is None:
        raise RuntimeError("No <=5s FLEURS English sample found")
    raise RuntimeError(
        f"No clean FLEURS sample routed below 0.5; best was idx={best[idx]} prob={best[degraded_prob]:.4f}"
    )


def apply_reverb(audio: np.ndarray, mix: float) -> np.ndarray:
    if mix <= 0:
        return audio
    taps_ms = [0, 17, 41, 73, 113, 181]
    amps = [1.0, 0.55, 0.32, 0.20, 0.12, 0.07]
    ir_len = int(TARGET_SR * 0.25)
    ir = np.zeros(ir_len, dtype=np.float32)
    for ms, amp in zip(taps_ms, amps):
        idx = min(ir_len - 1, int(TARGET_SR * ms / 1000.0))
        ir[idx] += amp
    decay = np.exp(-np.linspace(0.0, 4.5, ir_len)).astype(np.float32)
    ir += 0.03 * decay
    ir /= np.sum(np.abs(ir))
    wet = fftconvolve(audio, ir, mode="full")[: len(audio)].astype(np.float32)
    return ((1.0 - mix) * audio + mix * wet).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    noise = fftconvolve(
        noise, np.array([1.0, -0.85], dtype=np.float32), mode="same"
    ).astype(np.float32)
    sig_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise**2) + 1e-12))
    target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / max(noise_rms, 1e-12))
    return (audio + noise).astype(np.float32)


def distort(audio: np.ndarray, drive: float) -> np.ndarray:
    if drive <= 0:
        return audio
    return np.tanh(audio * (1.0 + drive)).astype(np.float32)


def make_degraded_clip(
    router: AudioQualityRouter, clean_path: Path
) -> tuple[Path, dict]:
    clean_audio, sr = sf.read(str(clean_path), dtype="float32")
    clean_audio = to_mono_16k(clean_audio, sr)
    degraded_path = FIXTURE_DIR / "degraded.wav"
    configs = [
        {"snr_db": 15.0, "reverb_mix": 0.15, "drive": 0.0},
        {"snr_db": 10.0, "reverb_mix": 0.25, "drive": 0.1},
        {"snr_db": 5.0, "reverb_mix": 0.35, "drive": 0.2},
        {"snr_db": 0.0, "reverb_mix": 0.45, "drive": 0.35},
        {"snr_db": -5.0, "reverb_mix": 0.6, "drive": 0.6},
    ]
    for attempt, cfg in enumerate(configs):
        rng = np.random.default_rng(20260521 + attempt)
        degraded = apply_reverb(clean_audio, cfg["reverb_mix"])
        degraded = add_noise(degraded, cfg["snr_db"], rng)
        degraded = distort(degraded, cfg["drive"])
        degraded = peak_normalize(degraded)
        save_wav(degraded_path, degraded)
        info = router_features(router, degraded_path)
        print(
            "degraded attempt",
            attempt,
            cfg,
            f"prob={info['degraded_prob']:.4f}",
            flush=True,
        )
        if info["degraded_prob"] >= 0.5:
            return degraded_path, info
    raise RuntimeError("Could not synthesize degraded clip routed above threshold 0.5")


def flatten_text(value) -> str:
    if isinstance(value, list):
        return " ".join(str(x).strip() for x in value if str(x).strip()).strip()
    return str(value).strip()


def run() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    ensure_checkpoint_layout()

    router = AudioQualityRouter(
        checkpoint_path=ROUTER_CKPT,
        device="cuda:0",
        threshold=0.5,
    )

    clean_path, clean_router = select_clean_clip(router)
    degraded_path, degraded_router = make_degraded_clip(router, clean_path)

    model = MegaASR(
        model_path=CKPT_DIR / "Qwen3-ASR-1.7B",
        lora_dir=CKPT_DIR / "mega-asr-merged",
        router_checkpoint=ROUTER_CKPT,
        routing_enabled=True,
        quality_threshold=0.5,
        device_map="cuda:0",
        quality_device="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=256,
        keep_delta_on_gpu=True,
    )

    clean_result = model.infer(clean_path, language="English", return_route=True)
    degraded_result = model.infer(degraded_path, language="English", return_route=True)

    if not (float(clean_result["degraded_prob"]) < 0.5):
        raise RuntimeError(f"clean.wav routed as degraded: {clean_result}")
    if not (float(degraded_result["degraded_prob"]) >= 0.5):
        raise RuntimeError(f"degraded.wav did not cross threshold: {degraded_result}")

    with safe_open(str(ROUTER_CKPT), framework="pt", device="cpu") as f:
        router_state = {key: list(f.get_tensor(key).shape) for key in f.keys()}

    reference = {
        "clean": {
            "degraded_prob": float(clean_result["degraded_prob"]),
            "use_lora": bool(clean_result["use_lora"]),
            "text": flatten_text(clean_result["text"]),
        },
        "degraded": {
            "degraded_prob": float(degraded_result["degraded_prob"]),
            "use_lora": bool(degraded_result["use_lora"]),
            "text": flatten_text(degraded_result["text"]),
        },
        "router_logmel_clean_first10": clean_router["logmel"][:10].tolist(),
        "router_logits_clean": [float(x) for x in clean_router["logits"]],
        "router_logits_degraded": [float(x) for x in degraded_router["logits"]],
    }

    (FIXTURE_DIR / "reference.json").write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n"
    )
    (FIXTURE_DIR / "router_state_dict_keys.json").write_text(
        json.dumps(router_state, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )

    print(json.dumps(reference, indent=2, ensure_ascii=False))
    print("router_state_keys", len(router_state))
    print("clean_prob", reference["clean"]["degraded_prob"])
    print("degraded_prob", reference["degraded"]["degraded_prob"])


if __name__ == "__main__":
    run()
