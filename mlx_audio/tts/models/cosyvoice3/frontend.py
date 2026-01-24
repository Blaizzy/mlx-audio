# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


from typing import Optional, Tuple

import mlx.core as mx
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from mlx_audio.codec.models.s3.model_v2 import S3TokenizerV2
from mlx_audio.audio_io import read as audio_read


class CosyVoice3Frontend:
    """
    Frontend for CosyVoice3 text-to-speech.

    Handles:
    - Text tokenization using the CosyVoice tokenizer
    - Speaker embedding extraction using CAMPPlus
    - Audio feature extraction for prompts
    """

    def __init__(
        self,
        model_path: str,
        campplus_model=None,
        sample_rate: int = 24000,
    ):
        """
        Initialize the frontend.

        Args:
            tokenizer_path: Path to tokenizer (HuggingFace format)
            campplus_model: CAMPPlus model instance (owned by parent Model)
            speech_tokenizer_path: Path to speech tokenizer safetensors model
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.tokenizer = None
        self.campplus_model = campplus_model
        self.speech_tokenizer_model = None

        # Load tokenizer
        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # Add CosyVoice3 special tokens (matching PyTorch CosyVoice3Tokenizer).
            # These tokens are used by the model but not in the base Qwen2 tokenizer.
            # The order must match training to get correct token IDs.
            self.tokenizer.add_special_tokens({
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "additional_special_tokens": [
                    "<|im_start|>", "<|im_end|>", "<|endofprompt|>",
                    "[breath]", "<strong>", "</strong>", "[noise]",
                    "[laughter]", "[cough]", "[clucking]", "[accent]",
                    "[quick_breath]",
                    "<laughter>", "</laughter>",
                    "[hissing]", "[sigh]", "[vocalized-noise]",
                    "[lipsmack]", "[mn]", "<|endofsystem|>",
                    "[AA]", "[AA0]", "[AA1]", "[AA2]", "[AE]", "[AE0]", "[AE1]", "[AE2]",
                    "[AH]", "[AH0]", "[AH1]", "[AH2]", "[AO]", "[AO0]", "[AO1]", "[AO2]",
                    "[AW]", "[AW0]", "[AW1]", "[AW2]", "[AY]", "[AY0]", "[AY1]", "[AY2]",
                    "[B]", "[CH]", "[D]", "[DH]", "[EH]", "[EH0]", "[EH1]", "[EH2]",
                    "[ER]", "[ER0]", "[ER1]", "[ER2]", "[EY]", "[EY0]", "[EY1]", "[EY2]",
                    "[F]", "[G]", "[HH]", "[IH]", "[IH0]", "[IH1]", "[IH2]",
                    "[IY]", "[IY0]", "[IY1]", "[IY2]", "[JH]", "[K]", "[L]", "[M]",
                    "[N]", "[NG]", "[OW]", "[OW0]", "[OW1]", "[OW2]", "[OY]", "[OY0]",
                    "[OY1]", "[OY2]", "[P]", "[R]", "[S]", "[SH]", "[T]", "[TH]",
                    "[UH]", "[UH0]", "[UH1]", "[UH2]", "[UW]", "[UW0]", "[UW1]", "[UW2]",
                    "[V]", "[W]", "[Y]", "[Z]", "[ZH]",
                    "[a]", "[ai]", "[an]", "[ang]", "[ao]", "[b]", "[c]", "[ch]",
                    "[d]", "[e]", "[ei]", "[en]", "[eng]", "[f]", "[g]", "[h]",
                    "[i]", "[ian]", "[in]", "[ing]", "[iu]",
                    "[ià]", "[iàn]", "[iàng]", "[iào]", "[iá]", "[ián]", "[iáng]", "[iáo]",
                    "[iè]", "[ié]", "[iòng]", "[ióng]", "[iù]", "[iú]",
                    "[iā]", "[iān]", "[iāng]", "[iāo]", "[iē]", "[iě]", "[iōng]", "[iū]",
                    "[iǎ]", "[iǎn]", "[iǎng]", "[iǎo]", "[iǒng]", "[iǔ]",
                    "[j]", "[k]", "[l]", "[m]", "[n]", "[o]", "[ong]", "[ou]",
                    "[p]", "[q]", "[r]", "[s]", "[sh]", "[t]",
                    "[u]", "[uang]", "[ue]", "[un]", "[uo]",
                    "[uà]", "[uài]", "[uàn]", "[uàng]", "[uá]", "[uái]", "[uán]", "[uáng]",
                    "[uè]", "[ué]", "[uì]", "[uí]", "[uò]", "[uó]",
                    "[uā]", "[uāi]", "[uān]", "[uāng]", "[uē]", "[uě]", "[uī]", "[uō]",
                    "[uǎ]", "[uǎi]", "[uǎn]", "[uǎng]", "[uǐ]", "[uǒ]",
                    "[vè]", "[w]", "[x]", "[y]", "[z]", "[zh]",
                    "[à]", "[ài]", "[àn]", "[àng]", "[ào]",
                    "[á]", "[ái]", "[án]", "[áng]", "[áo]",
                    "[è]", "[èi]", "[èn]", "[èng]", "[èr]",
                    "[é]", "[éi]", "[én]", "[éng]", "[ér]",
                    "[ì]", "[ìn]", "[ìng]", "[í]", "[ín]", "[íng]",
                    "[ò]", "[òng]", "[òu]", "[ó]", "[óng]", "[óu]",
                    "[ù]", "[ùn]", "[ú]", "[ún]",
                    "[ā]", "[āi]", "[ān]", "[āng]", "[āo]",
                    "[ē]", "[ēi]", "[ēn]", "[ēng]",
                    "[ě]", "[ěi]", "[ěn]", "[ěng]", "[ěr]",
                    "[ī]", "[īn]", "[īng]",
                    "[ō]", "[ōng]", "[ōu]",
                    "[ū]", "[ūn]",
                    "[ǎ]", "[ǎi]", "[ǎn]", "[ǎng]", "[ǎo]",
                    "[ǐ]", "[ǐn]", "[ǐng]",
                    "[ǒ]", "[ǒng]", "[ǒu]",
                    "[ǔ]", "[ǔn]",
                    "[ǘ]", "[ǚ]", "[ǜ]",
                ],
            })


        self.speech_tokenizer_model = S3TokenizerV2.from_pretrained("speech_tokenizer_v3", local_path=str(Path(model_path) / "speech_tokenizer_v3.safetensors"))

    def tokenize(self, text: str) -> mx.array:
        """
        Tokenize text using the CosyVoice tokenizer.

        Args:
            text: Input text

        Returns:
            Token IDs as mx.array (1, T)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Please provide tokenizer_path.")

        # CosyVoice tokenizer encodes raw text without special tokens.
        # The LLM handles its own structure (SOS, TASK_ID) separately.
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return mx.array([tokens], dtype=mx.int32)

    def _mel_filterbank(
        self,
        sr: int,
        n_fft: int,
        n_mels: int,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        norm: Optional[str] = "slaney",
        htk: bool = False,
    ) -> np.ndarray:
        """
        Compute a mel filterbank matrix (matching librosa.filters.mel).

        Args:
            sr: Sample rate
            n_fft: FFT size
            n_mels: Number of mel bands
            fmin: Minimum frequency
            fmax: Maximum frequency (defaults to sr/2)
            norm: Normalization type ('slaney' or None)
            htk: Use HTK formula (True) or Slaney/Auditory (False)

        Returns:
            Mel filterbank (n_mels, n_fft//2 + 1)
        """
        if fmax is None:
            fmax = sr / 2.0

        n_freqs = n_fft // 2 + 1
        all_freqs = np.linspace(0, sr / 2.0, n_freqs)

        # Hz <-> Mel conversions
        if htk:
            def hz_to_mel(f):
                return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

            def mel_to_hz(m):
                return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)
        else:
            # Slaney's Auditory Toolbox formula
            f_sp = 200.0 / 3.0
            min_log_hz = 1000.0
            min_log_mel = min_log_hz / f_sp
            logstep = np.log(6.4) / 27.0

            def hz_to_mel(f):
                f = np.asarray(f, dtype=np.float64)
                mel = np.where(
                    f < min_log_hz,
                    f / f_sp,
                    min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
                )
                return mel

            def mel_to_hz(m):
                m = np.asarray(m, dtype=np.float64)
                f = np.where(
                    m < min_log_mel,
                    m * f_sp,
                    min_log_hz * np.exp(logstep * (m - min_log_mel)),
                )
                return f

        # Compute mel band edges
        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Build filterbank
        filterbank = np.zeros((n_mels, n_freqs))
        for i in range(n_mels):
            lower = hz_points[i]
            center = hz_points[i + 1]
            upper = hz_points[i + 2]

            # Rising slope
            up_mask = (all_freqs >= lower) & (all_freqs <= center)
            if center > lower:
                filterbank[i, up_mask] = (all_freqs[up_mask] - lower) / (center - lower)

            # Falling slope
            down_mask = (all_freqs >= center) & (all_freqs <= upper)
            if upper > center:
                filterbank[i, down_mask] = (upper - all_freqs[down_mask]) / (upper - center)

        # Slaney normalization: normalize each band by its width
        if norm == "slaney":
            enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
            filterbank *= enorm[:, np.newaxis]

        return filterbank.astype(np.float32)

    def extract_speaker_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> mx.array:
        """
        Extract speaker embedding from audio using CAMPPlus (pure MLX).

        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Audio sample rate (should be 16000 for CAMPPlus)

        Returns:
            Speaker embedding as mx.array (1, 192)
        """
        if self.campplus_model is None:
            raise RuntimeError("CAMPPlus model not provided.")

        from mlx_audio.tts.models.cosyvoice3.campplus import kaldi_fbank

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal

            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)

        # Extract Kaldi-style fbank features
        audio_mx = mx.array(audio.astype(np.float32))
        features = kaldi_fbank(audio_mx, sample_rate=16000, num_mel_bins=80)
        mx.eval(features)

        # Add batch dimension: (T, 80) -> (1, T, 80)
        features = mx.expand_dims(features, 0)

        # Run CAMPPlus
        embedding = self.campplus_model(features)
        mx.eval(embedding)

        return embedding

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        audio, sr = audio_read(audio_path, dtype="float32")
        # Convert stereo to mono if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr

    def _get_flow_mel_filters(self) -> np.ndarray:
        """
        Get mel filterbank for the flow model (cached).

        Equivalent to librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=None)
        with slaney normalization.

        Returns:
            Mel filterbank matrix (n_mels, n_freq) = (80, 961)
        """
        if not hasattr(self, "_flow_mel_fb"):
            self._flow_mel_fb = self._mel_filterbank(
                sr=24000, n_fft=1920, n_mels=80, fmin=0.0, fmax=None,
                norm="slaney", htk=False
            )
        return self._flow_mel_fb

    def extract_mel_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> mx.array:
        """
        Extract mel spectrogram features for prompt audio.

        Matches PyTorch matcha.utils.audio.mel_spectrogram exactly:
        - n_fft=1920, hop_size=480, win_size=1920
        - Reflect padding before STFT (center=False after padding)
        - Periodic Hann window (matching torch.hann_window)
        - Magnitude spectrum with epsilon: sqrt(|spec|^2 + 1e-9)
        - Slaney-normalized mel filterbank (matching librosa.filters.mel)
        - Dynamic range compression: log(clamp(x, min=1e-5))

        Args:
            audio: Audio waveform
            sample_rate: Audio sample rate

        Returns:
            Mel features as mx.array (1, 80, T)
        """
        from scipy import signal

        # Resample to model sample rate if needed
        if sample_rate != self.sample_rate:
            num_samples = int(len(audio) * self.sample_rate / sample_rate)
            audio = signal.resample(audio, num_samples)

        # Ensure audio is float32 in range [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        # CosyVoice3 config from cosyvoice3.yaml
        n_fft = 1920
        hop_length = 480
        win_length = 1920

        # Reflect padding (matching PyTorch: pad with (n_fft - hop_size) / 2 on each side)
        pad_size = int((n_fft - hop_length) / 2)  # 720
        audio_padded = np.pad(audio, (pad_size, pad_size), mode="reflect")

        # Periodic Hann window (matching torch.hann_window)
        window = np.hanning(win_length + 1)[:-1]

        # Manual STFT (matching PyTorch torch.stft with center=False)
        num_frames = 1 + (len(audio_padded) - n_fft) // hop_length
        stft_result = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex64)

        for i in range(num_frames):
            start = i * hop_length
            frame = audio_padded[start : start + n_fft] * window
            stft_result[:, i] = np.fft.rfft(frame)

        # Magnitude with epsilon (matching PyTorch: sqrt(spec.pow(2).sum(-1) + 1e-9))
        magnitudes = np.sqrt(np.abs(stft_result) ** 2 + 1e-9)

        # Apply mel filterbank (slaney-normalized, matching librosa)
        mel_fb = self._get_flow_mel_filters()  # (80, 961)
        mel_spec = mel_fb @ magnitudes  # (80, T)

        # Dynamic range compression (matching PyTorch spectral_normalize_torch)
        # log(clamp(x, min=1e-5))
        log_mel = np.log(np.clip(mel_spec, a_min=1e-5, a_max=None))

        # Add batch dimension: (80, T) -> (1, 80, T)
        return mx.array(log_mel[np.newaxis, :, :].astype(np.float32))

    def _get_whisper_mel_filters(self, n_mels: int = 128) -> np.ndarray:
        """
        Get whisper-style mel filterbank (cached).

        Equivalent to librosa.filters.mel(sr=16000, n_fft=400, n_mels=N)
        with slaney normalization.

        Args:
            n_mels: Number of mel bins (80 or 128)

        Returns:
            Mel filterbank matrix (n_mels, n_fft//2 + 1)
        """
        cache_attr = f"_whisper_mel_fb_{n_mels}"
        if not hasattr(self, cache_attr):
            fb = self._mel_filterbank(
                sr=16000, n_fft=400, n_mels=n_mels, fmin=0.0, fmax=None,
                norm="slaney", htk=False
            )
            setattr(self, cache_attr, fb)
        return getattr(self, cache_attr)

    def _whisper_log_mel_spectrogram(
        self, audio: np.ndarray, n_mels: int = 128
    ) -> np.ndarray:
        """
        Compute whisper-style log mel spectrogram.

        Matches whisper.audio.log_mel_spectrogram exactly:
        - STFT with n_fft=400, hop_length=160, hann window, center=True (reflect pad)
        - Power spectrum with last frame dropped
        - Pre-computed librosa mel filterbank
        - log10(clamp(x, 1e-10))
        - Max normalization: max(x, x.max() - 8.0)
        - Shift/scale: (x + 4.0) / 4.0

        Args:
            audio: Audio waveform at 16kHz (float32)
            n_mels: Number of mel bins

        Returns:
            Log mel spectrogram (n_mels, T)
        """
        n_fft = 400
        hop_length = 160

        # Center padding (reflect) - matches torch.stft center=True
        pad_size = n_fft // 2  # 200
        audio_padded = np.pad(audio, (pad_size, pad_size), mode="reflect")

        # Hann window
        window = np.hanning(n_fft + 1)[:-1]  # Periodic hann (matches torch.hann_window)

        # STFT (manual to match torch.stft exactly)
        num_frames = 1 + (len(audio_padded) - n_fft) // hop_length
        stft_result = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex64)
        for i in range(num_frames):
            start = i * hop_length
            frame = audio_padded[start : start + n_fft] * window
            stft_result[:, i] = np.fft.rfft(frame)

        # Drop last frame (matches whisper: stft[..., :-1])
        magnitudes = np.abs(stft_result[:, :-1]) ** 2

        # Apply mel filterbank
        mel_fb = self._get_whisper_mel_filters(n_mels)
        mel_spec = mel_fb @ magnitudes

        # Log with floor
        log_spec = np.log10(np.maximum(mel_spec, 1e-10))

        # Max normalization (whisper-specific)
        log_spec = np.maximum(log_spec, log_spec.max() - 8.0)

        # Shift and scale
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec.astype(np.float32)

    def extract_speech_tokens(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> mx.array:
        """
        Extract speech tokens from audio using the speech tokenizer (pure MLX).

        Uses whisper-style 128-bin log mel spectrogram (matching the
        training pipeline of the speech tokenizer model).

        Args:
            audio: Audio waveform (numpy array)
            sample_rate: Audio sample rate

        Returns:
            Speech tokens as mx.array (1, T)
        """
        from scipy import signal

        if self.speech_tokenizer_model is None:
            raise RuntimeError(
                "Speech tokenizer not loaded. Please provide speech_tokenizer_path."
            )

        # Resample to 16kHz (whisper uses 16kHz)
        if sample_rate != 16000:
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)

        # Ensure float32
        audio = audio.astype(np.float32)

        # Limit to 30 seconds
        max_samples = 30 * 16000
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Compute whisper-style 128-bin log mel spectrogram
        log_mel = self._whisper_log_mel_spectrogram(audio, n_mels=128)

        # Shape: (1, 128, T) for MLX model
        feats = mx.array(log_mel[np.newaxis, :, :])
        feats_length = mx.array([log_mel.shape[1]], dtype=mx.int32)

        # Run speech tokenizer
        tokens, _ = self.speech_tokenizer_model.quantize(feats, feats_length)
        mx.eval(tokens)

        return tokens.astype(mx.int32)

    def frontend_zero_shot(
        self,
        text: str,
        ref_text: str,
        ref_audio: str,
    ) -> dict:
        """
        Prepare inputs for zero-shot inference.

        Args:
            text: Text to synthesize
            ref_text: Transcript of the reference audio
            ref_audio: Path to the reference audio file

        Returns:
            Dictionary with model inputs
        """
        # Tokenize text
        # CosyVoice3 expects ref_text in the format:
        #   "system instruction<|endofprompt|>transcript of reference audio"
        # The <|endofprompt|> separator tells the model where the instruction
        # ends and the reference transcript begins. Without it, the model
        # generates speech for the full combined text instead of just the target.
        text_tokens = self.tokenize(text)
        if "<|endofprompt|>" not in ref_text:
            prompt_text_formatted = (
                f"You are a helpful assistant.<|endofprompt|>{ref_text}"
            )
        else:
            prompt_text_formatted = ref_text
        prompt_text_tokens = self.tokenize(prompt_text_formatted)

        # Load and process reference audio
        prompt_audio, prompt_sr = self.load_audio(ref_audio)

        # Extract speaker embedding
        speaker_embedding = self.extract_speaker_embedding(prompt_audio, prompt_sr)

        # Extract mel features for prompt (for Flow model conditioning)
        # Returns (1, 80, T) format
        prompt_mel = self.extract_mel_features(prompt_audio, prompt_sr)

        # Extract speech tokens from prompt audio (for LLM context)
        prompt_speech_tokens = None
        if self.speech_tokenizer_model is not None:
            prompt_speech_tokens = self.extract_speech_tokens(prompt_audio, prompt_sr)

        # Align mel and token lengths (matching PyTorch CosyVoice3)
        # Force: mel_len = 2 * token_len (token_mel_ratio = 2)
        if prompt_speech_tokens is not None and prompt_mel is not None:
            # prompt_mel is (1, 80, T_mel)
            # prompt_speech_tokens is (1, T_tokens)
            mel_len = prompt_mel.shape[2]
            token_len = prompt_speech_tokens.shape[1]

            # Align: token_len = min(mel_len // 2, token_len)
            aligned_token_len = min(mel_len // 2, token_len)
            aligned_mel_len = aligned_token_len * 2

            # Truncate both to aligned lengths
            prompt_mel = prompt_mel[:, :, :aligned_mel_len]
            prompt_speech_tokens = prompt_speech_tokens[:, :aligned_token_len]

        # Transpose mel to (B, T, mel_dim) format to match PyTorch
        # PyTorch: speech_feat = feat.squeeze(0).transpose(0, 1).unsqueeze(0) -> (1, T, 80)
        if prompt_mel is not None:
            prompt_mel = mx.transpose(prompt_mel, (0, 2, 1))  # (1, 80, T) -> (1, T, 80)

        return {
            "text_tokens": text_tokens,
            "prompt_text_tokens": prompt_text_tokens,
            "speaker_embedding": speaker_embedding,
            "prompt_mel": prompt_mel,
            "prompt_speech_tokens": prompt_speech_tokens,
        }

    def frontend_instruct(
        self,
        text: str,
        ref_audio: str,
        instruct_text: str,
    ) -> dict:
        """
        Prepare inputs for instruct-mode inference.

        The instruction replaces the default system prompt content:
        Format: "You are a helpful assistant. {instruct_text}<|endofprompt|>"

        Key difference from zero-shot: the LLM does NOT receive prompt speech
        tokens. Only the Flow model uses them for mel conditioning.

        Args:
            text: Text to synthesize
            ref_audio: Path to reference audio file (for speaker identity)
            instruct_text: Style/language instruction
                (e.g., "Please speak as fast as possible.",
                       "Please express in Cantonese.")

        Returns:
            Dictionary with model inputs
        """
        text_tokens = self.tokenize(text)

        # Instruct format (matching PyTorch exactly):
        # 'You are a helpful assistant. {instruction}.<|endofprompt|>'
        prompt_text_formatted = (
            f"You are a helpful assistant. {instruct_text}.<|endofprompt|>"
        )
        prompt_text_tokens = self.tokenize(prompt_text_formatted)

        # Load and process reference audio
        prompt_audio, prompt_sr = self.load_audio(ref_audio)

        # Extract speaker embedding
        speaker_embedding = self.extract_speaker_embedding(prompt_audio, prompt_sr)

        # Extract mel features for Flow conditioning
        prompt_mel = self.extract_mel_features(prompt_audio, prompt_sr)

        # Extract speech tokens (for Flow only, NOT for LLM)
        flow_prompt_speech_tokens = None
        if self.speech_tokenizer_model is not None:
            flow_prompt_speech_tokens = self.extract_speech_tokens(prompt_audio, prompt_sr)

        # Align mel and token lengths
        if flow_prompt_speech_tokens is not None and prompt_mel is not None:
            mel_len = prompt_mel.shape[2]
            token_len = flow_prompt_speech_tokens.shape[1]
            aligned_token_len = min(mel_len // 2, token_len)
            aligned_mel_len = aligned_token_len * 2
            prompt_mel = prompt_mel[:, :, :aligned_mel_len]
            flow_prompt_speech_tokens = flow_prompt_speech_tokens[:, :aligned_token_len]

        # Transpose mel to (B, T, mel_dim)
        if prompt_mel is not None:
            prompt_mel = mx.transpose(prompt_mel, (0, 2, 1))

        return {
            "text_tokens": text_tokens,
            "prompt_text_tokens": prompt_text_tokens,
            "speaker_embedding": speaker_embedding,
            "prompt_mel": prompt_mel,
            "prompt_speech_tokens": None,  # LLM gets NO speech tokens
            "flow_prompt_speech_tokens": flow_prompt_speech_tokens,  # Flow only
        }

    def frontend_cross_lingual(
        self,
        text: str,
        ref_audio: str,
    ) -> dict:
        """
        Prepare inputs for cross-lingual / fine-grained control inference.

        The system prompt "You are a helpful assistant.<|endofprompt|>" is
        auto-prepended if not already present.

        Key difference from zero-shot: the LLM receives NEITHER prompt_text
        NOR prompt_speech_tokens. The full text (including system prompt) is
        passed as text_tokens. Only the Flow uses prompt speech tokens.

        Example text:
            "[breath]Hello world,[breath]this is a test."

        Args:
            text: Target text with optional control tokens ([breath], etc.)
            ref_audio: Path to reference audio file (for speaker identity)

        Returns:
            Dictionary with model inputs
        """
        # Auto-prepend system prompt if not already present
        if "<|endofprompt|>" not in text:
            text = f"You are a helpful assistant.<|endofprompt|>{text}"

        # In cross-lingual mode, the ENTIRE text (including system prompt)
        # goes as text_tokens. No separate prompt_text for the LLM.
        text_tokens = self.tokenize(text)

        # Load and process reference audio
        prompt_audio, prompt_sr = self.load_audio(ref_audio)

        # Extract speaker embedding
        speaker_embedding = self.extract_speaker_embedding(prompt_audio, prompt_sr)

        # Extract mel features for Flow conditioning
        prompt_mel = self.extract_mel_features(prompt_audio, prompt_sr)

        # Extract speech tokens (for Flow only, NOT for LLM)
        flow_prompt_speech_tokens = None
        if self.speech_tokenizer_model is not None:
            flow_prompt_speech_tokens = self.extract_speech_tokens(prompt_audio, prompt_sr)

        # Align mel and token lengths
        if flow_prompt_speech_tokens is not None and prompt_mel is not None:
            mel_len = prompt_mel.shape[2]
            token_len = flow_prompt_speech_tokens.shape[1]
            aligned_token_len = min(mel_len // 2, token_len)
            aligned_mel_len = aligned_token_len * 2
            prompt_mel = prompt_mel[:, :, :aligned_mel_len]
            flow_prompt_speech_tokens = flow_prompt_speech_tokens[:, :aligned_token_len]

        # Transpose mel to (B, T, mel_dim)
        if prompt_mel is not None:
            prompt_mel = mx.transpose(prompt_mel, (0, 2, 1))

        return {
            "text_tokens": text_tokens,
            "prompt_text_tokens": None,  # LLM gets NO prompt text
            "speaker_embedding": speaker_embedding,
            "prompt_mel": prompt_mel,
            "prompt_speech_tokens": None,  # LLM gets NO speech tokens
            "flow_prompt_speech_tokens": flow_prompt_speech_tokens,  # Flow only
        }

    def frontend_vc(
        self,
        source_audio: str,
        ref_audio: str,
    ) -> dict:
        """
        Prepare inputs for voice conversion.

        Extracts speech tokens from source audio and speaker characteristics
        from reference audio. No LLM step is needed.

        Args:
            source_audio: Path to source audio file (content to convert)
            ref_audio: Path to reference audio file (target voice)

        Returns:
            Dictionary with model inputs
        """
        # Load source audio and extract speech tokens
        source_wav, source_sr = self.load_audio(source_audio)
        if self.speech_tokenizer_model is None:
            raise RuntimeError(
                "Speech tokenizer not loaded. Cannot perform voice conversion."
            )
        source_speech_tokens = self.extract_speech_tokens(source_wav, source_sr)

        # Load reference audio
        ref_wav, ref_sr = self.load_audio(ref_audio)

        # Extract speaker embedding from reference
        speaker_embedding = self.extract_speaker_embedding(ref_wav, ref_sr)

        # Extract mel features from reference (for Flow conditioning)
        prompt_mel = self.extract_mel_features(ref_wav, ref_sr)

        # Extract speech tokens from reference (for Flow prompt)
        prompt_speech_tokens = self.extract_speech_tokens(ref_wav, ref_sr)

        # Align mel and token lengths for reference
        if prompt_speech_tokens is not None and prompt_mel is not None:
            mel_len = prompt_mel.shape[2]
            token_len = prompt_speech_tokens.shape[1]
            aligned_token_len = min(mel_len // 2, token_len)
            aligned_mel_len = aligned_token_len * 2
            prompt_mel = prompt_mel[:, :, :aligned_mel_len]
            prompt_speech_tokens = prompt_speech_tokens[:, :aligned_token_len]

        # Transpose mel to (B, T, mel_dim)
        if prompt_mel is not None:
            prompt_mel = mx.transpose(prompt_mel, (0, 2, 1))

        return {
            "source_speech_tokens": source_speech_tokens,
            "speaker_embedding": speaker_embedding,
            "prompt_mel": prompt_mel,
            "prompt_speech_tokens": prompt_speech_tokens,
        }
