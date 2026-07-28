from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import pytest

from mlx_audio.stt.models.canary.decoder import FixedPositionalEncoding
from mlx_audio.stt.models.cohere_asr.audio import CohereAudioFrontend
from mlx_audio.stt.models.cohere_asr.cohere_asr import RelPositionalEncoding
from mlx_audio.stt.models.cohere_asr.config import PreprocessorConfig
from mlx_audio.stt.models.granite_speech.config import EncoderConfig as GraniteConfig
from mlx_audio.stt.models.granite_speech.granite_speech import CTCEncoder
from mlx_audio.stt.models.granite_speech_nar import granite_speech_nar
from mlx_audio.stt.models.granite_speech_nar.encoder import ConformerAttention
from mlx_audio.stt.models.moonshine.moonshine import MoonshineRotaryEmbedding
from mlx_audio.stt.models.moss_music.audio import MossMusicFeatureExtractor
from mlx_audio.stt.models.moss_music.config import AudioEncoderConfig as MossConfig
from mlx_audio.stt.models.moss_music.moss_music import MossMusicEncoder
from mlx_audio.stt.models.qwen2_audio.config import EncoderConfig as Qwen2Config
from mlx_audio.stt.models.qwen2_audio.qwen2_audio import Qwen2AudioEncoder
from mlx_audio.stt.models.qwen3_asr.qwen3_asr import SinusoidalPositionEmbedding
from mlx_audio.stt.models.whisper.whisper import AudioEncoder, TextDecoder


def _qwen3_buffers():
    embedding = SinusoidalPositionEmbedding(32, 16)
    return [embedding._positional_embedding]


def _qwen2_buffers():
    encoder = Qwen2AudioEncoder(
        Qwen2Config(
            d_model=16,
            encoder_layers=0,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            num_mel_bins=8,
            max_source_positions=32,
        )
    )
    return [encoder._embed_positions]


def _whisper_buffers():
    encoder = AudioEncoder(8, 16, 16, 2, 0)
    decoder = TextDecoder(32, 16, 16, 2, 0)
    return [encoder._positional_embedding, decoder._mask]


def _moonshine_buffers():
    rope = MoonshineRotaryEmbedding(16)
    return [rope._inv_freq]


def _moss_buffers():
    encoder = MossMusicEncoder(
        MossConfig(
            d_model=16,
            output_dim=16,
            num_mel_bins=8,
            encoder_layers=0,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            downsample_hidden_size=4,
            max_source_positions=32,
            deepstack_encoder_layer_indexes=[],
        )
    )
    extractor = MossMusicFeatureExtractor(num_mel_bins=8, n_fft=32)
    return [encoder._embed_positions, extractor.window, extractor.filters]


def _canary_buffers():
    encoding = FixedPositionalEncoding(16, 32)
    return [encoding._pos_enc]


def _cohere_buffers():
    encoding = RelPositionalEncoding(16, 32)
    frontend = CohereAudioFrontend(
        PreprocessorConfig(features=8, n_fft=32, window_size=0.001)
    )
    return [encoding._pe, frontend.window, frontend.fb]


def _granite_buffers():
    encoder = CTCEncoder(
        GraniteConfig(
            input_dim=4,
            num_layers=0,
            hidden_dim=8,
            num_heads=2,
            dim_head=4,
            output_dim=4,
            context_size=8,
            max_pos_emb=16,
        )
    )
    return [encoder._attention_dists]


def _granite_nar_buffers():
    attention = ConformerAttention(
        hidden_dim=16,
        num_heads=2,
        dim_head=8,
        max_pos_emb=32,
        context_size=16,
    )
    return [attention._dist]


@pytest.mark.parametrize(
    "factory",
    [
        _qwen3_buffers,
        _qwen2_buffers,
        _whisper_buffers,
        _moonshine_buffers,
        _moss_buffers,
        _canary_buffers,
        _cohere_buffers,
        _granite_buffers,
        _granite_nar_buffers,
    ],
)
def test_fixed_buffers_can_cross_threads(factory):
    with ThreadPoolExecutor(max_workers=1) as executor:
        buffers = executor.submit(factory).result()

    mx.eval(*[buffer + 0 for buffer in buffers])


def test_granite_nar_module_buffers_can_cross_threads():
    def evaluate():
        mx.eval(granite_speech_nar._WINDOW + 0, granite_speech_nar._MEL_T + 0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(evaluate).result()
