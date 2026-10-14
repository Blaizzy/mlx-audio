import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.tts.models.fish_omni.config import AudioDecoderConfig
from mlx_audio.tts.models.fish_omni.prompt import History, split_text
from mlx_audio.tts.models.fish_omni.tower import AudioDecoder, sample_top_k_top_p

SEMANTIC_BEGIN = 100


def tiny_tower():
    config = AudioDecoderConfig(
        vocab_size=16,
        num_codebooks=4,
        dim=32,
        n_head=4,
        n_local_heads=2,
        head_dim=8,
        intermediate_size=48,
        shared_enrichment_layers=2,
    )
    tower = AudioDecoder(config, SEMANTIC_BEGIN, SEMANTIC_BEGIN + 15)
    return tower, config


def random_frames(length, rng):
    codes = rng.integers(0, 16, size=(length, 4))
    main = codes[:, :1] + SEMANTIC_BEGIN
    # A few text rows: main token outside the band, codes zero.
    main[:2] = 7
    codes[:2] = 0
    return mx.array(np.concatenate([main, codes], axis=1).astype(np.int32))[None]


def test_sampler_keeps_top_token_and_respects_top_k():
    mx.random.seed(0)
    logits = mx.array([[0.0, 5.0, 4.9, -3.0]] * 200)
    # top_p tiny: only the top-ranked token survives.
    assert (sample_top_k_top_p(logits, 1.0, 30, 1e-6) == 1).all().item()
    # top_k = 2 never draws ranks 3 or 4.
    draws = sample_top_k_top_p(logits, 1.0, 2, 1.0)
    assert set(np.array(draws).tolist()) == {1, 2}


def test_sampler_drops_the_token_crossing_top_p():
    mx.random.seed(0)
    # Probabilities ~[0.5, 0.3, 0.2]; cumsum crosses 0.6 at rank 2, which the
    # reference drops (unlike HF top-p, which keeps it).
    logits = mx.log(mx.array([[0.5, 0.3, 0.2]] * 400))
    draws = set(np.array(sample_top_k_top_p(logits, 1.0, 30, 0.6)).tolist())
    assert draws == {0}


def test_tower_prefill_matches_stepwise_decode():
    mx.random.seed(0)
    rng = np.random.default_rng(0)
    tower, config = tiny_tower()
    frames = random_frames(9, rng)
    hidden = mx.random.normal((1, 8, config.dim))

    full = tower.prefill(hidden, frames[:, :-1], frames[:, 1:], tower.make_cache())

    caches = tower.make_cache()
    split = tower.prefill(hidden[:, :5], frames[:, :5], frames[:, 1:6], caches)
    steps = [split]
    for row in range(5, 8):
        steps.append(
            tower.prefill(
                hidden[:, row : row + 1],
                frames[:, row : row + 1],
                frames[:, row + 1 : row + 2],
                caches,
            )
        )
    assert mx.allclose(full, mx.concatenate(steps, axis=1), atol=1e-4).item()


def test_tower_decode_step_shapes_and_code0():
    mx.random.seed(0)
    tower, config = tiny_tower()
    caches = tower.make_cache()
    frames = random_frames(1, np.random.default_rng(1))
    main = mx.array([[SEMANTIC_BEGIN + 5]])
    codes = tower.decode_step(
        main,
        mx.random.normal((1, 1, config.dim)),
        frames,
        caches,
        temperature=0.7,
        top_k=30,
        top_p=0.7,
    )
    assert codes.shape == (1, 1, config.num_codebooks)
    assert codes[0, 0, 0].item() == 5
    assert ((codes >= 0) & (codes < config.vocab_size)).all().item()
    assert all(c.offset == 1 for c in caches)


def test_embed_backbone_only_mixes_semantic_rows():
    tower, config = tiny_tower()
    frames = random_frames(4, np.random.default_rng(2))
    text = mx.random.normal((1, 4, config.dim))
    out = tower.embed_backbone(frames, text)
    assert mx.array_equal(out[:, :2], text[:, :2]).item()
    expected = (text[:, 2:] + tower.codebook_embeddings(frames[:, 2:, 2] + 16)) / 2**0.5
    assert mx.allclose(out[:, 2:], expected, atol=1e-5).item()


def test_split_text_keeps_speakers_and_protected_spans():
    text = (
        "<|speaker:0|> "
        + "Hello there. " * 30
        + "<|speaker:1|> [laughs] Sure, why not."
    )
    chunks = split_text(text, max_bytes=120)
    assert len(chunks) > 1
    assert all(chunk.startswith("<|speaker:") for chunk in chunks)
    assert any("[laughs]" in chunk for chunk in chunks)
    assert split_text("Short text.") == ["Short text."]


def test_history_pins_first_chunk():
    history = History()
    for i in range(3):
        history.push(f"chunk {i}", mx.zeros((10, 1)))
    candidates = [[t for t, _ in turns] for turns in history.candidates()]
    assert candidates == [["chunk 0", "chunk 2"], ["chunk 0"], []]


@pytest.mark.parametrize("bad", ["", "<|im_end|> hi", "<|speaker:16|> hi"])
def test_prompt_rejects_invalid_text(bad):
    from mlx_audio.tts.models.fish_omni.prompt import validate_text

    with pytest.raises(ValueError):
        validate_text(bad, "text")


def test_load_reference_upsamples_without_imaging(tmp_path):
    from mlx_audio.audio_io import write as audio_write
    from mlx_audio.tts.models.fish_omni.fish_omni import load_reference

    rate = 24000
    t = np.arange(rate) / rate
    path = tmp_path / "tone.wav"
    audio_write(
        str(path), (0.5 * np.sin(2 * np.pi * 9000 * t)).astype(np.float32), rate
    )

    audio = np.array(load_reference(path, 44100))
    assert abs(len(audio) - 44100) <= 1
    spectrum = np.abs(np.fft.rfft(audio)) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1 / 44100)
    # A linear upsampler leaves an image of the 9 kHz tone at 15 kHz (~3e-3 of the
    # energy); the codec encodes it, and that destabilises Drama 3.
    image = spectrum[(freqs > 13000) & (freqs < 20000)].sum() / spectrum.sum()
    assert image < 1e-4
