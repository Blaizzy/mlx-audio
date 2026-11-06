import mlx.core as mx

from mlx_audio.tts.models.chatterbox_turbo.models.t3.t3 import T3
from mlx_audio.tts.models.higgs_audio.generation import sample_audio


def test_chatterbox_turbo_top_k_one_keeps_only_the_best_logit():
    logits = mx.array([[0.1, 3.0, 0.2, 2.0]])
    filtered = T3._top_k_filtering(None, logits, 1)
    assert mx.isfinite(filtered).tolist() == [[False, True, False, False]]


def test_higgs_audio_top_k_one_samples_the_best_logit():
    logits = mx.array([[[[0.1, 3.0, 0.2, 2.0]]]])
    sampled = sample_audio(logits, temperature=1.0, top_p=1.0, top_k=1)
    assert sampled.reshape(-1).tolist() == [1]


def test_chatterbox_turbo_top_k_keeps_the_k_best_logits():
    logits = mx.array([[0.1, 3.0, 0.2, 2.0]])
    filtered = T3._top_k_filtering(None, logits, 2)
    assert mx.isfinite(filtered).tolist() == [[False, True, False, True]]


def test_higgs_audio_top_k_keeps_the_k_best_logits():
    mx.random.seed(0)
    logits = mx.broadcast_to(mx.array([0.1, 3.0, 0.2, 2.0]), (1, 256, 1, 4))
    sampled = sample_audio(logits, temperature=1.0, top_p=1.0, top_k=2)
    assert set(sampled.reshape(-1).tolist()) == {1, 3}


def test_higgs_audio_top_k_above_the_vocabulary_keeps_every_logit():
    mx.random.seed(0)
    logits = mx.zeros((1, 256, 1, 4))
    for top_k in (5, 6, 7):
        sampled = sample_audio(logits, temperature=1.0, top_p=1.0, top_k=top_k)
        assert set(sampled.reshape(-1).tolist()) == {0, 1, 2, 3}
