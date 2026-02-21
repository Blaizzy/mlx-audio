# MOSS Contract Fixtures

These JSON fixtures are a minimal, stable contract surface for MOSS audit tests.

- They intentionally do not depend on local `REFERENCE/` checkouts.
- They only include keys consumed by:
  - `mlx_audio.codec.models.moss_audio_tokenizer.config.load_moss_audio_tokenizer_config`
  - `mlx_audio.tts.models.moss_tts.audit.load_moss_variant_invariants`
  - `mlx_audio.tts.models.moss_tts.audit.load_moss_audio_tokenizer_audit`

If contract expectations change, update these fixtures and the corresponding tests together.
