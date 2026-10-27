# Vendored from MayaResearch/maya2-native (https://github.com/MayaResearch/maya2-native), Copyright Maya Research,
# licensed under the Apache License 2.0 (see LICENSE in this directory). Modified for mlx-audio: reformatted with black.
"""Vendored text normaliser (upstream commit fe5e6ec), via MayaResearch/maya2-native (maya3/text/tn, Apache-2.0).

In mlx-audio these modules are reformatted with black; their logic is unchanged (checked output-for-output against
the original on all 12 languages). Use mlx_audio.tts.models.maya2.text.normalise_text rather than this package.
"""

from .normalize import NormalizationError, normalize, resolve_language  # noqa: F401
