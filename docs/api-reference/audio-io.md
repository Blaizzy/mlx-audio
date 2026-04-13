# Audio I/O Reference

The `mlx_audio.audio_io` module provides functions for reading and writing audio files. It uses **miniaudio** for WAV decoding/encoding and **ffmpeg** for MP3, FLAC, OGG, Opus, Vorbis, and M4A/AAC formats.

## Reading Audio

::: mlx_audio.audio_io.read
    options:
      show_source: false
      heading_level: 3

## Writing Audio

::: mlx_audio.audio_io.write
    options:
      show_source: false
      heading_level: 3

## Soundfile-Compatible Aliases

Drop-in replacements for `soundfile.read()` and `soundfile.write()`:

::: mlx_audio.audio_io.sf_read
    options:
      show_source: false
      heading_level: 3

::: mlx_audio.audio_io.sf_write
    options:
      show_source: false
      heading_level: 3

## Supported Formats

### Reading

| Format | Backend | Notes |
|--------|---------|-------|
| WAV | miniaudio | No external dependencies |
| MP3 | miniaudio | No external dependencies |
| FLAC | miniaudio | No external dependencies |
| Vorbis / OGG | ffmpeg | Requires ffmpeg |
| M4A / AAC | ffmpeg | Requires ffmpeg |
| Opus | ffmpeg | Requires ffmpeg |

### Writing

| Format | Backend | Notes |
|--------|---------|-------|
| WAV | miniaudio | No external dependencies |
| MP3 | ffmpeg | Requires ffmpeg |
| FLAC | ffmpeg | Requires ffmpeg |
| OGG | ffmpeg | Uses FLAC codec in OGG container |
| Opus | ffmpeg | Requires ffmpeg |
| PCM / Raw | native | Raw int16 bytes |

## Installing ffmpeg

ffmpeg is required for non-WAV encoding and for M4A/AAC/OGG decoding:

=== "macOS"

    ```bash
    brew install ffmpeg
    ```

=== "Ubuntu / Debian"

    ```bash
    sudo apt install ffmpeg
    ```

!!! note
    WAV format works without ffmpeg. If you only need WAV, no extra installation is required.
