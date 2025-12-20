# Copyright © 2023-2024 Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)


def __getattr__(name):
    if name == "convert":
        from mlx_audio.convert import convert

        return convert
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["convert"]
