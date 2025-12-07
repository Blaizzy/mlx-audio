from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mlx-audio")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for editable installs without metadata
