import sys
from pathlib import Path

from setuptools import find_packages, setup

# Get the project root directory
root_dir = Path(__file__).parent

# Add the package directory to the Python path
package_dir = root_dir / "mlx_audio"
sys.path.append(str(package_dir))

# Read the requirements from the requirements.txt file
requirements_path = root_dir / "requirements.txt"
with open(requirements_path) as fid:
    requirements = [l.strip() for l in fid.readlines()]

# Import the version from the package
from mlx_audio.version import __version__

# Setup configuration
setup(
    name="mlx-audio",
    version=__version__,
    description="MLX-Audio is a package for inference of text-to-speech (TTS) and speech-to-speech (STS) models locally on your Mac using MLX",
    long_description=open(root_dir / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="prince.gdt@gmail.com",
    author="Prince Canuma",
    url="https://github.com/Blaizzy/mlx-audio",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(where=root_dir),
    include_package_data=True,
    package_data={
        "mlx_audio": [
            "tts/*.html",
            "tts/*.js",
            "tts/*.css",
            "tts/static/**/*",
            "stt/**/*.tiktoken",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    extras_require={
        "py38": ["importlib_resources"],
    },
    entry_points={
        "console_scripts": [
            "mlx_audio.tts.generate = mlx_audio.tts.generate:main",
            "mlx_audio.server = mlx_audio.server:main",
        ]
    },
)
