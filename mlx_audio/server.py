"""Main module for MLX Audio API server.

This module provides a FastAPI-based server for hosting MLX Audio models,
including Text-to-Speech (TTS), Speech-to-Text (STT), and Speech-to-Speech (S2S) models.
It offers an OpenAI-compatible API for Audio completions and model management.
"""

import argparse
import asyncio
import importlib
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_audio.stt.utils import MODEL_REMAPPING as MODEL_STT_REMAPPING
from mlx_audio.stt.utils import load_model as load_stt_model
from mlx_audio.tts.utils import MODEL_REMAPPING as MODEL_TTS_REMAPPING
from mlx_audio.tts.utils import load_config
from mlx_audio.tts.utils import load_model as load_tts_model


def get_available_models(model_type: str):
    """
    Get a list of all available model types by scanning the models directory.

    Returns:
        List[str]: A list of available model type names
    """
    models_dir = Path(__file__).parent / model_type / "models"
    available_models = []

    if models_dir.exists() and models_dir.is_dir():
        for item in models_dir.iterdir():
            if item.is_dir() and not item.name.startswith("__"):
                available_models.append(item.name)

    return available_models


def _try_resolve_model_for_category(
    category: str,
    initial_hint: str,
    model_name_components: List[str],
    remapping_dict: Dict[str, str],
) -> Tuple[bool, str]:
    """
    Attempts to resolve and verify a model architecture for a given category.

    Args:
        category (str): The model category (e.g., "tts", "stt").
        initial_hint (str): An initial hint for the model architecture.
        model_name_components (List[str]): List of model name components for further refinement.
        remapping_dict (Dict[str, str]): The remapping dictionary for the category.

    Returns:
        A tuple (success, arch_name):
        - success (bool): True if a valid architecture was found and its module imported.
        - arch_name (str): The specific architecture name that was processed or attempted.
    """
    # Stage 1: Initial remapping based on the hint
    arch_candidate = remapping_dict.get(initial_hint, initial_hint)

    # Stage 2: Refine with model_name parts
    if model_name_components:
        category_specific_architectures = get_available_models(category)
        for part in model_name_components:
            # Check if the part matches an available model directory name for the category
            if part in category_specific_architectures:
                arch_candidate = part

            # Check if the part is in the category's custom remapping dictionary
            # This remapping takes precedence and breaks the loop if found
            if part in remapping_dict:
                arch_candidate = remapping_dict[part]
                break

    # Stage 3: Import attempt
    try:
        importlib.import_module(f"mlx_audio.{category}.models.{arch_candidate}")
        return True, arch_candidate  # Success
    except ImportError:
        return (
            False,
            arch_candidate,
        )  # Failure, return the candidate for error messaging


def get_model_type(model_type: str, model_name: List[str]):
    """
    Retrieve the model category ("tts" or "stt") based on an architecture hint and model name parts.

    This function attempts to find the appropriate model category by:
    1. Trying to resolve a TTS model architecture based on the `model_type` hint and `model_name` parts.
    2. If TTS resolution fails, trying to resolve an STT model architecture using the same inputs.

    Args:
        model_type (str): An initial hint for the model architecture (e.g., "vits", "whisper").
                           This is used as a starting point and can be overridden by `model_name` parts.
        model_name (List[str]): List of model name components. These are checked against
                                available model directories and remapping dictionaries to refine
                                the architecture guess.

    Returns:
        str: The resolved model category ("tts" or "stt").

    Raises:
        ValueError: If a supported model architecture and category cannot be determined after checking both TTS and STT.
    """

    # Try to resolve as a TTS model
    is_tts, _ = _try_resolve_model_for_category(
        category="tts",
        initial_hint=model_type,
        model_name_components=model_name,
        remapping_dict=MODEL_TTS_REMAPPING,
    )
    if is_tts:
        return "tts"

    # If not TTS, try to resolve as an STT model
    is_stt, last_tried_arch = _try_resolve_model_for_category(
        category="stt",
        initial_hint=model_type,
        model_name_components=model_name,
        remapping_dict=MODEL_STT_REMAPPING,
    )
    if is_stt:
        return "stt"

    # If both TTS and STT resolution fail, return None
    return None


def get_model_name(model_path: Union[str, Path]) -> str:
    model_name = None
    if isinstance(model_path, str):
        model_name = model_path.lower().split("/")[-1].split("-")
    elif isinstance(model_path, Path):
        index = model_path.parts.index("hub")
        model_name = model_path.parts[index + 1].lower().split("--")[-1].split("-")
    else:
        raise ValueError(f"Invalid model path type: {type(model_path)}")
    return model_name


class ModelProvider:
    def __init__(self):
        self.models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    def load_model(self, model_name: str):
        if model_name not in self.models:
            config = load_config(model_name)
            model_type = config.get("model_type", None)

            model_name = get_model_name(model_name)
            model_type = get_model_type(model_type, model_name)
            if model_type == "tts":
                self.models[model_name] = load_tts_model(model_name)
            elif model_type == "stt":
                self.models[model_name] = load_stt_model(model_name)
            else:
                raise ValueError(f"Model type {model_type} not supported.")

        return self.models[model_name]

    async def remove_model(self, model_name: str) -> bool:
        async with self.lock:
            if model_name in self.models:
                del self.models[model_name]
                return True
            return False

    async def get_available_models(self):
        async with self.lock:
            return list(self.models.keys())


app = FastAPI()


def int_or_float(value):

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not an int or float")


def calculate_default_workers(workers: int = 2) -> int:
    if num_workers_env := os.getenv("FASTMLX_NUM_WORKERS"):
        try:
            workers = int(num_workers_env)
        except ValueError:
            workers = max(1, int(os.cpu_count() * float(num_workers_env)))
    return workers


# Add CORS middleware
def setup_cors(app: FastAPI, allowed_origins: List[str]):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# Initialize the ModelProvider
model_provider = ModelProvider()


@app.get("/v1/models")
async def list_models():
    """
    Get list of models - provided in OpenAI API compliant format.
    """
    models = await model_provider.get_available_models()
    models_data = []
    for model in models:
        models_data.append(
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "system",
            }
        )
    return {"object": "list", "data": models_data}


@app.post("/v1/models")
async def add_model(model_name: str):
    """
    Add a new model to the API.

    Args:
        model_name (str): The name of the model to add.

    Returns:
        dict (dict): A dictionary containing the status of the operation.
    """
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}


@app.delete("/v1/models")
async def remove_model(model_name: str):
    """
    Remove a model from the API.

    Args:
        model_name (str): The name of the model to remove.

    Returns:
        Response (str): A 204 No Content response if successful.

    Raises:
        HTTPException (str): If the model is not found.
    """
    model_name = unquote(model_name).strip('"')
    removed = await model_provider.remove_model(model_name)
    if removed:
        return Response(status_code=204)  # 204 No Content - successful deletion
    else:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


def run():
    parser = argparse.ArgumentParser(description="FastMLX API server")
    parser.add_argument(
        "--allowed-origins",
        nargs="+",
        default=["*"],
        help="List of allowed origins for CORS",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    parser.add_argument(
        "--reload",
        type=bool,
        default=False,
        help="Enable auto-reload of the server. Only works when 'workers' is set to None.",
    )

    parser.add_argument(
        "--workers",
        type=int_or_float,
        default=calculate_default_workers(),
        help="""Number of workers. Overrides the `FASTMLX_NUM_WORKERS` env variable.
        Can be either an int or a float.
        If an int, it will be the number of workers to use.
        If a float, number of workers will be this fraction of the  number of CPU cores available, with a minimum of 1.
        Defaults to the `FASTMLX_NUM_WORKERS` env variable if set and to 2 if not.
        To use all available CPU cores, set it to 1.0.

        Examples:
        --workers 1 (will use 1 worker)
        --workers 1.0 (will use all available CPU cores)
        --workers 0.5 (will use half the number of CPU cores available)
        --workers 0.0 (will use 1 worker)""",
    )

    args = parser.parse_args()
    if isinstance(args.workers, float):
        args.workers = max(1, int(os.cpu_count() * args.workers))

    setup_cors(app, args.allowed_origins)

    import uvicorn

    uvicorn.run(
        "mlx_audio:server",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        loop="asyncio",
    )


if __name__ == "__main__":
    run()
