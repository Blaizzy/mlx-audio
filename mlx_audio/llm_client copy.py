import os
import requests
import logging
from abc import ABC, abstractmethod
import torch

# Assuming you have a logger setup similar to server.py
logger = logging.getLogger("mlx_audio_server")

class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    @abstractmethod
    def chat(self, prompt: str) -> str:
        """Generate a conversational reply to the given prompt."""
        pass

class OllamaClient(LLMClient):
    """LLM client for Ollama."""
    def __init__(self, base_url: str, model: str, system_prompt: str):
        self.base_url = base_url
        self.model = model
        self.system_prompt = system_prompt

    def chat(self, prompt: str) -> str:
        """Return a conversational reply from the local Ollama server."""
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return "I'm sorry, I'm having trouble connecting to the language model." # More user-friendly fallback

class MlxClient(LLMClient):
    """LLM client for a local MLX model."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the MLX model and tokenizer using mlx-lm."""
        try:
            from mlx_lm import load, generate

            logger.info(f"Loading MLX LLM model from {self.model_path}...")
            
            # Load model and tokenizer with mlx-lm
            self.model, self.tokenizer = load(self.model_path)
            self.generate = generate
            
            logger.info("MLX LLM model loaded successfully.")
            
        except ImportError as e:
            logger.error(
                "mlx-lm is not installed. Please install it with `pip install mlx-lm`"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}")
            raise

    def chat(self, prompt: str) -> str:
        """Generate a conversational reply using the local MLX model."""
        if not self.model or not self.tokenizer:
            return "MLX model is not loaded. Please check the server logs."
        
        try:
            # The generate function from mlx_lm is used here.
            # You might need to adjust parameters based on your model's needs.
            response = self.generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt, 
                verbose=False,
                max_tokens=32000  # Adjust as needed
            )
            return response.strip()
            
        except Exception as e:
            logger.error(f"MLX inference failed: {e}")
            return "I'm sorry, I encountered an error during inference."
