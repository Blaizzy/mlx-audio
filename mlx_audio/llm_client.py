import os
import requests
import logging
from abc import ABC, abstractmethod
import torch
from datetime import datetime
import urllib.parse
from typing import Optional

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
    """LLM client for a local MLX model with memory and optional system prompt."""
    
    def __init__(self, model_path: str, system_prompt: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        self.search_cache = {}  # Cache for search results
        # Prefer explicit parameter, then environment variable, then fallback default
        self.system_prompt = (system_prompt or os.getenv("MLX_SYSTEM_PROMPT") or "You are an empathetic concise voice assistant.")
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

    def add_to_memory(self, user_message: str, assistant_response: str):
        """Add conversation exchange to memory."""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def get_memory_context(self) -> str:
        """Get formatted conversation history for context."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(context_parts)

    def web_search(self, query: str) -> str:
        """Perform web search using DuckDuckGo Instant Answer API."""
        # Search functionality temporarily disabled
        return "Search functionality is currently disabled"

    def chat(self, prompt: str) -> str:
        """Generate a conversational reply using the local MLX model with memory."""
        if not self.model or not self.tokenizer:
            return "MLX model is not loaded. Please check the server logs."
        
        try:
            prompt_clean = prompt.strip()
            prompt_lower = prompt_clean.lower()
            
            # Skip empty or too short prompts
            if not prompt_clean or len(prompt_clean) < 3:
                return "Please provide a more complete question."
            
            # Memory commands
            if "clear memory" in prompt_lower:
                self.conversation_history = []
                return "Memory cleared. I'll remember new conversations."
            
            if "what did we talk about" in prompt_lower:
                if self.conversation_history:
                    return "Our recent conversations:\n" + "\n".join([
                        f"• {m['user'][:50]}..." for m in self.conversation_history[-2:]
                    ])
                return "No previous conversations in this session."
            
            # Build efficient prompt (search functionality disabled)
            context_parts = []
            
            # Add memory context (brief)
            if self.conversation_history:
                recent = self.conversation_history[-2:]  # Only last 2
                if recent:
                    context_parts.append("Recent context: " + " | ".join([f"Q: {m['user'][:30]}..." for m in recent]))
            
            # Regular conversation - more natural prompt
            if context_parts:
                final_prompt = f"{context_parts[0]}\n\nUser: {prompt_clean}\nAssistant:"
            else:
                final_prompt = f"User: {prompt_clean}\nAssistant:"

            # Prepend system prompt if provided
            if self.system_prompt:
                final_prompt = f"{self.system_prompt}\n\n" + final_prompt
            
            logger.info(f"Regular response for: '{prompt_clean[:50]}...'")
            
            # Generate response
            response = self.generate(
                self.model, 
                self.tokenizer, 
                prompt=final_prompt, 
                verbose=False,
                max_tokens=2000
            )
            
            clean_response = response.strip()
            
            # Store in memory (shorter entries)
            self.add_to_memory(prompt_clean, clean_response[:300])
            
            return clean_response
            
        except Exception as e:
            logger.error(f"MLX inference failed: {e}")
            return "I'm sorry, I encountered an error. Please try rephrasing your question."
