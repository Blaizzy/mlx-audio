"""
MCP Client for integrating with Model Context Protocol servers
Provides memory persistence and web search capabilities
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger("mlx_audio_mcp")

class MCPSearchClient:
    """Client for MCP search servers"""
    
    def __init__(self):
        self.search_cache = {}
        self.cache_duration = 3600  # 1 hour cache
    
    def search(self, query: str) -> str:
        """Perform web search using MCP server"""
        try:
            # Check cache first
            cache_key = query.lower().strip()
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if (datetime.now() - cached_result['timestamp']).seconds < self.cache_duration:
                    return cached_result['result']
            
            # This would integrate with actual MCP search server
            # For now, using DuckDuckGo as fallback
            import requests
            
            search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(search_url, timeout=5)
            data = response.json()
            
            result = ""
            if 'Abstract' in data and data['Abstract']:
                result = data['Abstract']
            elif 'RelatedTopics' in data and data['RelatedTopics']:
                result = data['RelatedTopics'][0]['Text'][:500]
            else:
                result = "No relevant search results found."
            
            # Cache the result
            self.search_cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"MCP search failed: {e}")
            return f"Search unavailable: {str(e)}"

class MemoryManager:
    """Persistent conversation memory manager"""
    
    def __init__(self, memory_file: str = "conversation_memory.json"):
        self.memory_file = memory_file
        self.conversation_history = []
        self.max_history_length = 50
        self.load_memory()
    
    def load_memory(self):
        """Load conversation history from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r') as f:
                    self.conversation_history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            self.conversation_history = []
    
    def save_memory(self):
        """Save conversation history to file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def add_exchange(self, user_message: str, assistant_response: str):
        """Add conversation exchange to memory"""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        # Save to file immediately
        self.save_memory()
    
    def get_context(self, max_exchanges: int = 5) -> str:
        """Get formatted conversation context"""
        if not self.conversation_history:
            return ""
        
        recent_exchanges = self.conversation_history[-max_exchanges:]
        context_parts = []
        
        for exchange in recent_exchanges:
            context_parts.append(f"User: {exchange['user']}")
            context_parts.append(f"Assistant: {exchange['assistant']}")
        
        return "\n".join(context_parts)
    
    def clear_memory(self):
        """Clear all conversation memory"""
        self.conversation_history = []
        self.save_memory()
    
    def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search conversation history for relevant exchanges"""
        query_lower = query.lower()
        relevant_exchanges = []
        
        for exchange in self.conversation_history:
            if (query_lower in exchange['user'].lower() or 
                query_lower in exchange['assistant'].lower()):
                relevant_exchanges.append(exchange)
        
        return relevant_exchanges[-5:]  # Return last 5 relevant exchanges
