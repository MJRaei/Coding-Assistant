"""
Ollama embedding provider implementation.
"""

import logging
from typing import List, Dict, Any
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from .base_provider import BaseEmbeddingProvider


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Ollama embedding provider for local models"""
    
    POPULAR_MODELS = {
        'nomic-embed-text': 768,
        'all-minilm': 384,
        'snowflake-arctic-embed': 1024,
        'mxbai-embed-large': 1024,
        'bge-large': 1024,
        'bge-base': 768,
        'bge-small': 512,
    }
    
    def __init__(self, model_name: str, host: str = None, **kwargs):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama library not installed. "
                "Run: pip install ollama"
            )
        
        super().__init__(model_name, **kwargs)
        self.host = host or 'http://localhost:11434'
        self.client = None
        self.batch_size = kwargs.get('batch_size', 16)
        self.timeout = kwargs.get('timeout', 30)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> None:
        """Initialize Ollama client"""
        try:
            self.client = ollama.Client(host=self.host)
            
            try:
                response = self.client.embeddings(model=self.model_name, prompt="test")
                self._dimension = len(response['embedding'])
                self.logger.info(f"Ollama model {self.model_name} loaded successfully. Dimension: {self._dimension}")
            except Exception as e:
                self.logger.error(f"Model {self.model_name} not available. Error: {e}")
                self.logger.info("Available models can be listed with: ollama list")
                raise
                
        except Exception as e:
            self.logger.error(f"Error connecting to Ollama at {self.host}: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        if self._dimension is None:
            if self.model_name in self.POPULAR_MODELS:
                self._dimension = self.POPULAR_MODELS[self.model_name]
            else:
                if not self.client:
                    self.initialize()
        return self._dimension
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for texts using Ollama"""
        if not self.client:
            self.initialize()
        
        self.logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        
        embeddings = []
        
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                
                for text in batch:
                    response = self.client.embeddings(
                        model=self.model_name,
                        prompt=text
                    )
                    embedding = np.array(response['embedding'], dtype=np.float32)
                    embeddings.append(embedding)
                
                if i + self.batch_size < len(texts):
                    self.logger.debug(f"Processed {i + len(batch)}/{len(texts)} texts")
            
            self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if not self.client:
            self.initialize()
        
        try:
            response = self.client.embeddings(
                model=self.model_name,
                prompt=query
            )
            return np.array(response['embedding'], dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise
    
    def validate_model(self) -> bool:
        """Validate if the model is available"""
        try:
            if not self.client:
                self.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider"""
        return {
            'provider': 'ollama',
            'model': self.model_name,
            'dimension': self.get_embedding_dimension(),
            'host': self.host,
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'popular_models': list(self.POPULAR_MODELS.keys())
        } 