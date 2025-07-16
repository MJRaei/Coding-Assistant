"""
Updated embedding processor using the new flexible provider system.
"""

import logging
from typing import List, Optional
import numpy as np

from ..models import CodeChunk
from .embeddings import EmbeddingProviderFactory, BaseEmbeddingProvider
from ..config import DEFAULT_EMBEDDING_MODEL


class EmbeddingProcessor:
    """
    Flexible embedding processor supporting multiple providers.
    
    This replaces the old OpenAI-only implementation with a provider-based system
    that supports OpenAI, Hugging Face, Ollama, and other embedding services.
    """
    
    def __init__(self, provider_type: str = None, model_name: str = None, **kwargs):
        """
        Initialize embedding processor with flexible provider support
        
        Args:
            provider_type: Type of embedding provider ('openai', 'huggingface', 'ollama')
            model_name: Name of the embedding model
            **kwargs: Provider-specific configuration
        """
        self.provider_type = provider_type or 'openai'
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.logger = logging.getLogger(__name__)
        
        self.provider: BaseEmbeddingProvider = EmbeddingProviderFactory.create_provider(
            self.provider_type, 
            self.model_name, 
            **kwargs
        )
        
        self.logger.info(f"Initialized {self.provider_type} provider with model {self.model_name}")
    
    @classmethod
    def from_openai(cls, api_key: str = None, model_name: str = None) -> 'EmbeddingProcessor':
        """Create OpenAI processor"""
        return cls(
            provider_type='openai',
            model_name=model_name or DEFAULT_EMBEDDING_MODEL,
            api_key=api_key
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        return self.provider.get_embedding_dimension()
    
    def create_embedding_text(self, chunk: CodeChunk) -> str:
        """Create enriched text for embedding"""
        return self.provider.create_embedding_text(chunk)
    
    def generate_embeddings(self, chunks: List[CodeChunk]) -> List[np.ndarray]:
        """Generate embeddings for code chunks"""
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.provider_type}:{self.model_name}")
        
        embeddings = self.provider.embed_chunks(chunks)
        
        self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.provider.embed_query(query)
    
    def get_provider_info(self) -> dict:
        """Get information about the current provider"""
        return self.provider.get_provider_info()
    
    @classmethod
    def create_openai_processor(cls, model_name: str = None, api_key: str = None, **kwargs) -> 'EmbeddingProcessor':
        """Convenience method to create OpenAI processor (backward compatibility)"""
        return cls(
            provider_type='openai',
            model_name=model_name or 'text-embedding-3-small',
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def create_huggingface_processor(cls, model_name: str = None, device: str = None, **kwargs) -> 'EmbeddingProcessor':
        """Convenience method to create Hugging Face processor"""
        return cls(
            provider_type='huggingface',
            model_name=model_name or 'all-MiniLM-L6-v2',
            device=device or 'cpu',
            **kwargs
        )
    
    @classmethod
    def create_ollama_processor(cls, model_name: str = None, host: str = None, **kwargs) -> 'EmbeddingProcessor':
        """Convenience method to create Ollama processor"""
        return cls(
            provider_type='ollama',
            model_name=model_name or 'nomic-embed-text',
            host=host or 'http://localhost:11434',
            **kwargs
        )
    
    def _call_openai_embedding(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Legacy method for backward compatibility"""
        if self.provider_type != 'openai':
            raise ValueError("_call_openai_embedding only available for OpenAI provider")
        
        embeddings = self.provider.embed_texts(texts)
        return [emb.tolist() for emb in embeddings] 