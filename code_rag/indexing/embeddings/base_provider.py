"""
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from enum import Enum

from ...models import CodeChunk


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the embedding provider
        
        Args:
            model_name: Name of the embedding model
            **kwargs: Provider-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs
        self._dimension = None
        
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider (load model, setup client, etc.)"""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors as numpy arrays
        """
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query
        
        Args:
            query: Query text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    def create_embedding_text(self, chunk: CodeChunk) -> str:
        """Create enriched text for embedding (can be overridden)"""
        metadata = chunk.file_metadata
        
        context_parts = [
            f"File: {metadata.relative_path}",
            f"Type: {metadata.file_type}",
            f"Folder: {metadata.folder_path}",
        ]
        
        if metadata.functions:
            context_parts.append(f"Functions: {', '.join(metadata.functions[:5])}")
        
        if metadata.classes:
            context_parts.append(f"Classes: {', '.join(metadata.classes[:3])}")
        
        if metadata.imports:
            context_parts.append(f"Imports: {', '.join(metadata.imports[:5])}")
        
        context = " | ".join(context_parts)
        
        return f"{context}\n\n{chunk.content}"
    
    def embed_chunks(self, chunks: List[CodeChunk]) -> List[np.ndarray]:
        """
        Generate embeddings for code chunks
        
        Args:
            chunks: List of code chunks to embed
            
        Returns:
            List of embedding vectors
        """
        texts = [self.create_embedding_text(chunk) for chunk in chunks]
        return self.embed_texts(texts)
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider"""
        pass
    
    def validate_model(self) -> bool:
        """Validate if the model is supported (can be overridden)"""
        return True 