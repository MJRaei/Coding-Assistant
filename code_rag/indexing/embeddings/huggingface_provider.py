"""
Hugging Face embedding provider implementation.
"""

import logging
from typing import List, Dict, Any
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from .base_provider import BaseEmbeddingProvider


class HuggingFaceEmbeddingProvider(BaseEmbeddingProvider):
    """Hugging Face embedding provider using sentence-transformers"""
    
    POPULAR_MODELS = {
        'all-MiniLM-L6-v2': 384,
        'all-mpnet-base-v2': 768,
        'multi-qa-mpnet-base-dot-v1': 768,
        'all-distilroberta-v1': 768,
        'all-MiniLM-L12-v2': 384,
        'paraphrase-multilingual-MiniLM-L12-v2': 384,
        'distiluse-base-multilingual-cased': 512,
        'paraphrase-albert-small-v2': 768,
        'sentence-transformers/all-MiniLM-L6-v2': 384,
        'sentence-transformers/all-mpnet-base-v2': 768,
    }
    
    def __init__(self, model_name: str, device: str = None, **kwargs):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers library not installed. "
                "Run: pip install sentence-transformers"
            )
        
        super().__init__(model_name, **kwargs)
        self.device = device or 'cpu'
        self.model = None
        self.batch_size = kwargs.get('batch_size', 32)
        self.normalize_embeddings = kwargs.get('normalize_embeddings', True)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> None:
        """Initialize the sentence transformer model"""
        try:
            self.logger.info(f"Loading Hugging Face model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get dimension by encoding a sample text
            sample_embedding = self.model.encode(["sample text"])
            self._dimension = sample_embedding.shape[1]
            
            self.logger.info(f"Model loaded successfully. Dimension: {self._dimension}")
            
        except Exception as e:
            self.logger.error(f"Error loading model {self.model_name}: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        if self._dimension is None:
            if self.model_name in self.POPULAR_MODELS:
                self._dimension = self.POPULAR_MODELS[self.model_name]
            else:
                if not self.model:
                    self.initialize()
                # Dimension will be set during initialization
        return self._dimension
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for texts using Hugging Face model"""
        if not self.model:
            self.initialize()
        
        self.logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        
        try:
            # Generate embeddings in batches
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=len(texts) > 100
            )
            
            # Convert to list of numpy arrays
            embedding_list = [
                np.array(emb, dtype=np.float32) 
                for emb in embeddings
            ]
            
            self.logger.info(f"Successfully generated {len(embedding_list)} embeddings")
            return embedding_list
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if not self.model:
            self.initialize()
        
        try:
            embedding = self.model.encode(
                [query],
                normalize_embeddings=self.normalize_embeddings
            )
            return np.array(embedding[0], dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise
    
    def validate_model(self) -> bool:
        """Validate if the model can be loaded"""
        try:
            if not self.model:
                self.initialize()
            return True
        except Exception as e:
            self.logger.error(f"Model validation failed: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider"""
        return {
            'provider': 'huggingface',
            'model': self.model_name,
            'dimension': self.get_embedding_dimension(),
            'device': self.device,
            'batch_size': self.batch_size,
            'normalize_embeddings': self.normalize_embeddings,
            'popular_models': list(self.POPULAR_MODELS.keys())
        } 