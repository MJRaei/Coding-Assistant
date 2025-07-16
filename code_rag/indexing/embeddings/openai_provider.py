"""
OpenAI embedding provider implementation.
"""

import time
import logging
from typing import List, Dict, Any
import numpy as np

try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .base_provider import BaseEmbeddingProvider
from ...config import OPENAI_BATCH_SIZE, OPENAI_RATE_LIMIT_DELAY


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider"""
    
    SUPPORTED_MODELS = {
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
        'text-embedding-ada-002': 1536
    }
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run: pip install openai")
        
        super().__init__(model_name, **kwargs)
        self.api_key = api_key
        self.client = None
        self.batch_size = kwargs.get('batch_size', OPENAI_BATCH_SIZE)
        self.rate_limit_delay = kwargs.get('rate_limit_delay', OPENAI_RATE_LIMIT_DELAY)
        self.logger = logging.getLogger(__name__)
        
    def initialize(self) -> None:
        """Initialize OpenAI client"""
        self.client = OpenAI(api_key=self.api_key)
        
        if not self.validate_model():
            raise ValueError(f"Unsupported OpenAI model: {self.model_name}")
        
        self._dimension = self.SUPPORTED_MODELS[self.model_name]
        
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        if self._dimension is None:
            self._dimension = self.SUPPORTED_MODELS.get(self.model_name, 1536)
        return self._dimension
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for texts using OpenAI API"""
        if not self.client:
            self.initialize()
            
        self.logger.info(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    self.logger.debug(f"Processing batch {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                    
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    
                    batch_embeddings = [
                        np.array(data.embedding, dtype=np.float32) 
                        for data in response.data
                    ]
                    all_embeddings.extend(batch_embeddings)
                    
                    if i + self.batch_size < len(texts):
                        time.sleep(self.rate_limit_delay)
                    
                    break
                    
                except openai.RateLimitError as e:
                    retry_count += 1
                    wait_time = min(60, (2 ** retry_count))
                    self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                    
                except openai.APIError as e:
                    retry_count += 1
                    self.logger.error(f"OpenAI API error: {e}")
                    if retry_count >= max_retries:
                        raise
                    time.sleep(2 ** retry_count)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error in embedding generation: {e}")
                    raise
        
        self.logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        if not self.client:
            self.initialize()
            
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise
    
    def validate_model(self) -> bool:
        """Validate if the model is supported"""
        return self.model_name in self.SUPPORTED_MODELS
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider"""
        return {
            'provider': 'openai',
            'model': self.model_name,
            'dimension': self.get_embedding_dimension(),
            'batch_size': self.batch_size,
            'rate_limit_delay': self.rate_limit_delay,
            'supported_models': list(self.SUPPORTED_MODELS.keys())
        } 