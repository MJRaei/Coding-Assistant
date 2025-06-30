from typing import List
import logging
import time

import openai
from openai import OpenAI
import numpy as np

from ..models import CodeChunk
from ..config import DEFAULT_EMBEDDING_MODEL, OPENAI_BATCH_SIZE, OPENAI_RATE_LIMIT_DELAY, EMBEDDING_MODELS


class EmbeddingProcessor:
    """Handles embedding generation using OpenAI"""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        """
        Initialize OpenAI embedding processor
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model_name: OpenAI embedding model (if None, uses DEFAULT_EMBEDDING_MODEL from config)
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.logger = logging.getLogger(__name__)
        
        self.model_dimensions = EMBEDDING_MODELS
        
        if self.model_name not in self.model_dimensions:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        return self.model_dimensions[self.model_name]
    
    def create_embedding_text(self, chunk: CodeChunk) -> str:
        """Create enriched text for embedding"""
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
    
    def _call_openai_embedding(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Call OpenAI API with rate limiting and batching"""
        batch_size = batch_size or OPENAI_BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                    
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    if i + batch_size < len(texts):
                        time.sleep(OPENAI_RATE_LIMIT_DELAY)
                    
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
        
        return all_embeddings
    
    def generate_embeddings(self, chunks: List[CodeChunk]) -> List[np.ndarray]:
        """Generate embeddings for chunks using OpenAI"""
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.model_name}")
        
        texts = [self.create_embedding_text(chunk) for chunk in chunks]
        
        embeddings_list = self._call_openai_embedding(texts)
        
        embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_list]
        
        self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise 