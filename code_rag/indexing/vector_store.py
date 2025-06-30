from typing import List, Tuple
import logging

import faiss
import pickle
import numpy as np

from ..models import CodeChunk


class VectorStore:
    """Simple FAISS-based vector storage"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[CodeChunk] = []
        self.embeddings: List[np.ndarray] = []
        self.logger = logging.getLogger(__name__)
    
    def add_chunks(self, chunks: List[CodeChunk], embeddings: List[np.ndarray]):
        """Add chunks and their embeddings to the store"""
        normalized_embeddings = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized_embeddings.append(emb / norm)
            else:
                normalized_embeddings.append(emb)
        
        embeddings_array = np.array(normalized_embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.chunks.extend(chunks)
        self.embeddings.extend(normalized_embeddings)
        
        self.logger.info(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Search for similar chunks"""
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        faiss.write_index(self.index, f"{filepath}.faiss")
        with open(f"{filepath}.chunks", 'wb') as f:
            pickle.dump(self.chunks, f)
        self.logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the vector store from disk"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        with open(f"{filepath}.chunks", 'rb') as f:
            self.chunks = pickle.load(f)
        self.logger.info(f"Vector store loaded from {filepath}")
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store"""
        return {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_size': self.index.ntotal
        } 