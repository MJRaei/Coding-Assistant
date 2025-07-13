"""
Indexing components for building code embeddings and vector indices.
"""

from .file_processor import FileProcessor
from .chunk_processor import ChunkProcessor
from .embedding_processor import EmbeddingProcessor
from .vector_store import VectorStore
from .index_builder import IndexBuilder

# Import new chunking components
from .chunking import ChunkerFactory, ChunkingStrategy, BoundaryType, BaseLanguageChunker, PythonChunker, JSChunker

__all__ = [
    'FileProcessor',
    'ChunkProcessor', 
    'EmbeddingProcessor',
    'VectorStore',
    'IndexBuilder',
    # New chunking exports
    'ChunkerFactory',
    'ChunkingStrategy', 
    'BoundaryType',
    'BaseLanguageChunker',
    'PythonChunker',
    'JSChunker'
] 