"""
Factory for creating appropriate language-specific chunkers.
"""

from typing import Dict, Type, Optional, List
import logging

from .base_chunker import BaseLanguageChunker, ChunkingStrategy
from .python_chunker import PythonChunker


class ChunkerFactory:
    """Factory for creating language-specific chunkers"""
    
    # Registry of available chunkers
    _chunkers: Dict[str, Type[BaseLanguageChunker]] = {
        'py': PythonChunker,
        'pyx': PythonChunker,
        'pyi': PythonChunker,
    }
    
    @classmethod
    def register_chunker(cls, file_extension: str, chunker_class: Type[BaseLanguageChunker]):
        """Register a new chunker for a file extension"""
        cls._chunkers[file_extension] = chunker_class
    
    @classmethod
    def get_chunker(cls, file_extension: str, max_tokens: int = 2000, 
                   overlap_tokens: int = 200, 
                   strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_FIRST) -> BaseLanguageChunker:
        """Get appropriate chunker for file extension"""
        
        # Remove leading dot if present
        ext = file_extension.lstrip('.')
        
        chunker_class = cls._chunkers.get(ext)
        if chunker_class:
            return chunker_class(max_tokens, overlap_tokens, strategy)
        
        # Fallback to Python chunker as default (most generic)
        logger = logging.getLogger(__name__)
        logger.warning(f"No specific chunker found for '{ext}', using Python chunker as fallback")
        return PythonChunker(max_tokens, overlap_tokens, strategy)
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of all supported file extensions"""
        return list(cls._chunkers.keys())
    
    @classmethod
    def is_supported(cls, file_extension: str) -> bool:
        """Check if file extension is supported"""
        ext = file_extension.lstrip('.')
        return ext in cls._chunkers 