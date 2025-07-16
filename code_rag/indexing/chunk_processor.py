"""
Updated ChunkProcessor using the new modular chunking system.
"""

import re
from pathlib import Path
from typing import List, Optional
import logging

from ..models import FileMetadata, CodeChunk
from .file_processor import FileProcessor
from ..config import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS

from .chunking import ChunkerFactory, ChunkingStrategy


class ChunkProcessor:
    """
    Handles content chunking strategies using modular language-specific chunkers.
    
    This is the main interface that uses the new modular chunking system while
    maintaining backward compatibility with existing code.
    """
    
    def __init__(self, max_tokens: int = None, overlap_tokens: int = None, 
                 chunking_strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE_PRESERVING):
        """
        Initialize ChunkProcessor with new modular chunking system
        
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Overlap between chunks  
            chunking_strategy: Strategy to use for chunking
        """
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.overlap_tokens = overlap_tokens or DEFAULT_OVERLAP_TOKENS
        self.chunking_strategy = chunking_strategy
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"ChunkProcessor initialized with strategy: {chunking_strategy.value}")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters) - kept for backward compatibility"""
        return len(text) // 4
    
    def process_file(self, file_path: Path, file_metadata: FileMetadata) -> List[CodeChunk]:
        """
        Process a single file into chunks using appropriate language-specific chunker
        
        Args:
            file_path: Path to the file
            file_metadata: Metadata about the file
            
        Returns:
            List of CodeChunk objects
        """
        content = FileProcessor(str(file_path.parent)).read_file_content(file_path)
        if not content:
            return []
        
        chunker = ChunkerFactory.get_chunker(
            file_extension=file_metadata.file_type,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
            strategy=self.chunking_strategy
        )
        
        self.logger.info(f"Using {chunker.__class__.__name__} for {file_metadata.relative_path}")
        
        chunks = chunker.chunk_content(content, file_metadata)
        
        self.logger.info(f"Created {len(chunks)} chunks for {file_metadata.relative_path}")
        return chunks
    
    def chunk_by_functions(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """
        Legacy method: Chunk Python files by function/class boundaries
        Kept for backward compatibility - now delegates to PythonChunker
        """
        self.logger.warning("Using legacy chunk_by_functions method - consider using process_file instead")
        
        chunker = ChunkerFactory.get_chunker(
            file_extension='py',
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
                            strategy=ChunkingStrategy.STRUCTURE_PRESERVING
        )
        
        return chunker.chunk_content(content, file_metadata)
    
    def chunk_by_size(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """
        Legacy method: Chunk content by token size with overlap
        Kept for backward compatibility - now delegates to base chunker
        """
        self.logger.warning("Using legacy chunk_by_size method - consider using process_file instead")
        
        chunker = ChunkerFactory.get_chunker(
            file_extension=file_metadata.file_type,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
                            strategy=ChunkingStrategy.SIZE_BASED
        )
        
        return chunker.chunk_content(content, file_metadata)
    
    def set_chunking_strategy(self, strategy: ChunkingStrategy):
        """Change the chunking strategy"""
        self.chunking_strategy = strategy
        self.logger.info(f"Chunking strategy changed to: {strategy.value}")
    
    def get_supported_file_types(self) -> List[str]:
        """Get list of supported file types"""
        return ChunkerFactory.get_supported_extensions()
    
    def is_file_type_supported(self, file_extension: str) -> bool:
        """Check if a file type has specific chunking support"""
        return ChunkerFactory.is_supported(file_extension)
    
    def process_file_with_strategy(self, file_path: Path, file_metadata: FileMetadata, 
                                  strategy: ChunkingStrategy) -> List[CodeChunk]:
        """Process a file with a specific chunking strategy (without changing default)"""
        content = FileProcessor(str(file_path.parent)).read_file_content(file_path)
        if not content:
            return []
        
        chunker = ChunkerFactory.get_chunker(
            file_extension=file_metadata.file_type,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
            strategy=strategy
        )
        
        return chunker.chunk_content(content, file_metadata)
    
    def analyze_chunking_impact(self, file_path: Path, file_metadata: FileMetadata) -> dict:
        """
        Analyze how different chunking strategies would affect a file
        Returns comparison of different strategies
        """
        content = FileProcessor(str(file_path.parent)).read_file_content(file_path)
        if not content:
            return {}
        
        strategies = [
            ChunkingStrategy.STRUCTURE_PRESERVING,
            ChunkingStrategy.SIZE_BASED,
            ChunkingStrategy.ADAPTIVE_STRUCTURE
        ]
        
        analysis = {}
        
        for strategy in strategies:
            try:
                chunker = ChunkerFactory.get_chunker(
                    file_extension=file_metadata.file_type,
                    max_tokens=self.max_tokens,
                    overlap_tokens=self.overlap_tokens,
                    strategy=strategy
                )
                
                chunks = chunker.chunk_content(content, file_metadata)
                
                token_counts = [chunk.tokens_count for chunk in chunks]
                
                analysis[strategy.value] = {
                    'total_chunks': len(chunks),
                    'avg_tokens': sum(token_counts) / len(token_counts) if token_counts else 0,
                    'max_tokens': max(token_counts) if token_counts else 0,
                    'min_tokens': min(token_counts) if token_counts else 0,
                    'oversized_chunks': len([t for t in token_counts if t > self.max_tokens]),
                    'chunk_types': list(set(chunk.chunk_type for chunk in chunks))
                }
                
            except Exception as e:
                analysis[strategy.value] = {'error': str(e)}
        
        return analysis 