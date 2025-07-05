"""
Abstract base class for language-specific chunkers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import logging

from ...models import FileMetadata, CodeChunk


class BoundaryType(Enum):
    """Types of code boundaries that can be detected"""
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    DECORATOR = "decorator"
    IMPORT = "import"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    PROPERTY = "property"
    NESTED_FUNCTION = "nested_function"
    CONTROL_STRUCTURE = "control_structure"
    VARIABLE_ASSIGNMENT = "variable_assignment"
    BLANK_LINE = "blank_line"
    FUNCTION_CALL = "function_call"


class CodeBoundary:
    """Represents a detected code boundary"""
    def __init__(self, line_number: int, boundary_type: BoundaryType, 
                 content: str, name: str = None, parent: str = None, 
                 indent_level: int = 0):
        self.line_number = line_number
        self.boundary_type = boundary_type
        self.content = content  # The actual line content
        self.name = name  # Function/class name
        self.parent = parent  # Parent class for methods
        self.indent_level = indent_level
        self.child_boundaries: List['CodeBoundary'] = []
        
    def add_child(self, child: 'CodeBoundary'):
        """Add a child boundary (e.g., method to class)"""
        child.parent = self.name
        self.child_boundaries.append(child)
        
    def __repr__(self):
        return f"CodeBoundary({self.line_number}, {self.boundary_type}, {self.name})"


class ChunkingStrategy(Enum):
    """Different chunking strategies"""
    SEMANTIC_FIRST = "semantic_first"  # Preserve code structure, split only when necessary
    SIZE_FIRST = "size_first"          # Current behavior - split by size first
    HIERARCHICAL = "hierarchical"      # Create hierarchy-aware chunks
    BALANCED = "balanced"              # Balance between semantic and size considerations
    FUNCTION_AWARE = "function_aware"  # Split intelligently within functions while preserving relationships


class BaseLanguageChunker(ABC):
    """Abstract base class for language-specific chunking"""
    
    def __init__(self, max_tokens: int = 2000, overlap_tokens: int = 200, 
                 strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_FIRST):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.strategy = strategy
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def detect_boundaries(self, content: str, lines: List[str]) -> List[CodeBoundary]:
        """Detect code boundaries specific to this language"""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of file extensions this chunker supports"""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for this language (can be overridden)"""
        # Default estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def is_semantic_unit_complete(self, boundaries: List[CodeBoundary], 
                                 start_idx: int, end_idx: int) -> bool:
        """Check if a range of boundaries forms a complete semantic unit (can be overridden)"""
        return True  # Default: assume any range is complete
    
    def build_hierarchy(self, boundaries: List[CodeBoundary]) -> List[CodeBoundary]:
        """Build a hierarchy of code boundaries (e.g., methods under classes)"""
        root_boundaries = []
        parent_stack = []
        
        for boundary in boundaries:
            # Pop parents that are at same or higher indent level
            while parent_stack and parent_stack[-1].indent_level >= boundary.indent_level:
                parent_stack.pop()
            
            if parent_stack:
                # This boundary is a child of the current parent
                parent_stack[-1].add_child(boundary)
            else:
                # This is a root-level boundary
                root_boundaries.append(boundary)
            
            # If this boundary can have children, add it to the stack
            if boundary.boundary_type in [BoundaryType.CLASS, BoundaryType.FUNCTION]:
                parent_stack.append(boundary)
                
        return root_boundaries
    
    def chunk_by_semantic_units(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Chunk content preserving semantic units"""
        lines = content.splitlines()
        boundaries = self.detect_boundaries(content, lines)
        
        if not boundaries:
            self.logger.info("No boundaries detected, falling back to size-based chunking")
            return self.chunk_by_size(content, file_metadata)
        
        chunks = []
        current_chunk_lines = []
        current_tokens = 0
        start_line = 1
        chunk_type = 'section'
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_tokens = self.estimate_tokens(line)
            
            # Check if we're at a semantic boundary
            boundary_at_line = self._find_boundary_at_line(boundaries, i)
            
            # Decide whether to start a new chunk
            should_split = self._should_split_here(
                current_tokens, line_tokens, boundary_at_line, 
                current_chunk_lines, i, lines
            )
            
            if should_split and current_chunk_lines:
                # Create chunk from current content
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=len(chunks),
                    chunk_type=chunk_type,
                    start_line=start_line,
                    end_line=start_line + len(current_chunk_lines) - 1,
                    tokens_count=current_tokens
                ))
                
                # Start new chunk
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = i + 1
                chunk_type = boundary_at_line.boundary_type.value if boundary_at_line else 'section'
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
            
            i += 1
        
        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(CodeChunk(
                content=chunk_content,
                file_metadata=file_metadata,
                chunk_index=len(chunks),
                chunk_type=chunk_type,
                start_line=start_line,
                end_line=len(lines),
                tokens_count=current_tokens
            ))
        
        self.logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def chunk_by_size(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Fallback to simple size-based chunking"""
        chunks = []
        lines = content.splitlines()
        current_chunk = []
        current_tokens = 0
        start_line = 1
        
        for i, line in enumerate(lines):
            line_tokens = self.estimate_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=len(chunks),
                    chunk_type='section',
                    start_line=start_line,
                    end_line=i,
                    tokens_count=current_tokens
                ))
                
                # Handle overlap
                overlap_lines = self._calculate_overlap(current_chunk)
                current_chunk = overlap_lines + [line]
                current_tokens = sum(self.estimate_tokens(l) for l in current_chunk)
                start_line = i + 1 - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(CodeChunk(
                content=chunk_content,
                file_metadata=file_metadata,
                chunk_index=len(chunks),
                chunk_type='section',
                start_line=start_line,
                end_line=len(lines),
                tokens_count=current_tokens
            ))
        
        return chunks
    
    def chunk_content(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Main entry point for chunking content"""
        estimated_tokens = self.estimate_tokens(content)
        
        # Choose chunking strategy based on strategy and file size
        if self.strategy == ChunkingStrategy.SIZE_FIRST:
            # For size-first, still respect the original behavior for small files
            if estimated_tokens <= self.max_tokens:
                return [CodeChunk(
                    content=content,
                    file_metadata=file_metadata,
                    chunk_index=0,
                    chunk_type='full_file',
                    start_line=1,
                    end_line=file_metadata.line_count,
                    tokens_count=estimated_tokens
                )]
            return self.chunk_by_size(content, file_metadata)
        
        elif self.strategy == ChunkingStrategy.SEMANTIC_FIRST:
            return self.chunk_by_semantic_units(content, file_metadata)
        
        elif self.strategy == ChunkingStrategy.HIERARCHICAL:
            return self.chunk_by_hierarchy(content, file_metadata)
        
        elif self.strategy == ChunkingStrategy.FUNCTION_AWARE:
            return self.chunk_function_aware(content, file_metadata)
        
        else:  # BALANCED
            return self.chunk_balanced(content, file_metadata)
    
    def chunk_by_hierarchy(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Create hierarchy-aware chunks (default implementation)"""
        # Default implementation - subclasses should override
        return self.chunk_by_semantic_units(content, file_metadata)
    
    def chunk_balanced(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Balance semantic and size considerations (default implementation)"""
        # Default implementation - subclasses should override
        return self.chunk_by_semantic_units(content, file_metadata)
    
    def chunk_function_aware(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Function-aware chunking with intelligent splitting (default implementation)"""
        # Default implementation - subclasses should override with language-specific logic
        return self.chunk_by_semantic_units(content, file_metadata)
    
    def _find_boundary_at_line(self, boundaries: List[CodeBoundary], line_num: int) -> Optional[CodeBoundary]:
        """Find boundary at specific line number"""
        for boundary in boundaries:
            if boundary.line_number == line_num:
                return boundary
        return None
    
    def _should_split_here(self, current_tokens: int, line_tokens: int, 
                          boundary: Optional[CodeBoundary], current_lines: List[str], 
                          line_idx: int, all_lines: List[str]) -> bool:
        """Decide whether to split at this point"""
        if self.strategy == ChunkingStrategy.SIZE_FIRST:
            return current_tokens + line_tokens > self.max_tokens
        
        # For semantic-first strategy
        if not boundary:
            # Only split on size if no semantic boundary
            return current_tokens + line_tokens > self.max_tokens * 1.2  # Allow some overflow for semantics
        
        # We're at a semantic boundary - should we split?
        if current_tokens + line_tokens > self.max_tokens:
            return True  # Must split due to size
        
        # Split at major boundaries even if under size limit
        if boundary.boundary_type in [BoundaryType.CLASS, BoundaryType.FUNCTION]:
            return len(current_lines) > 0  # Don't split if this is the first line
        
        return False
    
    def _calculate_overlap(self, current_chunk: List[str]) -> List[str]:
        """Calculate overlap lines for smooth transitions"""
        if not current_chunk:
            return []
        
        overlap_lines = []
        overlap_tokens = 0
        
        for line in reversed(current_chunk):
            line_tokens = self.estimate_tokens(line)
            if overlap_tokens + line_tokens <= self.overlap_tokens:
                overlap_lines.insert(0, line)
                overlap_tokens += line_tokens
            else:
                break
        
        return overlap_lines 