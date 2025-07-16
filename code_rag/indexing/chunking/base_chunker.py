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
        self.content = content
        self.name = name
        self.parent = parent
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
    SEMANTIC_FIRST = "semantic_first"
    SIZE_FIRST = "size_first"
    FUNCTION_AWARE = "function_aware"


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
        return len(text) // 4
    
    def is_semantic_unit_complete(self, boundaries: List[CodeBoundary], 
                                 start_idx: int, end_idx: int) -> bool:
        """Check if a range of boundaries forms a complete semantic unit (can be overridden)"""
        return True
    
    def build_hierarchy(self, boundaries: List[CodeBoundary]) -> List[CodeBoundary]:
        """Build a hierarchy of code boundaries (e.g., methods under classes)"""
        root_boundaries = []
        parent_stack = []
        
        for boundary in boundaries:
            while parent_stack and parent_stack[-1].indent_level >= boundary.indent_level:
                parent_stack.pop()
            
            if parent_stack:
                parent_stack[-1].add_child(boundary)
            else:
                root_boundaries.append(boundary)
            
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
            
            boundary_at_line = self._find_boundary_at_line(boundaries, i)
            
            should_split = self._should_split_here(
                current_tokens, line_tokens, boundary_at_line, 
                current_chunk_lines, i, lines
            )
            
            if should_split and current_chunk_lines:
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
                
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = i + 1
                chunk_type = boundary_at_line.boundary_type.value if boundary_at_line else 'section'
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
            
            i += 1
        
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
                
                overlap_lines = self._calculate_overlap(current_chunk)
                current_chunk = overlap_lines + [line]
                current_tokens = sum(self.estimate_tokens(l) for l in current_chunk)
                start_line = i + 1 - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
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
        
        if self.strategy == ChunkingStrategy.SIZE_FIRST:
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
        
        elif self.strategy == ChunkingStrategy.FUNCTION_AWARE:
            return self.chunk_function_aware(content, file_metadata)
        
        else:
            return self.chunk_by_semantic_units(content, file_metadata)
    
    def chunk_function_aware(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Function-aware chunking with intelligent splitting (default implementation)"""
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
        
        if not boundary:
            return current_tokens + line_tokens > self.max_tokens * 1.2
        
        if current_tokens + line_tokens > self.max_tokens:
            return True
        
        if boundary.boundary_type in [BoundaryType.CLASS, BoundaryType.FUNCTION]:
            return len(current_lines) > 0
        
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