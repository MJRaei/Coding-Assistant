import re
from pathlib import Path
from typing import List
import logging

from ..models import FileMetadata, CodeChunk
from .file_processor import FileProcessor
from ..config import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS


class ChunkProcessor:
    """Handles content chunking strategies"""
    
    def __init__(self, max_tokens: int = None, overlap_tokens: int = None):
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.overlap_tokens = overlap_tokens or DEFAULT_OVERLAP_TOKENS
        self.logger = logging.getLogger(__name__)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def chunk_by_functions(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Chunk Python files by function/class boundaries"""
        chunks = []
        lines = content.splitlines()
        
        boundaries = []
        for i, line in enumerate(lines):
            if re.match(r'^(def|class)\s+\w+', line.strip()):
                boundaries.append(i)
        
        boundaries = [0] + boundaries + [len(lines)]
        
        for i in range(len(boundaries) - 1):
            start_line = boundaries[i]
            end_line = boundaries[i + 1]
            chunk_content = '\n'.join(lines[start_line:end_line])
            
            if chunk_content.strip():
                chunk_type = 'function' if 'def ' in lines[start_line] else 'class' if 'class ' in lines[start_line] else 'section'
                
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=i,
                    chunk_type=chunk_type,
                    start_line=start_line + 1,
                    end_line=end_line,
                    tokens_count=self.estimate_tokens(chunk_content)
                ))
        
        return chunks
    
    def chunk_by_size(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Chunk content by token size with overlap"""
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
                
                overlap_lines = []
                overlap_tokens = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    if overlap_tokens + self.estimate_tokens(current_chunk[j]) <= self.overlap_tokens:
                        overlap_lines.insert(0, current_chunk[j])
                        overlap_tokens += self.estimate_tokens(current_chunk[j])
                    else:
                        break
                
                current_chunk = overlap_lines + [line]
                current_tokens = overlap_tokens + line_tokens
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
    
    def process_file(self, file_path: Path, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Process a single file into chunks"""
        content = FileProcessor(str(file_path.parent)).read_file_content(file_path)
        if not content:
            return []
        
        estimated_tokens = self.estimate_tokens(content)
        
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
        elif file_metadata.file_type == 'py':
            return self.chunk_by_functions(content, file_metadata)
        else:
            return self.chunk_by_size(content, file_metadata) 