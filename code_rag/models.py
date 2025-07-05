from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class FileMetadata:
    """Metadata for each processed file"""
    file_path: str
    relative_path: str
    file_name: str
    file_type: str
    folder_path: str
    file_size: int
    last_modified: str
    content_hash: str
    imports: List[str]
    functions: List[str]
    classes: List[str]
    line_count: int


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    file_metadata: FileMetadata
    chunk_index: int
    chunk_type: str  # 'full_file', 'function', 'class', 'section'
    start_line: int
    end_line: int
    tokens_count: int
    parent_class: Optional[str] = None
    parent_function: Optional[str] = None
    function_part_index: Optional[int] = None
    related_chunks: List[int] = field(default_factory=list)
    semantic_id: Optional[str] = None
    chunk_metadata: Dict[str, Any] = field(default_factory=dict) 