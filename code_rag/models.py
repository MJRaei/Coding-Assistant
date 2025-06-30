from dataclasses import dataclass
from typing import List


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