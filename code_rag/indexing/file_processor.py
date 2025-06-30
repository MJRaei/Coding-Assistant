import os
import re
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import logging

from ..models import FileMetadata
from ..config import DEFAULT_EXCLUDED_DIRS, DEFAULT_INCLUDED_EXTENSIONS


class FileProcessor:
    """Handles file discovery and content extraction"""
    
    def __init__(self, root_path: str, excluded_dirs: List[str] = None, included_extensions: List[str] = None):
        self.root_path = Path(root_path)
        self.excluded_dirs = excluded_dirs or DEFAULT_EXCLUDED_DIRS
        self.included_extensions = included_extensions or DEFAULT_INCLUDED_EXTENSIONS
        self.logger = logging.getLogger(__name__)
    
    def discover_files(self) -> List[Path]:
        """Recursively discover all relevant files"""
        discovered_files = []
        
        for root, dirs, files in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.included_extensions:
                    discovered_files.append(file_path)
        
        self.logger.info(f"Discovered {len(discovered_files)} files")
        return discovered_files
    
    def extract_python_metadata(self, content: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract imports, functions, and classes from Python code"""
        imports = []
        functions = []
        classes = []
        
        try:
            import_pattern = r'^(?:from\s+[\w.]+\s+)?import\s+[\w.,\s*]+|^from\s+[\w.]+\s+import\s+[\w.,\s*]+'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            
            function_pattern = r'^def\s+(\w+)\s*\('
            functions = re.findall(function_pattern, content, re.MULTILINE)
            
            class_pattern = r'^class\s+(\w+)(?:\s*\([^)]*\))?\s*:'
            classes = re.findall(class_pattern, content, re.MULTILINE)
            
        except Exception as e:
            self.logger.warning(f"Error extracting Python metadata: {e}")
        
        return imports, functions, classes
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding handling"""
        encodings = ['utf-8', 'latin-1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                return None
        
        self.logger.error(f"Could not decode {file_path} with any encoding")
        return None
    
    def create_file_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Create metadata for a single file"""
        try:
            content = self.read_file_content(file_path)
            if content is None:
                return None
            
            stat = file_path.stat()
            relative_path = file_path.relative_to(self.root_path)
            
            imports, functions, classes = [], [], []
            if file_path.suffix == '.py':
                imports, functions, classes = self.extract_python_metadata(content)
            
            return FileMetadata(
                file_path=str(file_path),
                relative_path=str(relative_path),
                file_name=file_path.name,
                file_type=file_path.suffix[1:] if file_path.suffix else 'unknown',
                folder_path=str(file_path.parent.relative_to(self.root_path)),
                file_size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                content_hash=hashlib.md5(content.encode()).hexdigest(),
                imports=imports,
                functions=functions,
                classes=classes,
                line_count=len(content.splitlines())
            )
        
        except Exception as e:
            self.logger.error(f"Error creating metadata for {file_path}: {e}")
            return None 