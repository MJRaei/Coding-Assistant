"""
Python-specific chunker implementation.
"""

import re
import ast
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .base_chunker import BaseLanguageChunker, CodeBoundary, BoundaryType, ChunkingStrategy
from ...models import FileMetadata, CodeChunk


@dataclass
class PythonCodeElement:
    """Represents a Python code element with its context"""
    name: str
    type: str  # 'class', 'function', 'method', 'property'
    start_line: int
    end_line: int
    indent_level: int
    parent: Optional[str] = None
    decorators: List[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    is_property: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False


class PythonChunker(BaseLanguageChunker):
    """Python-specific chunker with advanced Python code understanding"""
    
    def __init__(self, max_tokens: int = 2000, overlap_tokens: int = 200, 
                 strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_FIRST):
        super().__init__(max_tokens, overlap_tokens, strategy)
        
        # Python-specific patterns
        self.class_pattern = re.compile(r'^(\s*)class\s+([A-Za-z_][A-Za-z0-9_]*)\s*[\(:]')
        self.function_pattern = re.compile(r'^(\s*)(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
        self.decorator_pattern = re.compile(r'^(\s*)@([A-Za-z_][A-Za-z0-9_.]*)')
        self.import_pattern = re.compile(r'^(\s*)(from\s+\S+\s+)?import\s+')
        
    def get_supported_extensions(self) -> List[str]:
        return ['.py', '.pyx', '.pyi']
    
    def estimate_tokens(self, text: str) -> int:
        """Python-specific token estimation"""
        # Python code tends to be more compact than average
        char_count = len(text)
        
        # More accurate estimation for Python:
        lines = text.splitlines()
        effective_chars = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue  # Skip empty lines and comments
            
            # Count only non-whitespace characters for token estimation
            effective_chars += len(stripped)
        
        # Python: roughly 1 token per 3.5 characters (more dense than average)
        return max(1, int(effective_chars / 3.5))
    
    def detect_boundaries(self, content: str, lines: List[str]) -> List[CodeBoundary]:
        """Detect Python-specific code boundaries using AST when possible"""
        boundaries = []
        
        # Try to use AST for more accurate parsing
        try:
            tree = ast.parse(content)
            boundaries = self._extract_boundaries_from_ast(tree, lines)
        except SyntaxError:
            # Fallback to regex-based detection for incomplete/malformed code
            self.logger.warning("Failed to parse with AST, falling back to regex")
            boundaries = self._extract_boundaries_with_regex(lines)
        
        return sorted(boundaries, key=lambda b: b.line_number)
    
    def _extract_boundaries_from_ast(self, tree: ast.AST, lines: List[str]) -> List[CodeBoundary]:
        """Extract boundaries using Python AST for accurate parsing"""
        boundaries = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                boundaries.append(CodeBoundary(
                    line_number=node.lineno - 1,  # Convert to 0-based
                    boundary_type=BoundaryType.CLASS,
                    content=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                    name=node.name,
                    indent_level=self._get_indent_level(lines[node.lineno - 1]) if node.lineno <= len(lines) else 0
                ))
                
                # Add methods within the class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        is_property = any(
                            isinstance(dec, ast.Name) and dec.id == 'property' 
                            for dec in item.decorator_list
                        )
                        
                        boundaries.append(CodeBoundary(
                            line_number=item.lineno - 1,
                            boundary_type=BoundaryType.PROPERTY if is_property else BoundaryType.METHOD,
                            content=lines[item.lineno - 1] if item.lineno <= len(lines) else "",
                            name=item.name,
                            parent=node.name,
                            indent_level=self._get_indent_level(lines[item.lineno - 1]) if item.lineno <= len(lines) else 0
                        ))
            
            elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                # Top-level function
                boundaries.append(CodeBoundary(
                    line_number=node.lineno - 1,
                    boundary_type=BoundaryType.FUNCTION,
                    content=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                    name=node.name,
                    indent_level=self._get_indent_level(lines[node.lineno - 1]) if node.lineno <= len(lines) else 0
                ))
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                boundaries.append(CodeBoundary(
                    line_number=node.lineno - 1,
                    boundary_type=BoundaryType.IMPORT,
                    content=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                    indent_level=self._get_indent_level(lines[node.lineno - 1]) if node.lineno <= len(lines) else 0
                ))
        
        return boundaries
    
    def _extract_boundaries_with_regex(self, lines: List[str]) -> List[CodeBoundary]:
        """Fallback regex-based boundary detection"""
        boundaries = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check for class definition
            class_match = self.class_pattern.match(line)
            if class_match:
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.CLASS,
                    content=line,
                    name=class_match.group(2),
                    indent_level=len(class_match.group(1))
                ))
                continue
            
            # Check for function/method definition
            func_match = self.function_pattern.match(line)
            if func_match:
                is_async = func_match.group(2) is not None
                function_name = func_match.group(3)
                indent = len(func_match.group(1))
                
                # Check if it's a method (indented) or function (top-level)
                boundary_type = BoundaryType.METHOD if indent > 0 else BoundaryType.FUNCTION
                
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=boundary_type,
                    content=line,
                    name=function_name,
                    indent_level=indent
                ))
                continue
            
            # Check for decorators
            decorator_match = self.decorator_pattern.match(line)
            if decorator_match:
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.DECORATOR,
                    content=line,
                    name=decorator_match.group(2),
                    indent_level=len(decorator_match.group(1))
                ))
                continue
            
            # Check for imports
            import_match = self.import_pattern.match(line)
            if import_match:
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.IMPORT,
                    content=line,
                    indent_level=len(import_match.group(1)) if import_match.group(1) else 0
                ))
        
        return boundaries
    
    def is_semantic_unit_complete(self, boundaries: List[CodeBoundary], 
                                 start_idx: int, end_idx: int) -> bool:
        """Check if the range contains complete Python semantic units"""
        if start_idx >= end_idx:
            return False
        
        # Check if we're cutting in the middle of a class
        for i in range(start_idx, end_idx):
            if boundaries[i].boundary_type == BoundaryType.CLASS:
                # Make sure all methods of this class are included
                class_name = boundaries[i].name
                for j in range(i + 1, len(boundaries)):
                    if j >= end_idx:
                        # Found a method that belongs to this class but is outside our range
                        if (boundaries[j].boundary_type == BoundaryType.METHOD and 
                            boundaries[j].parent == class_name):
                            return False
                    if (boundaries[j].boundary_type == BoundaryType.CLASS and 
                        boundaries[j].indent_level <= boundaries[i].indent_level):
                        break  # End of current class
        
        return True
    
    def chunk_by_hierarchy(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Create Python-specific hierarchical chunks"""
        lines = content.splitlines()
        boundaries = self.detect_boundaries(content, lines)
        
        if not boundaries:
            return self.chunk_by_size(content, file_metadata)
        
        chunks = []
        elements = self._group_related_elements(boundaries, lines)
        
        for element_group in elements:
            start_line = element_group[0]['start_line']
            end_line = element_group[-1]['end_line']
            
            chunk_content = '\n'.join(lines[start_line:end_line + 1])
            tokens = self.estimate_tokens(chunk_content)
            
            # If this group is too large, split it further
            if tokens > self.max_tokens:
                sub_chunks = self._split_large_group(element_group, lines, file_metadata)
                chunks.extend(sub_chunks)
            else:
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=len(chunks),
                    chunk_type=element_group[0]['type'],
                    start_line=start_line + 1,  # Convert back to 1-based
                    end_line=end_line + 1,
                    tokens_count=tokens
                ))
        
        return chunks
    
    def chunk_balanced(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Balance semantic preservation with size constraints"""
        # Start with semantic chunking but be more aggressive about splitting large units
        return self.chunk_by_semantic_units(content, file_metadata)
    
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)"""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
    
    def _get_indent_level(self, line: str) -> int:
        """Get the indentation level of a line"""
        return len(line) - len(line.lstrip())
    
    def _group_related_elements(self, boundaries: List[CodeBoundary], lines: List[str]) -> List[List[Dict]]:
        """Group related Python elements together (classes with their methods, etc.)"""
        groups = []
        current_group = []
        current_class = None
        
        for i, boundary in enumerate(boundaries):
            if boundary.boundary_type == BoundaryType.CLASS:
                # Start new group for class
                if current_group:
                    groups.append(current_group)
                
                current_class = boundary.name
                current_group = [{
                    'type': 'class',
                    'name': boundary.name,
                    'start_line': boundary.line_number,
                    'end_line': self._find_element_end_line(boundary, boundaries, i, lines)
                }]
            
            elif boundary.boundary_type == BoundaryType.METHOD and boundary.parent == current_class:
                # Add method to current class group
                current_group.append({
                    'type': 'method',
                    'name': boundary.name,
                    'start_line': boundary.line_number,
                    'end_line': self._find_element_end_line(boundary, boundaries, i, lines)
                })
            
            elif boundary.boundary_type == BoundaryType.FUNCTION:
                # Standalone function - its own group
                if current_group:
                    groups.append(current_group)
                
                current_group = [{
                    'type': 'function',
                    'name': boundary.name,
                    'start_line': boundary.line_number,
                    'end_line': self._find_element_end_line(boundary, boundaries, i, lines)
                }]
                current_class = None
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _find_element_end_line(self, boundary: CodeBoundary, boundaries: List[CodeBoundary], 
                              current_idx: int, lines: List[str]) -> int:
        """Find where a code element ends"""
        if current_idx + 1 < len(boundaries):
            next_boundary = boundaries[current_idx + 1]
            if next_boundary.indent_level <= boundary.indent_level:
                return next_boundary.line_number - 1
        
        # Find end by looking for next element at same or lower indentation
        start_line = boundary.line_number + 1
        target_indent = boundary.indent_level
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = self._get_indent_level(line)
                if current_indent <= target_indent:
                    return i - 1
        
        return len(lines) - 1  # End of file
    
    def _split_large_group(self, element_group: List[Dict], lines: List[str], 
                          file_metadata: FileMetadata) -> List[CodeChunk]:
        """Split a large element group into smaller chunks"""
        chunks = []
        
        for element in element_group:
            start_line = element['start_line']
            end_line = element['end_line']
            content = '\n'.join(lines[start_line:end_line + 1])
            tokens = self.estimate_tokens(content)
            
            chunks.append(CodeChunk(
                content=content,
                file_metadata=file_metadata,
                chunk_index=len(chunks),
                chunk_type=element['type'],
                start_line=start_line + 1,
                end_line=end_line + 1,
                tokens_count=tokens
            ))
        
        return chunks
    
    def chunk_function_aware(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Function-aware chunking with intelligent splitting within functions"""
        lines = content.splitlines()
        boundaries = self.detect_boundaries(content, lines)
        
        if not boundaries:
            return self.chunk_by_size(content, file_metadata)
        
        # Build hierarchy and analyze structure
        hierarchy = self.build_hierarchy(boundaries)
        elements = self._extract_code_elements(content, lines, boundaries)
        
        chunks = []
        related_chunks_map = {}
        
        for element in elements:
            element_chunks = self._chunk_element_function_aware(
                element, lines, file_metadata, len(chunks)
            )
            
            # Track relationships
            if len(element_chunks) > 1:
                chunk_indices = [chunk.chunk_index for chunk in element_chunks]
                for chunk in element_chunks:
                    chunk.related_chunks = [idx for idx in chunk_indices if idx != chunk.chunk_index]
            
            chunks.extend(element_chunks)
        
        return chunks
    
    def _extract_code_elements(self, content: str, lines: List[str], boundaries: List[CodeBoundary]) -> List[PythonCodeElement]:
        """Extract structured code elements from boundaries"""
        elements = []
        
        try:
            tree = ast.parse(content)
            elements = self._extract_elements_from_ast(tree, lines)
        except SyntaxError:
            elements = self._extract_elements_from_boundaries(boundaries, lines)
        
        return elements
    
    def _extract_elements_from_ast(self, tree: ast.AST, lines: List[str]) -> List[PythonCodeElement]:
        """Extract code elements using AST for precise analysis"""
        elements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_element = PythonCodeElement(
                    name=node.name,
                    type='class',
                    start_line=node.lineno,
                    end_line=self._get_node_end_line(node, lines),
                    indent_level=self._get_indent_level(lines[node.lineno - 1]) if node.lineno <= len(lines) else 0,
                    decorators=[self._get_decorator_name(dec) for dec in node.decorator_list]
                )
                
                # Estimate tokens for the entire class
                class_lines = lines[class_element.start_line - 1:class_element.end_line]
                class_content = '\n'.join(class_lines)
                class_tokens = self.estimate_tokens(class_content)
                
                # Only extract methods separately if class is too large
                if class_tokens > self.max_tokens:
                    # Class is too large, extract methods separately for intelligent splitting
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_element = PythonCodeElement(
                                name=item.name,
                                type='method',
                                start_line=item.lineno,
                                end_line=self._get_node_end_line(item, lines),
                                indent_level=self._get_indent_level(lines[item.lineno - 1]) if item.lineno <= len(lines) else 0,
                                parent=node.name,
                                decorators=[self._get_decorator_name(dec) for dec in item.decorator_list],
                                is_async=isinstance(item, ast.AsyncFunctionDef),
                                is_property=any(
                                    isinstance(dec, ast.Name) and dec.id == 'property' 
                                    for dec in item.decorator_list
                                )
                            )
                            elements.append(method_element)
                else:
                    # Class fits within limit, keep as single element
                    elements.append(class_element)
            
            elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                function_element = PythonCodeElement(
                    name=node.name,
                    type='function',
                    start_line=node.lineno,
                    end_line=self._get_node_end_line(node, lines),
                    indent_level=self._get_indent_level(lines[node.lineno - 1]) if node.lineno <= len(lines) else 0,
                    decorators=[self._get_decorator_name(dec) for dec in node.decorator_list],
                    is_async=isinstance(node, ast.AsyncFunctionDef)
                )
                elements.append(function_element)
        
        return sorted(elements, key=lambda e: e.start_line)
    
    def _extract_elements_from_boundaries(self, boundaries: List[CodeBoundary], lines: List[str]) -> List[PythonCodeElement]:
        """Fallback extraction from boundaries"""
        elements = []
        
        for boundary in boundaries:
            if boundary.boundary_type in [BoundaryType.CLASS, BoundaryType.FUNCTION, BoundaryType.METHOD]:
                element = PythonCodeElement(
                    name=boundary.name or "unknown",
                    type=boundary.boundary_type.value,
                    start_line=boundary.line_number + 1,
                    end_line=self._find_element_end_line(boundary, boundaries, 0, lines),
                    indent_level=boundary.indent_level,
                    parent=boundary.parent
                )
                elements.append(element)
        
        return sorted(elements, key=lambda e: e.start_line)
    
    def _chunk_element_function_aware(self, element: PythonCodeElement, lines: List[str], 
                                    file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Chunk a single code element with function-aware logic"""
        element_lines = lines[element.start_line - 1:element.end_line]
        element_content = '\n'.join(element_lines)
        element_tokens = self.estimate_tokens(element_content)
        
        # If element fits within limit, return as single chunk
        if element_tokens <= self.max_tokens:
            return [CodeChunk(
                content=element_content,
                file_metadata=file_metadata,
                chunk_index=base_index,
                chunk_type=element.type,
                start_line=element.start_line,
                end_line=element.end_line,
                tokens_count=element_tokens,
                parent_class=element.parent if element.type == 'method' else None,
                parent_function=element.parent if element.type != 'class' else None,
                semantic_id=f"{element.type}_{element.name}",
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'decorators': element.decorators or [],
                    'is_async': element.is_async,
                    'is_property': element.is_property
                }
            )]
        
        # Element is too large, need to split intelligently
        if element.type == 'class':
            return self._split_large_class(element, lines, file_metadata, base_index)
        elif element.type in ['function', 'method']:
            return self._split_large_function(element, lines, file_metadata, base_index)
        else:
            return self._split_generic_element(element, lines, file_metadata, base_index)
    
    def _split_large_class(self, element: PythonCodeElement, lines: List[str], 
                          file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Split large class by methods while preserving class context"""
        chunks = []
        class_lines = lines[element.start_line - 1:element.end_line]
        
        # Find class declaration and any class-level code
        class_header_end = self._find_class_header_end(class_lines)
        class_header = class_lines[:class_header_end]
        
        # Find method boundaries within the class
        method_boundaries = self._find_method_boundaries_in_class(class_lines, element.start_line)
        
        if not method_boundaries:
            # No methods found, split by size
            return self._split_by_size_with_metadata(element, lines, file_metadata, base_index)
        
        # Create chunks for each method, including class context
        current_chunk_index = base_index
        
        for i, method_boundary in enumerate(method_boundaries):
            method_start = method_boundary['start'] - element.start_line + 1
            method_end = method_boundary['end'] - element.start_line + 1
            
            # Include class header with first method
            if i == 0:
                chunk_lines = class_header + class_lines[method_start:method_end]
                chunk_start_line = element.start_line
            else:
                chunk_lines = class_lines[method_start:method_end]
                chunk_start_line = method_boundary['start']
            
            chunk_content = '\n'.join(chunk_lines)
            chunk_tokens = self.estimate_tokens(chunk_content)
            
            # If method is still too large, split it further
            if chunk_tokens > self.max_tokens:
                method_element = PythonCodeElement(
                    name=method_boundary['name'],
                    type='method',
                    start_line=method_boundary['start'],
                    end_line=method_boundary['end'],
                    indent_level=method_boundary['indent'],
                    parent=element.name
                )
                method_chunks = self._split_large_function(method_element, lines, file_metadata, current_chunk_index)
                chunks.extend(method_chunks)
                current_chunk_index += len(method_chunks)
            else:
                chunk = CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=current_chunk_index,
                    chunk_type='method',
                    start_line=chunk_start_line,
                    end_line=method_boundary['end'],
                    tokens_count=chunk_tokens,
                    parent_class=element.name,
                    semantic_id=f"method_{element.name}_{method_boundary['name']}",
                    chunk_metadata={
                        'element_name': method_boundary['name'],
                        'element_type': 'method',
                        'class_name': element.name,
                        'method_index': i
                    }
                )
                chunks.append(chunk)
                current_chunk_index += 1
        
        return chunks
    
    def _split_large_function(self, element: PythonCodeElement, lines: List[str], 
                            file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Split large function at safe breakpoints"""
        function_lines = lines[element.start_line - 1:element.end_line]
        breakpoints = self._find_safe_breakpoints(function_lines)
        
        if not breakpoints:
            return self._split_by_size_with_metadata(element, lines, file_metadata, base_index)
        
        chunks = []
        current_start = 0
        
        for i, breakpoint in enumerate(breakpoints):
            if i == len(breakpoints) - 1:
                chunk_lines = function_lines[current_start:]
            else:
                chunk_lines = function_lines[current_start:breakpoint]
            
            chunk_content = '\n'.join(chunk_lines)
            chunk_tokens = self.estimate_tokens(chunk_content)
            
            chunk = CodeChunk(
                content=chunk_content,
                file_metadata=file_metadata,
                chunk_index=base_index + i,
                chunk_type=element.type,
                start_line=element.start_line + current_start,
                end_line=element.start_line + (breakpoint - 1 if i < len(breakpoints) - 1 else len(function_lines) - 1),
                tokens_count=chunk_tokens,
                parent_class=element.parent if element.type == 'method' else None,
                parent_function=element.name,
                function_part_index=i,
                semantic_id=f"{element.type}_{element.name}_part_{i}",
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'part_index': i,
                    'total_parts': len(breakpoints),
                    'is_function_part': True
                }
            )
            chunks.append(chunk)
            current_start = breakpoint
        
        return chunks
    
    def _find_safe_breakpoints(self, function_lines: List[str]) -> List[int]:
        """Find safe places to split within a function"""
        breakpoints = []
        
        for i, line in enumerate(function_lines[1:], 1):  # Skip function definition
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                continue
            
            # Look for safe breakpoints
            if (stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ')) or
                stripped.startswith(('return ', 'yield ', 'raise ')) or
                ('=' in stripped and not stripped.startswith('def ')) or
                stripped.endswith(':') or
                (i > 0 and not function_lines[i-1].strip())):  # After blank line
                
                # Check if this creates a reasonable chunk size
                if i > 10:  # Minimum lines per chunk
                    breakpoints.append(i)
        
        # Ensure we don't create too many tiny chunks
        if len(breakpoints) > 5:
            breakpoints = breakpoints[::2]  # Take every other breakpoint
        
        return breakpoints
    
    def _find_class_header_end(self, class_lines: List[str]) -> int:
        """Find where class header ends (after class def and docstring)"""
        header_end = 1
        
        # Look for docstring
        for i, line in enumerate(class_lines[1:], 1):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Find end of docstring
                quote_type = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(quote_type) >= 2:
                    header_end = i + 1
                    break
                else:
                    for j, next_line in enumerate(class_lines[i+1:], i+1):
                        if quote_type in next_line:
                            header_end = j + 1
                            break
                break
            elif stripped and not stripped.startswith('#'):
                break
        
        return header_end
    
    def _find_method_boundaries_in_class(self, class_lines: List[str], class_start_line: int) -> List[Dict]:
        """Find method boundaries within a class"""
        methods = []
        
        for i, line in enumerate(class_lines):
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('async def '):
                match = self.function_pattern.match(line)
                if match:
                    method_name = match.group(3)
                    method_start = class_start_line + i
                    
                    # Find method end
                    method_end = self._find_method_end(class_lines, i)
                    
                    methods.append({
                        'name': method_name,
                        'start': method_start,
                        'end': class_start_line + method_end,
                        'indent': len(match.group(1))
                    })
        
        return methods
    
    def _find_method_end(self, class_lines: List[str], method_start_idx: int) -> int:
        """Find the end line of a method within class lines"""
        method_indent = len(class_lines[method_start_idx]) - len(class_lines[method_start_idx].lstrip())
        
        for i in range(method_start_idx + 1, len(class_lines)):
            line = class_lines[i]
            
            if line.strip() == '':
                continue
                
            line_indent = len(line) - len(line.lstrip())
            
            # If we find a line with same or less indentation, method ended
            if line_indent <= method_indent:
                return i - 1
        
        return len(class_lines) - 1
    
    def _split_by_size_with_metadata(self, element: PythonCodeElement, lines: List[str], 
                                   file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Fallback size-based splitting with enhanced metadata"""
        element_lines = lines[element.start_line - 1:element.end_line]
        chunks = []
        
        current_chunk = []
        current_tokens = 0
        
        for i, line in enumerate(element_lines):
            line_tokens = self.estimate_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunk = CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=base_index + len(chunks),
                    chunk_type=element.type,
                    start_line=element.start_line + i - len(current_chunk),
                    end_line=element.start_line + i - 1,
                    tokens_count=current_tokens,
                    parent_class=element.parent if element.type == 'method' else None,
                    parent_function=element.name if element.type != 'class' else None,
                    function_part_index=len(chunks),
                    semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                    chunk_metadata={
                        'element_name': element.name,
                        'element_type': element.type,
                        'part_index': len(chunks),
                        'is_size_split': True
                    }
                )
                chunks.append(chunk)
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunk = CodeChunk(
                content=chunk_content,
                file_metadata=file_metadata,
                chunk_index=base_index + len(chunks),
                chunk_type=element.type,
                start_line=element.start_line + len(element_lines) - len(current_chunk),
                end_line=element.end_line,
                tokens_count=current_tokens,
                parent_class=element.parent if element.type == 'method' else None,
                parent_function=element.name if element.type != 'class' else None,
                function_part_index=len(chunks),
                semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'part_index': len(chunks),
                    'is_size_split': True
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_generic_element(self, element: PythonCodeElement, lines: List[str], 
                             file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Generic element splitting"""
        return self._split_by_size_with_metadata(element, lines, file_metadata, base_index)
    
    def _get_node_end_line(self, node: ast.AST, lines: List[str]) -> int:
        """Get the end line of an AST node"""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        
        # Fallback: estimate based on node type and content
        if hasattr(node, 'lineno'):
            start_line = node.lineno
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Look for next definition at same or higher level
                for i in range(start_line, len(lines)):
                    line = lines[i]
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                        return i
                return len(lines)
            return start_line
        
        return len(lines)
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node"""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return "unknown" 