"""
JavaScript-specific chunker implementation.

JavaScript is a dynamic, multi-paradigm language with various function definitions,
class patterns, module systems, and asynchronous patterns. This chunker understands
modern JavaScript structure including ES6+ features, CommonJS, and various frameworks.
"""

import re
from typing import List, Optional, Dict, Set
from dataclasses import dataclass

from .base_chunker import BaseLanguageChunker, CodeBoundary, BoundaryType, ChunkingStrategy
from ...models import FileMetadata, CodeChunk


@dataclass
class JSCodeElement:
    """Represents a JavaScript code element with its context"""
    name: str
    type: str
    start_line: int
    end_line: int
    indent_level: int
    parent: Optional[str] = None
    function_type: Optional[str] = None
    is_async: bool = False
    is_export: bool = False
    is_default_export: bool = False
    decorators: List[str] = None
    parameters: List[str] = None
    is_iife: bool = False
    is_constructor: bool = False


class JSChunker(BaseLanguageChunker):
    """JavaScript-specific chunker with understanding of JS patterns"""
    
    def __init__(self, max_tokens: int = 2000, overlap_tokens: int = 200, 
                 strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE_PRESERVING):
        super().__init__(max_tokens, overlap_tokens, strategy)
        
        self.function_patterns = {
            'function_decl': re.compile(r'^\s*(export\s+)?(default\s+)?(async\s+)?function\s*\*?\s*(\w+)\s*\(([^)]*)\)\s*{?'),
            'arrow_func': re.compile(r'^\s*(export\s+)?(const|let|var)\s+(\w+)\s*=\s*(async\s+)?\(([^)]*)\)\s*=>\s*{?'),
            'method_def': re.compile(r'^\s*(async\s+)?(\w+)\s*\(([^)]*)\)\s*{'),
            'object_method': re.compile(r'^\s*(\w+):\s*(async\s+)?function\s*\*?\s*\(([^)]*)\)\s*{'),
            'iife': re.compile(r'^\s*\(function\s*\(([^)]*)\)\s*{'),
            'constructor': re.compile(r'^\s*function\s+([A-Z]\w*)\s*\(([^)]*)\)\s*{'),
            'generator': re.compile(r'^\s*(export\s+)?(async\s+)?function\s*\*\s*(\w+)\s*\(([^)]*)\)\s*{')
        }
        
        self.class_patterns = {
            'class_decl': re.compile(r'^\s*(export\s+)?(default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{'),
            'constructor_method': re.compile(r'^\s*constructor\s*\(([^)]*)\)\s*{'),
            'class_method': re.compile(r'^\s*(static\s+)?(async\s+)?(\w+)\s*\(([^)]*)\)\s*{'),
            'getter': re.compile(r'^\s*get\s+(\w+)\s*\(\s*\)\s*{'),
            'setter': re.compile(r'^\s*set\s+(\w+)\s*\(([^)]*)\)\s*{')
        }
        
        self.import_patterns = {
            'es6_import': re.compile(r'^\s*import\s+(.+?)\s+from\s+[\'"]([^\'"]+)[\'"]'),
            'es6_import_default': re.compile(r'^\s*import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]'),
            'es6_import_named': re.compile(r'^\s*import\s+\{([^}]+)\}\s+from\s+[\'"]([^\'"]+)[\'"]'),
            'commonjs_require': re.compile(r'^\s*(const|let|var)\s+(.+?)\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'),
            'es6_export': re.compile(r'^\s*export\s+(\{[^}]+\}|default\s+|const\s+|let\s+|var\s+|function\s+|class\s+|async\s+function\s+)'),
            'commonjs_exports': re.compile(r'^\s*(module\.exports|exports)(\.\w+)?\s*=')
        }
        
        self.variable_patterns = {
            'var_decl': re.compile(r'^\s*(const|let|var)\s+(\w+)\s*=\s*(.+)'),
            'object_literal': re.compile(r'^\s*(const|let|var)\s+(\w+)\s*=\s*\{'),
            'array_literal': re.compile(r'^\s*(const|let|var)\s+(\w+)\s*=\s*\[')
        }
        
        self.comment_patterns = {
            'single_line': re.compile(r'^\s*//(.*)'),
            'multi_line_start': re.compile(r'^\s*/\*'),
            'multi_line_end': re.compile(r'.*\*/\s*$'),
            'jsdoc': re.compile(r'^\s*/\*\*')
        }
        
        self.js_keywords = {
            'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
            'default', 'delete', 'do', 'else', 'export', 'extends', 'finally',
            'for', 'function', 'if', 'import', 'in', 'instanceof', 'let', 'new',
            'return', 'super', 'switch', 'this', 'throw', 'try', 'typeof', 'var',
            'void', 'while', 'with', 'yield', 'async', 'await', 'of'
        }
        
        self.framework_patterns = {
            'react_component': re.compile(r'^\s*(export\s+)?(default\s+)?(?:function|const)\s+(\w+)\s*=?\s*\(.*\)\s*=>\s*{?'),
            'react_hook': re.compile(r'^\s*(const|let)\s+\[(\w+),\s*set\w+\]\s*=\s*useState'),
            'vue_component': re.compile(r'^\s*(?:export\s+)?(?:default\s+)?\{'),
            'jquery_ready': re.compile(r'^\s*\$\(document\)\.ready\s*\('),
            'angular_controller': re.compile(r'^\s*\.controller\s*\(\s*[\'"]([^\'"]+)[\'"]')
        }

    def get_supported_extensions(self) -> List[str]:
        """Return supported JavaScript file extensions"""
        return ['.js', '.jsx', '.mjs', '.ts', '.tsx']

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens for JavaScript code"""
        base_tokens = len(text) // 3.5
        
        brace_depth = 0
        max_depth = 0
        for char in text:
            if char == '{':
                brace_depth += 1
                max_depth = max(max_depth, brace_depth)
            elif char == '}':
                brace_depth -= 1
        
        complexity_penalty = max_depth * 10
        
        return int(base_tokens + complexity_penalty)

    def detect_boundaries(self, content: str, lines: List[str]) -> List[CodeBoundary]:
        """Detect JavaScript-specific code boundaries"""
        boundaries = []
        brace_stack = []
        current_context = None
        in_multiline_comment = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if in_multiline_comment:
                if self.comment_patterns['multi_line_end'].match(stripped):
                    in_multiline_comment = False
                continue
            
            if not stripped:
                continue
            
            if self.comment_patterns['multi_line_start'].match(stripped):
                in_multiline_comment = True
                continue
            
            if self.comment_patterns['single_line'].match(stripped):
                continue
            
            for pattern_name, pattern in self.import_patterns.items():
                match = pattern.match(line)
                if match:
                    import_name = self._extract_import_name(match, pattern_name)
                    boundaries.append(CodeBoundary(
                        line_number=i,
                        boundary_type=BoundaryType.IMPORT,
                        content=line,
                        name=import_name,
                        indent_level=self._get_indent_level(line)
                    ))
                    break
            
            class_match = self.class_patterns['class_decl'].match(line)
            if class_match:
                class_name = class_match.group(3)
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.CLASS,
                    content=line,
                    name=class_name,
                    parent=current_context,
                    indent_level=self._get_indent_level(line)
                ))
                
                if '{' in line:
                    brace_stack.append(current_context)
                    current_context = class_name
                continue
            
            for pattern_name, pattern in self.function_patterns.items():
                match = pattern.match(line)
                if match:
                    func_info = self._extract_function_info(match, pattern_name)
                    if func_info:
                        boundaries.append(CodeBoundary(
                            line_number=i,
                            boundary_type=BoundaryType.FUNCTION,
                            content=line,
                            name=func_info['name'],
                            parent=current_context,
                            indent_level=self._get_indent_level(line)
                        ))
                        break
            
            for pattern_name, pattern in self.class_patterns.items():
                if pattern_name == 'class_decl':
                    continue
                    
                match = pattern.match(line)
                if match and current_context:
                    method_name = self._extract_method_name(match, pattern_name)
                    if method_name:
                        boundaries.append(CodeBoundary(
                            line_number=i,
                            boundary_type=BoundaryType.METHOD,
                            content=line,
                            name=method_name,
                            parent=current_context,
                            indent_level=self._get_indent_level(line)
                        ))
                        break
            
            for pattern_name, pattern in self.variable_patterns.items():
                match = pattern.match(line)
                if match:
                    var_name = match.group(2)
                    boundaries.append(CodeBoundary(
                        line_number=i,
                        boundary_type=BoundaryType.VARIABLE_ASSIGNMENT,
                        content=line,
                        name=var_name,
                        parent=current_context,
                        indent_level=self._get_indent_level(line)
                    ))
                    break
            
            if '}' in line and brace_stack:
                close_count = line.count('}')
                for _ in range(min(close_count, len(brace_stack))):
                    current_context = brace_stack.pop() if brace_stack else None
        
        return sorted(boundaries, key=lambda b: b.line_number)

    def _extract_import_name(self, match, pattern_name: str) -> str:
        """Extract meaningful name from import statement"""
        if pattern_name == 'es6_import_default':
            return match.group(1)
        elif pattern_name == 'es6_import_named':
            imports = match.group(1).split(',')
            return imports[0].strip()
        elif pattern_name == 'commonjs_require':
            return match.group(2)
        elif pattern_name == 'es6_import':
            return match.group(1).split(',')[0].strip()
        else:
            return match.group(2) if len(match.groups()) > 1 else "import"

    def _extract_function_info(self, match, pattern_name: str) -> Dict:
        """Extract function information from regex match"""
        if pattern_name == 'function_decl':
            return {
                'name': match.group(4),
                'is_async': bool(match.group(3)),
                'is_export': bool(match.group(1)),
                'is_default': bool(match.group(2)),
                'params': match.group(5).split(',') if match.group(5) else []
            }
        elif pattern_name == 'arrow_func':
            return {
                'name': match.group(3),
                'is_async': bool(match.group(4)),
                'is_export': bool(match.group(1)),
                'type': 'arrow',
                'params': match.group(5).split(',') if match.group(5) else []
            }
        elif pattern_name == 'method_def':
            return {
                'name': match.group(2),
                'is_async': bool(match.group(1)),
                'type': 'method',
                'params': match.group(3).split(',') if match.group(3) else []
            }
        elif pattern_name == 'constructor':
            return {
                'name': match.group(1),
                'type': 'constructor',
                'params': match.group(2).split(',') if match.group(2) else []
            }
        elif pattern_name == 'iife':
            return {
                'name': 'IIFE',
                'type': 'iife',
                'params': match.group(1).split(',') if match.group(1) else []
            }
        return None

    def _extract_method_name(self, match, pattern_name: str) -> str:
        """Extract method name from class method patterns"""
        if pattern_name == 'constructor_method':
            return 'constructor'
        elif pattern_name == 'class_method':
            return match.group(3)
        elif pattern_name == 'getter':
            return f"get {match.group(1)}"
        elif pattern_name == 'setter':
            return f"set {match.group(1)}"
        return None

    def _get_indent_level(self, line: str) -> int:
        """Calculate indentation level"""
        return len(line) - len(line.lstrip())

    def chunk_adaptive_structure(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """JavaScript-aware chunking with intelligent function and class splitting"""
        lines = content.splitlines()
        boundaries = self.detect_boundaries(content, lines)
        
        if not boundaries:
            return self.chunk_by_size(content, file_metadata)
        
        total_tokens = self.estimate_tokens(content)
        if total_tokens <= self.max_tokens:
            return self._create_single_file_chunk(content, file_metadata, boundaries)
        
        elements = self._extract_js_elements(content, lines, boundaries)
        chunks = []
        
        for element in elements:
            element_chunks = self._chunk_js_element(
                element, lines, file_metadata, len(chunks)
            )
            
            if len(element_chunks) > 1:
                chunk_indices = [chunk.chunk_index for chunk in element_chunks]
                for chunk in element_chunks:
                    chunk.related_chunks = [idx for idx in chunk_indices if idx != chunk.chunk_index]
            
            chunks.extend(element_chunks)
        
        return chunks

    def _create_single_file_chunk(self, content: str, file_metadata: FileMetadata, boundaries: List[CodeBoundary]) -> List[CodeChunk]:
        """Create a single chunk for small JS files with complete metadata"""
        imports = set()
        exports = set()
        functions = []
        classes = []
        variables = []
        async_functions = []
        
        for boundary in boundaries:
            if boundary.boundary_type == BoundaryType.IMPORT:
                imports.add(boundary.name)
            elif boundary.boundary_type == BoundaryType.CLASS:
                classes.append(boundary.name)
            elif boundary.boundary_type == BoundaryType.FUNCTION:
                functions.append(boundary.name)
                if 'async' in boundary.content:
                    async_functions.append(boundary.name)
            elif boundary.boundary_type == BoundaryType.METHOD:
                functions.append(f"{boundary.parent}.{boundary.name}")
                if 'async' in boundary.content:
                    async_functions.append(f"{boundary.parent}.{boundary.name}")
            elif boundary.boundary_type == BoundaryType.VARIABLE_ASSIGNMENT:
                variables.append(boundary.name)
        
        imports_list = sorted(list(imports))
        exports_list = sorted(list(exports))
        
        module_type = "ES6" if any('import' in line or 'export' in line for line in content.splitlines()) else "CommonJS"
        
        framework_type = self._detect_framework(content)
        
        return [CodeChunk(
            content=content,
            file_metadata=file_metadata,
            chunk_index=0,
            chunk_type='module',
            start_line=1,
            end_line=len(content.splitlines()),
            tokens_count=self.estimate_tokens(content),
            semantic_id=f"module_{file_metadata.file_name}",
            chunk_metadata={
                'js_imports': imports_list,
                'js_exports': exports_list,
                'js_functions': functions,
                'js_classes': classes,
                'js_variables': variables,
                'js_async_functions': async_functions,
                'framework_type': framework_type,
                'module_type': module_type,
                'is_complete_file': True
            }
        )]

    def _detect_framework(self, content: str) -> str:
        """Detect JavaScript framework being used"""
        content_lower = content.lower()
        
        if 'react' in content_lower or 'jsx' in content_lower or 'usestate' in content_lower:
            return 'React'
        
        if 'vue' in content_lower or 'v-' in content_lower:
            return 'Vue'
        
        if 'angular' in content_lower or '@component' in content_lower:
            return 'Angular'
        
        if '$(' in content or 'jquery' in content_lower:
            return 'jQuery'
        
        if 'require(' in content or 'module.exports' in content:
            return 'Node.js'
        
        return 'Vanilla'

    def _extract_js_elements(self, content: str, lines: List[str], boundaries: List[CodeBoundary]) -> List[JSCodeElement]:
        """Extract structured JavaScript elements from boundaries"""
        elements = []
        
        classes = []
        functions = []
        
        for boundary in boundaries:
            if boundary.boundary_type == BoundaryType.CLASS:
                end_line = self._find_element_end_line(boundary, lines)
                element = JSCodeElement(
                    name=boundary.name,
                    type='class',
                    start_line=boundary.line_number + 1,
                    end_line=end_line,
                    indent_level=boundary.indent_level,
                    parent=boundary.parent
                )
                classes.append(element)
                
            elif boundary.boundary_type == BoundaryType.FUNCTION:
                end_line = self._find_element_end_line(boundary, lines)
                element = JSCodeElement(
                    name=boundary.name,
                    type='function',
                    start_line=boundary.line_number + 1,
                    end_line=end_line,
                    indent_level=boundary.indent_level,
                    parent=boundary.parent,
                    is_async='async' in boundary.content
                )
                functions.append(element)
        
        elements.extend(classes)
        
        substantial_functions = [f for f in functions if f.indent_level == 0]
        elements.extend(substantial_functions)
        
        return sorted(elements, key=lambda e: e.start_line)

    def _find_element_end_line(self, boundary: CodeBoundary, lines: List[str]) -> int:
        """Find end line by matching braces for JavaScript elements"""
        start_line = boundary.line_number
        brace_count = 0
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and '{' in lines[start_line]:
                return i + 1
        
        return len(lines)

    def _chunk_js_element(self, element: JSCodeElement, lines: List[str], 
                          file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Chunk a single JavaScript element with intelligent splitting"""
        element_lines = lines[element.start_line - 1:element.end_line]
        element_content = '\n'.join(element_lines)
        element_tokens = self.estimate_tokens(element_content)
        
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
                    'function_type': element.function_type,
                    'is_async': element.is_async,
                    'is_export': element.is_export,
                    'is_constructor': element.is_constructor
                }
            )]
        
        return self._split_large_js_element(element, lines, file_metadata, base_index)

    def _split_large_js_element(self, element: JSCodeElement, lines: List[str], 
                               file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Split large JavaScript elements while respecting boundaries"""
        chunks = []
        element_lines = lines[element.start_line - 1:element.end_line]
        
        total_tokens = self.estimate_tokens('\n'.join(element_lines))
        num_chunks = (total_tokens + self.max_tokens - 1) // self.max_tokens
        target_chunk_size = total_tokens // num_chunks
        
        current_lines = []
        current_tokens = 0
        start_line_idx = 0
        brace_level = 0
        
        for i, line in enumerate(element_lines):
            line_tokens = self.estimate_tokens(line)
            
            brace_level += line.count('{') - line.count('}')
            
            should_split = (current_tokens + line_tokens > target_chunk_size and 
                          current_lines and 
                          brace_level == 0)
            
            if should_split:
                chunk_content = '\n'.join(current_lines)
                chunk_start_line = element.start_line + start_line_idx
                chunk_end_line = element.start_line + start_line_idx + len(current_lines) - 1
                
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=base_index + len(chunks),
                    chunk_type=element.type,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    tokens_count=current_tokens,
                    parent_class=element.parent if element.type == 'method' else None,
                    parent_function=element.parent if element.type != 'class' else None,
                    semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                    function_part_index=len(chunks),
                    chunk_metadata={
                        'element_name': element.name,
                        'element_type': element.type,
                        'part_index': len(chunks),
                        'function_type': element.function_type,
                        'is_async': element.is_async,
                        'is_export': element.is_export
                    }
                ))
                
                current_lines = [line]
                current_tokens = line_tokens
                start_line_idx = i
            else:
                current_lines.append(line)
                current_tokens += line_tokens
        
        if current_lines:
            chunk_content = '\n'.join(current_lines)
            chunk_start_line = element.start_line + start_line_idx
            chunk_end_line = element.end_line
            
            chunks.append(CodeChunk(
                content=chunk_content,
                file_metadata=file_metadata,
                chunk_index=base_index + len(chunks),
                chunk_type=element.type,
                start_line=chunk_start_line,
                end_line=chunk_end_line,
                tokens_count=current_tokens,
                parent_class=element.parent if element.type == 'method' else None,
                parent_function=element.parent if element.type != 'class' else None,
                semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                function_part_index=len(chunks),
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'part_index': len(chunks),
                    'function_type': element.function_type,
                    'is_async': element.is_async,
                    'is_export': element.is_export
                }
            ))
        
        return chunks 