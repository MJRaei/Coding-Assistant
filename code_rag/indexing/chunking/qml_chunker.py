"""
QML-specific chunker implementation.

QML (Qt Modeling Language) is a declarative language for designing 
user interface-centric applications. This chunker understands QML 
structure including components, properties, signals, and functions.
"""

import re
from typing import List, Optional
from dataclasses import dataclass

from .base_chunker import BaseLanguageChunker, CodeBoundary, BoundaryType, ChunkingStrategy
from ...models import FileMetadata, CodeChunk


@dataclass
class QMLCodeElement:
    """Represents a QML code element with its context"""
    name: str
    type: str  # 'component', 'property', 'function', 'signal', etc.
    start_line: int
    end_line: int
    indent_level: int
    parent: Optional[str] = None
    component_type: Optional[str] = None  # Rectangle, Button, etc.
    properties: List[str] = None
    signals: List[str] = None
    is_root_component: bool = False


class QMLChunker(BaseLanguageChunker):
    """QML-specific chunker with advanced QML understanding"""
    
    def __init__(self, max_tokens: int = 2000, overlap_tokens: int = 200, 
                 strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC_FIRST):
        super().__init__(max_tokens, overlap_tokens, strategy)
        
        # QML-specific patterns
        self.import_pattern = re.compile(r'^\s*import\s+(?:[\w\.]+|[\'"][^\'\"]*[\'"])\s*[\d\.]*\s*(?:as\s+\w+)?\s*$')
        self.pragma_pattern = re.compile(r'^\s*pragma\s+\w+\s*$')
        self.component_pattern = re.compile(r'^\s*(\w+)\s*\{')
        self.property_pattern = re.compile(r'^\s*(?:readonly\s+)?property\s+(\w+)\s+(\w+)(?:\s*:\s*(.+))?\s*$')
        self.signal_pattern = re.compile(r'^\s*signal\s+(\w+)\s*(?:\(([^)]*)\))?\s*$')
        self.function_pattern = re.compile(r'^\s*function\s+(\w+)\s*\([^)]*\)\s*\{?')
        self.signal_handler_pattern = re.compile(r'^\s*(on\w+)\s*:\s*(.+)$')
        self.binding_pattern = re.compile(r'^\s*(\w+)\s*:\s*(.+)$')
        
        # QML built-in components
        self.qml_components = {
            'Item', 'Rectangle', 'Text', 'Image', 'Button', 'TextField', 'Column', 
            'Row', 'Grid', 'StackView', 'ListView', 'GridView', 'ScrollView',
            'ApplicationWindow', 'Window', 'Dialog', 'Popup', 'Menu', 'ToolBar',
            'SwipeView', 'TabView', 'SplitView', 'Flickable', 'MouseArea',
            'Timer', 'Animation', 'PropertyAnimation', 'NumberAnimation',
            'Connections', 'Component', 'Loader', 'Repeater', 'Flow', 'GridLayout',
            'RowLayout', 'ColumnLayout', 'StackLayout', 'Frame', 'GroupBox',
            'Label', 'CheckBox', 'RadioButton', 'Switch', 'Slider', 'ProgressBar',
            'BusyIndicator', 'ComboBox', 'SpinBox', 'TextArea', 'ScrollBar',
            'TabBar', 'ToolButton', 'RoundButton', 'DelayButton', 'Tumbler'
        }
        
    def get_supported_extensions(self) -> List[str]:
        return ['.qml', '.qmldir']
    
    def estimate_tokens(self, text: str) -> int:
        """QML-specific token estimation"""
        char_count = len(text)
        lines = text.splitlines()
        effective_chars = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            effective_chars += len(stripped)
        
        # QML: roughly 1 token per 3.8 characters (denser than regular text)
        return max(1, int(effective_chars / 3.8))
    
    def detect_boundaries(self, content: str, lines: List[str]) -> List[CodeBoundary]:
        """Detect QML-specific code boundaries"""
        boundaries = []
        brace_stack = []
        current_component = None
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped or stripped.startswith('//'):
                continue
                
            # Check for imports
            if self.import_pattern.match(line):
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.IMPORT,
                    content=line,
                    name=self._extract_import_name(stripped),
                    indent_level=self._get_indent_level(line)
                ))
                continue
            
            # Check for pragma statements
            if self.pragma_pattern.match(line):
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.IMPORT,  # Treat pragma like import
                    content=line,
                    name="pragma",
                    indent_level=self._get_indent_level(line)
                ))
                continue
            
            # Check for component definitions
            comp_match = self.component_pattern.match(line)
            if comp_match:
                component_name = comp_match.group(1)
                
                if component_name in self.qml_components or component_name[0].isupper():
                    boundaries.append(CodeBoundary(
                        line_number=i,
                        boundary_type=BoundaryType.CLASS,
                        content=line,
                        name=component_name,
                        parent=current_component,
                        indent_level=self._get_indent_level(line)
                    ))
                    
                    if '{' in line:
                        brace_stack.append(current_component)
                        current_component = component_name
                continue
            
            # Check for properties
            prop_match = self.property_pattern.match(line)
            if prop_match:
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.PROPERTY,
                    content=line,
                    name=prop_match.group(2),
                    parent=current_component,
                    indent_level=self._get_indent_level(line)
                ))
                continue
            
            # Check for signals
            signal_match = self.signal_pattern.match(line)
            if signal_match:
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.FUNCTION,  # Treat signals as functions
                    content=line,
                    name=signal_match.group(1),
                    parent=current_component,
                    indent_level=self._get_indent_level(line)
                ))
                continue
            
            # Check for functions
            func_match = self.function_pattern.match(line)
            if func_match:
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.FUNCTION,
                    content=line,
                    name=func_match.group(1),
                    parent=current_component,
                    indent_level=self._get_indent_level(line)
                ))
                continue
            
            # Check for signal handlers
            handler_match = self.signal_handler_pattern.match(line)
            if handler_match:
                boundaries.append(CodeBoundary(
                    line_number=i,
                    boundary_type=BoundaryType.METHOD,  # Treat handlers as methods
                    content=line,
                    name=handler_match.group(1),
                    parent=current_component,
                    indent_level=self._get_indent_level(line)
                ))
                continue
            
            # Handle closing braces
            if '}' in line and brace_stack:
                close_count = line.count('}')
                for _ in range(min(close_count, len(brace_stack))):
                    current_component = brace_stack.pop() if brace_stack else None
        
        return sorted(boundaries, key=lambda b: b.line_number)
    
    def _extract_import_name(self, import_line: str) -> str:
        """Extract meaningful name from import statement"""
        parts = import_line.split()
        if len(parts) >= 2:
            if 'as' in parts:
                as_index = parts.index('as')
                if as_index + 1 < len(parts):
                    return parts[as_index + 1]
            
            # Handle different import patterns
            import_path = parts[1]
            if import_path.startswith('"') or import_path.startswith("'"):
                # Relative import like import './' as CustomComponents
                # Extract the quoted path and get a meaningful name
                path = import_path.strip('"\'')
                if path == './':
                    return "LocalComponents"
                elif path == '../':
                    return "ParentComponents"
                else:
                    # Use last part of path or the path itself
                    return path.split('/')[-1] or "RelativeImport"
            else:
                # Standard import like QtQuick.Controls - keep more specific name
                if '.' in import_path:
                    # For QtQuick.Controls, return "QtQuick.Controls" instead of just "QtQuick"
                    return import_path
                else:
                    # For simple imports like QtQuick, return as is
                    return import_path
        return "import"
    
    def _get_indent_level(self, line: str) -> int:
        """Calculate indentation level"""
        return len(line) - len(line.lstrip())
    
    def is_semantic_unit_complete(self, boundaries: List[CodeBoundary], 
                                 start_idx: int, end_idx: int) -> bool:
        """Check if a range represents a complete QML semantic unit"""
        if start_idx >= end_idx:
            return True
        
        # Check if we have a complete component definition
        start_boundary = boundaries[start_idx]
        if start_boundary.boundary_type == BoundaryType.CLASS:
            # This is a component - check if we include all its children
            component_indent = start_boundary.indent_level
            
            for i in range(start_idx + 1, min(end_idx + 1, len(boundaries))):
                boundary = boundaries[i]
                if boundary.indent_level <= component_indent:
                    # We've reached the end of this component
                    return i >= end_idx
            
            return True
        
        return True
    
    def chunk_function_aware(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """QML-aware chunking with intelligent component splitting"""
        lines = content.splitlines()
        boundaries = self.detect_boundaries(content, lines)
        
        if not boundaries:
            return self.chunk_by_size(content, file_metadata)
        
        # Check if file is small enough to keep as single chunk
        total_tokens = self.estimate_tokens(content)
        if total_tokens <= self.max_tokens:
            return self._create_single_file_chunk(content, file_metadata, boundaries)
        
        elements = self._extract_qml_elements(content, lines, boundaries)
        chunks = []
        
        for element in elements:
            element_chunks = self._chunk_qml_element(
                element, lines, file_metadata, len(chunks)
            )
            
            # Track relationships
            if len(element_chunks) > 1:
                chunk_indices = [chunk.chunk_index for chunk in element_chunks]
                for chunk in element_chunks:
                    chunk.related_chunks = [idx for idx in chunk_indices if idx != chunk.chunk_index]
            
            chunks.extend(element_chunks)
        
        return chunks
    
    def _create_single_file_chunk(self, content: str, file_metadata: FileMetadata, boundaries: List[CodeBoundary]) -> List[CodeChunk]:
        """Create a single chunk for small QML files with complete metadata"""
        # Extract metadata from boundaries with deduplication
        imports = set()  # Use set to automatically deduplicate
        components = []
        properties = []
        signals = []
        functions = []
        
        for boundary in boundaries:
            if boundary.boundary_type == BoundaryType.IMPORT:
                imports.add(boundary.name)
            elif boundary.boundary_type == BoundaryType.CLASS:
                components.append(boundary.name)
            elif boundary.boundary_type == BoundaryType.PROPERTY:
                properties.append(boundary.name)
            elif boundary.boundary_type == BoundaryType.FUNCTION:
                if boundary.name.startswith('on') and boundary.name[2:3].isupper():
                    # Signal handler
                    signals.append(boundary.name)
                else:
                    functions.append(boundary.name)
            elif boundary.boundary_type == BoundaryType.METHOD:
                # Also include methods as signal handlers
                signals.append(boundary.name)
        
        # Convert imports set back to sorted list for consistent output
        imports_list = sorted(list(imports))
        
        # Find the main component (usually the first one)
        main_component = components[0] if components else "QMLItem"
        is_root_component = len(components) == 1 or (len(components) > 1 and components[0] != components[1])
        
        return [CodeChunk(
            content=content,
            file_metadata=file_metadata,
            chunk_index=0,
            chunk_type='component',
            start_line=1,
            end_line=len(content.splitlines()),
            tokens_count=self.estimate_tokens(content),
            semantic_id=f"file_{main_component}",
            chunk_metadata={
                'component_type': main_component,
                'is_root_component': is_root_component,
                'qml_properties': properties,
                'qml_signals': signals,
                'qml_functions': functions,
                'qml_imports': imports_list,
                'is_complete_file': True
            }
        )]
    
    def _extract_qml_elements(self, content: str, lines: List[str], boundaries: List[CodeBoundary]) -> List[QMLCodeElement]:
        """Extract structured QML elements - conservative approach"""
        elements = []
        
        # Group boundaries by type and size
        root_components = []
        child_components = []
        functions = []
        
        for boundary in boundaries:
            if boundary.boundary_type == BoundaryType.CLASS:
                end_line = self._find_component_end_line(boundary, lines)
                element_lines = lines[boundary.line_number:end_line]
                element_tokens = sum(self.estimate_tokens(line) for line in element_lines)
                
                element = QMLCodeElement(
                    name=boundary.name,
                    type='component',
                    start_line=boundary.line_number + 1,
                    end_line=end_line,
                    indent_level=boundary.indent_level,
                    parent=boundary.parent,
                    component_type=boundary.name,
                    is_root_component=(boundary.indent_level == 0)
                )
                
                if boundary.indent_level == 0:
                    root_components.append((element, element_tokens))
                else:
                    child_components.append((element, element_tokens))
                    
            elif boundary.boundary_type == BoundaryType.FUNCTION:
                end_line = self._find_function_end_line(boundary, lines)
                element_lines = lines[boundary.line_number:end_line]
                element_tokens = sum(self.estimate_tokens(line) for line in element_lines)
                
                element = QMLCodeElement(
                    name=boundary.name,
                    type='function',
                    start_line=boundary.line_number + 1,
                    end_line=end_line,
                    indent_level=boundary.indent_level,
                    parent=boundary.parent
                )
                functions.append((element, element_tokens))
        
        # Ultra-conservative extraction strategy:
        # 1. Primarily extract root components only (main file structure)
        # 2. Avoid extracting child components unless they're very substantial
        # 3. Only extract standalone functions that are very substantial
        
        # Always include root components (this is the main content)
        for element, tokens in root_components:
            elements.append(element)
        
        # Only include very substantial standalone functions (>500 tokens)
        substantial_functions = [(e, t) for e, t in functions if t > 500 and e.indent_level == 0]
        for element, tokens in substantial_functions:
            elements.append(element)
        
        # Skip child components entirely to avoid fragmentation
        # Large files should be split at natural boundaries within the root component
        
        return sorted(elements, key=lambda e: e.start_line)
    
    def _find_component_end_line(self, boundary: CodeBoundary, lines: List[str]) -> int:
        """Find end line by matching braces"""
        start_line = boundary.line_number
        brace_count = 0
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0 and i > start_line:
                return i + 1
        
        return len(lines)
    
    def _find_function_end_line(self, boundary: CodeBoundary, lines: List[str]) -> int:
        """Find end line of function"""
        start_line = boundary.line_number
        
        if '{' in lines[start_line] and '}' in lines[start_line]:
            return start_line + 1
        
        brace_count = lines[start_line].count('{')
        if brace_count == 0:
            return start_line + 1
        
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            brace_count += line.count('{') - line.count('}')
            
            if brace_count == 0:
                return i + 1
        
        return len(lines)
    
    def _chunk_qml_element(self, element: QMLCodeElement, lines: List[str], 
                          file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Chunk a single QML element"""
        element_lines = lines[element.start_line - 1:element.end_line]
        element_content = '\n'.join(element_lines)
        element_tokens = self.estimate_tokens(element_content)
        
        # If fits within limit, return single chunk
        if element_tokens <= self.max_tokens:
            return [CodeChunk(
                content=element_content,
                file_metadata=file_metadata,
                chunk_index=base_index,
                chunk_type=element.type,
                start_line=element.start_line,
                end_line=element.end_line,
                tokens_count=element_tokens,
                parent_class=element.parent if element.type != 'component' else None,
                semantic_id=f"{element.type}_{element.name}",
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'component_type': element.component_type,
                    'is_root_component': element.is_root_component,
                    'qml_properties': element.properties or [],
                    'qml_signals': element.signals or []
                }
            )]
        
        # Split large elements
        return self._split_by_size_with_metadata(element, lines, file_metadata, base_index)
    
    def _split_by_size_with_metadata(self, element: QMLCodeElement, lines: List[str], 
                                    file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Split by size while preserving QML metadata and respecting component boundaries"""
        chunks = []
        element_lines = lines[element.start_line - 1:element.end_line]
        
        # For components that fit within limit, keep as single chunk
        total_tokens = sum(self.estimate_tokens(line) for line in element_lines)
        
        # If fits within the 2000 token limit, keep as single chunk
        if total_tokens <= self.max_tokens:
            return [CodeChunk(
                content='\n'.join(element_lines),
                file_metadata=file_metadata,
                chunk_index=base_index,
                chunk_type=element.type,
                start_line=element.start_line,
                end_line=element.end_line,
                tokens_count=total_tokens,
                parent_class=element.parent if element.type != 'component' else None,
                semantic_id=f"{element.type}_{element.name}",
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'component_type': element.component_type
                }
            )]
        
        # Calculate optimal chunk sizes for even distribution
        num_chunks = (total_tokens + self.max_tokens - 1) // self.max_tokens  # Ceiling division
        target_chunk_size = total_tokens // num_chunks
        max_chunk_size = min(self.max_tokens, target_chunk_size + 500)  # Allow some flexibility
        
        # Component-aware splitting that respects boundaries
        chunks = []
        current_lines = []
        current_tokens = 0
        start_line_idx = 0
        
        # Track component nesting to find safe split points
        component_stack = []
        brace_level = 0
        last_safe_split = 0
        
        for i, line in enumerate(element_lines):
            line_tokens = self.estimate_tokens(line)
            stripped = line.strip()
            
            # Track brace levels and component boundaries
            open_braces = stripped.count('{')
            close_braces = stripped.count('}')
            
            # Check for component definitions
            comp_match = self.component_pattern.match(line)
            if comp_match:
                component_name = comp_match.group(1)
                is_qml_component = (component_name in self.qml_components or 
                                  component_name[0].isupper() or 
                                  '.' in component_name or
                                  component_name.endswith('Components'))
                
                if is_qml_component and brace_level > 0:  # Not the root component
                    # This is a potential safe split point (start of new component)
                    # But only if we have accumulated enough content
                    if current_tokens > target_chunk_size * 0.5:  # At least 50% of target size
                        last_safe_split = i
            
            # Update brace level
            brace_level += open_braces - close_braces
            
            # Check for end of components (closing braces at lower levels)
            if close_braces > 0 and brace_level <= 1 and current_tokens > target_chunk_size * 0.5:
                # End of a component - another safe split point
                last_safe_split = i + 1
            
            # Check if we should split (using target size, not max size)
            should_split = (current_tokens + line_tokens > target_chunk_size and 
                          current_lines and 
                          current_tokens >= target_chunk_size * 0.7)  # At least 70% of target
            
            # Also split if we're approaching the hard limit
            if current_tokens + line_tokens > max_chunk_size and current_lines:
                should_split = True
            
            if should_split:
                # Look for the best split point
                split_point = None
                
                # Option 1: Use the last safe split point if it's recent enough
                if last_safe_split > 0 and i - last_safe_split < 50:  # Within 50 lines
                    split_point = last_safe_split
                    
                # Option 2: Look for a safe split point in next few lines
                elif i < len(element_lines) - 1:
                    for j in range(i + 1, min(i + 10, len(element_lines))):
                        next_line = element_lines[j].strip()
                        # Look for component start or function definitions
                        if (self.component_pattern.match(element_lines[j]) or 
                            self.function_pattern.match(element_lines[j]) or
                            next_line.startswith('}')):
                            split_point = j
                            break
                
                # If we found a safe split point, use it
                if split_point is not None and split_point > start_line_idx:
                    # Create chunk up to the split point
                    chunk_lines = element_lines[start_line_idx:split_point]
                    chunk_tokens = sum(self.estimate_tokens(line) for line in chunk_lines)
                    
                    chunk_content = '\n'.join(chunk_lines)
                    chunk_start_line = element.start_line + start_line_idx
                    chunk_end_line = element.start_line + split_point - 1
                    
                    chunks.append(CodeChunk(
                        content=chunk_content,
                        file_metadata=file_metadata,
                        chunk_index=base_index + len(chunks),
                        chunk_type=element.type,
                        start_line=chunk_start_line,
                        end_line=chunk_end_line,
                        tokens_count=chunk_tokens,
                        parent_class=element.parent if element.type != 'component' else None,
                        semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                        function_part_index=len(chunks),
                        chunk_metadata={
                            'element_name': element.name,
                            'element_type': element.type,
                            'component_type': element.component_type,
                            'part_index': len(chunks)
                        }
                    ))
                    
                    # Start new chunk from split point
                    current_lines = element_lines[split_point:i+1]
                    current_tokens = sum(self.estimate_tokens(line) for line in current_lines)
                    start_line_idx = split_point
                    last_safe_split = 0  # Reset
                else:
                    # No safe split found - create chunk at current position
                    # but ensure we don't break in the middle of a component
                    # Allow slightly over the target size to complete the current component
                    if brace_level > 0 and current_tokens < max_chunk_size:
                        # We're inside a component - continue until we close it or hit hard limit
                        current_lines.append(line)
                        current_tokens += line_tokens
                        continue
                    else:
                        # We're at component boundary or hit hard limit - safe to split
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
                            parent_class=element.parent if element.type != 'component' else None,
                            semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                            function_part_index=len(chunks),
                            chunk_metadata={
                                'element_name': element.name,
                                'element_type': element.type,
                                'component_type': element.component_type,
                                'part_index': len(chunks)
                            }
                        ))
                        
                        # Start new chunk
                        current_lines = [line]
                        current_tokens = line_tokens
                        start_line_idx = i
                        last_safe_split = 0
            else:
                # Add line to current chunk
                current_lines.append(line)
                current_tokens += line_tokens
        
        # Add final chunk if there are remaining lines
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
                parent_class=element.parent if element.type != 'component' else None,
                semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                function_part_index=len(chunks),
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'component_type': element.component_type,
                    'part_index': len(chunks)
                }
            ))
        
        return chunks
    
    def _find_natural_split_points(self, element_lines: List[str], start_line_number: int) -> List[int]:
        """Find natural places to split QML components (between child components and major sections)"""
        split_points = []
        brace_level = 0
        component_stack = []
        current_component_start = None
        found_root = False
        
        for i, line in enumerate(element_lines):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('//'):
                continue
            
            # Track brace levels
            open_braces = stripped.count('{')
            close_braces = stripped.count('}')
            
            # Check for component definitions
            comp_match = self.component_pattern.match(line)
            if comp_match:
                component_name = comp_match.group(1)
                
                # Check if this is a meaningful component
                is_qml_component = (component_name in self.qml_components or 
                                  component_name[0].isupper() or 
                                  '.' in component_name or
                                  component_name.endswith('Components'))
                
                if is_qml_component:
                    if not found_root:
                        # This is the root component, don't split here
                        found_root = True
                        current_component_start = i
                    else:
                        # This is a child component - potential split point
                        # Only add if we're not too deep in nesting and have reasonable chunk size
                        if brace_level <= 2 and i > 50:  # At least 50 lines for a meaningful split
                            split_points.append(i)
                            current_component_start = i
                    
                    component_stack.append((component_name, brace_level, i))
            
            # Check for function definitions (also good split points)
            func_match = self.function_pattern.match(line)
            if func_match and brace_level <= 1 and i > 50:
                split_points.append(i)
            
            # Update brace level after processing
            brace_level += open_braces - close_braces
            
            # Clean up component stack when components end
            if close_braces > 0:
                while component_stack and brace_level < component_stack[-1][1]:
                    component_stack.pop()
        
        # Filter split points to ensure reasonable chunk sizes - be very conservative
        filtered_split_points = []
        last_split = 0
        
        for split_point in split_points:
            # Ensure significant distance between splits (at least 200 lines or 1000 tokens)
            if split_point - last_split > 200:
                section_lines = element_lines[last_split:split_point]
                section_tokens = sum(self.estimate_tokens(line) for line in section_lines)
                
                # Only split if the section is substantially sized
                if section_tokens > 1000:
                    filtered_split_points.append(split_point)
                    last_split = split_point
        
        # Limit to maximum 3 splits to avoid over-fragmentation
        return filtered_split_points[:3]
    
    def _split_by_lines_with_metadata(self, element: QMLCodeElement, lines: List[str], 
                                     file_metadata: FileMetadata, base_index: int) -> List[CodeChunk]:
        """Fallback: Split by lines while preserving QML metadata"""
        chunks = []
        element_lines = lines[element.start_line - 1:element.end_line]
        
        current_chunk_lines = []
        current_tokens = 0
        
        for i, line in enumerate(element_lines):
            line_tokens = self.estimate_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk_lines:
                chunk_content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=base_index + len(chunks),
                    chunk_type=element.type,
                    start_line=element.start_line + i - len(current_chunk_lines),
                    end_line=element.start_line + i - 1,
                    tokens_count=current_tokens,
                    parent_class=element.parent if element.type != 'component' else None,
                    semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                    function_part_index=len(chunks),
                    chunk_metadata={
                        'element_name': element.name,
                        'element_type': element.type,
                        'component_type': element.component_type,
                        'part_index': len(chunks)
                    }
                ))
                
                # Start new chunk with overlap
                overlap_lines = current_chunk_lines[-min(3, len(current_chunk_lines)):]
                current_chunk_lines = overlap_lines + [line]
                current_tokens = sum(self.estimate_tokens(l) for l in current_chunk_lines)
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk_lines:
            chunk_content = '\n'.join(current_chunk_lines)
            chunks.append(CodeChunk(
                content=chunk_content,
                file_metadata=file_metadata,
                chunk_index=base_index + len(chunks),
                chunk_type=element.type,
                start_line=element.start_line + len(element_lines) - len(current_chunk_lines),
                end_line=element.end_line,
                tokens_count=current_tokens,
                parent_class=element.parent if element.type != 'component' else None,
                semantic_id=f"{element.type}_{element.name}_part_{len(chunks)}",
                function_part_index=len(chunks),
                chunk_metadata={
                    'element_name': element.name,
                    'element_type': element.type,
                    'component_type': element.component_type,
                    'part_index': len(chunks)
                }
            ))
        
        return chunks 