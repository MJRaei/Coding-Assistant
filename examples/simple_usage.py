"""
Simple usage example of the Code RAG system.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from code_rag.indexing.index_builder import IndexBuilder
from code_rag.indexing.chunking import ChunkingStrategy
from code_rag.retrieval.code_search import CodeSearch
from code_rag.retrieval.answer_generator import AnswerGenerator

def main():
    """Main function to demonstrate the Code RAG system."""
    
    # Target directory to chunk
    project_dir = Path("/Users/mjraei/Desktop/Projects/OplaSmart/resources/EngReportComponents")
    
    # Define paths - store data in the current code embedder project
    current_project = Path(__file__).parent.parent
    data_dir = current_project / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=== Code RAG System Demo ===")
    print(f"Target directory to analyze: {project_dir}")
    print(f"Data directory (for storing index): {data_dir}")
    
    if not project_dir.exists():
        print(f"❌ Error: Target directory does not exist: {project_dir}")
        return
    
    # 1. Build index using different chunking strategies
    print("\n1. Building index with different chunking strategies...")
    
    # Try ADAPTIVE_STRUCTURE strategy first (new!)
    print("\n--- Testing ADAPTIVE_STRUCTURE Strategy ---")
    builder = IndexBuilder(
        project_directory=str(project_dir),
        data_directory=str(data_dir),
        chunking_strategy=ChunkingStrategy.ADAPTIVE_STRUCTURE
    )
    
    try:
        index_path = builder.build_index()
        print(f"✓ ADAPTIVE_STRUCTURE index built successfully: {index_path}")
        
        # Inspect all chunks to see the new metadata
        print("\n--- Inspecting ADAPTIVE_STRUCTURE chunks ---")
        inspect_chunks(builder)
        
        # Test search with ADAPTIVE_STRUCTURE index
        print("\n--- Testing search with ADAPTIVE_STRUCTURE index ---")
        test_search(data_dir, "adaptive_structure")
        
    except Exception as e:
        print(f"✗ Error with ADAPTIVE_STRUCTURE strategy: {e}")
        print("This might be due to missing dependencies or configuration issues.")
    
    # Compare with STRUCTURE_PRESERVING strategy
    print("\n--- Testing STRUCTURE_PRESERVING Strategy (for comparison) ---")
    builder_semantic = IndexBuilder(
        project_directory=str(project_dir),
        data_directory=str(data_dir),
        chunking_strategy=ChunkingStrategy.STRUCTURE_PRESERVING
    )
    
    try:
        index_path_semantic = builder_semantic.build_index()
        print(f"✓ STRUCTURE_PRESERVING index built successfully: {index_path_semantic}")
        
        # Test search with STRUCTURE_PRESERVING index
        print("\n--- Testing search with STRUCTURE_PRESERVING index ---")
        test_search(data_dir, "structure_preserving")
        
    except Exception as e:
        print(f"✗ Error with STRUCTURE_PRESERVING strategy: {e}")
    
    # 2. Compare chunking strategies
    print("\n2. Comparing chunking strategies...")
    compare_strategies(project_dir, data_dir)

def inspect_chunks(builder, max_chunks=None):
    """Inspect all chunks from the built index with detailed metadata"""
    # Get all chunks from the vector store after building
    chunks = builder.get_vector_store().chunks
    
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    print(f"\n=== CHUNK INSPECTION ({len(chunks)} chunks) ===")
    
    for i, chunk in enumerate(chunks):
        print(f"\n{'='*60}")
        print(f"CHUNK {i+1} of {len(chunks)}")
        print(f"{'='*60}")
        
        # File metadata
        metadata = chunk.file_metadata
        print(f"File: {metadata.relative_path}")
        print(f"File type: {metadata.file_type}")
        print(f"File size: {metadata.file_size} bytes")
        print(f"Total lines: {metadata.line_count}")
        
        # Only show file-level metadata if it's populated (mainly for Python files)
        file_type = metadata.file_type.lower()
        if file_type != 'qml':  # For non-QML files, show file-level metadata
            print(f"Functions: {metadata.functions}")
            print(f"Classes: {metadata.classes}")
            print(f"Imports: {metadata.imports}")
        elif metadata.functions or metadata.classes or metadata.imports:
            # For QML files, only show if actually populated (future enhancement)
            print(f"Functions: {metadata.functions}")
            print(f"Classes: {metadata.classes}")
            print(f"Imports: {metadata.imports}")
        
        # Chunk info
        print(f"Chunk lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Token count: {chunk.tokens_count}")
        print(f"Chunk type: {chunk.chunk_type}")
        
        # Enhanced metadata display based on file type
        if chunk.parent_class or chunk.parent_function or chunk.semantic_id or chunk.chunk_metadata:
            
            if file_type == 'qml':
                print(f"\n🎯 QML Component Metadata:")
                
                # QML-specific information from chunk_metadata
                if chunk.chunk_metadata:
                    # Component information
                    if 'component_type' in chunk.chunk_metadata:
                        print(f"Component Type: {chunk.chunk_metadata['component_type']}")
                    if 'element_name' in chunk.chunk_metadata:
                        print(f"Element Name: {chunk.chunk_metadata['element_name']}")
                    if 'element_type' in chunk.chunk_metadata:
                        print(f"Element Type: {chunk.chunk_metadata['element_type']}")
                    if 'is_root_component' in chunk.chunk_metadata:
                        root_status = "Yes" if chunk.chunk_metadata['is_root_component'] else "No"
                        print(f"Root Component: {root_status}")
                    
                    # Part information for multi-chunk components
                    if 'part_index' in chunk.chunk_metadata:
                        print(f"Component Part: {chunk.chunk_metadata['part_index']}")
                    if chunk.function_part_index is not None:
                        print(f"Function Part Index: {chunk.function_part_index}")
                    
                    # Complete file indicator
                    if 'is_complete_file' in chunk.chunk_metadata and chunk.chunk_metadata['is_complete_file']:
                        print(f"Complete File: Yes")
                    
                    # QML imports
                    if 'qml_imports' in chunk.chunk_metadata and chunk.chunk_metadata['qml_imports']:
                        imports_list = chunk.chunk_metadata['qml_imports']
                        print(f"QML Imports: {', '.join(imports_list)}")
                    
                    # QML properties
                    if 'qml_properties' in chunk.chunk_metadata and chunk.chunk_metadata['qml_properties']:
                        properties_list = chunk.chunk_metadata['qml_properties']
                        print(f"QML Properties: {', '.join(properties_list)}")
                    
                    # QML signals
                    if 'qml_signals' in chunk.chunk_metadata and chunk.chunk_metadata['qml_signals']:
                        signals_list = chunk.chunk_metadata['qml_signals']
                        print(f"QML Signals: {', '.join(signals_list)}")
                    
                    # QML functions
                    if 'qml_functions' in chunk.chunk_metadata and chunk.chunk_metadata['qml_functions']:
                        functions_list = chunk.chunk_metadata['qml_functions']
                        print(f"QML Functions: {', '.join(functions_list)}")
                
                # General chunk relationship information
                if chunk.semantic_id:
                    print(f"Semantic ID: {chunk.semantic_id}")
                if chunk.parent_class:
                    print(f"Parent Component: {chunk.parent_class}")
                if chunk.related_chunks:
                    print(f"Related Chunks: {chunk.related_chunks}")
                
                # Show any additional metadata fields not explicitly handled
                if chunk.chunk_metadata:
                    handled_keys = {
                        'component_type', 'element_name', 'element_type', 'is_root_component',
                        'part_index', 'is_complete_file', 'qml_imports', 'qml_properties',
                        'qml_signals', 'qml_functions'
                    }
                    additional_metadata = {k: v for k, v in chunk.chunk_metadata.items() 
                                         if k not in handled_keys and v is not None}
                    if additional_metadata:
                        print(f"Additional Metadata: {additional_metadata}")
                    
            elif file_type == 'py':
                print(f"\n🐍 Python Code Metadata:")
                
                # Python-specific information
                if chunk.chunk_metadata:
                    if 'element_type' in chunk.chunk_metadata:
                        element_type = chunk.chunk_metadata['element_type']
                        print(f"Element Type: {element_type.title()}")
                    if 'decorators' in chunk.chunk_metadata and chunk.chunk_metadata['decorators']:
                        print(f"Decorators: {', '.join(chunk.chunk_metadata['decorators'])}")
                    if 'is_async' in chunk.chunk_metadata and chunk.chunk_metadata['is_async']:
                        print(f"Async Function: Yes")
                    if 'is_property' in chunk.chunk_metadata and chunk.chunk_metadata['is_property']:
                        print(f"Property Method: Yes")
                    if 'class_name' in chunk.chunk_metadata:
                        print(f"Class Name: {chunk.chunk_metadata['class_name']}")
                    if 'method_index' in chunk.chunk_metadata:
                        print(f"Method Index: {chunk.chunk_metadata['method_index']}")
                
                # General Python information
                if chunk.parent_class:
                    print(f"Parent Class: {chunk.parent_class}")
                if chunk.parent_function:
                    print(f"Parent Function: {chunk.parent_function}")
                if chunk.semantic_id:
                    print(f"Semantic ID: {chunk.semantic_id}")
                if chunk.function_part_index is not None:
                    print(f"Function Part: {chunk.function_part_index}")
                if chunk.related_chunks:
                    print(f"Related Chunks: {chunk.related_chunks}")
                    
            elif file_type in ['js', 'jsx', 'mjs', 'ts', 'tsx']:
                print(f"\n🟨 JavaScript Code Metadata:")
                
                # JavaScript-specific information from chunk_metadata
                if chunk.chunk_metadata:
                    # Framework and module information
                    if 'framework_type' in chunk.chunk_metadata and chunk.chunk_metadata['framework_type']:
                        print(f"Framework: {chunk.chunk_metadata['framework_type']}")
                    if 'module_type' in chunk.chunk_metadata and chunk.chunk_metadata['module_type']:
                        print(f"Module System: {chunk.chunk_metadata['module_type']}")
                    
                    # Element information
                    if 'element_name' in chunk.chunk_metadata:
                        print(f"Element Name: {chunk.chunk_metadata['element_name']}")
                    if 'element_type' in chunk.chunk_metadata:
                        print(f"Element Type: {chunk.chunk_metadata['element_type']}")
                    if 'function_type' in chunk.chunk_metadata and chunk.chunk_metadata['function_type']:
                        print(f"Function Type: {chunk.chunk_metadata['function_type']}")
                    
                    # Part information for multi-chunk elements
                    if 'part_index' in chunk.chunk_metadata:
                        print(f"Element Part: {chunk.chunk_metadata['part_index']}")
                    if chunk.function_part_index is not None:
                        print(f"Function Part Index: {chunk.function_part_index}")
                    
                    # Complete file indicator
                    if 'is_complete_file' in chunk.chunk_metadata and chunk.chunk_metadata['is_complete_file']:
                        print(f"Complete File: Yes")
                    
                    # JavaScript imports and exports
                    if 'js_imports' in chunk.chunk_metadata and chunk.chunk_metadata['js_imports']:
                        imports_list = chunk.chunk_metadata['js_imports']
                        print(f"JS Imports: {', '.join(imports_list)}")
                    if 'js_exports' in chunk.chunk_metadata and chunk.chunk_metadata['js_exports']:
                        exports_list = chunk.chunk_metadata['js_exports']
                        print(f"JS Exports: {', '.join(exports_list)}")
                    
                    # JavaScript functions and classes
                    if 'js_functions' in chunk.chunk_metadata and chunk.chunk_metadata['js_functions']:
                        functions_list = chunk.chunk_metadata['js_functions']
                        print(f"JS Functions: {', '.join(functions_list)}")
                    if 'js_classes' in chunk.chunk_metadata and chunk.chunk_metadata['js_classes']:
                        classes_list = chunk.chunk_metadata['js_classes']
                        print(f"JS Classes: {', '.join(classes_list)}")
                    
                    # JavaScript variables and async functions
                    if 'js_variables' in chunk.chunk_metadata and chunk.chunk_metadata['js_variables']:
                        variables_list = chunk.chunk_metadata['js_variables']
                        print(f"JS Variables: {', '.join(variables_list)}")
                    if 'js_async_functions' in chunk.chunk_metadata and chunk.chunk_metadata['js_async_functions']:
                        async_functions_list = chunk.chunk_metadata['js_async_functions']
                        print(f"Async Functions: {', '.join(async_functions_list)}")
                    
                    # Special function types
                    if 'is_async' in chunk.chunk_metadata and chunk.chunk_metadata['is_async']:
                        print(f"Is Async: Yes")
                    if 'is_export' in chunk.chunk_metadata and chunk.chunk_metadata['is_export']:
                        print(f"Is Export: Yes")
                    if 'is_constructor' in chunk.chunk_metadata and chunk.chunk_metadata['is_constructor']:
                        print(f"Is Constructor: Yes")
                
                # General chunk relationship information
                if chunk.semantic_id:
                    print(f"Semantic ID: {chunk.semantic_id}")
                if chunk.parent_class:
                    print(f"Parent Class: {chunk.parent_class}")
                if chunk.parent_function:
                    print(f"Parent Function: {chunk.parent_function}")
                if chunk.related_chunks:
                    print(f"Related Chunks: {chunk.related_chunks}")
                
                # Show any additional metadata fields not explicitly handled
                if chunk.chunk_metadata:
                    handled_keys = {
                        'framework_type', 'module_type', 'element_name', 'element_type', 
                        'function_type', 'part_index', 'is_complete_file', 'js_imports', 
                        'js_exports', 'js_functions', 'js_classes', 'js_variables', 
                        'js_async_functions', 'is_async', 'is_export', 'is_constructor'
                    }
                    additional_metadata = {k: v for k, v in chunk.chunk_metadata.items() 
                                         if k not in handled_keys and v is not None}
                    if additional_metadata:
                        print(f"Additional Metadata: {additional_metadata}")
                        
            else:
                # Generic metadata for other file types
                print(f"\n🎯 Code Metadata ({file_type.upper()}):")
                if chunk.parent_class:
                    print(f"Parent Class: {chunk.parent_class}")
                if chunk.parent_function:
                    print(f"Parent Function: {chunk.parent_function}")
                if chunk.function_part_index is not None:
                    print(f"Function Part: {chunk.function_part_index}")
                if chunk.semantic_id:
                    print(f"Semantic ID: {chunk.semantic_id}")
                if chunk.related_chunks:
                    print(f"Related Chunks: {chunk.related_chunks}")
                if chunk.chunk_metadata:
                    print(f"Metadata: {chunk.chunk_metadata}")
        
        # Content
        print(f"\nContent:")
        print("-" * 60)
        print(chunk.content)
        print("-" * 60)
        
        # Pause between chunks for readability
        if i < len(chunks) - 1:  # Don't pause after last chunk
            input("Press Enter to see next chunk...")
    
    print(f"\n✅ Finished inspecting {len(chunks)} chunks!")

def test_search(data_dir, strategy_name):
    """Test search functionality"""
    try:
        index_path = data_dir / "reporting_module_index"
        searcher = CodeSearch(index_path=str(index_path))
        
        # Test different types of queries
        queries = [
            "chunking strategy implementation",
            "class methods and functions",
            "file processing code",
            "Python code parsing"
        ]
        
        for query in queries:
            print(f"\nQuery: '{query}'")
            results = searcher.search(query, top_k=2)
            
            for j, result in enumerate(results):
                print(f"  Result {j+1}:")
                print(f"    File: {result['file_path']}")
                print(f"    Score: {result['score']:.3f}")
                print(f"    Chunk type: {result.get('chunk_type', 'N/A')}")
                print(f"    Line range: {result.get('line_range', 'N/A')}")
                
                # Show enhanced metadata based on file type
                file_ext = result['file_path'].split('.')[-1].lower()
                
                if file_ext == 'qml':
                    print(f"    🎯 QML Info:")
                    if 'chunk_metadata' in result and result['chunk_metadata']:
                        metadata = result['chunk_metadata']
                        if 'component_type' in metadata:
                            print(f"      Component: {metadata['component_type']}")
                        if 'is_root_component' in metadata:
                            root_status = "Yes" if metadata['is_root_component'] else "No"
                            print(f"      Root Component: {root_status}")
                        if 'qml_imports' in metadata and metadata['qml_imports']:
                            print(f"      Imports: {', '.join(metadata['qml_imports'])}")
                        if 'is_complete_file' in metadata and metadata['is_complete_file']:
                            print(f"      Complete File: Yes")
                    if 'parent_class' in result:
                        print(f"      Parent Component: {result['parent_class']}")
                    if 'semantic_id' in result:
                        print(f"      Semantic ID: {result['semantic_id']}")
                        
                elif file_ext == 'py':
                    print(f"    🐍 Python Info:")
                    if 'chunk_metadata' in result and result['chunk_metadata']:
                        metadata = result['chunk_metadata']
                        if 'element_type' in metadata:
                            print(f"      Element: {metadata['element_type'].title()}")
                        if 'decorators' in metadata and metadata['decorators']:
                            print(f"      Decorators: {', '.join(metadata['decorators'])}")
                        if 'is_async' in metadata and metadata['is_async']:
                            print(f"      Async: Yes")
                    if 'parent_class' in result:
                        print(f"      Parent Class: {result['parent_class']}")
                    if 'parent_function' in result:
                        print(f"      Parent Function: {result['parent_function']}")
                    if 'semantic_id' in result:
                        print(f"      Semantic ID: {result['semantic_id']}")
                        
                elif file_ext in ['js', 'jsx', 'mjs', 'ts', 'tsx']:
                    print(f"    🟨 JavaScript Info:")
                    if 'chunk_metadata' in result and result['chunk_metadata']:
                        metadata = result['chunk_metadata']
                        if 'framework_type' in metadata:
                            print(f"      Framework: {metadata['framework_type']}")
                        if 'module_type' in metadata:
                            print(f"      Module System: {metadata['module_type']}")
                        if 'element_name' in metadata:
                            print(f"      Element Name: {metadata['element_name']}")
                        if 'element_type' in metadata:
                            print(f"      Element Type: {metadata['element_type']}")
                        if 'function_type' in metadata:
                            print(f"      Function Type: {metadata['function_type']}")
                        if 'is_async' in metadata:
                            print(f"      Is Async: Yes")
                        if 'is_export' in metadata:
                            print(f"      Is Export: Yes")
                        if 'is_constructor' in metadata:
                            print(f"      Is Constructor: Yes")
                        if 'js_imports' in metadata and metadata['js_imports']:
                            print(f"      JS Imports: {', '.join(metadata['js_imports'])}")
                        if 'js_exports' in metadata and metadata['js_exports']:
                            print(f"      JS Exports: {', '.join(metadata['js_exports'])}")
                        if 'js_functions' in metadata and metadata['js_functions']:
                            print(f"      JS Functions: {', '.join(metadata['js_functions'])}")
                        if 'js_classes' in metadata and metadata['js_classes']:
                            print(f"      JS Classes: {', '.join(metadata['js_classes'])}")
                        if 'js_variables' in metadata and metadata['js_variables']:
                            print(f"      JS Variables: {', '.join(metadata['js_variables'])}")
                        if 'js_async_functions' in metadata and metadata['js_async_functions']:
                            print(f"      Async Functions: {', '.join(metadata['js_async_functions'])}")
                    if 'parent_class' in result:
                        print(f"      Parent Class: {result['parent_class']}")
                    if 'parent_function' in result:
                        print(f"      Parent Function: {result['parent_function']}")
                    if 'semantic_id' in result:
                        print(f"      Semantic ID: {result['semantic_id']}")
                        
                else:
                    # Generic metadata for other file types
                    if 'parent_class' in result:
                        print(f"    Parent Class: {result['parent_class']}")
                    if 'parent_function' in result:
                        print(f"    Parent Function: {result['parent_function']}")
                    if 'semantic_id' in result:
                        print(f"    Semantic ID: {result['semantic_id']}")
                
                if 'function_part_index' in result:
                    print(f"    Part Index: {result['function_part_index']}")
                
                print(f"    Content preview: {result['content'][:150]}...")
                print()
            
    except Exception as e:
        print(f"Error in search test: {e}")

def compare_strategies(project_dir, data_dir):
    """Compare different chunking strategies"""
    from code_rag.indexing.chunk_processor import ChunkProcessor
    from code_rag.indexing.file_processor import FileProcessor
    
    file_processor = FileProcessor(str(project_dir))
    
    # Find a Python file to analyze
    python_files = [f for f in file_processor.discover_files() if f.suffix == '.py']
    if not python_files:
        print("No Python files found for comparison")
        return
    
    test_file = python_files[0]
    try:
        print(f"\nAnalyzing: {test_file.relative_to(project_dir)}")
    except ValueError:
        print(f"\nAnalyzing: {test_file}")  # Fallback if relative path fails
    
    file_metadata = file_processor.create_file_metadata(test_file)
    if not file_metadata:
        print("Could not process file metadata")
        return
    
    chunk_processor = ChunkProcessor()
    analysis = chunk_processor.analyze_chunking_impact(test_file, file_metadata)
    
    print("\n--- Chunking Strategy Comparison ---")
    for strategy, stats in analysis.items():
        print(f"\n{strategy.upper()}:")
        if 'error' in stats:
            print(f"  Error: {stats['error']}")
        else:
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Average tokens: {stats['avg_tokens']:.1f}")
            print(f"  Max tokens: {stats['max_tokens']}")
            print(f"  Min tokens: {stats['min_tokens']}")
            print(f"  Oversized chunks: {stats['oversized_chunks']}")
            print(f"  Chunk types: {', '.join(stats['chunk_types'])}")

if __name__ == "__main__":
    main() 