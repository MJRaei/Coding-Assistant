"""
Simple usage example of the Code RAG system.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_rag.indexing.index_builder import IndexBuilder
from code_rag.indexing.chunking import ChunkingStrategy
from code_rag.retrieval.code_search import CodeSearch
from code_rag.retrieval.answer_generator import AnswerGenerator

def main():
    """Main function to demonstrate the Code RAG system."""
    
    # Target directory to chunk
    project_dir = Path("/Users/mjraei/Desktop/Projects/OplaSmart/engReports")
    
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
    
    # Try FUNCTION_AWARE strategy first (new!)
    print("\n--- Testing FUNCTION_AWARE Strategy ---")
    builder = IndexBuilder(
        project_directory=str(project_dir),
        data_directory=str(data_dir),
        chunking_strategy=ChunkingStrategy.FUNCTION_AWARE
    )
    
    try:
        index_path = builder.build_index()
        print(f"✓ FUNCTION_AWARE index built successfully: {index_path}")
        
        # Inspect all chunks to see the new metadata
        print("\n--- Inspecting FUNCTION_AWARE chunks ---")
        inspect_chunks(builder)
        
        # Test search with FUNCTION_AWARE index
        print("\n--- Testing search with FUNCTION_AWARE index ---")
        test_search(data_dir, "function_aware")
        
    except Exception as e:
        print(f"✗ Error with FUNCTION_AWARE strategy: {e}")
        print("This might be due to missing dependencies or configuration issues.")
    
    # Compare with SEMANTIC_FIRST strategy
    print("\n--- Testing SEMANTIC_FIRST Strategy (for comparison) ---")
    builder_semantic = IndexBuilder(
        project_directory=str(project_dir),
        data_directory=str(data_dir),
        chunking_strategy=ChunkingStrategy.SEMANTIC_FIRST
    )
    
    try:
        index_path_semantic = builder_semantic.build_index()
        print(f"✓ SEMANTIC_FIRST index built successfully: {index_path_semantic}")
        
        # Test search with SEMANTIC_FIRST index
        print("\n--- Testing search with SEMANTIC_FIRST index ---")
        test_search(data_dir, "semantic_first")
        
    except Exception as e:
        print(f"✗ Error with SEMANTIC_FIRST strategy: {e}")
    
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
        print(f"Functions: {metadata.functions}")
        print(f"Classes: {metadata.classes}")
        print(f"Imports: {metadata.imports}")
        
        # Chunk info
        print(f"Chunk lines: {chunk.start_line}-{chunk.end_line}")
        print(f"Token count: {chunk.tokens_count}")
        print(f"Chunk type: {chunk.chunk_type}")
        
        # FUNCTION_AWARE enhanced metadata
        if chunk.parent_class or chunk.parent_function or chunk.semantic_id:
            print(f"\n🎯 FUNCTION_AWARE Metadata:")
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
                print(f"Extra Metadata: {chunk.chunk_metadata}")
        
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
                
                # Show enhanced FUNCTION_AWARE metadata if available
                if 'parent_class' in result:
                    print(f"    Parent Class: {result['parent_class']}")
                if 'parent_function' in result:
                    print(f"    Parent Function: {result['parent_function']}")
                if 'semantic_id' in result:
                    print(f"    Semantic ID: {result['semantic_id']}")
                if 'function_part_index' in result:
                    print(f"    Function Part: {result['function_part_index']}")
                
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