"""
Simple example: Chunk a codebase and ask questions about it.

This example demonstrates:
1. Chunking a specific directory with ADAPTIVE_STRUCTURE strategy
2. Asking a simple question about the codebase
3. Getting detailed answers with source information

You can easily modify the QUESTION variable to ask different questions.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_rag.indexing.index_builder import IndexBuilder
from code_rag.indexing.chunking import ChunkingStrategy
from code_rag.retrieval.answer_generator import AnswerGenerator
from code_rag.retrieval.code_search import CodeSearch


def main():
    """Main function to chunk codebase and ask questions"""
    
    # =============================================================================
    # CONFIGURATION - Modify these as needed
    # =============================================================================
    
    # Directory to analyze
    TARGET_DIRECTORY = "/Users/mjraei/Desktop/Projects/OplaSmart/resources/EngReportComponents"
    
    # Question to ask (MODIFY THIS to ask different questions)
    QUESTION = "What is settings list model of displacement?"
    
    # Where to save the index
    CURRENT_PROJECT = Path(__file__).parent.parent
    DATA_DIR = CURRENT_PROJECT / "data"
    INDEX_NAME = "reporting_module_index"  # This matches the IndexBuilder default
    
    # =============================================================================
    # BUILD INDEX
    # =============================================================================
    
    print("🔍 Code RAG: Chunk and Query Example")
    print("=" * 50)
    print(f"Target Directory: {TARGET_DIRECTORY}")
    print(f"Question: {QUESTION}")
    print(f"Index will be saved to: {DATA_DIR / INDEX_NAME}")
    print()
    
    # Check if target directory exists
    if not Path(TARGET_DIRECTORY).exists():
        print(f"❌ Error: Target directory does not exist: {TARGET_DIRECTORY}")
        print("Please update the TARGET_DIRECTORY variable in this script.")
        return
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Build index with ADAPTIVE_STRUCTURE strategy (best for code understanding)
    print("🏗️  Building index with ADAPTIVE_STRUCTURE chunking strategy...")
    
    try:
        builder = IndexBuilder(
            project_directory=TARGET_DIRECTORY,
            data_directory=str(DATA_DIR),
            chunking_strategy=ChunkingStrategy.ADAPTIVE_STRUCTURE
        )
        
        # Build the index
        index_path = builder.build_index()
        
        # Get stats
        stats = builder.get_vector_store().get_stats()
        print(f"✅ Index built successfully!")
        print(f"   📁 Files processed: {stats['total_chunks']} chunks created")
        print(f"   💾 Saved to: {index_path}")
        print()
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        print("Make sure you have:")
        print("1. OPENAI_API_KEY set in your .env file")
        print("2. Required dependencies installed (pip install -r requirements.txt)")
        return
    
    # =============================================================================
    # ASK QUESTION
    # =============================================================================
    
    print("🤔 Asking question about the codebase...")
    print(f"Question: {QUESTION}")
    print()
    
    try:
        # Create answer generator
        answer_gen = AnswerGenerator(index_path=index_path)
        
        # Ask the question
        result = answer_gen.ask_question(QUESTION, k=5)
        
        # Display results
        print("💡 ANSWER:")
        print("-" * 30)
        print(result['answer'])
        print()
        
        # Show source information
        print("📚 SOURCES:")
        print("-" * 30)
        print(f"Files referenced: {len(result['source_files'])}")
        for i, file in enumerate(result['source_files'], 1):
            print(f"  {i}. {file}")
        print()
        
        if result['functions_mentioned']:
            print(f"Functions mentioned: {', '.join(result['functions_mentioned'])}")
        
        if result['classes_mentioned']:
            print(f"Classes mentioned: {', '.join(result['classes_mentioned'])}")
        
        print(f"Chunks analyzed: {result['chunks_used']}")
        print(f"Model used: {result['model_used']}")
        
    except Exception as e:
        print(f"❌ Error asking question: {e}")
        print("Make sure your OpenAI API key is valid and you have credits.")
        return
    
    # =============================================================================
    # SEMANTIC SEARCH EXAMPLE
    # =============================================================================
    
    print("\n" + "=" * 50)
    print("🔍 BONUS: Semantic Search Example")
    print("=" * 50)
    
    try:
        # Create code search instance
        searcher = CodeSearch(index_path=index_path)
        
        # Search for code related to the question
        search_query = "pump calculation implementation"
        search_results = searcher.search(search_query, top_k=3)
        
        print(f"Search Query: '{search_query}'")
        print(f"Found {len(search_results)} relevant code chunks:")
        print()
        
        for i, result in enumerate(search_results, 1):
            print(f"📄 Result {i} (Score: {result['score']:.3f})")
            print(f"   File: {result['file_path']}")
            print(f"   Lines: {result['line_range']}")
            print(f"   Type: {result['chunk_type']}")
            
            # Show enhanced metadata based on file type
            file_ext = result['file_path'].split('.')[-1].lower()
            
            if file_ext == 'qml':
                print(f"   🎯 QML Component:")
                if 'chunk_metadata' in result and result['chunk_metadata']:
                    metadata = result['chunk_metadata']
                    if 'component_type' in metadata:
                        print(f"     Type: {metadata['component_type']}")
                    if 'is_root_component' in metadata:
                        root_status = "Yes" if metadata['is_root_component'] else "No"
                        print(f"     Root: {root_status}")
                    if 'qml_imports' in metadata and metadata['qml_imports']:
                        print(f"     Imports: {', '.join(metadata['qml_imports'])}")
                    if 'is_complete_file' in metadata and metadata['is_complete_file']:
                        print(f"     Complete File: Yes")
                if 'parent_class' in result:
                    print(f"     Parent: {result['parent_class']}")
                if 'semantic_id' in result:
                    print(f"     ID: {result['semantic_id']}")
                    
            elif file_ext == 'py':
                print(f"   🐍 Python Code:")
                if 'chunk_metadata' in result and result['chunk_metadata']:
                    metadata = result['chunk_metadata']
                    if 'element_type' in metadata:
                        print(f"     Type: {metadata['element_type'].title()}")
                    if 'decorators' in metadata and metadata['decorators']:
                        print(f"     Decorators: {', '.join(metadata['decorators'])}")
                if 'parent_class' in result:
                    print(f"     Class: {result['parent_class']}")
                if 'parent_function' in result:
                    print(f"     Function: {result['parent_function']}")
                if 'semantic_id' in result:
                    print(f"     ID: {result['semantic_id']}")
                    
            else:
                # Generic metadata for other file types
                if 'parent_class' in result:
                    print(f"   Class: {result['parent_class']}")
                if 'parent_function' in result:
                    print(f"   Function: {result['parent_function']}")
                if 'semantic_id' in result:
                    print(f"   ID: {result['semantic_id']}")
            
            print(f"   Preview: {result['content'][:150]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error in semantic search: {e}")
    
    # =============================================================================
    # COMPLETION
    # =============================================================================
    
    print("=" * 50)
    print("✅ Example completed successfully!")
    print()
    print("💡 To ask different questions:")
    print("   1. Modify the QUESTION variable at the top of this script")
    print("   2. Run the script again (it will reuse the existing index)")
    print()
    print("🗂️  Index files created:")
    print(f"   • {DATA_DIR / INDEX_NAME}.faiss")
    print(f"   • {DATA_DIR / INDEX_NAME}.chunks")
    
    # Show QML chunking support
    print("\n🎯 QML Chunking Support:")
    print("   QML files will now be chunked with component-aware strategy!")
    print("   Supported QML features:")
    print("   • Component definitions (Rectangle, Button, etc.)")
    print("   • Property declarations and bindings")
    print("   • Signal declarations and handlers")
    print("   • JavaScript functions within QML")
    print("   • Import statements and pragmas")


if __name__ == "__main__":
    main() 