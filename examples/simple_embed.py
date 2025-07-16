"""
Simple example: Chunk and embed a project directory and save to data folder.

This example demonstrates the core functionality:
1. Takes a project directory as input
2. Chunks and embeds the code using ADAPTIVE_STRUCTURE strategy
3. Saves the index to the data folder

Modify the project_dir variable to point to your target directory.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_rag.indexing.index_builder import IndexBuilder
from code_rag.indexing.chunking import ChunkingStrategy


def main():
    """Main function to chunk and embed a project directory."""
    
    # =============================================================================
    # CONFIGURATION - Modify this path to your target directory
    # =============================================================================
    project_dir = Path("/Users/mjraei/Desktop/Projects/OplaSmart/engReports")
    
    # =============================================================================
    # SETUP
    # =============================================================================
    
    # Define data directory (where the index will be saved)
    current_project = Path(__file__).parent.parent
    data_dir = current_project / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("🔍 Code RAG: Simple Project Embedding")
    print("=" * 50)
    print(f"Target directory: {project_dir}")
    print(f"Data directory: {data_dir}")
    print()
    
    # Check if target directory exists
    if not project_dir.exists():
        print(f"❌ Error: Target directory does not exist: {project_dir}")
        print("Please update the project_dir variable in this script.")
        return
    
    # =============================================================================
    # BUILD INDEX
    # =============================================================================
    
    print("🏗️  Building index with ADAPTIVE_STRUCTURE chunking strategy...")
    
    try:
        builder = IndexBuilder(
            project_directory=str(project_dir),
            data_directory=str(data_dir),
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
        
        print("🎯 Index files created:")
        index_name = "reporting_module_index"  # Default name used by IndexBuilder
        print(f"   • {data_dir / f'{index_name}.faiss'}")
        print(f"   • {data_dir / f'{index_name}.chunks'}")
        
    except Exception as e:
        print(f"❌ Error building index: {e}")
        print("Make sure you have:")
        print("1. OPENAI_API_KEY set in your .env file")
        print("2. Required dependencies installed (pip install -r requirements.txt)")
        return
    
    print("\n✅ Project successfully chunked and embedded!")
    print("You can now use the index for code search and question answering.")


if __name__ == "__main__":
    main() 