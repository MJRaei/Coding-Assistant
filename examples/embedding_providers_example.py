"""
Example demonstrating the new flexible embedding providers.

This example shows how to use different embedding providers:
1. OpenAI embeddings (cloud-based)
2. Hugging Face embeddings (local)  
3. Ollama embeddings (local)
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_rag.indexing.index_builder import IndexBuilder
from code_rag.indexing.chunking import ChunkingStrategy
from code_rag.indexing.embedding_processor import EmbeddingProcessor
from code_rag.retrieval.answer_generator import AnswerGenerator


def test_openai_embeddings():
    """Test OpenAI embeddings (requires API key)"""
    print("\n=== Testing OpenAI Embeddings ===")
    
    try:
        # Create OpenAI processor
        processor = EmbeddingProcessor.create_openai_processor(
            model_name='text-embedding-3-small'
        )
        
        print(f"✅ OpenAI processor created successfully")
        print(f"Provider info: {processor.get_provider_info()}")
        
        return processor
        
    except Exception as e:
        print(f"❌ OpenAI processor failed: {e}")
        return None


def test_huggingface_embeddings():
    """Test Hugging Face embeddings (local, no API key needed)"""
    print("\n=== Testing Hugging Face Embeddings ===")
    
    try:
        # Create Hugging Face processor
        processor = EmbeddingProcessor.create_huggingface_processor(
            model_name='all-MiniLM-L6-v2',
            device='cpu'
        )
        
        print(f"✅ Hugging Face processor created successfully")
        print(f"Provider info: {processor.get_provider_info()}")
        
        return processor
        
    except Exception as e:
        print(f"❌ Hugging Face processor failed: {e}")
        print("Make sure to install: pip install sentence-transformers")
        return None


def test_ollama_embeddings():
    """Test Ollama embeddings (local, requires Ollama server)"""
    print("\n=== Testing Ollama Embeddings ===")
    
    try:
        # Create Ollama processor
        processor = EmbeddingProcessor.create_ollama_processor(
            model_name='nomic-embed-text',
            host='http://localhost:11434'
        )
        
        print(f"✅ Ollama processor created successfully")
        print(f"Provider info: {processor.get_provider_info()}")
        
        return processor
        
    except Exception as e:
        print(f"❌ Ollama processor failed: {e}")
        print("Make sure Ollama is running and the model is installed:")
        print("  - Install Ollama: https://ollama.ai/")
        print("  - Run: ollama pull nomic-embed-text")
        return None


def build_index_with_provider(provider_name, embedding_config, target_directory):
    """Build an index using a specific embedding provider"""
    print(f"\n=== Building Index with {provider_name} ===")
    
    try:
        # Set up data directory
        current_project = Path(__file__).parent.parent
        data_dir = current_project / "data" / f"{provider_name.lower()}_embeddings"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create index builder with embedding provider configuration
        builder = IndexBuilder(
            project_directory=target_directory,
            data_directory=str(data_dir),
            chunking_strategy=ChunkingStrategy.ADAPTIVE_STRUCTURE,
            **embedding_config
        )
        
        # Build the index
        index_path = builder.build_index()
        
        print(f"✅ Index built successfully with {provider_name}")
        print(f"   Index saved to: {index_path}")
        
        return index_path
        
    except Exception as e:
        print(f"❌ Error building index with {provider_name}: {e}")
        return None


def main():
    """Main function to demonstrate different embedding providers"""
    
    # Configuration
    target_directory = "/Users/mjraei/Desktop/Projects/OplaSmart/resources/EngReportComponents"
    
    print("🚀 Embedding Providers Demo")
    print("=" * 50)
    print(f"Target directory: {target_directory}")
    
    # Check if target directory exists
    if not Path(target_directory).exists():
        print(f"❌ Target directory not found: {target_directory}")
        print("Please update the target_directory variable in this script.")
        return
    
    # Test different providers
    providers = []
    
    # Test OpenAI
    openai_processor = test_openai_embeddings()
    if openai_processor:
        providers.append(("OpenAI", {
            'embedding_provider': 'openai',
            'embedding_model': 'text-embedding-3-small'
        }))
    
    # Test Hugging Face
    hf_processor = test_huggingface_embeddings()
    if hf_processor:
        providers.append(("HuggingFace", {
            'embedding_provider': 'huggingface',
            'embedding_model': 'all-MiniLM-L6-v2',
            'embedding_config': {'device': 'cpu'}
        }))
    
    # Test Ollama
    ollama_processor = test_ollama_embeddings()
    if ollama_processor:
        providers.append(("Ollama", {
            'embedding_provider': 'ollama',
            'embedding_model': 'nomic-embed-text',
            'embedding_config': {'host': 'http://localhost:11434'}
        }))
    
    if not providers:
        print("❌ No embedding providers are available!")
        return
    
    # Build indices with different providers
    print(f"\n📊 Building indices with {len(providers)} different providers...")
    
    for provider_name, embedding_config in providers:
        index_path = build_index_with_provider(provider_name, embedding_config, target_directory)
        
        if index_path:
            print(f"✅ {provider_name} index ready for use!")
    
    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Choose your preferred embedding provider")
    print("2. Set the appropriate environment variables")
    print("3. Use the provider in your production code")
    print("\nEnvironment variables:")
    print("- EMBEDDING_PROVIDER=openai|huggingface|ollama")
    print("- EMBEDDING_MODEL=<model_name>")
    print("- Provider-specific settings (API keys, device, etc.)")
    print("\nExample usage:")
    print("# OpenAI")
    print("export EMBEDDING_PROVIDER=openai")
    print("export EMBEDDING_MODEL=text-embedding-3-small")
    print("export OPENAI_API_KEY=your_api_key")
    print("\n# Hugging Face")
    print("export EMBEDDING_PROVIDER=huggingface")
    print("export EMBEDDING_MODEL=all-MiniLM-L6-v2")
    print("export HF_DEVICE=cpu")
    print("\n# Ollama")
    print("export EMBEDDING_PROVIDER=ollama")
    print("export EMBEDDING_MODEL=nomic-embed-text")
    print("export OLLAMA_HOST=http://localhost:11434")


if __name__ == "__main__":
    main() 