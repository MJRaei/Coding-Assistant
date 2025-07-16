import logging

from .file_processor import FileProcessor
from .chunk_processor import ChunkProcessor
from .embedding_processor import EmbeddingProcessor
from .vector_store import VectorStore
from .chunking import ChunkingStrategy
from ..config import get_openai_api_key, DEFAULT_EMBEDDING_MODEL, DEFAULT_LOG_LEVEL, ensure_directories_exist


class IndexBuilder:
    """Main orchestrator for building code embeddings and vector indices"""
    
    def __init__(self, project_directory: str = None, data_directory: str = None, 
                 root_path: str = None, openai_api_key: str = None, 
                 embedding_model: str = None, 
                 chunking_strategy: ChunkingStrategy = ChunkingStrategy.STRUCTURE_PRESERVING):
        """
        Initialize the Index Builder
        
        Args:
            project_directory: Path to your code repository (new parameter)
            data_directory: Path to store index data (new parameter)
            root_path: Path to your code repository (legacy parameter, use project_directory instead)
            openai_api_key: OpenAI API key (if None, uses config to get from env)
            embedding_model: OpenAI embedding model to use (uses config default if None)
            chunking_strategy: Chunking strategy to use for processing files
        """
        if project_directory:
            self.root_path = project_directory
        elif root_path:
            self.root_path = root_path
        else:
            raise ValueError("Either project_directory or root_path must be provided")
        
        self.data_directory = data_directory
        
        api_key = openai_api_key or get_openai_api_key()
        embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        
        self.file_processor = FileProcessor(self.root_path)
        self.chunk_processor = ChunkProcessor(chunking_strategy=chunking_strategy)
        self.embedding_processor = EmbeddingProcessor(api_key, embedding_model)
        
        embedding_dim = self.embedding_processor.get_embedding_dimension()
        self.vector_store = VectorStore(dimension=embedding_dim)
        
        self.logger = logging.getLogger(__name__)
        
        logging.basicConfig(level=getattr(logging, DEFAULT_LOG_LEVEL))
        ensure_directories_exist()
    
    def build_index(self, save_to_data_dir: bool = True):
        """Build the complete RAG index"""
        self.logger.info("Starting RAG index building...")
        
        files = self.file_processor.discover_files()
        
        all_chunks = []
        for file_path in files:
            self.logger.info(f"Processing {file_path}")
            
            metadata = self.file_processor.create_file_metadata(file_path)
            if metadata is None:
                continue
            
            chunks = self.chunk_processor.process_file(file_path, metadata)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(files)} files")
        
        embeddings = self.embedding_processor.generate_embeddings(all_chunks)
        
        self.vector_store.add_chunks(all_chunks, embeddings)
        
        self.logger.info("RAG index building completed!")
        
        index_path = None
        if save_to_data_dir and self.data_directory:
            import os
            index_path = os.path.join(self.data_directory, "reporting_module_index")
            self.save_index(index_path)
        
        return index_path or {
            'total_files': len(files),
            'total_chunks': len(all_chunks),
            'vector_store_stats': self.vector_store.get_stats()
        }
    
    def save_index(self, filepath: str):
        """Save the built index to disk"""
        self.vector_store.save(filepath)
        self.logger.info(f"Index saved to {filepath}")
    
    def get_vector_store(self) -> VectorStore:
        """Get the vector store (for use by retrieval components)"""
        return self.vector_store 