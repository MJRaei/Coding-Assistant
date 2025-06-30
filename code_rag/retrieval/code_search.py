from typing import List, Dict, Any
from dataclasses import asdict
import logging

from ..indexing.vector_store import VectorStore
from ..indexing.embedding_processor import EmbeddingProcessor
from ..config import DEFAULT_EMBEDDING_MODEL, DEFAULT_SEARCH_K


class CodeSearch:
    """Pure semantic search functionality without AI generation"""
    
    def __init__(self, index_path: str = None, vector_store: VectorStore = None,
                 openai_api_key: str = None, embedding_model: str = None):
        """
        Initialize Code Search
        
        Args:
            index_path: Path to load pre-built index from
            vector_store: Pre-built vector store (alternative to index_path)
            openai_api_key: OpenAI API key for query embeddings
            embedding_model: OpenAI embedding model for queries (uses config default if None)
        """
        self.logger = logging.getLogger(__name__)
        
        self.embedding_processor = EmbeddingProcessor(openai_api_key, embedding_model)
        
        if vector_store is not None:
            self.vector_store = vector_store
        elif index_path is not None:
            embedding_dim = self.embedding_processor.get_embedding_dimension()
            self.vector_store = VectorStore(dimension=embedding_dim)
            self.vector_store.load(index_path)
        else:
            raise ValueError("Either index_path or vector_store must be provided")
    
    def search_similar_code(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Search for code chunks similar to the query
        
        Args:
            query: Search query
            k: Number of results to return (uses config default if None)
            
        Returns:
            List of formatted results with similarity scores
        """
        k = k or DEFAULT_SEARCH_K
        self.logger.info(f"Searching for: {query}")
        
        query_embedding = self.embedding_processor.generate_query_embedding(query)
        
        results = self.vector_store.search(query_embedding, k)
        
        formatted_results = []
        for chunk, score in results:
            formatted_results.append({
                'content': chunk.content,
                'score': score,
                'file_path': chunk.file_metadata.relative_path,
                'file_type': chunk.file_metadata.file_type,
                'chunk_type': chunk.chunk_type,
                'line_range': f"{chunk.start_line}-{chunk.end_line}",
                'functions': chunk.file_metadata.functions,
                'classes': chunk.file_metadata.classes,
                'metadata': asdict(chunk.file_metadata)
            })
        
        self.logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    def search_by_file_type(self, query: str, file_type: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Search for code chunks of a specific file type
        
        Args:
            query: Search query
            file_type: File extension (e.g., 'py', 'js', 'md')
            k: Number of results to return (uses config default if None)
            
        Returns:
            List of formatted results filtered by file type
        """
        k = k or DEFAULT_SEARCH_K
        all_results = self.search_similar_code(query, k * 3)
        
        filtered_results = [
            result for result in all_results 
            if result['file_type'] == file_type
        ]
        
        return filtered_results[:k]
    
    def search_by_function_name(self, function_name: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code chunks containing a specific function
        
        Args:
            function_name: Name of the function to search for
            k: Number of results to return
            
        Returns:
            List of chunks containing the function
        """
        query = f"function {function_name} definition implementation"
        all_results = self.search_similar_code(query, k * 2)
        
        filtered_results = [
            result for result in all_results
            if function_name in result['functions']
        ]
        
        return filtered_results[:k]
    
    def search_by_class_name(self, class_name: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code chunks containing a specific class
        
        Args:
            class_name: Name of the class to search for
            k: Number of results to return
            
        Returns:
            List of chunks containing the class
        """
        query = f"class {class_name} definition implementation"
        all_results = self.search_similar_code(query, k * 2)
        
        filtered_results = [
            result for result in all_results
            if class_name in result['classes']
        ]
        
        return filtered_results[:k]
    
    def search_in_folder(self, query: str, folder_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for code chunks within a specific folder
        
        Args:
            query: Search query
            folder_path: Folder path to search in
            k: Number of results to return
            
        Returns:
            List of formatted results from the specified folder
        """
        all_results = self.search_similar_code(query, k * 3)
        
        filtered_results = [
            result for result in all_results
            if result['metadata']['folder_path'].startswith(folder_path)
        ]
        
        return filtered_results[:k]
    
    def get_file_summary(self, file_path: str) -> Dict[str, Any]:
        """
        Get a summary of a specific file's contents
        
        Args:
            file_path: Relative path to the file
            
        Returns:
            Summary of the file's contents
        """
        all_chunks = [
            chunk for chunk in self.vector_store.chunks
            if chunk.file_metadata.relative_path == file_path
        ]
        
        if not all_chunks:
            return {'error': f'File {file_path} not found in index'}
        
        metadata = all_chunks[0].file_metadata
        
        return {
            'file_path': file_path,
            'file_type': metadata.file_type,
            'total_chunks': len(all_chunks),
            'total_lines': metadata.line_count,
            'file_size': metadata.file_size,
            'functions': metadata.functions,
            'classes': metadata.classes,
            'imports': metadata.imports,
            'last_modified': metadata.last_modified
        } 