from typing import List, Dict, Any, Tuple
import logging

from openai import OpenAI

from ..indexing.vector_store import VectorStore
from ..indexing.embedding_processor import EmbeddingProcessor
from ..config import DEFAULT_EMBEDDING_MODEL, DEFAULT_CHAT_MODEL, DEFAULT_MAX_CHUNKS_FOR_CONTEXT


class AnswerGenerator:
    """Generates answers using OpenAI chat completion based on retrieved chunks"""
    
    def __init__(self, index_path: str = None, vector_store: VectorStore = None, 
                 openai_api_key: str = None, embedding_model: str = None,
                 chat_model: str = None):
        """
        Initialize OpenAI answer generator
        
        Args:
            index_path: Path to load pre-built index from
            vector_store: Pre-built vector store (alternative to index_path)
            openai_api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            embedding_model: OpenAI embedding model for queries (uses config default if None)
            chat_model: OpenAI chat model (uses config default if None)
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.chat_model = chat_model or DEFAULT_CHAT_MODEL
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
    
    def create_system_prompt(self) -> str:
        """Create the system prompt for the code assistant"""
        return """You are an expert code assistant specializing in helping developers understand and work with their codebase.
            Your role is to:
            1. Analyze the provided code chunks and answer questions about them
            2. Provide clear, accurate explanations of how code works
            3. Suggest improvements or modifications when asked
            4. Help with debugging and troubleshooting
            5. Explain code relationships and dependencies

            Guidelines:
            - Always base your answers on the provided code chunks
            - If the code chunks don't contain enough information to answer the question, say so clearly
            - Provide specific examples from the code when possible
            - Include relevant file paths and line numbers when referencing code
            - If suggesting code changes, make them clear and actionable
            - For complex questions, break down your answer into logical steps

            Remember: You're helping developers understand their own codebase, so be practical and specific."""
    
    def format_chunks_for_context(self, chunks_with_scores: List[Tuple[Any, float]], max_chunks: int = None) -> str:
        """Format retrieved chunks into context for the LLM"""
        max_chunks = max_chunks or DEFAULT_MAX_CHUNKS_FOR_CONTEXT
        context_parts = []
        
        for i, (chunk, score) in enumerate(chunks_with_scores[:max_chunks]):
            metadata = chunk.file_metadata
            
            context_part = f"""
                === CODE CHUNK {i+1} (Relevance Score: {score:.3f}) ===
                File: {metadata.relative_path}
                Lines: {chunk.start_line}-{chunk.end_line}
                Type: {chunk.chunk_type}
                Functions: {', '.join(metadata.functions) if metadata.functions else 'None'}
                Classes: {', '.join(metadata.classes) if metadata.classes else 'None'}

                Code:
                ```{metadata.file_type}
                {chunk.content}
                ```
                """
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def ask_question(self, question: str, k: int = 5, 
                    conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Ask a question about the codebase and get a comprehensive answer
        
        Args:
            question: The question about your code
            k: Number of relevant code chunks to retrieve
            conversation_history: Previous conversation messages for context
        
        Returns:
            Dict containing the answer and supporting information
        """
        self.logger.info(f"Processing question: {question}")
        
        query_embedding = self.embedding_processor.generate_query_embedding(question)
        
        retrieved_chunks = self.vector_store.search(query_embedding, k)
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant code chunks to answer your question. The codebase might not contain information related to your query.",
                'source_files': [],
                'functions_mentioned': [],
                'classes_mentioned': [],
                'chunks_used': 0
            }
        
        result = self.generate_answer(question, retrieved_chunks, conversation_history)
        
        self.logger.info(f"Generated answer using {len(retrieved_chunks)} code chunks")
        return result
    
    def generate_answer(self, question: str, retrieved_chunks: List[Tuple[Any, float]], 
                       conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Generate an answer based on the question and retrieved code chunks
        
        Args:
            question: The user's question
            retrieved_chunks: List of (chunk, score) tuples from vector search
            conversation_history: Previous conversation for context
        
        Returns:
            Dict containing the answer and metadata
        """
        try:
            context = self.format_chunks_for_context(retrieved_chunks)
            
            messages = [{"role": "system", "content": self.create_system_prompt()}]
            
            if conversation_history:
                messages.extend(conversation_history[-6:])
            
            user_message = f"""Based on the following code chunks from the codebase, please answer this question:

                QUESTION: {question}

                RELEVANT CODE CHUNKS:
                {context}

                Please provide a comprehensive answer based on the code provided above."""
            
            messages.append({"role": "user", "content": user_message})
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1,
                max_tokens=1500,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            source_files = list(set([chunk.file_metadata.relative_path for chunk, _ in retrieved_chunks]))
            functions_mentioned = []
            classes_mentioned = []
            
            for chunk, _ in retrieved_chunks:
                functions_mentioned.extend(chunk.file_metadata.functions)
                classes_mentioned.extend(chunk.file_metadata.classes)
            
            return {
                'answer': answer,
                'source_files': source_files,
                'functions_mentioned': list(set(functions_mentioned)),
                'classes_mentioned': list(set(classes_mentioned)),
                'chunks_used': len(retrieved_chunks),
                'model_used': self.chat_model,
                'retrieved_chunks': [
                    {
                        'file_path': chunk.file_metadata.relative_path,
                        'score': score,
                        'line_range': f"{chunk.start_line}-{chunk.end_line}",
                        'chunk_type': chunk.chunk_type
                    }
                    for chunk, score in retrieved_chunks
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating answer: {e}")
            return {
                'answer': f"I encountered an error while generating the answer: {str(e)}",
                'error': True,
                'source_files': [],
                'functions_mentioned': [],
                'classes_mentioned': [],
                'chunks_used': 0
            }
    
    def chat(self, conversation_history: List[Dict[str, str]], new_question: str, k: int = 5) -> Dict[str, Any]:
        """
        Continue a conversation about your codebase
        
        Args:
            conversation_history: List of previous messages [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
            new_question: New question to ask
            k: Number of relevant chunks to retrieve
        
        Returns:
            Dict containing the answer and supporting information
        """
        return self.ask_question(new_question, k, conversation_history) 