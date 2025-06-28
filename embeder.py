import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# For embeddings and chat completion - OpenAI API
import openai
from openai import OpenAI
import numpy as np
import time

# For vector storage - you can replace with your preferred vector DB
# Example: Pinecone, Weaviate, Chroma, or simple FAISS
import faiss
import pickle

@dataclass
class FileMetadata:
    """Metadata for each processed file"""
    file_path: str
    relative_path: str
    file_name: str
    file_type: str
    folder_path: str
    file_size: int
    last_modified: str
    content_hash: str
    imports: List[str]
    functions: List[str]
    classes: List[str]
    line_count: int

@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata"""
    content: str
    file_metadata: FileMetadata
    chunk_index: int
    chunk_type: str  # 'full_file', 'function', 'class', 'section'
    start_line: int
    end_line: int
    tokens_count: int

class FileProcessor:
    """Handles file discovery and content extraction"""
    
    def __init__(self, root_path: str, excluded_dirs: List[str] = None, included_extensions: List[str] = None):
        self.root_path = Path(root_path)
        self.excluded_dirs = excluded_dirs or ['__pycache__', '.git', '.pytest_cache', 'node_modules', '.venv', 'venv']
        self.included_extensions = included_extensions or ['.py', '.md', '.txt', '.yml', '.yaml', '.json', '.sql']
        self.logger = logging.getLogger(__name__)
    
    def discover_files(self) -> List[Path]:
        """Recursively discover all relevant files"""
        discovered_files = []
        
        for root, dirs, files in os.walk(self.root_path):
            # Remove excluded directories from traversal
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.included_extensions:
                    discovered_files.append(file_path)
        
        self.logger.info(f"Discovered {len(discovered_files)} files")
        return discovered_files
    
    def extract_python_metadata(self, content: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract imports, functions, and classes from Python code"""
        imports = []
        functions = []
        classes = []
        
        try:
            # Extract imports
            import_pattern = r'^(?:from\s+[\w.]+\s+)?import\s+[\w.,\s*]+|^from\s+[\w.]+\s+import\s+[\w.,\s*]+'
            imports = re.findall(import_pattern, content, re.MULTILINE)
            
            # Extract function definitions
            function_pattern = r'^def\s+(\w+)\s*\('
            functions = re.findall(function_pattern, content, re.MULTILINE)
            
            # Extract class definitions
            class_pattern = r'^class\s+(\w+)(?:\s*\([^)]*\))?\s*:'
            classes = re.findall(class_pattern, content, re.MULTILINE)
            
        except Exception as e:
            self.logger.warning(f"Error extracting Python metadata: {e}")
        
        return imports, functions, classes
    
    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with encoding handling"""
        encodings = ['utf-8', 'latin-1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {e}")
                return None
        
        self.logger.error(f"Could not decode {file_path} with any encoding")
        return None
    
    def create_file_metadata(self, file_path: Path) -> Optional[FileMetadata]:
        """Create metadata for a single file"""
        try:
            content = self.read_file_content(file_path)
            if content is None:
                return None
            
            # Basic file info
            stat = file_path.stat()
            relative_path = file_path.relative_to(self.root_path)
            
            # Extract Python-specific metadata
            imports, functions, classes = [], [], []
            if file_path.suffix == '.py':
                imports, functions, classes = self.extract_python_metadata(content)
            
            return FileMetadata(
                file_path=str(file_path),
                relative_path=str(relative_path),
                file_name=file_path.name,
                file_type=file_path.suffix[1:] if file_path.suffix else 'unknown',
                folder_path=str(file_path.parent.relative_to(self.root_path)),
                file_size=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                content_hash=hashlib.md5(content.encode()).hexdigest(),
                imports=imports,
                functions=functions,
                classes=classes,
                line_count=len(content.splitlines())
            )
        
        except Exception as e:
            self.logger.error(f"Error creating metadata for {file_path}: {e}")
            return None

class ChunkProcessor:
    """Handles content chunking strategies"""
    
    def __init__(self, max_tokens: int = 2000, overlap_tokens: int = 200):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.logger = logging.getLogger(__name__)
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def chunk_by_functions(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Chunk Python files by function/class boundaries"""
        chunks = []
        lines = content.splitlines()
        
        # Find function and class boundaries
        boundaries = []
        for i, line in enumerate(lines):
            if re.match(r'^(def|class)\s+\w+', line.strip()):
                boundaries.append(i)
        
        # Add file start and end
        boundaries = [0] + boundaries + [len(lines)]
        
        for i in range(len(boundaries) - 1):
            start_line = boundaries[i]
            end_line = boundaries[i + 1]
            chunk_content = '\n'.join(lines[start_line:end_line])
            
            if chunk_content.strip():  # Skip empty chunks
                chunk_type = 'function' if 'def ' in lines[start_line] else 'class' if 'class ' in lines[start_line] else 'section'
                
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=i,
                    chunk_type=chunk_type,
                    start_line=start_line + 1,
                    end_line=end_line,
                    tokens_count=self.estimate_tokens(chunk_content)
                ))
        
        return chunks
    
    def chunk_by_size(self, content: str, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Chunk content by token size with overlap"""
        chunks = []
        lines = content.splitlines()
        current_chunk = []
        current_tokens = 0
        start_line = 1
        
        for i, line in enumerate(lines):
            line_tokens = self.estimate_tokens(line)
            
            if current_tokens + line_tokens > self.max_tokens and current_chunk:
                # Create chunk
                chunk_content = '\n'.join(current_chunk)
                chunks.append(CodeChunk(
                    content=chunk_content,
                    file_metadata=file_metadata,
                    chunk_index=len(chunks),
                    chunk_type='section',
                    start_line=start_line,
                    end_line=i,
                    tokens_count=current_tokens
                ))
                
                # Handle overlap
                overlap_lines = []
                overlap_tokens = 0
                for j in range(len(current_chunk) - 1, -1, -1):
                    if overlap_tokens + self.estimate_tokens(current_chunk[j]) <= self.overlap_tokens:
                        overlap_lines.insert(0, current_chunk[j])
                        overlap_tokens += self.estimate_tokens(current_chunk[j])
                    else:
                        break
                
                current_chunk = overlap_lines + [line]
                current_tokens = overlap_tokens + line_tokens
                start_line = i + 1 - len(overlap_lines)
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append(CodeChunk(
                content=chunk_content,
                file_metadata=file_metadata,
                chunk_index=len(chunks),
                chunk_type='section',
                start_line=start_line,
                end_line=len(lines),
                tokens_count=current_tokens
            ))
        
        return chunks
    
    def process_file(self, file_path: Path, file_metadata: FileMetadata) -> List[CodeChunk]:
        """Process a single file into chunks"""
        content = FileProcessor(str(file_path.parent)).read_file_content(file_path)
        if not content:
            return []
        
        # Decide chunking strategy based on file type and size
        estimated_tokens = self.estimate_tokens(content)
        
        if estimated_tokens <= self.max_tokens:
            # Small file - keep as single chunk
            return [CodeChunk(
                content=content,
                file_metadata=file_metadata,
                chunk_index=0,
                chunk_type='full_file',
                start_line=1,
                end_line=file_metadata.line_count,
                tokens_count=estimated_tokens
            )]
        elif file_metadata.file_type == 'py':
            # Python file - chunk by functions
            return self.chunk_by_functions(content, file_metadata)
        else:
            # Other files - chunk by size
            return self.chunk_by_size(content, file_metadata)

class EmbeddingProcessor:
    """Handles embedding generation using OpenAI"""
    
    def __init__(self, api_key: str = None, model_name: str = 'text-embedding-3-small'):
        """
        Initialize OpenAI embedding processor
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model_name: OpenAI embedding model ('text-embedding-3-small' or 'text-embedding-3-large')
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Model dimensions for vector store initialization
        self.model_dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536
        }
        
        if model_name not in self.model_dimensions:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding model"""
        return self.model_dimensions[self.model_name]
    
    def create_embedding_text(self, chunk: CodeChunk) -> str:
        """Create enriched text for embedding"""
        metadata = chunk.file_metadata
        
        # Build context string
        context_parts = [
            f"File: {metadata.relative_path}",
            f"Type: {metadata.file_type}",
            f"Folder: {metadata.folder_path}",
        ]
        
        if metadata.functions:
            context_parts.append(f"Functions: {', '.join(metadata.functions[:5])}")
        
        if metadata.classes:
            context_parts.append(f"Classes: {', '.join(metadata.classes[:3])}")
        
        if metadata.imports:
            context_parts.append(f"Imports: {', '.join(metadata.imports[:5])}")
        
        context = " | ".join(context_parts)
        
        # Combine context with content
        return f"{context}\n\n{chunk.content}"
    
    def _call_openai_embedding(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Call OpenAI API with rate limiting and batching"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                    
                    response = self.client.embeddings.create(
                        input=batch,
                        model=self.model_name
                    )
                    
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Rate limiting - OpenAI allows 3000 RPM for text-embedding-3-small
                    if i + batch_size < len(texts):
                        time.sleep(0.1)  # Small delay between batches
                    
                    break
                    
                except openai.RateLimitError as e:
                    retry_count += 1
                    wait_time = min(60, (2 ** retry_count))
                    self.logger.warning(f"Rate limit hit, waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                    time.sleep(wait_time)
                    
                except openai.APIError as e:
                    retry_count += 1
                    self.logger.error(f"OpenAI API error: {e}")
                    if retry_count >= max_retries:
                        raise
                    time.sleep(2 ** retry_count)
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error in embedding generation: {e}")
                    raise
        
        return all_embeddings
    
    def generate_embeddings(self, chunks: List[CodeChunk]) -> List[np.ndarray]:
        """Generate embeddings for chunks using OpenAI"""
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks using {self.model_name}")
        
        texts = [self.create_embedding_text(chunk) for chunk in chunks]
        
        # Call OpenAI API
        embeddings_list = self._call_openai_embedding(texts)
        
        # Convert to numpy arrays
        embeddings = [np.array(emb, dtype=np.float32) for emb in embeddings_list]
        
        self.logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        try:
            response = self.client.embeddings.create(
                input=[query],
                model=self.model_name
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise

class AnswerGenerator:
    """Generates answers using OpenAI chat completion based on retrieved chunks"""
    
    def __init__(self, api_key: str = None, model_name: str = 'gpt-4o-mini'):
        """
        Initialize OpenAI answer generator
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model_name: OpenAI chat model ('gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo')
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
    
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
    
    def format_chunks_for_context(self, chunks_with_scores: List[Tuple[Any, float]], max_chunks: int = 5) -> str:
        """Format retrieved chunks into context for the LLM"""
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
            # Format the code context
            context = self.format_chunks_for_context(retrieved_chunks)
            
            # Build the conversation
            messages = [{"role": "system", "content": self.create_system_prompt()}]
            
            # Add conversation history if provided
            if conversation_history:
                messages.extend(conversation_history[-6:])  # Keep last 6 messages for context
            
            # Add the current question with context
            user_message = f"""Based on the following code chunks from the codebase, please answer this question:

                QUESTION: {question}

                RELEVANT CODE CHUNKS:
                {context}

                Please provide a comprehensive answer based on the code provided above."""
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,  # Low temperature for more consistent, factual responses
                max_tokens=1500,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Extract metadata from retrieved chunks
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
                'model_used': self.model_name,
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
        
class VectorStore:
    """Simple FAISS-based vector storage"""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks: List[CodeChunk] = []
        self.embeddings: List[np.ndarray] = []
    
    def add_chunks(self, chunks: List[CodeChunk], embeddings: List[np.ndarray]):
        """Add chunks and their embeddings to the store"""
        # Normalize embeddings for cosine similarity
        normalized_embeddings = []
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 0:
                normalized_embeddings.append(emb / norm)
            else:
                normalized_embeddings.append(emb)
        
        embeddings_array = np.array(normalized_embeddings).astype('float32')
        self.index.add(embeddings_array)
        self.chunks.extend(chunks)
        self.embeddings.extend(normalized_embeddings)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Search for similar chunks"""
        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, filepath: str):
        """Save the vector store to disk"""
        faiss.write_index(self.index, f"{filepath}.faiss")
        with open(f"{filepath}.chunks", 'wb') as f:
            pickle.dump(self.chunks, f)
    
    def load(self, filepath: str):
        """Load the vector store from disk"""
        self.index = faiss.read_index(f"{filepath}.faiss")
        with open(f"{filepath}.chunks", 'rb') as f:
            self.chunks = pickle.load(f)

class CodeRAGSystem:
    """Main orchestrator for the Code RAG system with full Q&A capabilities"""
    
    def __init__(self, root_path: str, openai_api_key: str = None, 
                 embedding_model: str = 'text-embedding-3-small',
                 chat_model: str = 'gpt-4o-mini'):
        """
        Initialize the Code RAG system
        
        Args:
            root_path: Path to your code repository
            openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
            embedding_model: OpenAI embedding model to use
            chat_model: OpenAI chat model for generating answers
        """
        self.root_path = root_path
        self.file_processor = FileProcessor(root_path)
        self.chunk_processor = ChunkProcessor()
        self.embedding_processor = EmbeddingProcessor(openai_api_key, embedding_model)
        self.answer_generator = AnswerGenerator(openai_api_key, chat_model)
        
        # Initialize vector store with correct dimensions
        embedding_dim = self.embedding_processor.get_embedding_dimension()
        self.vector_store = VectorStore(dimension=embedding_dim)
        
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def build_index(self):
        """Build the complete RAG index"""
        self.logger.info("Starting RAG index building...")
        
        # 1. Discover files
        files = self.file_processor.discover_files()
        
        # 2. Process files and create chunks
        all_chunks = []
        for file_path in files:
            self.logger.info(f"Processing {file_path}")
            
            # Create metadata
            metadata = self.file_processor.create_file_metadata(file_path)
            if metadata is None:
                continue
            
            # Create chunks
            chunks = self.chunk_processor.process_file(file_path, metadata)
            all_chunks.extend(chunks)
        
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(files)} files")
        
        # 3. Generate embeddings using OpenAI
        embeddings = self.embedding_processor.generate_embeddings(all_chunks)
        
        # 4. Store in vector database
        self.vector_store.add_chunks(all_chunks, embeddings)
        
        self.logger.info("RAG index building completed!")
    
    def ask_question(self, question: str, k: int = 5, 
                    conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Ask a question about your codebase and get a comprehensive answer
        
        Args:
            question: The question about your code
            k: Number of relevant code chunks to retrieve
            conversation_history: Previous conversation messages for context
        
        Returns:
            Dict containing the answer and supporting information
        """
        self.logger.info(f"Processing question: {question}")
        
        # 1. Generate query embedding using OpenAI
        query_embedding = self.embedding_processor.generate_query_embedding(question)
        
        # 2. Search for similar chunks
        retrieved_chunks = self.vector_store.search(query_embedding, k)
        
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant code chunks to answer your question. The codebase might not contain information related to your query.",
                'source_files': [],
                'functions_mentioned': [],
                'classes_mentioned': [],
                'chunks_used': 0
            }
        
        # 3. Generate answer using retrieved chunks
        result = self.answer_generator.generate_answer(
            question, retrieved_chunks, conversation_history
        )
        
        self.logger.info(f"Generated answer using {len(retrieved_chunks)} code chunks")
        return result
    
    def query(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Legacy method - returns raw chunks (kept for backward compatibility)
        Use ask_question() for full RAG experience
        """
        query_embedding = self.embedding_processor.generate_query_embedding(question)
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
        
        return formatted_results
    
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
    
    def save_index(self, filepath: str):
        """Save the built index to disk"""
        self.vector_store.save(filepath)
        self.logger.info(f"Index saved to {filepath}")
    
    def load_index(self, filepath: str):
        """Load a previously built index"""
        self.vector_store.load(filepath)
        self.logger.info(f"Index loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system with OpenAI
    rag_system = CodeRAGSystem(
        root_path=r"C:\Users\mjavadraei\Projects\OplaSmart\engReports",
        openai_api_key="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",  # or set OPENAI_API_KEY env var
        embedding_model="text-embedding-3-small",
        chat_model="gpt-4o-mini"
    )
    
    # Build the index
    rag_system.build_index()
    
    # Save the index
    rag_system.save_index("reporting_module_index")
    
    # Ask questions about your code
    result = rag_system.ask_question("How the pumps output is calculated in the reports?")
    
    print("ANSWER:")
    print(result['answer'])
    print(f"\nSource files: {', '.join(result['source_files'])}")
    print(f"Functions mentioned: {', '.join(result['functions_mentioned'])}")
    
    # Continue the conversation
    # conversation = [
    #     {"role": "user", "content": "How do I generate a sales report?"},
    #     {"role": "assistant", "content": result['answer']}
    # ]
    
    # follow_up = rag_system.chat(conversation, "What parameters does it accept?")
    # print("\nFOLLOW-UP ANSWER:")
    # print(follow_up['answer'])