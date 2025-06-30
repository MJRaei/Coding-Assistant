"""
Example usage of the Code RAG system with the new modular structure.
Now uses .env file for configuration.

This example shows how to:
1. Build an index from your codebase
2. Ask questions about your code
3. Perform pure semantic search
4. Continue conversations
"""

import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code_rag.indexing import IndexBuilder
from code_rag.retrieval import AnswerGenerator, CodeSearch
from code_rag.config import get_indices_dir, get_logs_dir, DEFAULT_LOG_LEVEL

def setup_logging():
    """Setup logging for the example"""
    log_dir = get_logs_dir()
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, DEFAULT_LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'example.log')),
            logging.StreamHandler()
        ]
    )

def build_index_example():
    """Example of building an index from a codebase"""
    print("=== Building Index ===")
    
    builder = IndexBuilder(
        root_path=r"/Users/mjraei/Desktop/Projects/OplaSmart/engReports"
    )
    
    stats = builder.build_index()
    print(f"Index built successfully!")
    print(f"- Total files processed: {stats['total_files']}")
    print(f"- Total chunks created: {stats['total_chunks']}")
    
    index_path = os.path.join(get_indices_dir(), "reporting_module_index")
    builder.save_index(index_path)
    print(f"Index saved to: {index_path}")
    
    return index_path

def answer_questions_example(index_path: str):
    """Example of asking questions about the code"""
    print("\n=== Asking Questions ===")
    
    answer_gen = AnswerGenerator(
        index_path=index_path
    )
    
    question = "How the pumps output is calculated in the reports?"
    result = answer_gen.ask_question(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Source files: {', '.join(result['source_files'])}")
    print(f"Functions mentioned: {', '.join(result['functions_mentioned'])}")
    
    return answer_gen, result

def conversational_example(answer_gen: AnswerGenerator, previous_result: dict):
    """Example of continuing a conversation"""
    print("\n=== Conversational Follow-up ===")
    
    conversation_history = [
        {"role": "user", "content": "How the pumps output is calculated in the reports?"},
        {"role": "assistant", "content": previous_result['answer']}
    ]
    
    follow_up_question = "What parameters does it accept?"
    follow_up_result = answer_gen.chat(conversation_history, follow_up_question)
    
    print(f"Follow-up Question: {follow_up_question}")
    print(f"Answer: {follow_up_result['answer']}")

def semantic_search_example(index_path: str):
    """Example of pure semantic search without AI generation"""
    print("\n=== Semantic Search ===")
    
    search = CodeSearch(
        index_path=index_path
    )
    
    query = "pump calculation efficiency"
    results = search.search_similar_code(query)
    
    print(f"Search Query: {query}")
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['score']:.3f}) ---")
        print(f"File: {result['file_path']}")
        print(f"Lines: {result['line_range']}")
        print(f"Type: {result['chunk_type']}")
        print(f"Content Preview: {result['content'][:200]}...")
    
    print(f"\n=== Search by Function Name ===")
    function_results = search.search_by_function_name("calculate")
    
    for i, result in enumerate(function_results, 1):
        print(f"\n--- Function Result {i} ---")
        print(f"File: {result['file_path']}")
        print(f"Functions: {result['functions']}")
    
    if results:
        file_path = results[0]['file_path']
        summary = search.get_file_summary(file_path)
        print(f"\n=== File Summary for {file_path} ===")
        print(f"File type: {summary['file_type']}")
        print(f"Total lines: {summary['total_lines']}")
        print(f"Functions: {summary['functions']}")
        print(f"Classes: {summary['classes']}")

def main():
    """Main example function"""
    setup_logging()
    
    print("Code RAG System - Modular Example with .env Configuration")
    print("=" * 60)
    
    try:
        index_path = build_index_example()
        
        answer_gen, result = answer_questions_example(index_path)
        
        conversational_example(answer_gen, result)
        
        semantic_search_example(index_path)
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        print("All configuration was loaded from your .env file!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your .env file is properly configured with OPENAI_API_KEY")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 